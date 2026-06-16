"""Notify the AstaBench team when a new leaderboard submission lands.

Background: submissions to https://huggingface.co/spaces/allenai/asta-bench-leaderboard
push commits to the HF datasets backing the leaderboard, but the team got *no*
signal when one appeared (see allenai/asta-bench-leaderboard#119,
allenai/gas2own#86). This module wires a HuggingFace webhook into the existing
Gradio Space so that every submission commit fans out to:

  * Slack ``#asta-bench`` (no submitter email -- privacy), and
  * email ``asta-support@allenai.org`` (subject carries HF username + agent name
    so it is greppable in the mailbox; body includes the submitter email).

Deployment / secret wiring is documented in ``docs/submission-notifier.md``.

The webhook payload itself does not carry the commit message or changed files,
so on each relevant commit we read the latest commit message (which encodes the
submission folder URL) and then read the per-submission ``submission.json``
metadata from the submissions dataset. Both the public and internal datasets, as
well as the private contact-info dataset, use the same commit-message shape
(``Submission from hf user "X" to "hf://datasets/.../{config}/{split}/{name}"``),
so a commit on any of them resolves to the same submission and dedupes to one
notification.
"""

import json
import logging
import os
import re
import smtplib
import tempfile
from email.message import EmailMessage

import requests
from huggingface_hub import HfApi, hf_hub_download

logger = logging.getLogger(__name__)

# --- Configuration (all via env / HF Space secrets) ------------------------

# Datasets we notify on. The private contact-info dataset receives a commit for
# *every* submission (including opt-out submitters), so it is the authoritative
# trigger; the public/internal submission datasets are included so a notification
# still fires if only those commit.
NOTIFY_REPOS = {
    "allenai/asta-bench-submissions",
    "allenai/asta-bench-internal-submissions",
    "allenai/asta-bench-internal-contact-info",
}

SLACK_CHANNEL = os.getenv("NOTIFIER_SLACK_CHANNEL", "#asta-bench")
EMAIL_TO = os.getenv("NOTIFIER_EMAIL_TO", "asta-support@allenai.org")

# The read-only token minted for the notifier (see #119). Falls back to the
# Space's own HF_TOKEN so local/dev runs work without extra wiring.
HF_TOKEN = os.getenv("NOTIFIER_HF_TOKEN") or os.getenv("HF_TOKEN")

# Persisted dedup state. DATA_DIR is created by the Space; fall back to a temp
# dir so importing this module never fails outside the Space.
try:
    from config import DATA_DIR

    _STATE_DIR = DATA_DIR
except Exception:  # pragma: no cover - config import only fails off-Space
    _STATE_DIR = tempfile.gettempdir()
_SEEN_PATH = os.path.join(_STATE_DIR, "notifier_seen_submissions.json")

# Commit message shape from submission.py:upload_submission and the contact-info
# push_to_hub: ... to "hf://datasets/{repo}/{config}/{split}/{submission_name}"
_COMMIT_RE = re.compile(
    r'from hf user "(?P<username>[^"]*)" to '
    r'"hf://datasets/(?P<repo>[^/]+/[^/]+)/(?P<config>[^/]+)/'
    r'(?P<split>[^/]+)/(?P<name>[^/"]+)"'
)


# --- Dedup -----------------------------------------------------------------


def _load_seen():
    try:
        with open(_SEEN_PATH, "r", encoding="utf-8") as fp:
            return set(json.load(fp))
    except (FileNotFoundError, ValueError):
        return set()


def _save_seen(seen):
    try:
        os.makedirs(_STATE_DIR, exist_ok=True)
        with open(_SEEN_PATH, "w", encoding="utf-8") as fp:
            json.dump(sorted(seen), fp)
    except OSError as e:  # pragma: no cover - best effort
        logger.warning("notifier: could not persist dedup state: %s", e)


# --- Parsing / metadata -----------------------------------------------------


def parse_commit_message(message):
    """Pull submitter + submission-folder coordinates out of a commit message.

    Returns a dict with ``username``, ``repo``, ``config``, ``split``, ``name``
    and a synthesized ``dataset_url``, or ``None`` if the message is not a
    submission commit (e.g. a manual edit or a results-scoring commit).
    """
    m = _COMMIT_RE.search(message or "")
    if not m:
        return None
    d = m.groupdict()
    d["dataset_url"] = (
        f"hf://datasets/{d['repo']}/{d['config']}/{d['split']}/{d['name']}"
    )
    return d


def submission_folder_url(coords):
    """Browser URL for the submission folder on the HF Hub."""
    return (
        f"https://huggingface.co/datasets/{coords['repo']}/tree/main/"
        f"{coords['config']}/{coords['split']}/{coords['name']}"
    )


def _metadata_filename():
    try:
        from agenteval.cli import SUBMISSION_METADATA_FILENAME

        return SUBMISSION_METADATA_FILENAME
    except Exception:  # pragma: no cover - agenteval only present on the Space
        return "submission.json"


# The submission datasets we can read submission.json from (the contact-info
# dataset stores metadata as parquet rows instead, handled separately).
_SUBMISSION_DATASETS = {
    "allenai/asta-bench-submissions",
    "allenai/asta-bench-internal-submissions",
}


def fetch_submission_metadata(coords):
    """Read the per-submission ``submission.json`` for these coords.

    Tries the submissions dataset named in the commit first, then its twin, so a
    contact-info commit (which has no submission.json) still resolves metadata.
    Returns a dict of the metadata fields, or ``{}`` if it cannot be read.
    """
    filename = _metadata_filename()
    candidates = [coords["repo"]] if coords["repo"] in _SUBMISSION_DATASETS else []
    candidates += [r for r in _SUBMISSION_DATASETS if r not in candidates]
    path_in_repo = f"{coords['config']}/{coords['split']}/{coords['name']}/{filename}"
    for repo in candidates:
        try:
            local = hf_hub_download(
                repo_id=repo,
                repo_type="dataset",
                filename=path_in_repo,
                token=HF_TOKEN,
            )
            with open(local, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception as e:
            logger.debug("notifier: no submission.json in %s: %s", repo, e)
    return {}


def fetch_submitter_email(coords):
    """Look up the submitter email from the private contact-info dataset.

    Matches on ``dataset_url`` (unique per submission). Returns ``None`` if not
    found or unreadable -- email notifications still go out, just without it.
    """
    try:
        from datasets import load_dataset

        ds = load_dataset(
            "allenai/asta-bench-internal-contact-info",
            coords["config"],
            split=coords["split"],
            token=HF_TOKEN,
            download_mode="force_redownload",
        )
    except Exception as e:
        logger.warning("notifier: could not load contact-info dataset: %s", e)
        return None
    for row in reversed(list(ds)):
        if row.get("dataset_url") == coords["dataset_url"]:
            return row.get("email")
    return None


# --- Message rendering ------------------------------------------------------


def _field(meta, *keys, default="(not provided)"):
    for k in keys:
        v = meta.get(k)
        if v not in (None, ""):
            return v
    return default


def build_slack_payload(coords, meta):
    """Slack ``chat.postMessage`` body. Never includes the submitter email."""
    agent_name = _field(meta, "agent_name", default=coords["name"])
    lines = [
        f"*New AstaBench submission: {agent_name}*",
        f"• *Submitter (HF):* `{_field(meta, 'username', default=coords['username'])}`",
        f"• *Track / split:* {coords['split']} (config `{coords['config']}`)",
        f"• *Description:* {_field(meta, 'agent_description')}",
        f"• *Agent URL:* {_field(meta, 'agent_url')}",
        f"• *Openness:* {_field(meta, 'openness')}",
        f"• *Tool usage:* {_field(meta, 'tool_usage', 'degree_of_control')}",
        f"• *Submitted:* {_field(meta, 'submit_time')}",
        f"• *Submission folder:* {submission_folder_url(coords)}",
    ]
    # cost cap / model are only present once scoring runs; surface if available.
    for label, *keys in (("Cost cap", "cost_cap", "cost"), ("Model", "model")):
        val = _field(meta, *keys, default=None)
        if val:
            lines.append(f"• *{label}:* {val}")
    return {"channel": SLACK_CHANNEL, "text": "\n".join(lines)}


def build_email(coords, meta, submitter_email):
    """(subject, body) for the asta-support notification email."""
    agent_name = _field(meta, "agent_name", default=coords["name"])
    username = _field(meta, "username", default=coords["username"])
    subject = f"[AstaBench submission] {username} — {agent_name} ({coords['split']})"
    body = "\n".join(
        [
            "A new submission landed on the AstaBench leaderboard.",
            "",
            f"Submitter (HF): {username}",
            f"Submitter email: {submitter_email or '(not found)'}",
            f"Agent name: {agent_name}",
            f"Description: {_field(meta, 'agent_description')}",
            f"Agent URL: {_field(meta, 'agent_url')}",
            f"Track / split: {coords['split']} (config {coords['config']})",
            f"Openness: {_field(meta, 'openness')}",
            f"Tool usage: {_field(meta, 'tool_usage', 'degree_of_control')}",
            f"Submitted: {_field(meta, 'submit_time')}",
            "",
            f"Submission folder: {submission_folder_url(coords)}",
        ]
    )
    return subject, body


# --- Delivery ---------------------------------------------------------------


def send_slack(payload):
    token = os.getenv("SLACK_BOT_TOKEN")
    if token:
        resp = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=15,
        )
        data = resp.json()
        if not data.get("ok"):
            raise RuntimeError(f"Slack API error: {data.get('error')}")
        return
    url = os.getenv("SLACK_WEBHOOK_URL")
    if url:
        resp = requests.post(url, json={"text": payload["text"]}, timeout=15)
        resp.raise_for_status()
        return
    raise RuntimeError("No SLACK_BOT_TOKEN or SLACK_WEBHOOK_URL configured")


def send_email(subject, body):
    host = os.getenv("SMTP_HOST")
    if not host:
        raise RuntimeError("No SMTP_HOST configured")
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.getenv("NOTIFIER_EMAIL_FROM", "asta-support@allenai.org")
    msg["To"] = EMAIL_TO
    msg.set_content(body)
    port = int(os.getenv("SMTP_PORT", "587"))
    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.ehlo()
        if os.getenv("SMTP_STARTTLS", "true").lower() == "true":
            smtp.starttls()
            smtp.ehlo()
        user, password = os.getenv("SMTP_USERNAME"), os.getenv("SMTP_PASSWORD")
        if user and password:
            smtp.login(user, password)
        smtp.send_message(msg)


# --- Orchestration ----------------------------------------------------------


def handle_commit(repo_id, head_sha=None):
    """Process one dataset commit; returns True if a notification was sent.

    Errors in any single channel are logged but do not prevent the other, and
    never raise out of the webhook handler (HF would otherwise mark delivery
    failed and retry, duplicating work).
    """
    if repo_id not in NOTIFY_REPOS:
        logger.info("notifier: ignoring commit on unrelated repo %s", repo_id)
        return False

    api = HfApi(token=HF_TOKEN)
    try:
        commits = api.list_repo_commits(repo_id, repo_type="dataset")
        message = commits[0].title if commits else ""
    except Exception as e:
        logger.error("notifier: could not read commits for %s: %s", repo_id, e)
        return False

    coords = parse_commit_message(message)
    if not coords:
        logger.info("notifier: commit on %s is not a submission: %r", repo_id, message)
        return False

    seen = _load_seen()
    if coords["dataset_url"] in seen:
        logger.info("notifier: already notified for %s", coords["dataset_url"])
        return False

    meta = fetch_submission_metadata(coords)
    submitter_email = fetch_submitter_email(coords)

    ok = False
    try:
        send_slack(build_slack_payload(coords, meta))
        ok = True
    except Exception as e:
        logger.error(
            "notifier: Slack delivery failed for %s: %s", coords["dataset_url"], e
        )
    try:
        subject, body = build_email(coords, meta, submitter_email)
        send_email(subject, body)
        ok = True
    except Exception as e:
        logger.error(
            "notifier: email delivery failed for %s: %s", coords["dataset_url"], e
        )

    # Only mark seen once at least one channel delivered, so a fully-failed
    # commit can be retried by HF rather than silently dropped.
    if ok:
        seen.add(coords["dataset_url"])
        _save_seen(seen)
    return ok


def attach_to(demo, webhook_secret=None):
    """Wrap a Gradio Blocks ``demo`` in a HuggingFace WebhooksServer.

    Returns the server (whose ``.launch`` mirrors ``demo.launch``) when a
    webhook secret is available, else returns ``None`` so the caller can fall
    back to a plain ``demo.launch()``. Register the webhook on each dataset in
    NOTIFY_REPOS pointing at ``<space-url>/webhooks/submissions``.
    """
    secret = (
        webhook_secret
        or os.getenv("NOTIFIER_WEBHOOK_SECRET")
        or os.getenv("WEBHOOK_SECRET")
    )
    if not secret:
        logger.warning("notifier: no webhook secret set; webhook endpoint disabled")
        return None

    from huggingface_hub import WebhookPayload, WebhooksServer

    app = WebhooksServer(ui=demo, webhook_secret=secret)

    @app.add_webhook("/webhooks/submissions")
    async def _on_submission(payload: WebhookPayload):
        try:
            if payload.event.action != "update" or not payload.event.scope.startswith(
                "repo"
            ):
                return {"processed": False, "reason": "ignored event"}
            head = getattr(payload.repo, "head_sha", None)
            sent = handle_commit(payload.repo.name, head)
            return {"processed": sent}
        except Exception as e:  # never let the handler 500 -> HF retry storm
            logger.exception("notifier: unhandled error: %s", e)
            return {"processed": False, "error": str(e)}

    return app
