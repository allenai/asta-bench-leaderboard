"""Notify the AstaBench team when a new leaderboard submission lands.

Background: submissions to https://huggingface.co/spaces/allenai/asta-bench-leaderboard
push commits to the HF datasets backing the leaderboard, but the team got *no*
signal when one appeared (see allenai/asta-bench-leaderboard#119,
allenai/gas2own#86). This module wires a HuggingFace webhook into the existing
Gradio Space so that every submission commit fans out to:

  * Slack ``#asta-bench`` -- ambient awareness, and
  * a triage ticket on the **S2 Forever / On-call board** (GitHub Project #64):
    a GitHub issue opened on ``allenai/scholar`` (by S2 convention, tickets are
    filed on ``scholar`` even when they reference code elsewhere), added to the
    project and set to the ``Triage Needed`` status so it lands in front of
    whoever is on-call.

The ticket is the actionable channel that replaced an earlier email design: the
GitHub API is reachable from the HF Space (an internal Ai2 SMTP relay is not),
and a card on the board has a lifecycle (assignable, closeable) a
fire-and-forget email did not. Submission volume is low (roughly one per month
or fewer), so one ticket per submission is intentional and manageable; revisit
if volume grows.

Privacy: the submitter's email is **never** posted to Slack and **never** put in
the (org-visible) ticket. It remains only in the private contact-info dataset,
which the ticket points the on-call to.

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
import tempfile

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

# GitHub triage ticket. By S2 convention tickets are filed on allenai/scholar
# even when the code lives elsewhere; the issue is then added to the S2 Forever /
# On-call board (Project #64) under the "Triage Needed" status. All overridable
# via env so the target can move without a code change. The board IDs are
# resolved at runtime from the org + project number (see _resolve_project), so
# they never go stale.
ISSUE_REPO = os.getenv("NOTIFIER_GITHUB_REPO", "allenai/scholar")
PROJECT_ORG = os.getenv("NOTIFIER_PROJECT_ORG", "allenai")
PROJECT_NUMBER = int(os.getenv("NOTIFIER_PROJECT_NUMBER", "64"))
PROJECT_STATUS = os.getenv("NOTIFIER_PROJECT_STATUS", "Triage Needed")

# Token for opening the issue and editing the project. Needs repo + project
# scope (a classic PAT with repo+project, or a fine-grained token with Issues:RW
# and Projects:RW on allenai). Falls back to a generic GITHUB_TOKEN if present.
GITHUB_TOKEN = os.getenv("NOTIFIER_GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN")

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


def build_issue(coords, meta):
    """(title, body) for the on-call triage ticket.

    PII-free: the submitter email is deliberately omitted because Project #64 is
    org-visible. The on-call looks it up in the private contact-info dataset,
    which the body points to.
    """
    agent_name = _field(meta, "agent_name", default=coords["name"])
    username = _field(meta, "username", default=coords["username"])
    title = f"AstaBench submission: {agent_name} ({coords['split']})"
    body = "\n".join(
        [
            "A new submission landed on the AstaBench leaderboard and needs "
            "triage by the on-call.",
            "",
            f"- **Submitter (HF):** `{username}`",
            f"- **Agent name:** {agent_name}",
            f"- **Description:** {_field(meta, 'agent_description')}",
            f"- **Agent URL:** {_field(meta, 'agent_url')}",
            f"- **Track / split:** {coords['split']} (config `{coords['config']}`)",
            f"- **Openness:** {_field(meta, 'openness')}",
            f"- **Tool usage:** {_field(meta, 'tool_usage', 'degree_of_control')}",
            f"- **Submitted:** {_field(meta, 'submit_time')}",
            f"- **Submission folder:** {submission_folder_url(coords)}",
            "",
            "The submitter's email is intentionally omitted from this org-visible "
            "ticket. It is recorded in the private contact-info dataset "
            "`allenai/asta-bench-internal-contact-info` "
            f"(config `{coords['config']}`, split `{coords['split']}`).",
            "",
            "_Filed automatically by the leaderboard submission notifier "
            "(allenai/asta-bench-leaderboard#119)._",
        ]
    )
    return title, body


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


_GITHUB_API = "https://api.github.com"
_GITHUB_GRAPHQL = "https://api.github.com/graphql"


def _gh_headers():
    if not GITHUB_TOKEN:
        raise RuntimeError(
            "No NOTIFIER_GITHUB_TOKEN (or GITHUB_TOKEN) configured for triage tickets"
        )
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _graphql(query, variables):
    resp = requests.post(
        _GITHUB_GRAPHQL,
        headers=_gh_headers(),
        json={"query": query, "variables": variables},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("errors"):
        raise RuntimeError(f"GitHub GraphQL error: {data['errors']}")
    return data["data"]


def _resolve_project():
    """Resolve (project_id, status_field_id, option_id) for the configured board.

    Done at runtime from PROJECT_ORG / PROJECT_NUMBER / PROJECT_STATUS so the
    node IDs can never go stale if the board is rebuilt.
    """
    data = _graphql(
        """
        query($org: String!, $number: Int!) {
          organization(login: $org) {
            projectV2(number: $number) {
              id
              field(name: "Status") {
                ... on ProjectV2SingleSelectField {
                  id
                  options { id name }
                }
              }
            }
          }
        }
        """,
        {"org": PROJECT_ORG, "number": PROJECT_NUMBER},
    )
    project = (data.get("organization") or {}).get("projectV2")
    if not project:
        raise RuntimeError(f"project #{PROJECT_NUMBER} not found in org {PROJECT_ORG}")
    field = project.get("field")
    if not field:
        raise RuntimeError(
            f"project #{PROJECT_NUMBER} has no single-select Status field"
        )
    option = next((o for o in field["options"] if o["name"] == PROJECT_STATUS), None)
    if option is None:
        raise RuntimeError(
            f"status option {PROJECT_STATUS!r} not found on project #{PROJECT_NUMBER}"
        )
    return project["id"], field["id"], option["id"]


def create_triage_ticket(title, body):
    """Open the issue on ISSUE_REPO, add it to the board, set status; return URL."""
    owner, repo = ISSUE_REPO.split("/", 1)
    resp = requests.post(
        f"{_GITHUB_API}/repos/{owner}/{repo}/issues",
        headers=_gh_headers(),
        json={"title": title, "body": body},
        timeout=20,
    )
    resp.raise_for_status()
    issue = resp.json()

    project_id, status_field_id, option_id = _resolve_project()
    added = _graphql(
        """
        mutation($project: ID!, $content: ID!) {
          addProjectV2ItemById(input: {projectId: $project, contentId: $content}) {
            item { id }
          }
        }
        """,
        {"project": project_id, "content": issue["node_id"]},
    )
    item_id = added["addProjectV2ItemById"]["item"]["id"]
    _graphql(
        """
        mutation($project: ID!, $item: ID!, $field: ID!, $option: String!) {
          updateProjectV2ItemFieldValue(input: {
            projectId: $project, itemId: $item, fieldId: $field,
            value: {singleSelectOptionId: $option}
          }) { projectV2Item { id } }
        }
        """,
        {
            "project": project_id,
            "item": item_id,
            "field": status_field_id,
            "option": option_id,
        },
    )
    return issue["html_url"]


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

    ok = False
    try:
        send_slack(build_slack_payload(coords, meta))
        ok = True
    except Exception as e:
        logger.error(
            "notifier: Slack delivery failed for %s: %s", coords["dataset_url"], e
        )
    try:
        title, body = build_issue(coords, meta)
        issue_url = create_triage_ticket(title, body)
        logger.info(
            "notifier: opened triage ticket %s for %s", issue_url, coords["dataset_url"]
        )
        ok = True
    except Exception as e:
        logger.error(
            "notifier: triage ticket creation failed for %s: %s",
            coords["dataset_url"],
            e,
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
