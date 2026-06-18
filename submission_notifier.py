"""Notify the AstaBench on-call when a new web-form submission lands.

Background: submissions to https://huggingface.co/spaces/allenai/asta-bench-leaderboard
need human review by the on-call (per the submission form, runs are reviewed
within 5-7 business days), but the team got *no* signal when one appeared (see
allenai/asta-bench-leaderboard#119, allenai/gas2own#86).

**Trigger.** This module is called *in-process from* ``submission.add_new_eval``
right after a submission's files are uploaded. ``add_new_eval`` runs only when a
logged-in user actually submits through the web form, so it is exactly the
on-call-actionable event: rescoring, programmatic/API dataset writes, and
internal users self-publishing results never enter that function, so they never
notify. This is why we fire here rather than on a HuggingFace dataset-commit
webhook -- the form is the precise signal, and the submitter metadata is
first-class in-process data rather than something reconstructed from a commit
message after the fact.

On each web-form submission it fans out to:

  * Slack ``#asta-bench`` -- ambient awareness, and
  * a triage ticket on the **S2 Forever / On-call board** (GitHub Project #64):
    a GitHub issue opened on ``allenai/scholar`` (by S2 convention, tickets are
    filed on ``scholar`` even when they reference code elsewhere), added to the
    project and set to the ``Triage Needed`` status so it lands in front of
    whoever is on-call.

The ticket is the actionable channel; the GitHub API is reachable from the HF
Space (an internal Ai2 SMTP relay is not), and a card on the board has a
lifecycle (assignable, closeable) a fire-and-forget email did not. Submission
volume is low (roughly one per month or fewer), so one ticket per submission is
intentional and manageable; revisit if volume grows.

**Best-effort.** ``notify_submission`` never raises: any Slack/GitHub failure is
logged but does not fail or block the user's submission. With no Slack or GitHub
credentials configured it is a silent no-op -- which is how the notifier is kept
*off* on the internal Space and in CI (only the public Space gets the secrets).

Privacy: the submitter's email is **never** posted to Slack and **never** put in
the (org-visible) ticket. It remains only in the private contact-info dataset,
which the ticket points the on-call to.

Deployment / secret wiring is documented in ``docs/submission-notifier.md``.
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)

# --- Configuration (all via env / HF Space secrets) ------------------------

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


def submission_folder_url(coords):
    """Browser URL for the submission folder on the HF Hub."""
    return (
        f"https://huggingface.co/datasets/{coords['repo']}/tree/main/"
        f"{coords['config']}/{coords['split']}/{coords['name']}"
    )


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
            "A new submission was made through the AstaBench leaderboard web "
            "form and needs review by the on-call.",
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


def _slack_configured():
    return bool(os.getenv("SLACK_BOT_TOKEN") or os.getenv("SLACK_WEBHOOK_URL"))


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


def notify_submission(
    *,
    submission_dataset,
    config_name,
    split,
    submission_name,
    agent_name=None,
    agent_description=None,
    agent_url=None,
    openness=None,
    tool_usage=None,
    username=None,
    submit_time=None,
):
    """Fan out a new web-form submission to Slack + an on-call triage ticket.

    Called in-process from ``submission.add_new_eval`` after a successful upload,
    with the submission metadata it already has in hand. Returns the list of
    channels that delivered (e.g. ``["slack", "ticket"]``).

    Best-effort: each channel is independent, every failure is logged but never
    raised, so a notifier problem can never fail or block the user's submission.
    A channel with no credentials configured is skipped, so with neither Slack
    nor GitHub configured (the internal Space, CI) this is a silent no-op.
    """
    coords = {
        "repo": submission_dataset,
        "config": config_name,
        "split": split,
        "name": submission_name,
        "username": username or "",
    }
    meta = {
        "agent_name": agent_name,
        "agent_description": agent_description,
        "agent_url": agent_url,
        "openness": openness,
        "tool_usage": tool_usage,
        "username": username,
        "submit_time": str(submit_time) if submit_time is not None else None,
    }

    delivered = []
    if _slack_configured():
        try:
            send_slack(build_slack_payload(coords, meta))
            delivered.append("slack")
        except Exception as e:
            logger.error(
                "notifier: Slack delivery failed for %s: %s", submission_name, e
            )
    else:
        logger.info(
            "notifier: no Slack credentials; skipping Slack for %s", submission_name
        )

    if GITHUB_TOKEN:
        try:
            title, body = build_issue(coords, meta)
            issue_url = create_triage_ticket(title, body)
            logger.info(
                "notifier: opened triage ticket %s for %s", issue_url, submission_name
            )
            delivered.append("ticket")
        except Exception as e:
            logger.error(
                "notifier: triage ticket creation failed for %s: %s", submission_name, e
            )
    else:
        logger.info(
            "notifier: no GitHub token; skipping triage ticket for %s", submission_name
        )

    return delivered
