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

On each web-form submission it opens a GitHub issue on a configured repo and,
optionally, attaches it to a GitHub Project board under a chosen status column,
so it lands in front of whoever is on call. Where the ticket goes is set entirely
through environment / Space secrets (see Configuration below) -- the module
carries no deployment-specific values.

The ticket is the actionable channel and the only notification this module sends.
A board card has a lifecycle (assignable, closeable) that a fire-and-forget email
did not, and the GitHub API is reachable from anywhere a HF Space runs. Submission
volume is low (roughly one per month or fewer), so one ticket per submission is
intentional and manageable; revisit if volume grows.

**Best-effort.** ``notify_submission`` never raises: any GitHub failure is logged
but does not fail or block the user's submission. With no token / repo configured
it is a silent no-op -- which is how the notifier is kept *off* where it should be
(e.g. an internal-only Space, or CI): only deployments that set the secrets fire.

Privacy: the submitter's email is **never** put in the (org-visible) ticket. It
remains only in the private contact-info dataset, which the ticket points the
on-call to.

Deployment / secret wiring is documented in ``docs/submission-notifier.md``.
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)

# --- Configuration (all via env / HF Space secrets) ------------------------
#
# The notifier carries *no* deployment-specific values in code: where tickets go
# is set entirely through Space secrets, so a fork can point it at its own repo
# and board without editing this file.
#
#   NOTIFIER_GITHUB_TOKEN   token to open the issue (and edit the board, if set).
#                           Falls back to GITHUB_TOKEN. Without it, no-op.
#   NOTIFIER_GITHUB_REPO    "owner/repo" to open the issue on. Without it, no-op.
#   NOTIFIER_PROJECT_NUMBER optional org/user Project to attach the issue to;
#                           omit to just open a plain issue (no board).
#   NOTIFIER_PROJECT_ORG    org that owns the Project; defaults to the repo owner.
#   NOTIFIER_PROJECT_STATUS status column to set once attached; defaults to
#                           "Triage Needed". Set empty to add the card without
#                           touching its status. Ignored when no Project is set.
#
# When a Project is configured its node IDs are resolved at runtime from the org
# + number (see _resolve_project), so they never go stale if the board changes.
GITHUB_TOKEN = os.getenv("NOTIFIER_GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN")
ISSUE_REPO = os.getenv("NOTIFIER_GITHUB_REPO", "")

_project_number = os.getenv("NOTIFIER_PROJECT_NUMBER")
PROJECT_NUMBER = int(_project_number) if _project_number else None
PROJECT_ORG = os.getenv("NOTIFIER_PROJECT_ORG") or (
    ISSUE_REPO.split("/", 1)[0] if "/" in ISSUE_REPO else ""
)
PROJECT_STATUS = os.getenv("NOTIFIER_PROJECT_STATUS", "Triage Needed")


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


def build_issue(coords, meta):
    """(title, body) for the on-call triage ticket.

    PII-free: the submitter email is deliberately omitted because the ticket is
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
    node IDs can never go stale if the board is rebuilt. When PROJECT_STATUS is
    empty, the status field/option are returned as None (card added, status
    untouched).
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
    if not PROJECT_STATUS:
        return project["id"], None, None
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

    # No Project configured -> a plain issue is the whole notification.
    if PROJECT_NUMBER is None:
        return issue["html_url"]

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
    if option_id is not None:
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
    """Open an on-call triage ticket for a new web-form submission.

    Called in-process from ``submission.add_new_eval`` after a successful upload,
    with the submission metadata it already has in hand. Returns the issue URL on
    success, or ``None`` when unconfigured or on failure.

    Best-effort: every failure is logged but never raised, so a notifier problem
    can never fail or block the user's submission. With no GitHub token
    configured (the internal Space, CI) this is a silent no-op.
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

    if not GITHUB_TOKEN or not ISSUE_REPO:
        logger.info(
            "notifier: not configured (need NOTIFIER_GITHUB_TOKEN + "
            "NOTIFIER_GITHUB_REPO); skipping ticket for %s",
            submission_name,
        )
        return None

    try:
        title, body = build_issue(coords, meta)
        issue_url = create_triage_ticket(title, body)
        logger.info(
            "notifier: opened triage ticket %s for %s", issue_url, submission_name
        )
        return issue_url
    except Exception as e:
        logger.error(
            "notifier: triage ticket creation failed for %s: %s", submission_name, e
        )
        return None


if __name__ == "__main__":
    # Local smoke test: fire a real ticket to verify the token + board wiring
    # end-to-end without going through the Space. Requires at least a token and a
    # repo in the environment; add a Project to also exercise the board path. It
    # opens a REAL issue (and a REAL card if a Project is set), so point it at a
    # throwaway repo/board you own and close the test ticket afterwards:
    #
    #   NOTIFIER_GITHUB_TOKEN=... \
    #   NOTIFIER_GITHUB_REPO=<you>/<sandbox-repo> \
    #   NOTIFIER_PROJECT_NUMBER=<n> \
    #   NOTIFIER_PROJECT_STATUS="<a status option on that board>" \
    #   python submission_notifier.py
    #
    # Omit NOTIFIER_PROJECT_NUMBER to test the plain-issue path only.
    logging.basicConfig(level=logging.INFO)
    url = notify_submission(
        submission_dataset="allenai/asta-bench-submissions",
        config_name="1.0.0-dev1",
        split="test",
        submission_name="smoketest_LocalAgent_2026-06-18_00-00-00",
        agent_name="LocalAgent (notifier smoke test)",
        agent_description="Local verification of the submission notifier -- safe to close.",
        agent_url="https://example.com",
        openness="Open Source",
        tool_usage="Standard",
        username="local-tester",
        submit_time="2026-06-18T00:00:00+00:00",
    )
    print(
        f"triage ticket: {url}"
        if url
        else "no ticket created -- set NOTIFIER_GITHUB_TOKEN and check the logs above"
    )
