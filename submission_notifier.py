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

On each web-form submission it opens a triage ticket on the **S2 Forever /
On-call board** (GitHub Project #64): a GitHub issue opened on ``allenai/scholar``
(by S2 convention, tickets are filed on ``scholar`` even when they reference code
elsewhere), added to the project and set to the ``Triage Needed`` status so it
lands in front of whoever is on-call.

The ticket is the on-call's actionable channel and the only notification this
module sends. The GitHub API is reachable from the HF Space (an internal Ai2
SMTP relay is not), and a card on the board has a lifecycle (assignable,
closeable) a fire-and-forget email did not. Submission volume is low (roughly one
per month or fewer), so one ticket per submission is intentional and manageable;
revisit if volume grows.

**Best-effort.** ``notify_submission`` never raises: any GitHub failure is logged
but does not fail or block the user's submission. With no GitHub token configured
it is a silent no-op -- which is how the notifier is kept *off* on the internal
Space and in CI (only the public Space gets the secret).

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

    if not GITHUB_TOKEN:
        logger.info(
            "notifier: no GitHub token; skipping triage ticket for %s", submission_name
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
    # Local smoke test: fire a real triage ticket to verify token + board wiring
    # end-to-end without going through the Space. Requires NOTIFIER_GITHUB_TOKEN
    # (or GITHUB_TOKEN) in the environment.
    #
    # This opens a REAL issue and a REAL card. To avoid posting to the live
    # on-call board, point it at a throwaway repo/board via env, e.g.:
    #
    #   NOTIFIER_GITHUB_TOKEN=... \
    #   NOTIFIER_GITHUB_REPO=<you>/<sandbox-repo> \
    #   NOTIFIER_PROJECT_ORG=<you> NOTIFIER_PROJECT_NUMBER=<n> \
    #   NOTIFIER_PROJECT_STATUS="<a status option on that board>" \
    #   python submission_notifier.py
    #
    # With the defaults it files against allenai/scholar + Project #64 -- only do
    # that as a deliberate production check, and close the test card afterwards.
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
