"""Open a GitHub triage ticket when a new web-form submission lands on the leaderboard.

Called in-process from ``submission.add_new_eval`` after a submission uploads. It
opens an issue on a configured repo and optionally attaches it to a GitHub Project
board. Where the ticket goes is set entirely via environment / Space secrets (see
Configuration); with no token/repo set it is a silent no-op, and ``notify_submission``
is best-effort and never raises. See ``docs/submission-notifier.md`` for deployment.
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)

# --- Configuration (all via env / HF Space secrets) ------------------------
#   NOTIFIER_GITHUB_TOKEN    token to open the issue (and edit the board); falls
#                            back to GITHUB_TOKEN. Without it (or the repo), no-op.
#   NOTIFIER_GITHUB_REPO     "owner/repo" to open the issue on. Without it, no-op.
#   NOTIFIER_PROJECT_NUMBER  optional Project to attach the issue to; omit for a
#                            plain issue (no board).
#   NOTIFIER_PROJECT_ORG     org that owns the Project; defaults to the repo owner.
#   NOTIFIER_PROJECT_STATUS  status column to set; defaults to "Triage Needed".
#                            Empty -> add the card without setting status.
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


def build_issue(coords):
    """(title, body) for the on-call triage ticket."""
    dataset_url = (
        f"hf://datasets/{coords['repo']}/{coords['config']}/"
        f"{coords['split']}/{coords['name']}"
    )
    title = f"AstaBench submission from {coords['username']} ({coords['split']})"
    body = "\n".join(
        [
            f'Submission from hf user "{coords["username"]}" to "{dataset_url}"',
            "",
            f"Folder: {submission_folder_url(coords)}",
            "",
            "Filed automatically by the leaderboard submission notifier; "
            "needs on-call review.",
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
    username=None,
):
    """Open an on-call triage ticket for a new web-form submission.

    Called in-process from ``submission.add_new_eval`` after a successful upload.
    Returns the issue URL on success, or ``None`` when unconfigured or on failure.

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

    if not GITHUB_TOKEN or not ISSUE_REPO:
        logger.info(
            "notifier: not configured (need NOTIFIER_GITHUB_TOKEN + "
            "NOTIFIER_GITHUB_REPO); skipping ticket for %s",
            submission_name,
        )
        return None

    try:
        title, body = build_issue(coords)
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
        username="local-tester",
    )
    print(
        f"triage ticket: {url}"
        if url
        else "no ticket created -- set NOTIFIER_GITHUB_TOKEN and check the logs above"
    )
