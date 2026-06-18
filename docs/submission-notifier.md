# Submission notifier

Opens a GitHub ticket whenever someone submits an agent through the leaderboard
web form, so submissions that need human review (scoring, publishing) don't sit
unnoticed. Optional and entirely configured through Space secrets — set none and
the leaderboard behaves exactly as before. Implements
allenai/asta-bench-leaderboard#119.

## How it works

The notifier fires **in-process from the leaderboard app**, not from a webhook.
`submission.add_new_eval` — the function the web form runs — calls
`submission_notifier.notify_submission(...)` right after a submission's files are
uploaded, passing the submission coordinates it already has in hand.

`add_new_eval` runs **only when the web form is used**, which is what makes it the
right trigger: rescoring writes the *results* datasets, programmatic/API writes
don't go through the form, and users who self-publish results never call it. So
"the form ran" is exactly the actionable event, and the submitter identity is the
OAuth-verified username rather than something parsed from a commit message.

For each submission it opens an issue on the configured repo and, if a Project is
configured, attaches the issue to that board and sets a status column. The issue
is intentionally minimal — the submitter's HF username and the submission's
dataset path (the same line that appears on the HF dataset commit), plus a link
to the submission folder. The Project's node IDs are resolved **at runtime** from
the org + number, so they don't go stale if the board is rebuilt.

`notify_submission` is **best-effort**: any failure is logged (`notifier: ...` in
the Space logs) but never raised, so a notifier problem can never block or fail a
user's submission.

**Privacy:** the submitter's email is **never** placed in the (org-visible)
ticket. It stays only in the private contact-info dataset.

## Configuration

All settings come from environment variables / Space secrets — the code carries
no deployment-specific values, so a fork can point the notifier at its own repo
and board without editing it.

| Secret | Required | Notes |
|---|---|---|
| `NOTIFIER_GITHUB_TOKEN` | to fire | Token to open issues (and edit the board, if used). Classic PAT with `repo` + `project` scope, or a fine-grained token with Issues: R/W on the repo and Projects: R/W on the org. Falls back to `GITHUB_TOKEN`. |
| `NOTIFIER_GITHUB_REPO` | to fire | `owner/repo` to open the issue on. |
| `NOTIFIER_PROJECT_NUMBER` | no | Project (board) number to attach the issue to. Omit to open a plain issue with no board. |
| `NOTIFIER_PROJECT_ORG` | no | Org that owns the Project. Defaults to the owner of `NOTIFIER_GITHUB_REPO`. |
| `NOTIFIER_PROJECT_STATUS` | no | Status column to set after attaching. Defaults to `Triage Needed`; set empty to add the card without touching its status. |

**Minimum to fire:** a token and a repo — that opens a plain issue per
submission. Add `NOTIFIER_PROJECT_NUMBER` to also land it on a board.

**Disabled by default:** with no token or no repo, `notify_submission` is a
silent no-op. This is how the notifier stays off where it should be — e.g. an
internal-only Space, or CI, where the secrets simply aren't set — without any
code change. The token owner authors every issue, so use a bot/service account
if you don't want a person's name on each ticket.

> Deployment specifics for the Ai2/Semantic Scholar instance (which repo, board,
> and credential to use) live in allenai/gas2own#86, not here.

## Verifying

### Locally (without deploying)

`submission_notifier.py` has a `__main__` smoke test that fires one real ticket
end-to-end, so you can confirm the wiring before go-live. It opens a **real**
issue (and a **real** card if a Project is set), so point it at a throwaway
repo/board you own and close the test ticket after:

```bash
pip install requests
NOTIFIER_GITHUB_TOKEN=<token> \
NOTIFIER_GITHUB_REPO=<you>/<sandbox-repo> \
NOTIFIER_PROJECT_NUMBER=<n> \
NOTIFIER_PROJECT_STATUS="<a status option on that board>" \
python submission_notifier.py
```

Omit `NOTIFIER_PROJECT_NUMBER` to test the plain-issue path only. It prints the
created issue URL, or a hint to check the logs if no token/repo is set.

### Through the deployed Space

1. Make a test submission through the Space's submission form.
2. Confirm a new issue on the configured repo (and a card on the board, if set).
3. Any error is logged to the Space logs (`notifier: ...`) without affecting the
   submitter's experience.

## Limitations

- One ticket fires per successful web-form submission. At low (≈monthly) volume
  this is intended; add a submitter-identity filter if a steady stream of
  routine submissions ever makes it noisy.
- Cost cap / model are only known once the scoring pipeline runs, so the
  on-submission ticket does not include them.
