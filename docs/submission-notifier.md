# Submission notifier

Notifies the AstaBench on-call whenever a new submission is made through the
leaderboard web form, so external submissions — which require on-call action to
score and publish — no longer sit unnoticed. Implements
allenai/asta-bench-leaderboard#119 (parent: allenai/gas2own#86).

When a logged-in user submits through the form, the submission handler fans out
to:

- **Slack `#asta-bench`** — ambient awareness: submitter HF username, agent
  name/description/URL, track/split, openness, tool-usage, timestamp, and a link
  to the submission folder.
- **A triage ticket on the S2 Forever / On-call board** ([Project #64](https://github.com/orgs/allenai/projects/64)) —
  the actionable channel. A GitHub issue is opened on **`allenai/scholar`** (by
  S2 convention tickets are filed there even when they reference code elsewhere),
  added to the project, and set to the **Triage Needed** status so it surfaces
  to whoever is on-call. A card has a lifecycle (assignable, closeable) that a
  fire-and-forget email did not, and the GitHub API is reachable from the Space
  where an internal Ai2 SMTP relay is not.

**Privacy:** the submitter's email is **never** posted to Slack and **never**
placed in the (org-visible) ticket. It remains only in the private contact-info
dataset, which the ticket points the on-call to.

**Volume:** web-form submissions land roughly once a month or less, so one
ticket per submission is intentional and the board stays manageable. Revisit
(e.g. Slack-only for routine submissions, ticket on a condition) if volume grows.

## How it works

The notifier fires **in-process from the leaderboard app**, not from a webhook.
`submission.add_new_eval` — the function the form runs — calls
`submission_notifier.notify_submission(...)` right after a submission's files are
uploaded, passing the agent metadata it already has in hand.

This is deliberately the trigger rather than a HuggingFace dataset-commit
webhook, because `add_new_eval` runs **only when the web form is used**:

- **Rescoring** writes the *results* datasets, never the submissions dataset, so
  it never reaches `add_new_eval`.
- **Programmatic / API** writes to the submissions dataset don't go through the
  form, so they never reach `add_new_eval`.
- **Internal users self-publishing** results run `agenteval` themselves rather
  than the web form, so they never reach `add_new_eval`.

So "the form ran" is exactly the on-call-actionable event, and the submitter
metadata is first-class in-process data (the OAuth-verified username, the agent
fields the user typed) rather than something reconstructed from a commit message.

`notify_submission` is **best-effort**: each channel is independent, every
failure is logged (`notifier: ...` in the Space logs) but never raised, so a
notifier problem can never fail or block the user's submission. A channel with
no credentials configured is skipped, so with neither Slack nor GitHub
configured the call is a silent no-op.

The triage ticket is created with the REST API (open issue on `allenai/scholar`)
followed by two GraphQL mutations (`addProjectV2ItemById` to attach it to the
board, then `updateProjectV2ItemFieldValue` to set the **Triage Needed**
status). The board's node IDs (project, Status field, "Triage Needed" option)
are resolved **at runtime** from the org + project number, so they never go
stale if the board is rebuilt.

## One-time setup

There is **no webhook to register** and **no second Space to run** — the notifier
lives inside the leaderboard app. Go-live is just setting Space secrets.

### 1. Space secrets

Set these on the **public leaderboard Space** (Settings → Variables and secrets).
Configure them on the public Space only: the notifier is a no-op without
credentials, so the internal Space (which backs the internal-only leaderboard)
stays silent by simply not having them set.

| Secret | Required | Notes |
|---|---|---|
| `NOTIFIER_GITHUB_TOKEN` | yes (ticket) | GitHub token with **`repo`** + **`project`** scope (classic PAT), or a fine-grained token with **Issues: Read/Write** on `allenai/scholar` and **Projects: Read/Write** on `allenai`. Used to open the issue and add it to the board. **Recommended: reuse the existing `allenai-dev-role` PAT** that `s2-ticket-factory` already uses to file on-call cards — see step 3. Falls back to `GITHUB_TOKEN`. |
| `SLACK_BOT_TOKEN` | one of these | Bot token with `chat:write`; bot must be invited to `#asta-bench`. |
| `SLACK_WEBHOOK_URL` | one of these | Incoming-webhook URL for `#asta-bench` (less metadata flexibility). |
| `NOTIFIER_SLACK_CHANNEL` | no | Defaults to `#asta-bench`. |
| `NOTIFIER_GITHUB_REPO` | no | Repo to open the issue on. Defaults to `allenai/scholar`. |
| `NOTIFIER_PROJECT_ORG` | no | Org owning the board. Defaults to `allenai`. |
| `NOTIFIER_PROJECT_NUMBER` | no | Project number. Defaults to `64` (S2 Forever / On-call). |
| `NOTIFIER_PROJECT_STATUS` | no | Status option to set. Defaults to `Triage Needed`. |

The GitHub token's owner becomes the issue author, so use a service/bot account
if you don't want a person's name on every ticket. The recommended source is the
existing `allenai-dev-role` PAT (see step 3), which already authors the on-call
cards filed by `s2-ticket-factory`.

### 2. Slack

Prefer an existing AstaBench Slack app; otherwise create one with the
`chat:write` scope, install it to the workspace, invite the bot to
`#asta-bench`, and set `SLACK_BOT_TOKEN`.

### 3. GitHub

Provision a token (`NOTIFIER_GITHUB_TOKEN`) that can open issues on
`allenai/scholar` and edit [Project #64](https://github.com/orgs/allenai/projects/64).

**Recommended: reuse the existing `allenai-dev-role` PAT.** The
[`allenai/s2-ticket-factory`](https://github.com/allenai/s2-ticket-factory)
Lambda already files on-call cards on `allenai/scholar` + Project #64 using a
classic PAT owned by the `allenai-dev-role` bot account — so its scopes (issues
write + projectV2 mutation) are exactly what this notifier needs, and reusing it
keeps every auto-filed on-call card under one consistent author. The PAT lives in
1Password under the entry **"Github AllenAI Dev Role"**. Copy that token into the
Space's `NOTIFIER_GITHUB_TOKEN` secret.

If you'd rather isolate blast radius (so revoking one automation's token doesn't
affect the other), mint a dedicated token for the notifier instead — same scopes,
ideally still owned by a bot account rather than a person.

No board configuration is needed — the **Triage Needed** column already exists
and the IDs are resolved at runtime. To file tickets on a different board or
column, set `NOTIFIER_PROJECT_NUMBER` / `NOTIFIER_PROJECT_STATUS` (and
`NOTIFIER_GITHUB_REPO` to open the issue elsewhere).

## Verifying

1. Make a test submission through the public Space's submission form.
2. Confirm a Slack message in `#asta-bench` and a new card in **Triage Needed**
   on [Project #64](https://github.com/orgs/allenai/projects/64) (an issue on
   `allenai/scholar`).
3. Any delivery error is logged to the Space logs (`notifier: ...`) without
   affecting the submitter's experience.

## Limitations / follow-ups

- One ticket fires per successful web-form submission. At ~monthly volume this
  is the desired behaviour; if internal users start routinely using the public
  form (they normally run `agenteval` themselves) it could add a few easy-to-
  close tickets. Revisit with a submitter-identity filter only if that happens.
- Cost cap / model are only known once the scoring pipeline runs, so the
  on-submission notification does not include them.
