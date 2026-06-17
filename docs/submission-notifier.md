# Submission notifier

Notifies the AstaBench team whenever a new leaderboard submission lands, so
submissions no longer sit unnoticed until a submitter happens to email
`asta-support@`. Implements allenai/asta-bench-leaderboard#119 (parent:
allenai/gas2own#86).

On every submission commit to the backing HF datasets, a HuggingFace webhook
hits this Space and fans out to:

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

**Volume:** submissions land roughly once a month or less, so one ticket per
submission is intentional and the board stays manageable. Revisit (e.g.
Slack-only for routine submissions, ticket on a condition) if volume grows.

Duplicate commits for the same submission folder (retries/edits, or the twin
commit on the contact-info dataset) dedupe to a single notification, keyed on
the submission's `hf://datasets/.../<config>/<split>/<name>` URL.

## How it works

`submission_notifier.py` wraps the existing Gradio app in a
[`huggingface_hub.WebhooksServer`](https://huggingface.co/docs/huggingface_hub/guides/webhooks_server)
and registers `POST /webhooks/submissions`. The server verifies the
`X-Webhook-Secret` header against `NOTIFIER_WEBHOOK_SECRET` on every request.

The webhook payload doesn't carry the commit message or changed files, so for
each relevant commit the handler reads the dataset's latest commit message —
which encodes the submission folder URL (`Submission from hf user "X" to
"hf://datasets/.../<config>/<split>/<name>"`, written by
`submission.py:upload_submission` and the contact-info `push_to_hub`) — then
reads the per-submission `submission.json` for the full agent metadata.

The triage ticket is created with the REST API (open issue on `allenai/scholar`)
followed by two GraphQL mutations (`addProjectV2ItemById` to attach it to the
board, then `updateProjectV2ItemFieldValue` to set the **Triage Needed**
status). The board's node IDs (project, Status field, "Triage Needed" option)
are resolved **at runtime** from the org + project number, so they never go
stale if the board is rebuilt.

If `NOTIFIER_WEBHOOK_SECRET` is unset the Space launches exactly as before, with
no webhook endpoint — the notifier is purely additive.

## One-time setup

### 1. Space secrets

Set these on the **public leaderboard Space** (Settings → Variables and secrets):

| Secret | Required | Notes |
|---|---|---|
| `NOTIFIER_WEBHOOK_SECRET` | yes | Random string. Also registered as the HF webhook secret (step 4). Enables the endpoint. |
| `NOTIFIER_HF_TOKEN` | recommended | Read-only HF token scoped to the three datasets (minted 2026-05-19, see #119). Falls back to the Space's `HF_TOKEN`. |
| `NOTIFIER_GITHUB_TOKEN` | yes (ticket) | GitHub token with **`repo`** + **`project`** scope (classic PAT), or a fine-grained token with **Issues: Read/Write** on `allenai/scholar` and **Projects: Read/Write** on `allenai`. Used to open the issue and add it to the board. Falls back to `GITHUB_TOKEN`. |
| `SLACK_BOT_TOKEN` | one of these | Bot token with `chat:write`; bot must be invited to `#asta-bench`. |
| `SLACK_WEBHOOK_URL` | one of these | Incoming-webhook URL for `#asta-bench` (less metadata flexibility). |
| `NOTIFIER_SLACK_CHANNEL` | no | Defaults to `#asta-bench`. |
| `NOTIFIER_GITHUB_REPO` | no | Repo to open the issue on. Defaults to `allenai/scholar`. |
| `NOTIFIER_PROJECT_ORG` | no | Org owning the board. Defaults to `allenai`. |
| `NOTIFIER_PROJECT_NUMBER` | no | Project number. Defaults to `64` (S2 Forever / On-call). |
| `NOTIFIER_PROJECT_STATUS` | no | Status option to set. Defaults to `Triage Needed`. |

The GitHub token's owner becomes the issue author, so use a service/bot account
if you don't want a person's name on every ticket.

### 2. Slack

Prefer an existing AstaBench Slack app; otherwise create one with the
`chat:write` scope, install it to the workspace, invite the bot to
`#asta-bench`, and set `SLACK_BOT_TOKEN`.

### 3. GitHub

Provision a token (`NOTIFIER_GITHUB_TOKEN`) that can open issues on
`allenai/scholar` and edit [Project #64](https://github.com/orgs/allenai/projects/64).
No board configuration is needed — the **Triage Needed** column already exists
and the IDs are resolved at runtime. To file tickets on a different board or
column, set `NOTIFIER_PROJECT_NUMBER` / `NOTIFIER_PROJECT_STATUS` (and
`NOTIFIER_GITHUB_REPO` to open the issue elsewhere).

### 4. HF webhooks

As an HF org admin, create a webhook (Settings → Webhooks) for **each** dataset:

- `allenai/asta-bench-internal-contact-info` (private; commits on every submission)
- `allenai/asta-bench-submissions` (public)
- `allenai/asta-bench-internal-submissions` (internal)

For each: target URL `https://<space-subdomain>.hf.space/webhooks/submissions`,
events **repo content / update**, secret = `NOTIFIER_WEBHOOK_SECRET`.

## Verifying

1. Use the **Test** button on an HF webhook, or make a real submission.
2. Confirm a Slack message in `#asta-bench` and a new card in **Triage Needed**
   on [Project #64](https://github.com/orgs/allenai/projects/64) (an issue on
   `allenai/scholar`).
3. Errors are logged to the Space logs (`notifier: ...`). A commit that fails to
   deliver on every channel is **not** marked seen, so HF retries it.

## Limitations / follow-ups

- Dedup state lives in `DATA_DIR/notifier_seen_submissions.json` on the Space's
  ephemeral disk; a Space restart between a commit and its retry could
  re-notify. The leaderboard Space already restarts hourly. If this becomes
  noisy, persist dedup state to a dataset.
- "Seen" is marked when **either** channel delivers, so a Slack success paired
  with a ticket failure won't auto-retry the ticket; the failure is logged
  loudly (`notifier: triage ticket creation failed ...`). Given ~monthly volume
  this is an acceptable trade against duplicate tickets on retry.
- Cost cap / model are only surfaced once the scoring pipeline writes them; the
  on-upload notification shows them only if already present in `submission.json`.
