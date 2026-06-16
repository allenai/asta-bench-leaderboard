# Submission notifier

Notifies the AstaBench team whenever a new leaderboard submission lands, so
submissions no longer sit unnoticed until a submitter happens to email
`asta-support@`. Implements allenai/asta-bench-leaderboard#119 (parent:
allenai/gas2own#86).

On every submission commit to the backing HF datasets, a HuggingFace webhook
hits this Space and fans out to:

- **Slack `#asta-bench`** — submitter HF username, agent name/description/URL,
  track/split, openness, tool-usage, timestamp, and a link to the submission
  folder. The submitter **email is never posted to Slack** (privacy).
- **Email `asta-support@allenai.org`** — the same payload **plus** the submitter
  email. The subject is `[AstaBench submission] <hf-username> — <agent-name>
  (<split>)` so it is greppable in the mailbox.

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
reads the per-submission `submission.json` for the full agent metadata and the
contact-info dataset for the submitter email.

If `NOTIFIER_WEBHOOK_SECRET` is unset the Space launches exactly as before, with
no webhook endpoint — the notifier is purely additive.

## One-time setup

### 1. Space secrets

Set these on the **public leaderboard Space** (Settings → Variables and secrets):

| Secret | Required | Notes |
|---|---|---|
| `NOTIFIER_WEBHOOK_SECRET` | yes | Random string. Also registered as the HF webhook secret (step 3). Enables the endpoint. |
| `NOTIFIER_HF_TOKEN` | recommended | Read-only HF token scoped to the three datasets (minted 2026-05-19, see #119). Falls back to the Space's `HF_TOKEN`. |
| `SLACK_BOT_TOKEN` | one of these | Bot token with `chat:write`; bot must be invited to `#asta-bench`. |
| `SLACK_WEBHOOK_URL` | one of these | Incoming-webhook URL for `#asta-bench` (less metadata flexibility). |
| `NOTIFIER_SLACK_CHANNEL` | no | Defaults to `#asta-bench`. |
| `SMTP_HOST` | yes (email) | SMTP relay host (Mailgun/SES/Ai2 SMTP). |
| `SMTP_PORT` | no | Defaults to `587`. |
| `SMTP_STARTTLS` | no | Defaults to `true`. |
| `SMTP_USERNAME` / `SMTP_PASSWORD` | if relay requires auth | |
| `NOTIFIER_EMAIL_FROM` | no | Defaults to `asta-support@allenai.org`. |
| `NOTIFIER_EMAIL_TO` | no | Defaults to `asta-support@allenai.org`. |

### 2. Slack

Prefer an existing AstaBench Slack app; otherwise create one with the
`chat:write` scope, install it to the workspace, invite the bot to
`#asta-bench`, and set `SLACK_BOT_TOKEN`.

### 3. HF webhooks

As an HF org admin, create a webhook (Settings → Webhooks) for **each** dataset:

- `allenai/asta-bench-internal-contact-info` (private; commits on every submission)
- `allenai/asta-bench-submissions` (public)
- `allenai/asta-bench-internal-submissions` (internal)

For each: target URL `https://<space-subdomain>.hf.space/webhooks/submissions`,
events **repo content / update**, secret = `NOTIFIER_WEBHOOK_SECRET`.

## Verifying

1. Use the **Test** button on an HF webhook, or make a real submission.
2. Confirm a Slack message in `#asta-bench` and an email at `asta-support@`.
3. Errors are logged to the Space logs (`notifier: ...`). A commit that fails to
   deliver on every channel is **not** marked seen, so HF retries it.

## Limitations / follow-ups

- Dedup state lives in `DATA_DIR/notifier_seen_submissions.json` on the Space's
  ephemeral disk; a Space restart between a commit and its retry could
  re-notify. The leaderboard Space already restarts hourly. If this becomes
  noisy, persist dedup state to a dataset.
- Cost cap / model are only surfaced once the scoring pipeline writes them; the
  on-upload notification shows them only if already present in `submission.json`.
