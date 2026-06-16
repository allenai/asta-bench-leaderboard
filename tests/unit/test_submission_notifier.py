"""Unit tests for submission_notifier message construction and dedup.

These are pure-function tests (no network / no HF / no Slack / no SMTP). The CI
workflow only runs tests/integration, but these are runnable locally with:

    pytest tests/unit/ -v
"""

import submission_notifier as sn

SUBMISSIONS_COMMIT = (
    'Submission from hf user "alice" to '
    '"hf://datasets/allenai/asta-bench-submissions/1.0.0-dev1/test/'
    'alice_RoboPhD_2026-05-14_09-30-00"'
)
CONTACT_COMMIT = (
    'Submission from hf user "alice" to '
    '"hf://datasets/allenai/asta-bench-submissions/1.0.0-dev1/test/'
    'alice_RoboPhD_2026-05-14_09-30-00"'
)

META = {
    "agent_name": "RoboPhD",
    "agent_description": "A seed agent",
    "agent_url": "https://arxiv.org/abs/2604.04347",
    "openness": "Open Source",
    "tool_usage": "Standard",
    "username": "alice",
    "submit_time": "2026-05-14T09:30:00+00:00",
}


def test_parse_commit_message_extracts_coords():
    coords = sn.parse_commit_message(SUBMISSIONS_COMMIT)
    assert coords["username"] == "alice"
    assert coords["repo"] == "allenai/asta-bench-submissions"
    assert coords["config"] == "1.0.0-dev1"
    assert coords["split"] == "test"
    assert coords["name"] == "alice_RoboPhD_2026-05-14_09-30-00"
    assert coords["dataset_url"].startswith(
        "hf://datasets/allenai/asta-bench-submissions/"
    )


def test_parse_commit_message_ignores_non_submission():
    assert sn.parse_commit_message("Update README") is None
    assert sn.parse_commit_message("") is None
    assert sn.parse_commit_message(None) is None


def test_contact_and_submission_commits_dedupe_to_same_key():
    a = sn.parse_commit_message(SUBMISSIONS_COMMIT)
    b = sn.parse_commit_message(CONTACT_COMMIT)
    assert a["dataset_url"] == b["dataset_url"]


def test_slack_payload_omits_email_and_targets_channel():
    coords = sn.parse_commit_message(SUBMISSIONS_COMMIT)
    payload = sn.build_slack_payload(coords, META)
    assert payload["channel"] == sn.SLACK_CHANNEL
    text = payload["text"]
    assert "RoboPhD" in text
    assert "alice" in text
    assert "test" in text
    # Privacy: the submitter email must never appear in Slack.
    assert "@" not in text or "asta" not in text  # no email address leaked
    assert "alice@" not in text


def test_slack_payload_falls_back_to_coords_without_metadata():
    coords = sn.parse_commit_message(SUBMISSIONS_COMMIT)
    payload = sn.build_slack_payload(coords, {})
    # Folder name is used as the agent-name fallback.
    assert "alice_RoboPhD" in payload["text"]


def test_email_subject_is_greppable_and_body_has_email():
    coords = sn.parse_commit_message(SUBMISSIONS_COMMIT)
    subject, body = sn.build_email(coords, META, "alice@example.com")
    # Subject must carry HF username + agent name for mailbox grepping.
    assert "alice" in subject
    assert "RoboPhD" in subject
    # Email body carries the submitter email (Slack does not).
    assert "alice@example.com" in body


def test_email_handles_missing_submitter_email():
    coords = sn.parse_commit_message(SUBMISSIONS_COMMIT)
    subject, body = sn.build_email(coords, META, None)
    assert "(not found)" in body


def test_submission_folder_url():
    coords = sn.parse_commit_message(SUBMISSIONS_COMMIT)
    url = sn.submission_folder_url(coords)
    assert url == (
        "https://huggingface.co/datasets/allenai/asta-bench-submissions/tree/main/"
        "1.0.0-dev1/test/alice_RoboPhD_2026-05-14_09-30-00"
    )


def test_dedup_state_roundtrip(tmp_path, monkeypatch):
    state = tmp_path / "seen.json"
    monkeypatch.setattr(sn, "_SEEN_PATH", str(state))
    monkeypatch.setattr(sn, "_STATE_DIR", str(tmp_path))
    assert sn._load_seen() == set()
    sn._save_seen({"hf://datasets/x/1/test/a"})
    assert "hf://datasets/x/1/test/a" in sn._load_seen()
