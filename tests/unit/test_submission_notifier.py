"""Unit tests for submission_notifier message construction and fan-out.

These are pure-function tests (no real network / Slack / GitHub). The CI workflow
only runs tests/integration, but these are runnable locally with:

    pytest tests/unit/ -v
"""

import submission_notifier as sn

# Coordinates as add_new_eval would pass them in-process (no commit parsing).
COORDS = {
    "repo": "allenai/asta-bench-submissions",
    "config": "1.0.0-dev1",
    "split": "test",
    "name": "alice_RoboPhD_2026-05-14_09-30-00",
    "username": "alice",
}

META = {
    "agent_name": "RoboPhD",
    "agent_description": "A seed agent",
    "agent_url": "https://arxiv.org/abs/2604.04347",
    "openness": "Open Source",
    "tool_usage": "Standard",
    "username": "alice",
    "submit_time": "2026-05-14T09:30:00+00:00",
}


def test_submission_folder_url():
    url = sn.submission_folder_url(COORDS)
    assert url == (
        "https://huggingface.co/datasets/allenai/asta-bench-submissions/tree/main/"
        "1.0.0-dev1/test/alice_RoboPhD_2026-05-14_09-30-00"
    )


def test_slack_payload_omits_email_and_targets_channel():
    payload = sn.build_slack_payload(COORDS, META)
    assert payload["channel"] == sn.SLACK_CHANNEL
    text = payload["text"]
    assert "RoboPhD" in text
    assert "alice" in text
    assert "test" in text
    # Privacy: the submitter email must never appear in Slack.
    assert "@" not in text or "asta" not in text  # no email address leaked
    assert "alice@" not in text


def test_slack_payload_falls_back_to_coords_without_metadata():
    payload = sn.build_slack_payload(COORDS, {})
    # Folder name is used as the agent-name fallback.
    assert "alice_RoboPhD" in payload["text"]


def test_issue_title_and_body_carry_metadata():
    title, body = sn.build_issue(COORDS, META)
    # Title carries agent name + split so the board card is self-describing.
    assert "RoboPhD" in title
    assert "test" in title
    # Body carries the submission detail the on-call needs.
    assert "alice" in body  # HF username
    assert "A seed agent" in body
    assert "Open Source" in body
    assert "huggingface.co/datasets" in body  # link to the submission folder
    # Points the on-call to where the (omitted) email lives.
    assert "asta-bench-internal-contact-info" in body


def test_issue_is_pii_free():
    # build_issue never even receives the submitter email; the org-visible
    # ticket must not contain an email address.
    title, body = sn.build_issue(COORDS, META)
    assert "@" not in body
    assert "@" not in title


def test_issue_falls_back_to_coords_without_metadata():
    title, _body = sn.build_issue(COORDS, {})
    # Folder name is used as the agent-name fallback.
    assert "alice_RoboPhD" in title


def test_notify_submission_noop_when_unconfigured(monkeypatch):
    # With neither Slack nor GitHub configured, notify is a silent no-op and
    # delivers nothing -- and must not raise (it runs inside add_new_eval).
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    monkeypatch.setattr(sn, "GITHUB_TOKEN", None)
    delivered = sn.notify_submission(
        submission_dataset="allenai/asta-bench-submissions",
        config_name="1.0.0-dev1",
        split="test",
        submission_name="alice_RoboPhD_2026-05-14_09-30-00",
        agent_name="RoboPhD",
        username="alice",
        submit_time="2026-05-14T09:30:00+00:00",
    )
    assert delivered == []


def test_notify_submission_swallows_channel_errors(monkeypatch):
    # A failing channel is logged, not raised, and does not block the other.
    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
    monkeypatch.setattr(sn, "GITHUB_TOKEN", "ghp_test")

    def boom_slack(_payload):
        raise RuntimeError("slack down")

    calls = {}

    def fake_ticket(title, body):
        calls["title"] = title
        return "https://github.com/allenai/scholar/issues/1"

    monkeypatch.setattr(sn, "send_slack", boom_slack)
    monkeypatch.setattr(sn, "create_triage_ticket", fake_ticket)

    delivered = sn.notify_submission(
        submission_dataset="allenai/asta-bench-submissions",
        config_name="1.0.0-dev1",
        split="test",
        submission_name="alice_RoboPhD_2026-05-14_09-30-00",
        agent_name="RoboPhD",
        username="alice",
        submit_time="2026-05-14T09:30:00+00:00",
    )
    # Slack failed, ticket succeeded -> only the ticket is reported delivered.
    assert delivered == ["ticket"]
    assert "RoboPhD" in calls["title"]
