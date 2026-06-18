"""Unit tests for submission_notifier message construction and fan-out.

These are pure-function tests (no real network / GitHub). The CI workflow
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
    # With no GitHub token configured, notify is a silent no-op: it returns None
    # and must not raise (it runs inside add_new_eval).
    monkeypatch.setattr(sn, "GITHUB_TOKEN", None)
    monkeypatch.setattr(sn, "ISSUE_REPO", "owner/repo")
    result = sn.notify_submission(
        submission_dataset="allenai/asta-bench-submissions",
        config_name="1.0.0-dev1",
        split="test",
        submission_name="alice_RoboPhD_2026-05-14_09-30-00",
        agent_name="RoboPhD",
        username="alice",
        submit_time="2026-05-14T09:30:00+00:00",
    )
    assert result is None


def test_notify_submission_noop_without_repo(monkeypatch):
    # A token alone isn't enough: with no repo configured (the unset default),
    # there's nowhere to open the issue, so notify is a silent no-op.
    monkeypatch.setattr(sn, "GITHUB_TOKEN", "ghp_test")
    monkeypatch.setattr(sn, "ISSUE_REPO", "")
    result = sn.notify_submission(
        submission_dataset="allenai/asta-bench-submissions",
        config_name="1.0.0-dev1",
        split="test",
        submission_name="alice_RoboPhD_2026-05-14_09-30-00",
        agent_name="RoboPhD",
        username="alice",
    )
    assert result is None


def test_notify_submission_files_ticket_with_metadata(monkeypatch):
    # When configured, notify opens a ticket and returns its URL.
    monkeypatch.setattr(sn, "GITHUB_TOKEN", "ghp_test")
    monkeypatch.setattr(sn, "ISSUE_REPO", "owner/repo")

    calls = {}

    def fake_ticket(title, body):
        calls["title"] = title
        calls["body"] = body
        return "https://github.com/allenai/scholar/issues/1"

    monkeypatch.setattr(sn, "create_triage_ticket", fake_ticket)

    result = sn.notify_submission(
        submission_dataset="allenai/asta-bench-submissions",
        config_name="1.0.0-dev1",
        split="test",
        submission_name="alice_RoboPhD_2026-05-14_09-30-00",
        agent_name="RoboPhD",
        username="alice",
        submit_time="2026-05-14T09:30:00+00:00",
    )
    assert result == "https://github.com/allenai/scholar/issues/1"
    assert "RoboPhD" in calls["title"]


def test_create_ticket_skips_board_without_project(monkeypatch):
    # With no Project configured, a plain issue is the whole notification: the
    # issue is opened and no board GraphQL is attempted.
    monkeypatch.setattr(sn, "GITHUB_TOKEN", "ghp_test")
    monkeypatch.setattr(sn, "ISSUE_REPO", "owner/repo")
    monkeypatch.setattr(sn, "PROJECT_NUMBER", None)

    class FakeResp:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "html_url": "https://github.com/owner/repo/issues/7",
                "node_id": "I_abc",
            }

    monkeypatch.setattr(sn.requests, "post", lambda *a, **k: FakeResp())

    def fail_graphql(*_a, **_k):
        raise AssertionError("no board GraphQL should run without a Project")

    monkeypatch.setattr(sn, "_graphql", fail_graphql)

    url = sn.create_triage_ticket("title", "body")
    assert url == "https://github.com/owner/repo/issues/7"


def test_notify_submission_swallows_ticket_errors(monkeypatch):
    # A ticket failure is logged, not raised, and returns None so the submission
    # is never blocked.
    monkeypatch.setattr(sn, "GITHUB_TOKEN", "ghp_test")
    monkeypatch.setattr(sn, "ISSUE_REPO", "owner/repo")

    def boom_ticket(_title, _body):
        raise RuntimeError("github down")

    monkeypatch.setattr(sn, "create_triage_ticket", boom_ticket)

    result = sn.notify_submission(
        submission_dataset="allenai/asta-bench-submissions",
        config_name="1.0.0-dev1",
        split="test",
        submission_name="alice_RoboPhD_2026-05-14_09-30-00",
        agent_name="RoboPhD",
        username="alice",
        submit_time="2026-05-14T09:30:00+00:00",
    )
    assert result is None
