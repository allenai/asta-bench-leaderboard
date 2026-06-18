"""Unit tests for submission_notifier message construction and fan-out.

These are pure-function tests (no real network / GitHub). The CI workflow
only runs tests/integration, but these are runnable locally with:

    pytest tests/unit/ -v
"""

import submission_notifier as sn


def test_notify_submission_noop_when_unconfigured(monkeypatch):
    # With no GitHub token configured, notify is a silent no-op: it returns None
    # and must not raise (it runs inside add_new_eval).
    monkeypatch.setattr(sn, "GITHUB_TOKEN", None)
    monkeypatch.setattr(sn, "ISSUE_REPO", "owner/repo")
    result = sn.notify_submission(
        submission_dataset="allenai/asta-bench-submissions",
        config_name="1.0.0-dev1",
        split="test",
        submission_name="alice_Wonderbot_2026-05-14_09-30-00",
        username="alice",
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
        submission_name="alice_Wonderbot_2026-05-14_09-30-00",
        username="alice",
    )
    assert result is None


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
        submission_name="alice_Wonderbot_2026-05-14_09-30-00",
        username="alice",
    )
    assert result is None
