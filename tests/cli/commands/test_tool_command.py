"""Tests for the ``claritymed tool`` headless command."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from claritymed.cli.main import app

runner = CliRunner()


@pytest.fixture
def _bootstrap_user(monkeypatch):
    """Initialize the test user so CLI commands have a target."""
    from claritymed.stores.account import init_user

    init_user("test", "Test")
    monkeypatch.setenv("CLARITYMED_HEADLESS", "1")


def test_tool_run_help_lists_subcommands(_bootstrap_user):
    result = runner.invoke(app, ["tool", "--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout
    assert "rule-list" in result.stdout


def test_tool_run_rejects_interactive_without_auto_approve(_bootstrap_user):
    result = runner.invoke(
        app,
        [
            "tool",
            "run",
            "save_allergy",
            json.dumps(
                {"substance": "peanut", "severity": "mild", "source": "self_report"}
            ),
            "--user",
            "test",
        ],
        input="",  # non-empty stdin in click context counts as supplied
    )
    # Without --auto-approve we error out cleanly.
    assert result.exit_code == 1


def test_tool_run_rule_list_empty(_bootstrap_user):
    result = runner.invoke(app, ["tool", "rule-list", "--user", "test"])
    assert result.exit_code == 0
    assert "no active rules" in result.stdout or "Approval rules" in result.stdout


def test_tool_run_rule_revoke_unknown_id(_bootstrap_user):
    result = runner.invoke(app, ["tool", "rule-revoke", "deadbeef", "--user", "test"])
    assert result.exit_code == 1
