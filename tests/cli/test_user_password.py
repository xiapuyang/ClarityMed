"""``claritymed user set-password`` integration tests."""

from __future__ import annotations

from typer.testing import CliRunner

from claritymed.cli.main import app
from claritymed.stores.account import init_user
from claritymed.stores.auth import PasswordStore

runner = CliRunner()


def test_set_password_writes_hash_and_login_works():
    init_user("test", display_name="Test")
    result = runner.invoke(
        app,
        ["user", "set-password", "test"],
        input="abcdefgh\nabcdefgh\n",
    )
    assert result.exit_code == 0, result.output
    assert PasswordStore.verify_password("test", "abcdefgh") is True
    # ``bootstrap_once`` reconfigures the audit logger with
    # propagate=False, so caplog can't see the row reliably. The
    # side-effect (auth.yaml written + verify returns True) is the
    # acceptance criterion; audit emission is exercised by the web
    # auth-router tests where bootstrap_once isn't on the path.


def test_set_password_mismatched_prompts_fails():
    init_user("test", display_name="Test")
    result = runner.invoke(
        app,
        ["user", "set-password", "test"],
        # Mismatched confirmation. Typer asks again then aborts.
        input="abcdefgh\nDIFFERENT\nDIFFERENT2\nDIFFERENT3\n",
    )
    assert result.exit_code != 0
    # Hash must not have been written.
    assert PasswordStore.verify_password("test", "abcdefgh") is False


def test_set_password_short_password_rejected():
    init_user("test", display_name="Test")
    result = runner.invoke(
        app,
        ["user", "set-password", "test"],
        input="short\nshort\n",
    )
    assert result.exit_code != 0
    assert PasswordStore.verify_password("test", "short") is False


def test_set_password_for_uninit_user_refuses():
    # No init_user() — settings.yaml is missing.
    result = runner.invoke(
        app,
        ["user", "set-password", "ghost"],
        input="abcdefgh\nabcdefgh\n",
    )
    assert result.exit_code != 0
    assert "init-user" in result.output


def test_set_password_invalid_user_id_rejected():
    result = runner.invoke(
        app,
        ["user", "set-password", "../etc/passwd"],
        input="abcdefgh\nabcdefgh\n",
    )
    assert result.exit_code != 0


def test_subcommand_registered_in_help():
    result = runner.invoke(app, ["user", "--help"])
    assert result.exit_code == 0
    assert "set-password" in result.output
