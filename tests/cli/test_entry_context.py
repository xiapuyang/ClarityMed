"""Tests for ``claritymed.cli.entry.inject_context``."""

from __future__ import annotations

import json
import re

import pytest

from claritymed.cli import inject_context
from claritymed.context import language_ctx, request_id_ctx, user_id_ctx
from claritymed.core.observability.logging import setup_logging

REQUEST_ID_PATTERN = re.compile(r"^[0-9]{14}[0-9A-F]{8}$")


def test_explicit_args_take_effect():
    setup_logging("test", console_level=None)
    with inject_context(user_id="alice", language="zh") as (rid, uid, lang):
        assert uid == "alice"
        assert lang == "zh"
        assert REQUEST_ID_PATTERN.match(rid)
        assert request_id_ctx.get() == rid
    assert request_id_ctx.get() is None
    assert user_id_ctx.get() is None
    assert language_ctx.get() is None


def test_falls_back_to_default_user_with_warning(tmp_path):
    """The 'claritymed' parent logger has propagate=False, so caplog cannot
    see records from claritymed.cli.entry — verify via the app.log file
    that production logging actually emitted the warning."""
    setup_logging("test", console_level=None)
    with inject_context() as (_, uid, _):
        assert uid == "default"
    text = (tmp_path / "logs" / "app.log").read_text(encoding="utf-8")
    assert "default" in text.lower()


def test_env_user_when_no_explicit(monkeypatch):
    setup_logging("test", console_level=None)
    monkeypatch.setenv("CLARITYMED_USER", "bob")
    with inject_context() as (_, uid, _):
        assert uid == "bob"


def test_explicit_user_beats_env(monkeypatch):
    setup_logging("test", console_level=None)
    monkeypatch.setenv("CLARITYMED_USER", "bob")
    with inject_context(user_id="alice") as (_, uid, _):
        assert uid == "alice"


def test_invalid_env_lang_warns_and_falls_back(monkeypatch, tmp_path):
    setup_logging("test", console_level=None)
    monkeypatch.setenv("CLARITYMED_LANG", "ja")
    with inject_context() as (_, _, lang):
        assert lang == "en"
    text = (tmp_path / "logs" / "app.log").read_text(encoding="utf-8")
    assert "ja" in text  # warning message references the rejected value


def test_audit_events_around_block(tmp_path):
    setup_logging("test", console_level=None)
    with inject_context(user_id="alice", language="en"):
        pass
    audit_path = tmp_path / "logs" / "audit.log"
    lines = audit_path.read_text(encoding="utf-8").splitlines()
    kinds = []
    for line in lines:
        if line.strip():
            payload_json = line.split("] ", 3)[-1]
            kinds.append(json.loads(payload_json)["kind"])
    assert "request_start" in kinds
    assert "request_end" in kinds


def test_audit_event_on_exception(tmp_path):
    setup_logging("test", console_level=None)
    try:
        with inject_context(user_id="alice", language="en"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    text = (tmp_path / "logs" / "audit.log").read_text(encoding="utf-8")
    assert '"status":"exception"' in text  # pydantic JSON has no space after colon


# --- check_user_exists ----------------------------------------------------


def test_check_user_exists_raises_when_user_missing():
    """check_user_exists=True raises UserNotFoundError for unknown users."""
    from claritymed.errors import UserNotFoundError

    setup_logging("test", console_level=None)
    with pytest.raises(UserNotFoundError, match="ghost"):
        with inject_context(user_id="ghost", check_user_exists=True):
            pass


def test_check_user_exists_passes_when_user_present(tmp_path):
    """check_user_exists=True does not raise when settings.yaml exists."""
    from claritymed.stores.account import init_user

    setup_logging("test", console_level=None)
    init_user("alice")
    with inject_context(user_id="alice", check_user_exists=True) as (_, uid, _):
        assert uid == "alice"


def test_check_user_exists_false_skips_check():
    """Default (check_user_exists=False) never raises even for unknown users."""
    setup_logging("test", console_level=None)
    with inject_context(user_id="nobody") as (_, uid, _):
        assert uid == "nobody"
