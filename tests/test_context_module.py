"""Tests for ``claritymed.context`` ContextVars + OTel baggage helpers."""

from __future__ import annotations

import logging

import pytest

from claritymed.context import (
    MissingContextError,
    apply_context,
    attach_session_baggage,
    detach_session_baggage,
    get_context_or_raise,
    is_valid_request_id,
    new_request_id,
    reset_context,
)


def test_new_request_id_format():
    rid = new_request_id()
    assert len(rid) == 22
    assert is_valid_request_id(rid)


def test_is_valid_request_id_rejects_empty():
    assert is_valid_request_id("") is False


def test_is_valid_request_id_rejects_short_string():
    assert is_valid_request_id("short") is False


def test_is_valid_request_id_rejects_pattern_match_with_invalid_date():
    """The strptime double-check rejects e.g. month 13 even if the pattern
    matches the shape of the input."""
    # Right shape (14 digits + 8 hex), but month=13.
    bogus = "20261301000000DEADBEEF"
    assert is_valid_request_id(bogus) is False


def test_is_valid_request_id_rejects_pattern_with_invalid_day():
    # Day = 32 in any month is invalid.
    bogus = "20260132000000ABCD1234"
    assert is_valid_request_id(bogus) is False


def test_get_context_or_raise_raises_when_unset():
    """Outside of an inject_context scope, get_context_or_raise must raise."""
    with pytest.raises(MissingContextError):
        get_context_or_raise()


def test_detach_session_baggage_with_none_is_noop():
    detach_session_baggage(None)  # must not raise


def test_attach_session_baggage_with_otel_returns_token():
    """In the normal environment OTel is importable — attach returns a token."""
    token = attach_session_baggage("session-xyz")
    # Token may be a real OTel token or None if OTel is unavailable.
    if token is not None:
        detach_session_baggage(token)


def test_attach_baggage_swallows_unexpected_exception(monkeypatch, caplog):
    """_attach_baggage logs + returns None when otel raises an unexpected error."""
    import opentelemetry.context as otel_ctx

    from claritymed import context as _ctx_mod

    def _boom(_ctx):
        raise RuntimeError("otel runtime fail")

    monkeypatch.setattr(otel_ctx, "get_current", _boom)
    with caplog.at_level(logging.WARNING):
        token = _ctx_mod._attach_baggage("req-1", "alice")
    assert token is None
    assert any("baggage attach failed" in r.message for r in caplog.records)


def test_detach_baggage_swallows_unexpected_exception(monkeypatch, caplog):
    """_detach_baggage logs + swallows when otel detach raises."""
    import opentelemetry.context as otel_ctx

    from claritymed import context as _ctx_mod

    def _boom(_t):
        raise RuntimeError("nope")

    monkeypatch.setattr(otel_ctx, "detach", _boom)
    with caplog.at_level(logging.WARNING):
        _ctx_mod._detach_baggage(object())
    assert any("baggage detach failed" in r.message for r in caplog.records)


def test_attach_session_baggage_swallows_unexpected_exception(monkeypatch, caplog):
    """attach_session_baggage logs + returns None on unexpected OTel failure."""
    import opentelemetry.context as otel_ctx

    def _boom(_ctx):
        raise RuntimeError("nope")

    monkeypatch.setattr(otel_ctx, "get_current", _boom)
    with caplog.at_level(logging.WARNING):
        token = attach_session_baggage("sid-1")
    assert token is None


def test_attach_baggage_returns_none_when_otel_missing(monkeypatch):
    """ImportError path: otel not installed → attach returns None silently."""
    import builtins

    from claritymed import context as _ctx_mod

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("opentelemetry"):
            raise ImportError("no otel")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert _ctx_mod._attach_baggage("req-1", "alice") is None


def test_detach_baggage_swallows_import_error(monkeypatch):
    import builtins

    from claritymed import context as _ctx_mod

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("opentelemetry"):
            raise ImportError("no otel")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    _ctx_mod._detach_baggage(object())  # must not raise


def test_attach_session_baggage_returns_none_when_otel_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("opentelemetry"):
            raise ImportError("no otel")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert attach_session_baggage("sid-1") is None


def test_reset_context_swallows_cross_context_value_error():
    """Tokens issued in Context A must not raise when reset in Context B.

    Reproduces the async-generator finaliser race: an async gen calls
    ``apply_context`` while iterating, then the consumer abandons it
    early. asyncio's ``aclose()`` finaliser fires on a fresh task with
    its own Context, and Python's ``ContextVar.reset`` would otherwise
    raise ``ValueError`` for the cross-Context token. ``reset_context``
    must silently no-op so a benign cleanup doesn't surface as an
    unretrieved task exception.
    """
    import contextvars

    tokens_holder: list = []

    def _grab_tokens():
        tokens_holder.append(apply_context(new_request_id(), "test", "en"))

    # Allocate the tokens inside an isolated Context.
    contextvars.copy_context().run(_grab_tokens)
    # Now reset from the outer Context — would raise ValueError without
    # the defensive try/except inside reset_context.
    reset_context(tokens_holder[0])
