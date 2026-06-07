"""Tests for ``claritymed.core.observability.audit``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from claritymed.context import (
    MissingContextError,
    apply_context,
    reset_context,
)
from claritymed.core.observability.audit import AuditEvent, audit_event
from claritymed.core.observability.logging import setup_logging


def _audit_path(tmp_path: Path) -> Path:
    return tmp_path / "logs" / "audit.log"


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_happy_event_round_trips_via_json(tmp_path):
    setup_logging("test", console_level=None)
    tokens = apply_context("20260606222522A1B2C3D4", "alice", "en")
    try:
        ev = audit_event("retrieval", payload={"chunks": ["a", "b"]})
    finally:
        reset_context(tokens)
    assert ev.kind == "retrieval"

    line = _read_lines(_audit_path(tmp_path))[-1]
    # The audit format wraps the JSON message with the formatter prefix; the
    # JSON itself is the trailing portion after the bracketed labels.
    payload_json = line.split("] ", 3)[-1]
    parsed = json.loads(payload_json)
    assert parsed["kind"] == "retrieval"
    assert parsed["payload"] == {"chunks": ["a", "b"]}
    assert parsed["user_id"] == "alice"
    assert parsed["language"] == "en"


def test_missing_user_id_blocks_audit():
    with pytest.raises(MissingContextError):
        audit_event("retrieval", payload={})


def test_invalid_kind_rejected_at_construction():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AuditEvent(
            kind="not_in_enum",  # type: ignore[arg-type]
            payload={},
            request_id="20260606222522DEADBEEF",
            user_id="alice",
            language="en",
        )


def test_one_line_per_event(tmp_path):
    setup_logging("test", console_level=None)
    tokens = apply_context("20260606222522A1B2C3D4", "alice", "en")
    try:
        for i in range(20):
            audit_event("tool_invoke", payload={"i": i})
    finally:
        reset_context(tokens)
    lines = _read_lines(_audit_path(tmp_path))
    # Each line must parse independently.
    for line in lines[-20:]:
        payload_json = line.split("] ", 3)[-1]
        parsed = json.loads(payload_json)
        assert parsed["kind"] == "tool_invoke"
