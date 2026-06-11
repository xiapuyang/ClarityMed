"""Tests for ``claritymed.core.observability.audit``."""

from __future__ import annotations

import json

import pytest

from claritymed.context import (
    MissingContextError,
    apply_context,
    reset_context,
)
from claritymed.core.observability.audit import AuditEvent, audit_event


def test_happy_event_round_trips_via_json(caplog):
    import logging

    tokens = apply_context("20260606222522A1B2C3D4", "alice", "en")
    try:
        with caplog.at_level(logging.INFO, logger="claritymed.audit"):
            ev = audit_event("retrieval", payload={"chunks": ["a", "b"]})
    finally:
        reset_context(tokens)
    assert ev.kind == "retrieval"

    audit_records = [r for r in caplog.records if r.name == "claritymed.audit"]
    assert audit_records, "No claritymed.audit records captured"
    parsed = json.loads(audit_records[-1].getMessage())
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


def test_one_line_per_event(caplog):
    import logging

    tokens = apply_context("20260606222522A1B2C3D4", "alice", "en")
    try:
        with caplog.at_level(logging.INFO, logger="claritymed.audit"):
            for i in range(20):
                audit_event("tool_invoke", payload={"i": i})
    finally:
        reset_context(tokens)

    audit_records = [r for r in caplog.records if r.name == "claritymed.audit"]
    assert len(audit_records) == 20, (
        f"Expected 20 audit records, got {len(audit_records)}"
    )
    for record in audit_records:
        parsed = json.loads(record.getMessage())
        assert parsed["kind"] == "tool_invoke"
