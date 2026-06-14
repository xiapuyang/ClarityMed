"""Symptoms feature ``AuditKind`` extension — typed vocabulary tests.

Covers the 10 kinds added by Unit 14 of the disease-prediction plan: every
kind must be accepted by ``AuditEvent`` construction, and a typo must fall
through to pydantic's ``ValidationError`` rather than emitting a
free-form event.
"""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError

from claritymed.context import apply_context, reset_context
from claritymed.core.observability.audit import AuditEvent, AuditKind, audit_event

SYMPTOMS_KINDS = (
    "tool.predict_disease_from_symptoms",
    "symptoms.session.started",
    "symptoms.session.turn",
    "symptoms.session.completed",
    "symptoms.session.cancelled",
    "symptoms.session.cap_hit",
    "symptoms.session.ineligible",
    "symptoms.eligibility.checked",
    "symptoms.eligibility.strategy_unavailable",
    "symptoms.safety_keywords.missing",
)


@pytest.mark.parametrize("kind", SYMPTOMS_KINDS)
def test_symptoms_kind_is_in_audit_kind_literal(kind: str) -> None:
    """Every documented symptoms kind appears in the ``AuditKind`` Literal."""
    assert kind in get_args(AuditKind)


@pytest.mark.parametrize("kind", SYMPTOMS_KINDS)
def test_audit_event_accepts_symptoms_kind(kind: str) -> None:
    """Construction round-trip — pydantic validates ``kind`` against the Literal."""
    event = AuditEvent(
        kind=kind,
        payload={},
        request_id="REQ-test",
        user_id="test",
        language="en",
    )
    assert event.kind == kind


def test_audit_event_rejects_bogus_symptoms_kind() -> None:
    """Typed-vocabulary guard: invented kinds are rejected at construction."""
    with pytest.raises(ValidationError):
        AuditEvent(
            kind="symptoms.bogus.kind",  # type: ignore[arg-type]
            payload={},
            request_id="REQ-test",
            user_id="test",
            language="en",
        )


def test_audit_event_via_helper_round_trips_through_logger() -> None:
    """``audit_event`` uses ContextVars + logs; verify the symptoms path works."""
    tokens = apply_context(
        request_id="REQ-symptoms-test",
        user_id="test",
        language="en",
    )
    try:
        event = audit_event(
            "symptoms.session.started",
            {
                "dataset_id": "ddxplus",
                "model_id": "typed_basd_v1",
                "session_id": "sess-abc",
            },
        )
    finally:
        reset_context(tokens)
    assert event.kind == "symptoms.session.started"
    assert event.user_id == "test"
    assert event.payload["dataset_id"] == "ddxplus"


def test_all_symptoms_kinds_distinct_from_existing_vocab() -> None:
    """Sanity check: the 10 new kinds collide with no pre-existing entry."""
    seen: set[str] = set()
    for k in get_args(AuditKind):
        assert k not in seen, f"duplicate AuditKind: {k!r}"
        seen.add(k)
