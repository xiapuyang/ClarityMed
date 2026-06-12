"""Regression tests for ``ToolDispatcher.gate`` argument validation.

Captures the exact failure modes observed during e2e against the
Qwen3.6-MLX 35B local model (issue: every list / bool / number value
arrives as a quoted JSON string instead of its native type). The
dispatcher must reject these cleanly with an error string the LLM can
read and self-correct on, not a raw pydantic ``ValidationError`` dump
that the model gives up on after a few retries.

If pydantic-ai's error-handling contract changes, or if the dispatcher
swallows the validation error somewhere, these tests catch the
regression at unit-test speed instead of waiting for an e2e flake.
"""

from __future__ import annotations

import pytest

from claritymed.context import apply_context, reset_context
from claritymed.errors import PathOutsideUserDomain, UnknownSha256
from claritymed.orchestrator.services.tool_dispatcher import ToolDispatcher


@pytest.fixture
def _ctx():
    tokens = apply_context("20260612000000ABCDEF12", "test", "en")
    yield
    reset_context(tokens)


@pytest.fixture
def _gate() -> ToolDispatcher:
    return ToolDispatcher()


# --- string-coerced primitives (the Qwen3.6 anti-pattern) -------------


def test_gate_rejects_string_coerced_list(_ctx, _gate):
    """``"attachments": "[]"`` (string) instead of ``[]`` (list)."""
    with pytest.raises(ValueError, match=r"invalid args for save_record"):
        _gate.gate(
            "save_record",
            {
                "category": "checkups",
                "kind": "checkup",
                "title": "Annual",
                "attachments": "[]",
            },
        )


def test_gate_accepts_string_coerced_bool(_ctx, _gate):
    """``"public": "false"`` is coerced by pydantic to ``False`` and
    accepted. This is intentionally lenient — the model's string-bool
    confusion costs nothing at the gate, so we don't over-fit the
    schema. Pin the behavior so a future ``strict=True`` flip is a
    deliberate trade-off, not a silent break.
    """
    # Should NOT raise.
    _gate.gate(
        "save_to_library",
        {"title": "Some title", "public": "false"},
    )


def test_gate_rejects_string_coerced_year(_ctx, _gate):
    """``"year": "abc"`` is not a valid int."""
    with pytest.raises(ValueError, match=r"invalid args for save_to_library"):
        _gate.gate(
            "save_to_library",
            {"title": "Some title", "year": "abc"},
        )


# --- error message readability ----------------------------------------


def test_gate_error_message_names_offending_field(_ctx, _gate):
    """The error string must surface the field name so the LLM can fix
    just that field on retry. The exact phrasing of pydantic's message
    is allowed to drift; field name visibility is the contract."""
    with pytest.raises(ValueError) as exc_info:
        _gate.gate(
            "save_record",
            {
                "category": "checkups",
                "kind": "checkup",
                "title": "Annual",
                "attachments": "[]",
            },
        )
    msg = str(exc_info.value)
    assert "attachments" in msg, f"error must name the offending field; got: {msg!r}"


def test_gate_error_message_names_tool(_ctx, _gate):
    """Tool name must appear in the error so a multi-tool turn surfaces
    which call failed in the model's view of the conversation."""
    with pytest.raises(ValueError, match=r"save_allergy"):
        _gate.gate("save_allergy", {"substance": "x"})  # missing required fields


# --- enum / literal violations ----------------------------------------


def test_gate_rejects_invalid_severity_literal(_ctx, _gate):
    """``severity`` is a Literal; ``"Severe"`` (capitalized) must fail."""
    with pytest.raises(ValueError, match=r"save_allergy"):
        _gate.gate(
            "save_allergy",
            {
                "substance": "penicillin",
                "severity": "Severe",
                "source": "self_report",
            },
        )


def test_gate_rejects_non_whitelisted_profile_field(_ctx, _gate):
    """``field=age`` is not in ``ProfileField``."""
    with pytest.raises(ValueError, match=r"update_profile_field"):
        _gate.gate(
            "update_profile_field",
            {"field": "age", "value": 35},
        )


# --- extra-field rejection (extra="forbid") ---------------------------


def test_gate_rejects_hallucinated_field(_ctx, _gate):
    """Models sometimes copy a field from a different tool's schema
    (``confirm_kind`` belongs to ``delete_record``, not ``save_record``).
    ``extra="forbid"`` must catch this."""
    with pytest.raises(ValueError, match=r"save_record"):
        _gate.gate(
            "save_record",
            {
                "category": "checkups",
                "kind": "checkup",
                "title": "Annual",
                "confirm_kind": "checkup",
            },
        )


# --- typed errors that should NOT fall through to ValueError ----------


def test_gate_raises_unknown_sha256_not_value_error(_ctx, _gate):
    """``UnknownSha256`` is a typed exception the orchestrator surfaces
    distinctly; it must not be re-wrapped as a generic ``ValueError``."""
    with pytest.raises(UnknownSha256):
        _gate.gate(
            "save_record",
            {
                "category": "checkups",
                "kind": "checkup",
                "title": "Annual",
                "attachments": [{"sha256": "0" * 64, "filename": "x.pdf"}],
            },
        )


def test_gate_raises_path_outside_domain_not_value_error(_ctx, _gate):
    """Absolute / parent-escape paths must surface as their typed
    exception."""
    with pytest.raises(PathOutsideUserDomain):
        _gate.gate(
            "delete_record",
            {
                "record_path": "/etc/passwd",
                "confirm_kind": "checkup",
            },
        )
