"""Regression tests for ``ToolDispatcher.gate`` argument validation.

Captures the exact failure modes observed during e2e against the
Qwen3.6-MLX 35B local model (issue: every list / bool / number value
arrives as a quoted JSON string instead of its native type). The
dispatcher must surface these as ``pydantic_ai.exceptions.ModelRetry``
so the tool loop hands the message back to the model for self-correction
on the next agent step. Plain ``ValueError`` would escape ``agent.run``
and abort the whole turn — small-model parameter typos would never get
the one retry they need.

Security / integrity errors (``UnknownSha256``, ``PathOutsideUserDomain``)
deliberately do NOT get re-wrapped as ``ModelRetry`` — those are not
"the model emitted a typo" failures, and inviting the model to retry
sha guesses or path-traversal attempts would defeat the gate.
"""

from __future__ import annotations

import pytest
from pydantic_ai.exceptions import ModelRetry

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
    with pytest.raises(ModelRetry, match=r"invalid args for save_record"):
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
    with pytest.raises(ModelRetry, match=r"invalid args for save_to_library"):
        _gate.gate(
            "save_to_library",
            {"title": "Some title", "year": "abc"},
        )


# --- error message readability ----------------------------------------


def test_gate_error_message_names_offending_field(_ctx, _gate):
    """The error string must surface the field name so the LLM can fix
    just that field on retry. The exact phrasing of pydantic's message
    is allowed to drift; field name visibility is the contract."""
    with pytest.raises(ModelRetry) as exc_info:
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
    with pytest.raises(ModelRetry, match=r"save_allergy"):
        _gate.gate("save_allergy", {"substance": "x"})  # missing required fields


# --- retry-message coaching for small-model failure modes -------------


def test_gate_error_message_teaches_array_literal_not_string(_ctx, _gate):
    """The dominant local-model failure: ``'[]'`` (string) for a list field.

    Raw pydantic phrasing (``Input should be a valid list``) doesn't
    teach Qwen3-MLX 35B how to fix the call — it loops emitting the
    same stringified value. The retry message must explicitly say
    "JSON array, not quoted string" so the next retry has a concrete
    edit target."""
    with pytest.raises(ModelRetry) as exc_info:
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
    assert "JSON array" in msg, f"hint must name the expected type; got: {msg!r}"
    assert "not" in msg and "string" in msg, (
        f"hint must contrast array vs string; got: {msg!r}"
    )


def test_gate_error_message_suggests_field_for_typo(_ctx, _gate):
    """``extra_forbidden`` with a near-miss field name should yield a
    "Did you mean" hint via difflib. Observed in bench: model passes
    ``name='penicillin'`` to ``save_allergy`` (real field: ``substance``)
    or ``type='object'`` (no real match) to ``save_record``."""
    with pytest.raises(ModelRetry) as exc_info:
        _gate.gate(
            "save_record",
            {
                "category": "checkups",
                "kind": "checkup",
                "title": "Annual",
                "confirm_kind": "checkup",  # belongs to delete_record
            },
        )
    msg = str(exc_info.value)
    assert "Valid fields" in msg, f"typo hint must list valid fields; got: {msg!r}"
    # ``confirm_kind`` is closest to ``kind`` by difflib's ratio.
    assert "Did you mean" in msg or "kind" in msg, (
        f"typo hint should suggest closest match; got: {msg!r}"
    )


def test_gate_error_message_flags_missing_field(_ctx, _gate):
    """``missing`` errors must surface as "required — include it" so
    the model knows the fix is to ADD the field, not change a value."""
    with pytest.raises(ModelRetry) as exc_info:
        _gate.gate("save_allergy", {"substance": "x"})  # no severity, no source
    msg = str(exc_info.value)
    assert "required" in msg, f"hint must say required; got: {msg!r}"
    assert "severity" in msg, f"hint must name the missing field; got: {msg!r}"


# --- enum / literal violations ----------------------------------------


def test_gate_rejects_invalid_severity_literal(_ctx, _gate):
    """``severity`` is a Literal; ``"Severe"`` (capitalized) must fail."""
    with pytest.raises(ModelRetry, match=r"save_allergy"):
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
    with pytest.raises(ModelRetry, match=r"update_profile_field"):
        _gate.gate(
            "update_profile_field",
            {"field": "age", "value": 35},
        )


# --- extra-field rejection (extra="forbid") ---------------------------


def test_gate_rejects_hallucinated_field(_ctx, _gate):
    """Models sometimes copy a field from a different tool's schema
    (``confirm_kind`` belongs to ``delete_record``, not ``save_record``).
    ``extra="forbid"`` must catch this."""
    with pytest.raises(ModelRetry, match=r"save_record"):
        _gate.gate(
            "save_record",
            {
                "category": "checkups",
                "kind": "checkup",
                "title": "Annual",
                "confirm_kind": "checkup",
            },
        )


# --- string-"None"/"null" sentinel normalization ----------------------


def test_validate_args_logs_warning_on_failure(_ctx, _gate, caplog):
    """A validation failure must emit a structured warning to the
    ``claritymed`` logger (which routes to ``app.log``) carrying tool
    name + the cleaned args + the pydantic error list. A maintainer
    grepping app.log for a single retry cycle should be able to read
    each attempt's args and root cause without re-running the session."""
    import logging

    with caplog.at_level(
        logging.WARNING, logger="claritymed.orchestrator.services.tool_dispatcher"
    ):
        with pytest.raises(ModelRetry):
            _gate.gate(
                "save_allergy",
                {"substance": "p", "severity": "bogus", "source": "self_report"},
            )
    matching = [
        r
        for r in caplog.records
        if r.name == "claritymed.orchestrator.services.tool_dispatcher"
        and "tool_args_invalid" in r.getMessage()
    ]
    assert matching, "expected a tool_args_invalid warning"
    msg = matching[0].getMessage()
    assert "tool=save_allergy" in msg, msg
    assert "severity" in msg, "args dict must appear in the log line"


def test_gate_accepts_string_none_for_optional_date(_ctx, _gate):
    """Small local models often emit ``"onset_date": "None"`` (Python
    literal as a string) instead of omitting the field. The dispatcher
    must normalize this to real ``None`` before pydantic sees it; the
    previous behavior raised ``date_from_datetime_parsing`` on the
    six-character string."""
    # Should NOT raise.
    _gate.gate(
        "save_allergy",
        {
            "substance": "penicillin",
            "severity": "moderate",
            "source": "self_report",
            "onset_date": "None",
            "end_date": "null",
        },
    )


def test_gate_accepts_string_null_in_nested_list(_ctx, _gate):
    """The normalizer walks nested dicts/lists so sentinels inside
    ``extracted_labs[*]`` don't slip past — these fields are typed
    ``float | None`` and would otherwise fail Decimal coercion on
    the literal ``"None"`` / ``"null"`` strings."""
    # Should NOT raise.
    _gate.gate(
        "save_record",
        {
            "category": "checkups",
            "kind": "checkup",
            "title": "Annual",
            "extracted_labs": [
                {
                    "name": "LDL",
                    "value": 110.0,
                    "unit": "mg/dL",
                    "ref_low": "None",
                    "ref_high": "null",
                    "flag": "None",
                }
            ],
        },
    )


# --- typed errors that must NOT get re-wrapped as a retry --------------


def test_gate_raises_unknown_sha256_not_retry(_ctx, _gate):
    """``UnknownSha256`` is a typed exception the orchestrator surfaces
    distinctly. It is a security / integrity error, not an LLM args
    typo, so it must escape the dispatcher as-is — *not* get re-wrapped
    as ``ModelRetry`` (which would let the model keep guessing shas
    until it stumbles onto a real one) and *not* as a plain
    ``ValueError`` (which would lose its semantic category)."""
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


def test_gate_raises_path_outside_domain_not_retry(_ctx, _gate):
    """Absolute / parent-escape paths must surface as their typed
    exception — same rationale as the sha test above: a path-traversal
    attempt is not something the model should be invited to retry."""
    with pytest.raises(PathOutsideUserDomain):
        _gate.gate(
            "delete_record",
            {
                "record_path": "/etc/passwd",
                "confirm_kind": "checkup",
            },
        )
