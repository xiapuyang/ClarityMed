"""Tests for ``GroundedAnswer`` plus the pydantic-ai ``result_type`` smoke test."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from claritymed.core.schemas import (
    Citation,
    Disclaimer,
    GroundedAnswer,
    UncertaintyResult,
)


def _uncertainty(level: str = "medium") -> UncertaintyResult:
    return UncertaintyResult(
        type="epistemic",
        source="retrieval",
        level=level,  # type: ignore[arg-type]
        reasons=["context recall thin"],
    )


def _answer(**overrides) -> GroundedAnswer:
    payload = {
        "request_id": "20260606222522A1B2C3D4",
        "user_id": "alice",
        "language": "en",
        "text": "Try drinking water and resting; if pain persists, see a clinician.",
        "citations": [
            Citation(
                source_id="med-123",
                title="Patient guide: tension headache",
                language="en",
            )
        ],
        "uncertainty": _uncertainty("low"),
        "disclaimer": Disclaimer(text="General guidance only.", language="en"),
    }
    payload.update(overrides)
    return GroundedAnswer(**payload)


def test_happy_path():
    a = _answer()
    assert a.language == "en"


def test_round_trip_json():
    a = _answer()
    restored = GroundedAnswer.model_validate_json(a.model_dump_json())
    assert restored == a


def test_no_citations_requires_uncertainty():
    """The "no evidence cannot be confident" contract."""
    with pytest.raises(ValidationError):
        _answer(citations=[], uncertainty=_uncertainty("low"))


def test_no_citations_with_medium_uncertainty_ok():
    a = _answer(citations=[], uncertainty=_uncertainty("medium"))
    assert a.citations == []


def test_invalid_language_rejected():
    with pytest.raises(ValidationError):
        _answer(language="ja")  # type: ignore[arg-type]


def test_empty_text_rejected():
    with pytest.raises(ValidationError):
        _answer(text="")


def test_request_id_pattern_enforced():
    with pytest.raises(ValidationError):
        _answer(request_id="a1b2c3d4")  # old 8-hex form


def test_citation_authority_tier_defaults_to_general():
    """Existing citations that omit authority_tier stay schema-valid."""
    c = Citation(source_id="med-1", title="Tension headache guide", language="en")
    assert c.authority_tier == 3


def test_citation_authority_tier_accepts_all_three_levels():
    for tier in (1, 2, 3):
        c = Citation(
            source_id=f"src-{tier}",
            title="ref",
            language="en",
            authority_tier=tier,
        )
        assert c.authority_tier == tier


def test_citation_authority_tier_rejects_out_of_range():
    for bad in (0, 4, -1):
        with pytest.raises(ValidationError):
            Citation(
                source_id="src",
                title="ref",
                language="en",
                authority_tier=bad,
            )


def test_grounded_answer_works_as_pydantic_ai_result_type():
    """Foundation contract: GroundedAnswer can be a pydantic-ai Agent output_type.

    We feed TestModel a hand-built valid payload because TestModel's synthetic
    generator cannot satisfy our cross-field validators (e.g., "no citations
    means uncertainty >= medium"). The test still proves:

    1. The schema is accepted by pydantic-ai as an output_type.
    2. A valid GroundedAnswer round-trips through the Agent runtime.
    """
    valid_payload = {
        "request_id": "20260606222522A1B2C3D4",
        "user_id": "alice",
        "language": "en",
        "text": "Drink water and rest.",
        "citations": [
            {
                "source_id": "med-1",
                "title": "Tension headache guide",
                "language": "en",
            }
        ],
        "uncertainty": {
            "type": "epistemic",
            "source": "retrieval",
            "level": "low",
            "reasons": ["context recall thin"],
        },
        "disclaimer": {"text": "General guidance only.", "language": "en"},
        "provenance": {},
    }
    agent = Agent(
        model=TestModel(custom_output_args=valid_payload),
        output_type=GroundedAnswer,
    )
    result = agent.run_sync("does not matter")
    assert isinstance(result.output, GroundedAnswer)
    assert result.output.user_id == "alice"
