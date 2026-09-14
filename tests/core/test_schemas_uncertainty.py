"""Tests for ``UncertaintyResult``."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from claritymed.core.schemas import UncertaintyResult


def _ok(**overrides) -> UncertaintyResult:
    payload = {
        "type": "epistemic",
        "source": "retrieval",
        "level": "medium",
        "reasons": ["top-1 and top-2 documents disagree"],
    }
    payload.update(overrides)
    return UncertaintyResult(**payload)


def test_happy_path_minimal_construction():
    u = _ok()
    assert u.provenance == {}
    assert u.score is None


def test_score_none_allowed_with_high_level():
    u = _ok(level="high", score=None)
    assert u.level == "high"


def test_score_out_of_range_rejected():
    with pytest.raises(ValidationError):
        _ok(score=1.5)


def test_type_literal_enforced():
    with pytest.raises(ValidationError):
        _ok(type="random")


def test_frozen_blocks_mutation():
    u = _ok()
    with pytest.raises(ValidationError):
        u.level = "low"  # type: ignore[misc]


def test_reasons_must_not_be_empty():
    with pytest.raises(ValidationError):
        _ok(reasons=[])


def test_extra_fields_rejected():
    with pytest.raises(ValidationError):
        UncertaintyResult(
            type="epistemic",
            source="retrieval",
            level="medium",
            reasons=["r"],
            phi_field="leaking",  # type: ignore[call-arg]
        )


def test_fuse_is_explicitly_deferred():
    with pytest.raises(NotImplementedError):
        UncertaintyResult.fuse([_ok(), _ok()])
