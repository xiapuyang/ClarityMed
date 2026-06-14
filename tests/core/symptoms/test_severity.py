"""Tests for the severity → tier mapping."""

from __future__ import annotations

import pytest

from claritymed.core.symptoms.severity import (
    tier_for_differential,
    tier_for_severity,
)


@pytest.mark.parametrize(
    "severity,expected",
    [
        (1, "Critical"),
        (2, "Urgent"),
        (3, "Moderate"),
        (4, "Moderate"),
        (5, "Mild"),
    ],
)
def test_tier_for_severity(severity: int, expected: str) -> None:
    assert tier_for_severity(severity) == expected


@pytest.mark.parametrize("bad", [0, 6, -1, 99])
def test_tier_for_severity_out_of_range_raises(bad: int) -> None:
    with pytest.raises(ValueError, match="1-5"):
        tier_for_severity(bad)


def test_tier_for_differential_takes_max_urgency() -> None:
    """A Critical row dominates even when downstream rows are Mild."""
    assert tier_for_differential([5, 5, 1, 4]) == "Critical"
    assert tier_for_differential([3, 4, 2]) == "Urgent"
    assert tier_for_differential([5]) == "Mild"


def test_tier_for_differential_empty_defaults_to_mild() -> None:
    """No signal → Mild so the LLM falls back to a normal answer."""
    assert tier_for_differential([]) == "Mild"


def test_tier_for_differential_works_with_generator() -> None:
    rows = (r for r in [3, 1, 4])
    assert tier_for_differential(rows) == "Critical"


def test_tier_for_differential_propagates_out_of_range_error() -> None:
    """A 0 severity in the corpus should surface here, not silently degrade."""
    with pytest.raises(ValueError):
        tier_for_differential([5, 0])
