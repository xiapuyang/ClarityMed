"""Tests for the XGBoost tune CLI helpers.

The full sweep requires real DDXPlus data + a trained checkpoint, so
that lives outside the unit-test suite. Here we verify the pure
selection helper — the DSR-floor + IL-elbow gate that ports from
:mod:`.ddxplus.tune._elbow_maxstep`.
"""

from __future__ import annotations

from claritymed.ingest.symptoms.xgb.tune import _elbow_maxstep


def test_elbow_at_first_saturating_maxstep():
    """When marginal IL drops below SATURATION_RATE, return that ms."""
    # IL grows fast at first, then plateaus.
    il_by_ms = {4: 4.0, 6: 6.0, 8: 6.05, 10: 6.10}
    # 4→6 gain rate = 1.0 (above); 6→8 gain rate = 0.025 (below saturation).
    assert _elbow_maxstep([4, 6, 8, 10], il_by_ms) == 6


def test_elbow_falls_back_to_smallest_when_no_saturation():
    """Monotone IL growth with no plateau → smallest ms fallback."""
    il_by_ms = {4: 4.0, 6: 6.0, 8: 8.0, 10: 10.0}
    assert _elbow_maxstep([4, 6, 8, 10], il_by_ms) == 4


def test_elbow_with_unsorted_input():
    """Order of the input list shouldn't matter."""
    il_by_ms = {12: 6.10, 4: 4.0, 8: 6.05, 6: 6.0, 10: 6.08}
    assert _elbow_maxstep([12, 4, 8, 6, 10], il_by_ms) == 6
