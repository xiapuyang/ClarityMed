"""Generic feasibility scoring exercised against BUSI's floors.

Lives under ``busi/`` because the BUSI U-Net ModelSpec is the test
fixture; the scoring math itself is now in
:mod:`claritymed.ingest.vision.forge.scoring` and shared by every
forge-built model. A future refactor that re-merges the floors or
drops the feasibility region must fail these tests.
"""

from __future__ import annotations

import pytest

from claritymed.ingest.vision.busi.models.unet_resnet50 import UNET_RESNET50
from claritymed.ingest.vision.forge.scoring import (
    FEASIBLE_OFFSET,
    PhaseFloorGateError,
    feasibility_aware_score,
    gate_or_raise,
    is_feasible,
    study_feasibility_summary,
)

_TASK = UNET_RESNET50.task
_SEARCH = _TASK.phase_floors("search")
_TRAIN = _TASK.phase_floors("train")
_TUNE = _TASK.phase_floors("tune")
_WEIGHTS = _TASK.composite_weights


def _bd(*, recall=0.0, accuracy=0.0, dice=0.0, composite=None):
    """Build a breakdown dict with sensible defaults.

    ``composite`` defaults to BUSI's historical formula
    ``0.6 * malignant_recall + 0.4 * dice`` so tests don't have to
    compute it. Override when a test cares about an explicit composite
    value separate from the components.
    """
    if composite is None:
        composite = _WEIGHTS["malignant_recall"] * recall + _WEIGHTS["dice"] * dice
    return {
        "malignant_recall": recall,
        "accuracy": accuracy,
        "dice": dice,
        "composite": composite,
    }


# --- progressive-floor invariants ----------------------------------------


def test_floors_increase_monotonically_across_phases() -> None:
    """Each later phase must demand at least as much as the earlier one."""
    for name in ("malignant_recall", "accuracy", "dice"):
        assert _SEARCH.floors[name] <= _TRAIN.floors[name]
        assert _TRAIN.floors[name] <= _TUNE.floors[name]


def test_tune_floor_equals_deploy_floor() -> None:
    """Tune is the last selection step; its bar must equal deploy's."""
    assert _TUNE.floors == _TASK.deploy_floors_map()


# --- feasibility region invariants --------------------------------------


def test_feasible_trial_scores_above_offset() -> None:
    breakdown = _bd(recall=0.9, accuracy=0.88, dice=0.72)
    score = feasibility_aware_score(breakdown, _TUNE, _WEIGHTS)
    assert score == pytest.approx(FEASIBLE_OFFSET + breakdown["composite"])
    assert score > FEASIBLE_OFFSET


def test_recall_only_degenerate_is_rejected_at_tune_floor() -> None:
    """The 2026-06-15 failure mode: recall=1, dice≈0, acc≈0.27."""
    degenerate = _bd(recall=1.0, accuracy=0.27, dice=0.04)
    feasible = _bd(recall=0.92, accuracy=0.85, dice=0.70)
    assert feasibility_aware_score(degenerate, _TUNE, _WEIGHTS) < 0
    assert feasibility_aware_score(feasible, _TUNE, _WEIGHTS) >= FEASIBLE_OFFSET
    assert feasibility_aware_score(degenerate, _TUNE, _WEIGHTS) < (
        feasibility_aware_score(feasible, _TUNE, _WEIGHTS)
    )


def test_closer_to_feasibility_scores_higher_in_infeasible_region() -> None:
    """Infeasible trials still have a gradient toward feasibility."""
    near = _bd(recall=0.84, accuracy=0.86, dice=0.71)
    far = _bd(recall=0.50, accuracy=0.86, dice=0.71)
    near_score = feasibility_aware_score(near, _TUNE, _WEIGHTS)
    far_score = feasibility_aware_score(far, _TUNE, _WEIGHTS)
    assert near_score < 0 and far_score < 0
    assert near_score > far_score


def test_multiple_violations_aggregate() -> None:
    one = _bd(recall=0.75, accuracy=0.85, dice=0.70)
    two = _bd(recall=0.75, accuracy=0.75, dice=0.70)
    assert feasibility_aware_score(two, _TUNE, _WEIGHTS) < (
        feasibility_aware_score(one, _TUNE, _WEIGHTS)
    )


# --- progressive scoring -------------------------------------------------


def test_breakdown_passes_search_but_fails_tune() -> None:
    """5-epoch HP combo at the search bar, short of train & tune."""
    breakdown = _bd(recall=0.82, accuracy=0.70, dice=0.45)
    assert is_feasible(breakdown, _SEARCH)
    assert not is_feasible(breakdown, _TRAIN)
    assert not is_feasible(breakdown, _TUNE)
    assert feasibility_aware_score(breakdown, _SEARCH, _WEIGHTS) >= FEASIBLE_OFFSET
    assert feasibility_aware_score(breakdown, _TRAIN, _WEIGHTS) < 0
    assert feasibility_aware_score(breakdown, _TUNE, _WEIGHTS) < 0


def test_breakdown_passes_train_but_fails_tune() -> None:
    breakdown = _bd(recall=0.87, accuracy=0.81, dice=0.64)
    assert is_feasible(breakdown, _SEARCH)
    assert is_feasible(breakdown, _TRAIN)
    assert not is_feasible(breakdown, _TUNE)


# --- gate_or_raise ------------------------------------------------------


def test_gate_or_raise_passes_silently_when_feasible() -> None:
    breakdown = _bd(recall=0.90, accuracy=0.88, dice=0.72)
    gate_or_raise(phase_label="tune", breakdown=breakdown, floors=_TUNE, force=False)


def test_gate_or_raise_raises_phase_floor_gate_error_on_failure() -> None:
    breakdown = _bd(recall=0.50, accuracy=0.50, dice=0.20)
    with pytest.raises(PhaseFloorGateError) as exc:
        gate_or_raise(
            phase_label="search", breakdown=breakdown, floors=_SEARCH, force=False
        )
    msg = str(exc.value)
    assert "search" in msg
    assert any(m in msg for m in ("malignant_recall", "accuracy", "dice"))
    assert "--force" in msg


def test_gate_or_raise_warns_when_forced(caplog) -> None:
    breakdown = _bd(recall=0.50, accuracy=0.50, dice=0.20)
    with caplog.at_level("WARNING"):
        gate_or_raise(
            phase_label="search", breakdown=breakdown, floors=_SEARCH, force=True
        )
    assert any("forced through" in r.message for r in caplog.records)


# --- study_feasibility_summary ------------------------------------------


class _FakeTrial:
    def __init__(self, breakdown):
        self.user_attrs = {"breakdown": breakdown} if breakdown is not None else {}

        class _State:
            name = "COMPLETE"

        self.state = _State()


class _FakeStudy:
    def __init__(self, trials):
        self.trials = trials


def test_study_feasibility_summary_counts_against_named_phase() -> None:
    trials = [
        _FakeTrial(_bd(recall=0.80, accuracy=0.70, dice=0.45)),  # search only
        _FakeTrial(_bd(recall=0.85, accuracy=0.82, dice=0.55)),  # train
        _FakeTrial(_bd(recall=0.92, accuracy=0.90, dice=0.75)),  # tune
    ]
    study = _FakeStudy(trials)
    assert study_feasibility_summary(study, _SEARCH)["feasible_trials"] == 3
    assert study_feasibility_summary(study, _TUNE)["feasible_trials"] == 1


def test_study_feasibility_summary_reports_worst_deficit() -> None:
    trials = [
        _FakeTrial(_bd(recall=0.70, accuracy=0.90, dice=0.90)),  # recall short by .20
        _FakeTrial(_bd(recall=0.80, accuracy=0.50, dice=0.90)),  # acc short by .35
    ]
    summary = study_feasibility_summary(_FakeStudy(trials), _TUNE)
    assert summary["feasible_trials"] == 0
    assert summary["infeasible_trials"] == 2
    assert summary["worst_deficits"]["malignant_recall"] == pytest.approx(0.20)
    assert summary["worst_deficits"]["accuracy"] == pytest.approx(0.35)
