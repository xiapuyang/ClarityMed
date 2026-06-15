"""Progressive feasibility scoring across the BUSI pipeline.

The historical "raw composite" objective let a degenerate trial
(recall = 1, dice ≈ 0) win the search and silently hand the deploy
gate a doomed model. These tests pin both halves of the fix:

1. Two-region scoring at each phase's floors keeps the optimizer
   from preferring degenerate operating points.
2. Progressive floors (search < train < tune == deploy) keep early
   phases from punishing learnable HP combos for failing to converge
   in the per-trial budget.

A future refactor that re-merges the floors or drops the feasibility
region must fail these tests.
"""

from __future__ import annotations

import pytest

from claritymed.ingest.vision.busi import scoring
from claritymed.ingest.vision.busi.scoring import (
    DEPLOY_FLOORS,
    FEASIBLE_OFFSET,
    SEARCH_FLOORS,
    TRAIN_FLOORS,
    TUNE_FLOORS,
    PhaseFloorGateError,
    feasibility_aware_score,
    gate_or_raise,
    is_feasible,
)


def _bd(*, recall=0.0, accuracy=0.0, dice=0.0, composite=None):
    """Build a breakdown dict with sensible defaults.

    ``composite`` defaults to ``0.6 * recall + 0.4 * dice`` so tests
    don't have to compute it. Override when a test cares about an
    explicit composite value separate from the components.
    """
    if composite is None:
        composite = (
            scoring.COMPOSITE_RECALL_WEIGHT * recall
            + scoring.COMPOSITE_DICE_WEIGHT * dice
        )
    return {
        "malignant_recall": recall,
        "accuracy": accuracy,
        "dice": dice,
        "composite": composite,
    }


# --- progressive-floor invariants ---------------------------------------


def test_floors_increase_monotonically_across_phases() -> None:
    """Each later phase must demand at least as much as the earlier one."""
    for name in ("malignant_recall", "accuracy", "dice"):
        assert getattr(SEARCH_FLOORS, name) <= getattr(TRAIN_FLOORS, name)
        assert getattr(TRAIN_FLOORS, name) <= getattr(TUNE_FLOORS, name)


def test_tune_floor_equals_deploy_floor() -> None:
    """Tune is the last selection step; its bar must equal deploy's."""
    assert TUNE_FLOORS == DEPLOY_FLOORS


def test_deploy_module_re_exports_match_tune_floors() -> None:
    """deploy.py keeps three top-level constants for back-compat; they
    must equal TUNE_FLOORS so the gate code reads one source of truth.
    """
    from claritymed.ingest.vision.busi import deploy

    assert deploy.FLOOR_MALIGNANT_RECALL == TUNE_FLOORS.malignant_recall
    assert deploy.FLOOR_ACCURACY == TUNE_FLOORS.accuracy
    assert deploy.FLOOR_DICE == TUNE_FLOORS.dice


# --- feasibility region invariants --------------------------------------


def test_feasible_trial_scores_above_offset() -> None:
    """A trial meeting every floor scores in ``(FEASIBLE_OFFSET, FEASIBLE_OFFSET + 1]``."""
    breakdown = _bd(recall=0.9, accuracy=0.88, dice=0.72)
    score = feasibility_aware_score(breakdown, TUNE_FLOORS)
    assert score == pytest.approx(FEASIBLE_OFFSET + breakdown["composite"])
    assert score > FEASIBLE_OFFSET


def test_recall_only_degenerate_is_rejected_at_tune_floor() -> None:
    """The 2026-06-15 failure mode: recall=1, dice≈0, acc≈0.27.

    Must score worse than any feasible trial — otherwise Optuna's
    maximize would still pick this as the winner.
    """
    degenerate = _bd(recall=1.0, accuracy=0.27, dice=0.04)
    feasible = _bd(recall=0.85, accuracy=0.85, dice=0.70)
    assert feasibility_aware_score(degenerate, TUNE_FLOORS) < 0
    assert feasibility_aware_score(feasible, TUNE_FLOORS) >= FEASIBLE_OFFSET
    assert feasibility_aware_score(degenerate, TUNE_FLOORS) < (
        feasibility_aware_score(feasible, TUNE_FLOORS)
    )


def test_closer_to_feasibility_scores_higher_in_infeasible_region() -> None:
    """Infeasible trials still have a gradient toward feasibility."""
    near = _bd(recall=0.84, accuracy=0.86, dice=0.71)
    far = _bd(recall=0.50, accuracy=0.86, dice=0.71)
    near_score = feasibility_aware_score(near, TUNE_FLOORS)
    far_score = feasibility_aware_score(far, TUNE_FLOORS)
    assert near_score < 0 and far_score < 0
    assert near_score > far_score


def test_multiple_violations_aggregate() -> None:
    """Violating two floors penalizes more than violating one."""
    one = _bd(recall=0.75, accuracy=0.85, dice=0.70)
    two = _bd(recall=0.75, accuracy=0.75, dice=0.70)
    assert feasibility_aware_score(two, TUNE_FLOORS) < (
        feasibility_aware_score(one, TUNE_FLOORS)
    )


# --- progressive scoring: same breakdown, different phase = different verdict


def test_breakdown_passes_search_but_fails_tune() -> None:
    """The whole point of progressive floors: a 5-epoch HP combo with
    recall 0.70, dice 0.45, acc 0.70 is "alive" (search-feasible) but
    nowhere near deploy. The new scoring captures this.
    """
    breakdown = _bd(recall=0.70, accuracy=0.70, dice=0.45)
    assert is_feasible(breakdown, SEARCH_FLOORS)
    assert not is_feasible(breakdown, TRAIN_FLOORS)
    assert not is_feasible(breakdown, TUNE_FLOORS)
    assert feasibility_aware_score(breakdown, SEARCH_FLOORS) >= FEASIBLE_OFFSET
    assert feasibility_aware_score(breakdown, TRAIN_FLOORS) < 0
    assert feasibility_aware_score(breakdown, TUNE_FLOORS) < 0


def test_breakdown_passes_train_but_fails_tune() -> None:
    """A near-convergence checkpoint that needs tune to clear deploy."""
    breakdown = _bd(recall=0.82, accuracy=0.81, dice=0.64)
    assert is_feasible(breakdown, SEARCH_FLOORS)
    assert is_feasible(breakdown, TRAIN_FLOORS)
    assert not is_feasible(breakdown, TUNE_FLOORS)


# --- gate_or_raise ------------------------------------------------------


def test_gate_or_raise_passes_silently_when_feasible() -> None:
    """A breakdown meeting the phase's floors goes through with no exception."""
    breakdown = _bd(recall=0.90, accuracy=0.88, dice=0.72)
    gate_or_raise(
        phase_label="tune",
        breakdown=breakdown,
        floors=TUNE_FLOORS,
        force=False,
    )  # does not raise


def test_gate_or_raise_raises_phase_floor_gate_error_on_failure() -> None:
    """Default behaviour: hard-bail before the next phase runs."""
    breakdown = _bd(recall=0.50, accuracy=0.50, dice=0.20)
    with pytest.raises(PhaseFloorGateError) as exc:
        gate_or_raise(
            phase_label="search",
            breakdown=breakdown,
            floors=SEARCH_FLOORS,
            force=False,
        )
    msg = str(exc.value)
    # The message must surface the phase name, the violating metrics,
    # and the operator's next move — these are the breadcrumbs that
    # avoid a "why did my pipeline stop?" follow-up.
    assert "search" in msg
    assert any(m in msg for m in ("malignant_recall", "accuracy", "dice"))
    assert "--force" in msg


def test_gate_or_raise_warns_when_forced(caplog) -> None:
    """``force=True`` downgrades the gate to a warning, doesn't raise."""
    breakdown = _bd(recall=0.50, accuracy=0.50, dice=0.20)
    with caplog.at_level("WARNING"):
        gate_or_raise(
            phase_label="search",
            breakdown=breakdown,
            floors=SEARCH_FLOORS,
            force=True,
        )
    assert any("forced through" in r.message for r in caplog.records)


# --- study_feasibility_summary ------------------------------------------


class _FakeTrial:
    """Minimal stand-in for an Optuna trial — only the surfaces the
    summary helper actually reads.
    """

    def __init__(self, breakdown):
        self.user_attrs = {"breakdown": breakdown} if breakdown is not None else {}

        class _State:
            name = "COMPLETE"

        self.state = _State()


class _FakeStudy:
    def __init__(self, trials):
        self.trials = trials


def test_study_feasibility_summary_counts_against_named_phase() -> None:
    """Same trials, different phase floors → different feasibility counts."""
    trials = [
        _FakeTrial(_bd(recall=0.70, accuracy=0.70, dice=0.45)),  # search only
        _FakeTrial(_bd(recall=0.82, accuracy=0.82, dice=0.64)),  # train
        _FakeTrial(_bd(recall=0.90, accuracy=0.90, dice=0.75)),  # tune
    ]
    study = _FakeStudy(trials)
    assert (
        scoring.study_feasibility_summary(study, SEARCH_FLOORS)["feasible_trials"] == 3
    )
    assert scoring.study_feasibility_summary(study, TUNE_FLOORS)["feasible_trials"] == 1


def test_study_feasibility_summary_reports_worst_deficit() -> None:
    """Worst-deficit-per-metric across infeasible trials."""
    trials = [
        _FakeTrial(_bd(recall=0.70, accuracy=0.90, dice=0.90)),  # recall short by .15
        _FakeTrial(_bd(recall=0.80, accuracy=0.50, dice=0.90)),  # acc short by .35
    ]
    summary = scoring.study_feasibility_summary(_FakeStudy(trials), TUNE_FLOORS)
    assert summary["feasible_trials"] == 0
    assert summary["infeasible_trials"] == 2
    assert summary["worst_deficits"]["malignant_recall"] == pytest.approx(0.15)
    assert summary["worst_deficits"]["accuracy"] == pytest.approx(0.35)
