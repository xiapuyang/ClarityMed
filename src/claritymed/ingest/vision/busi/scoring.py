"""Progressive feasibility scoring for the BUSI vision pipeline.

The BUSI pipeline has **four** selection points, each operating on a
different budget:

1. ``hparam.py`` — Optuna search, ~5 epochs per trial. Tiny per-trial
   budget; goal is "is this HP combo *learnable*?", not "is it
   deployable?".
2. ``train.py`` — full training with early stopping, ~100 epochs.
   Goal is "is this checkpoint *within reach* of the deploy bar after
   inference-time tuning?".
3. ``tune.py`` — Optuna search over inference-time params (zero
   training budget). Goal is "is this operating point *at* the deploy
   bar?".
4. ``deploy.py`` — promotion gate. Goal: enforce "shippable" verbatim.

Using one set of floors across all four (the historical mistake) means
search burns trials in a region where *no* HP combo can clear the bar
in 5 epochs, and the optimizer has no gradient toward feasibility. The
fix is **progressive floors**: each phase has its own graduation bar
("good enough to invest in the next phase"), monotonically rising
toward the deploy floor. A trial that can't clear its phase's bar is
infeasible *at that phase*, regardless of where it sits on the
absolute scale.

The composite weights (0.6 recall / 0.4 dice) are unchanged; only the
gating floors differ across phases.

The principle "前一层floor没有达到，往下一层走大概率无效" — if the
previous layer's bar isn't met, going to the next layer is likely
wasted compute — is enforced by :func:`gate_or_raise` (called from
``pipeline.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Composite weights — read in one place so loss/eval/scoring code paths
# don't drift apart on "what does 'composite' mean?".
COMPOSITE_RECALL_WEIGHT = 0.6
COMPOSITE_DICE_WEIGHT = 0.4


@dataclass(frozen=True)
class PhaseFloors:
    """A phase's "good enough to invest in the next phase" bar.

    Three medically-meaningful metrics: malignant recall (don't miss
    the cancer), accuracy (don't overcall), dice (segmentation is
    useful, not noise). All three must be cleared for a trial /
    checkpoint / operating point to be feasible at this phase.
    """

    malignant_recall: float
    accuracy: float
    dice: float
    label: str  # short tag used in log messages and JSON output

    def deficits(self, breakdown: dict[str, float]) -> dict[str, float]:
        """Return ``{metric: floor - actual}`` for each violated floor.

        Empty dict ↔ feasible. Used by callers that need to surface
        precisely-which-floor-failed for the operator.
        """
        out: dict[str, float] = {}
        if breakdown["malignant_recall"] < self.malignant_recall:
            out["malignant_recall"] = (
                self.malignant_recall - breakdown["malignant_recall"]
            )
        if breakdown["accuracy"] < self.accuracy:
            out["accuracy"] = self.accuracy - breakdown["accuracy"]
        if breakdown["dice"] < self.dice:
            out["dice"] = self.dice - breakdown["dice"]
        return out


# Final deploy floors — "shippable". Tune's bar equals these because
# tune is the last selection step before promotion; if tune can't pull
# the model up to here, deploy is going to reject anyway.
FLOOR_MALIGNANT_RECALL = 0.85
FLOOR_ACCURACY = 0.85
FLOOR_DICE = 0.70

# Per-phase graduation bars. Rationale for the gap sizes:
#
# * search (~5 epochs): the model has barely started learning. The bar
#   is "shows signs of life" — recall + accuracy clearly above random
#   for 3-class (~0.33), dice clearly above noise (~0.10). A HP combo
#   that can't even get here in 5 epochs has no chance after 100.
#
# * train (~100 epochs, full run): the bar is "within tune's reach of
#   deploy". Tune can shift recall/accuracy a few points (operating
#   point), and dice 1-2 points (TTA + seg threshold). So train's
#   floor is ~5 points below deploy on recall/accuracy, ~8 below on
#   dice — enough headroom for tune to close, not so close that train
#   passes garbage downstream.
#
# * tune & deploy: identical. Tune's job is to land on the deploy bar.

SEARCH_FLOORS = PhaseFloors(
    malignant_recall=0.65,
    accuracy=0.65,
    dice=0.40,
    label="search",
)

TRAIN_FLOORS = PhaseFloors(
    malignant_recall=0.80,
    accuracy=0.80,
    dice=0.62,
    label="train",
)

TUNE_FLOORS = PhaseFloors(
    malignant_recall=FLOOR_MALIGNANT_RECALL,
    accuracy=FLOOR_ACCURACY,
    dice=FLOOR_DICE,
    label="tune",
)

DEPLOY_FLOORS = TUNE_FLOORS  # alias for callers that read intent rather than budget

# Feasible trials are lifted into ``(FEASIBLE_OFFSET, FEASIBLE_OFFSET + 1]``
# so they strictly beat any infeasible trial (which lives in
# ``(-inf, 0]``). The gap of ``[0, FEASIBLE_OFFSET]`` is intentional:
# any feasible trial wins over any infeasible one regardless of how
# close the latter got. Caller can detect "no feasible trial in the
# budget" via ``best_score < FEASIBLE_OFFSET``.
FEASIBLE_OFFSET = 1.0


def is_feasible(breakdown: dict[str, float], floors: PhaseFloors) -> bool:
    """Predicate matching :func:`feasibility_aware_score`'s feasible region."""
    return not floors.deficits(breakdown)


def feasibility_aware_score(breakdown: dict[str, float], floors: PhaseFloors) -> float:
    """Two-region scoring at the given phase's floors.

    Feasible breakdowns score ``FEASIBLE_OFFSET + composite``
    (in ``(1, 2]``). Among feasible breakdowns, higher composite wins.

    Infeasible breakdowns score ``-sum(deficit / floor)`` across the
    violated floors. Always ``≤ 0``; closer to zero is closer to
    feasibility. Optuna's acquisition function has a gradient on
    either side of the gap.

    Phases differ only in *what counts as feasible* — the composite
    used for the feasible-region ordering is the same everywhere.
    """
    deficits = floors.deficits(breakdown)
    if deficits:
        # Relative deficit per violated metric, summed. `getattr` reads
        # the matching floor field — PhaseFloors fields and deficits
        # keys share the same names by design.
        return -sum(
            deficit / getattr(floors, name) for name, deficit in deficits.items()
        )
    composite = (
        COMPOSITE_RECALL_WEIGHT * breakdown["malignant_recall"]
        + COMPOSITE_DICE_WEIGHT * breakdown["dice"]
    )
    return FEASIBLE_OFFSET + composite


def study_feasibility_summary(study, floors: PhaseFloors) -> dict[str, Any]:
    """Diagnostic snapshot of an Optuna study against a phase's floors.

    Returns ``{feasible_trials, infeasible_trials, worst_deficits}``
    where ``worst_deficits`` maps each violated metric to the worst
    deficit observed across all infeasible trials. Used to log a clear
    warning when a budget produced no feasible trial — the operator
    needs to know that searching/tuning harder won't help (the model
    is the bottleneck, not the search budget).

    Trial breakdowns must have been stashed on
    ``trial.user_attrs["breakdown"]`` by the objective; trials missing
    that attribute count as infeasible without contributing to
    ``worst_deficits``.
    """
    feasible = 0
    infeasible = 0
    worst_by_floor: dict[str, float] = {}
    for trial in study.trials:
        if trial.state.name != "COMPLETE":
            continue
        breakdown = trial.user_attrs.get("breakdown") or {}
        if breakdown and is_feasible(breakdown, floors):
            feasible += 1
            continue
        infeasible += 1
        for name, deficit in floors.deficits(breakdown or {}).items():
            if deficit > worst_by_floor.get(name, 0.0):
                worst_by_floor[name] = deficit
    return {
        "feasible_trials": feasible,
        "infeasible_trials": infeasible,
        "worst_deficits": worst_by_floor,
    }


class PhaseFloorGateError(SystemExit):
    """Raised when a phase's winner failed to clear that phase's floor.

    ``SystemExit`` (not a generic exception) so the pipeline CLI exits
    with a nonzero code without a traceback — the message itself is
    the operator-facing report.
    """


def gate_or_raise(
    *,
    phase_label: str,
    breakdown: dict[str, float],
    floors: PhaseFloors,
    force: bool = False,
) -> None:
    """Inter-phase prerequisite check.

    Called by ``pipeline.py`` between phases. If the winning breakdown
    of the just-finished phase doesn't clear *that phase's* floor, the
    next phase is almost certainly wasted compute (search budget that
    didn't make recall go above 0.65 in 5 epochs won't produce a
    checkpoint that hits 0.85 in 100 epochs; a checkpoint that didn't
    reach train's floor can't be tuned into the deploy floor).

    ``force=True`` downgrades the gate to a warning — for operators
    who want to push through (e.g. for diagnostic runs against a
    known-broken HP combo).
    """
    deficits = floors.deficits(breakdown)
    if not deficits:
        return
    summary = ", ".join(
        f"{name}={breakdown[name]:.3f} (floor {getattr(floors, name):.3f}, "
        f"deficit {deficit:.3f})"
        for name, deficit in deficits.items()
    )
    msg = (
        f"phase {phase_label!r} did not clear its floors: {summary}. "
        f"Going to the next phase is likely wasted compute. "
        f"Re-run with --force to override; otherwise iterate on the "
        f"current phase (more trials, wider search space, stronger model)."
    )
    if force:
        import logging

        logging.getLogger(__name__).warning("%s [forced through]", msg)
        return
    raise PhaseFloorGateError(msg)
