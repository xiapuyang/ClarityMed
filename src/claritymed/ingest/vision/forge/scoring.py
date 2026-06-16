"""Generic feasibility scoring shared by every forge task.

Each :class:`~claritymed.ingest.vision.forge.tasks.base.Task` plugs in
its own per-phase floor dicts and composite weights; the math here is
the same as ``ingest/vision/busi/scoring.py`` was — only the field
names are pulled off the Task at call time rather than baked into a
per-dataset dataclass.

Three concepts:

* :class:`PhaseFloors` — ``{metric_name: minimum_value}`` plus a
  short label used in log lines.
* :func:`feasibility_aware_score` — two-region scoring. Feasible
  trials (every floor met) score ``FEASIBLE_OFFSET + composite``,
  in ``(1, 2]``. Infeasible ones score ``-sum(deficit/floor)``,
  always ``≤ 0``; closer to zero is closer to feasibility.
* :func:`gate_or_raise` — inter-phase prerequisite. Raises
  :class:`PhaseFloorGateError` when the previous phase's winner
  didn't clear its floor and ``force=False`` (default), or logs a
  warning when ``force=True``.

Feasible trials always strictly beat infeasible ones — Optuna's
``maximize`` direction naturally lands on the best feasible operating
point when one exists, while infeasible trials still provide a
gradient on the deficit side. See ``docs/vision-model-workflow.md``
for the rationale.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Gap between the infeasible region (``(-inf, 0]``) and the feasible
# one (``(FEASIBLE_OFFSET, FEASIBLE_OFFSET + 1]``). Any feasible trial
# wins over any infeasible one — callers can detect "no feasible trial
# in budget" via ``best_score < FEASIBLE_OFFSET``.
FEASIBLE_OFFSET = 1.0


@dataclass(frozen=True)
class PhaseFloors:
    """A phase's "good enough to invest in the next phase" bar.

    ``floors`` carries ``{metric_name: minimum_value}``. The Task
    decides which metrics it scores (cls-only uses
    ``cancer_recall + accuracy``; cls+seg uses
    ``malignant_recall + accuracy + dice``); this struct stays generic.

    ``label`` is the short phase tag used in log lines and JSON output
    (``"search"`` / ``"train"`` / ``"tune"``).
    """

    floors: dict[str, float]
    label: str

    def deficits(self, breakdown: dict[str, float]) -> dict[str, float]:
        """Return ``{metric: floor - actual}`` for each violated floor.

        Empty dict ↔ feasible. Used by ``gate_or_raise`` and by
        ``study_feasibility_summary`` to surface precisely-which-floor
        failed for the operator.
        """
        out: dict[str, float] = {}
        for name, floor in self.floors.items():
            actual = breakdown.get(name, 0.0)
            if actual < floor:
                out[name] = floor - actual
        return out


def is_feasible(breakdown: dict[str, float], floors: PhaseFloors) -> bool:
    """Predicate matching :func:`feasibility_aware_score`'s feasible region."""
    return not floors.deficits(breakdown)


def feasibility_aware_score(
    breakdown: dict[str, float],
    floors: PhaseFloors,
    composite_weights: dict[str, float],
) -> float:
    """Two-region scoring at the given phase's floors.

    Feasible breakdowns score ``FEASIBLE_OFFSET + composite``, where
    ``composite = sum(weight[m] * breakdown[m] for m in weights)``.
    Infeasible breakdowns score ``-sum(deficit / floor)`` across the
    violated floors. Always ``≤ 0``; closer to zero is closer to
    feasibility.
    """
    deficits = floors.deficits(breakdown)
    if deficits:
        return -sum(deficit / floors.floors[name] for name, deficit in deficits.items())
    composite = sum(
        weight * breakdown.get(metric, 0.0)
        for metric, weight in composite_weights.items()
    )
    return FEASIBLE_OFFSET + composite


def study_feasibility_summary(study, floors: PhaseFloors) -> dict[str, Any]:
    """Diagnostic snapshot of an Optuna study against a phase's floors.

    Returns ``{feasible_trials, infeasible_trials, worst_deficits}``
    where ``worst_deficits`` maps each violated metric to the worst
    deficit observed across all infeasible trials. Used to log a clear
    warning when a budget produced no feasible trial — the operator
    needs to know that searching harder won't help.

    Trial breakdowns must have been stashed on
    ``trial.user_attrs["breakdown"]`` by the objective. Trials missing
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

    Subclassing ``SystemExit`` so the pipeline CLI exits with a nonzero
    code without a traceback — the message itself is the
    operator-facing report.
    """


def gate_or_raise(
    *,
    phase_label: str,
    breakdown: dict[str, float],
    floors: PhaseFloors,
    force: bool = False,
) -> None:
    """Inter-phase prerequisite check.

    Called by ``framework.run_pipeline`` between phases. If the winning
    breakdown of the just-finished phase doesn't clear *that phase's*
    floor, the next phase is almost certainly wasted compute.

    ``force=True`` downgrades the gate to a warning — diagnostic runs.
    """
    deficits = floors.deficits(breakdown)
    if not deficits:
        return
    summary = ", ".join(
        f"{name}={breakdown.get(name, 0.0):.3f} "
        f"(floor {floors.floors[name]:.3f}, deficit {deficit:.3f})"
        for name, deficit in deficits.items()
    )
    msg = (
        f"phase {phase_label!r} did not clear its floors: {summary}. "
        f"Going to the next phase is likely wasted compute. "
        f"Re-run with --force to override; otherwise iterate on the "
        f"current phase (more trials, wider search space, stronger model)."
    )
    if force:
        logger.warning("%s [forced through]", msg)
        return
    raise PhaseFloorGateError(msg)


__all__ = [
    "FEASIBLE_OFFSET",
    "PhaseFloorGateError",
    "PhaseFloors",
    "feasibility_aware_score",
    "gate_or_raise",
    "is_feasible",
    "study_feasibility_summary",
]
