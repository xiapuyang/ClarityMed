"""Aggregate per-profile predictions into :class:`ProfileMetrics`.

Why a separate module from ``runner.py``: the runner deals with I/O
(YAML loading, async gate calls); metrics is a pure function over
``(Case, Prediction)`` pairs. Keeping them split means metrics is
trivially unit-testable without bringing the gate into the test
fixture.

The headline number for the gate is **critical recall** at the
``balanced`` profile — missing a critical case = patient harm. F-beta
(β=2) weights recall 4× precision to surface this in one scalar.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from claritymed.core.emergency.schemas import EmergencyLevel

from evals.emergency.schemas import (
    Case,
    ConfusionCell,
    Prediction,
    ProfileMetrics,
)

# Predicted urgent or higher counts as an "alert" for the alert-rate
# metric. Routine + moderate are not surfaced to the user as
# safety-flagged.
_ALERT_LEVELS: frozenset[EmergencyLevel] = frozenset({"critical", "urgent"})
_BETA: float = 2.0


def _f_beta(precision: float, recall: float, beta: float = _BETA) -> float:
    """Compute F-beta. Returns 0.0 when both precision and recall are 0."""
    if precision == 0.0 and recall == 0.0:
        return 0.0
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1.0 + b2) * precision * recall / denom


def _binary_critical(
    pairs: list[tuple[Case, Prediction]],
) -> tuple[int, int, int]:
    """Return (TP, FP, FN) treating ``critical`` as the positive class."""
    tp = fp = fn = 0
    for case, pred in pairs:
        actual_critical = case.ground_truth_level == "critical"
        pred_critical = pred.predicted_level == "critical"
        if actual_critical and pred_critical:
            tp += 1
        elif not actual_critical and pred_critical:
            fp += 1
        elif actual_critical and not pred_critical:
            fn += 1
    return tp, fp, fn


def _per_rule_recall(
    pairs: list[tuple[Case, Prediction]],
) -> dict[str, float]:
    """Recall per ``ground_truth_rule_id``.

    Cases without a ground_truth_rule_id are excluded — they label the
    level (e.g. "urgent" for ambiguous chest pain) without pinning a
    specific rule, so per-rule recall is undefined.
    """
    total: dict[str, int] = defaultdict(int)
    hit: dict[str, int] = defaultdict(int)
    for case, pred in pairs:
        rid = case.ground_truth_rule_id
        if not rid:
            continue
        total[rid] += 1
        if rid in pred.matched_rule_ids:
            hit[rid] += 1
    return {
        rid: (hit[rid] / total[rid]) if total[rid] else 0.0 for rid in sorted(total)
    }


def _adversarial_fpr(pairs: list[tuple[Case, Prediction]]) -> float:
    """Of cases marked adversarial, fraction predicted ``critical``.

    Plan §"Eval Strategy" pins adversarial FPR as the panic-attack-as-MI
    canary: a panic attack firing the ambiguous catch-all at ``urgent``
    is a precision/recall tradeoff worth measuring separately (via
    ``alert_rate_per_100``), but is not the gate calling a benign
    presentation a life-threat. The ``critical``-only definition keeps
    the FPR metric focused on the failure mode the plan targets
    (≤ 15% under ``balanced``).
    """
    adv = [(c, p) for c, p in pairs if c.is_adversarial]
    if not adv:
        return 0.0
    fires = sum(1 for _, p in adv if p.predicted_level == "critical")
    return fires / len(adv)


def _confusion(
    pairs: list[tuple[Case, Prediction]],
) -> list[ConfusionCell]:
    """Build the (actual, predicted) confusion matrix as a flat list."""
    counts: dict[tuple[EmergencyLevel, EmergencyLevel], int] = defaultdict(int)
    for case, pred in pairs:
        counts[(case.ground_truth_level, pred.predicted_level)] += 1
    cells = [
        ConfusionCell(actual=a, predicted=p, count=n) for (a, p), n in counts.items()
    ]
    cells.sort(key=lambda c: (c.actual, c.predicted))
    return cells


def aggregate(
    cases: Iterable[Case],
    predictions: Iterable[Prediction],
    *,
    profile: str,
) -> ProfileMetrics:
    """Combine cases + predictions into one :class:`ProfileMetrics`.

    Predictions are matched to cases by id. Predictions whose case is
    missing fail loud — that's a runner bug, not an eval gap.
    """
    case_by_id = {c.id: c for c in cases}
    preds = list(predictions)
    pairs: list[tuple[Case, Prediction]] = []
    skipped = 0
    for p in preds:
        case = case_by_id.get(p.case_id)
        if case is None:
            raise ValueError(f"prediction references unknown case {p.case_id!r}")
        if p.skipped:
            skipped += 1
            continue
        pairs.append((case, p))

    total = len(preds)
    scored = len(pairs)
    tp, fp, fn = _binary_critical(pairs)
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0

    alerts = sum(1 for _, p in pairs if p.predicted_level in _ALERT_LEVELS)
    alert_rate = (alerts / scored * 100.0) if scored else 0.0

    return ProfileMetrics(
        profile=profile,  # type: ignore[arg-type]
        total=total,
        scored=scored,
        skipped=skipped,
        critical_recall=recall,
        critical_precision=precision,
        f_beta_2=_f_beta(precision, recall),
        adversarial_fpr=_adversarial_fpr(pairs),
        per_rule_recall=_per_rule_recall(pairs),
        alert_rate_per_100=alert_rate,
        confusion=_confusion(pairs),
    )
