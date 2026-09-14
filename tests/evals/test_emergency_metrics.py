"""Unit tests for ``evals.emergency.metrics``.

The metrics module is pure (no I/O, no gate calls), so the entire
surface area is exercised against hand-built (Case, Prediction) pairs.
F-beta β=2 weights recall 4× precision — this file pins the formula
with a worked example so a future change to the weighting raises a red
flag at review time, not in production after the eval baseline drifts.
"""

from __future__ import annotations

import math

from claritymed.core.emergency.schemas import ExtractedSymptoms

from evals.emergency.metrics import _f_beta, aggregate
from evals.emergency.schemas import Case, Prediction


def _case(
    cid: str,
    *,
    level: str,
    rule_id: str | None = None,
    adversarial: bool = False,
) -> Case:
    return Case(
        id=cid,
        source="public_vignettes",
        language="en",
        symptoms=ExtractedSymptoms(primary_complaint="chest_pain"),
        ground_truth_level=level,  # type: ignore[arg-type]
        ground_truth_rule_id=rule_id,
        is_adversarial=adversarial,
    )


def _pred(
    cid: str,
    *,
    level: str,
    matched: list[str] | None = None,
    skipped: bool = False,
) -> Prediction:
    return Prediction(
        case_id=cid,
        profile="balanced",
        predicted_level=level,  # type: ignore[arg-type]
        matched_rule_ids=matched or [],
        skipped=skipped,
    )


def test_f_beta_zero_when_both_zero():
    assert _f_beta(0.0, 0.0) == 0.0


def test_f_beta_recall_weighted_4x_precision():
    # β=2 → recall weighted 4× precision in F-beta.
    # P=0.5, R=1.0 should beat P=1.0, R=0.5 because the second has
    # half the recall.
    assert _f_beta(0.5, 1.0) > _f_beta(1.0, 0.5)


def test_f_beta_perfect_classifier_is_one():
    assert math.isclose(_f_beta(1.0, 1.0), 1.0)


def test_aggregate_critical_recall_with_one_miss():
    cases = [
        _case("a", level="critical", rule_id="acs"),
        _case("b", level="critical", rule_id="acs"),
        _case("c", level="routine"),
    ]
    preds = [
        _pred("a", level="critical", matched=["acs"]),
        _pred("b", level="routine"),  # miss
        _pred("c", level="routine"),
    ]
    m = aggregate(cases, preds, profile="balanced")
    assert m.scored == 3 and m.skipped == 0
    # 1 TP, 0 FP, 1 FN → recall = 1/2, precision = 1/1.
    assert math.isclose(m.critical_recall, 0.5)
    assert math.isclose(m.critical_precision, 1.0)


def test_aggregate_skipped_predictions_excluded_from_scoring():
    cases = [_case("a", level="critical", rule_id="acs")]
    preds = [_pred("a", level="routine", skipped=True)]
    m = aggregate(cases, preds, profile="balanced")
    assert m.total == 1
    assert m.scored == 0
    assert m.skipped == 1
    assert m.critical_recall == 0.0  # no scored TP+FN


def test_aggregate_per_rule_recall():
    cases = [
        _case("a", level="critical", rule_id="acs"),
        _case("b", level="critical", rule_id="acs"),
        _case("c", level="critical", rule_id="anaphylaxis"),
        # case without rule_id — must be excluded from per-rule recall.
        _case("d", level="urgent"),
    ]
    preds = [
        _pred("a", level="critical", matched=["acs"]),  # hit
        _pred("b", level="routine", matched=[]),  # miss
        _pred("c", level="critical", matched=["anaphylaxis"]),  # hit
        _pred("d", level="urgent", matched=["chest_pain_ambiguous"]),
    ]
    m = aggregate(cases, preds, profile="balanced")
    assert math.isclose(m.per_rule_recall["acs"], 0.5)
    assert math.isclose(m.per_rule_recall["anaphylaxis"], 1.0)
    assert "chest_pain_ambiguous" not in m.per_rule_recall


def test_aggregate_adversarial_fpr_only_counts_critical_on_adv_cases():
    """Urgent fires on an adversarial chest_pain case are NOT FPR.

    The catch-all firing ``urgent`` on a panic-attack-style chest pain
    is the precision tradeoff measured by ``alert_rate_per_100``.
    Adversarial FPR pins the harder failure: gate calling a benign
    presentation life-threatening.
    """
    cases = [
        _case("a", level="routine", adversarial=True),
        _case("b", level="routine", adversarial=True),
        _case("c", level="routine", adversarial=True),
        _case("d", level="routine", adversarial=False),
    ]
    preds = [
        _pred("a", level="critical"),  # adv → critical → counts
        _pred("b", level="urgent"),  # adv → urgent → NOT FPR
        _pred("c", level="routine"),  # adv → silent → does not count
        _pred("d", level="critical"),  # non-adv → not in FPR pool
    ]
    m = aggregate(cases, preds, profile="balanced")
    assert math.isclose(m.adversarial_fpr, 1 / 3)


def test_aggregate_alert_rate_counts_critical_and_urgent():
    cases = [
        _case("a", level="critical"),
        _case("b", level="urgent"),
        _case("c", level="routine"),
        _case("d", level="routine"),
    ]
    preds = [
        _pred("a", level="critical"),  # alert
        _pred("b", level="urgent"),  # alert
        _pred("c", level="moderate"),  # not alert
        _pred("d", level="routine"),  # not alert
    ]
    m = aggregate(cases, preds, profile="balanced")
    # 2 alerts out of 4 cases → 50 per 100.
    assert math.isclose(m.alert_rate_per_100, 50.0)


def test_aggregate_unknown_case_id_raises():
    import pytest

    cases = [_case("a", level="critical")]
    preds = [_pred("missing", level="critical")]
    with pytest.raises(ValueError):
        aggregate(cases, preds, profile="balanced")


def test_aggregate_confusion_matrix_groups_pairs():
    cases = [
        _case("a", level="critical"),
        _case("b", level="critical"),
        _case("c", level="routine"),
    ]
    preds = [
        _pred("a", level="critical"),
        _pred("b", level="critical"),
        _pred("c", level="routine"),
    ]
    m = aggregate(cases, preds, profile="balanced")
    by_pair = {(c.actual, c.predicted): c.count for c in m.confusion}
    assert by_pair[("critical", "critical")] == 2
    assert by_pair[("routine", "routine")] == 1
