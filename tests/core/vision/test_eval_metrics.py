"""Tests for ``binary_clinical_metrics`` — the cross-dataset bench primitive."""

from __future__ import annotations

import numpy as np
import pytest

from claritymed.core.vision.eval_metrics import (
    BinaryMetrics,
    binary_clinical_metrics,
)


# --- helpers -------------------------------------------------------------


def _three_class_probs(
    *,
    n_per_class: int,
    correct_prob: float,
) -> tuple[list[int], np.ndarray]:
    """Build deterministic (gt, probs) for a 3-class ``(benign, malignant,
    normal)`` setup where every sample's ground-truth class wins by
    exactly ``correct_prob`` with the remainder split evenly across the
    other two columns.

    With ``correct_prob > 1/3`` every sample is correctly argmax'd, so
    sensitivity + specificity collapse cleanly to closed-form values
    independent of any threshold.
    """
    remainder = (1.0 - correct_prob) / 2
    gt: list[int] = []
    rows: list[list[float]] = []
    for cls in range(3):
        for _ in range(n_per_class):
            row = [remainder, remainder, remainder]
            row[cls] = correct_prob
            rows.append(row)
            gt.append(cls)
    return gt, np.array(rows, dtype=np.float64)


# --- happy path: 3-class BUSI-shaped collapse ----------------------------


def test_three_class_busi_shaped_collapse_positive_malignant() -> None:
    """30 samples, 10 per class, all argmax-correct. positive={malignant}
    gives sensitivity=specificity=accuracy=1.0; AUC=1.0 because the
    positive class's probability column perfectly separates positives
    from negatives.
    """
    gt, probs = _three_class_probs(n_per_class=10, correct_prob=0.7)
    result = binary_clinical_metrics(
        gt_labels=gt,
        probs=probs,
        label_tuple=("benign", "malignant", "normal"),
        positive_labels=frozenset({"malignant"}),
    )

    assert result.sensitivity == 1.0
    assert result.specificity == 1.0
    assert result.accuracy == 1.0
    assert result.auc == 1.0
    assert result.n_total == 30
    assert result.n_positive == 10


def test_three_class_returns_frozen_binary_metrics() -> None:
    """The return type is a frozen pydantic model — mutation raises."""
    gt, probs = _three_class_probs(n_per_class=3, correct_prob=0.7)
    result = binary_clinical_metrics(
        gt_labels=gt,
        probs=probs,
        label_tuple=("benign", "malignant", "normal"),
        positive_labels=frozenset({"malignant"}),
    )

    assert isinstance(result, BinaryMetrics)
    with pytest.raises(Exception):  # pydantic raises ValidationError on frozen mutation
        result.sensitivity = 0.5  # type: ignore[misc]


# --- happy path: 2-class breast_us_kaggle-shaped -------------------------


def test_two_class_breast_us_kaggle_shaped() -> None:
    """20 samples, 10 benign + 10 malignant, all argmax-correct."""
    gt = [0] * 10 + [1] * 10
    probs = np.array(
        [[0.8, 0.2]] * 10 + [[0.2, 0.8]] * 10,
        dtype=np.float64,
    )
    result = binary_clinical_metrics(
        gt_labels=gt,
        probs=probs,
        label_tuple=("benign", "malignant"),
        positive_labels=frozenset({"malignant"}),
    )

    assert result.sensitivity == 1.0
    assert result.specificity == 1.0
    assert result.accuracy == 1.0
    assert result.auc == 1.0
    assert result.n_total == 20
    assert result.n_positive == 10


def test_imperfect_classifier_drops_sensitivity() -> None:
    """5 of 10 positives mispredicted as negative → sensitivity=0.5,
    specificity=1.0, accuracy=0.75. AUC stays high because the
    probability column still separates groups even when argmax flips.
    """
    gt = [0] * 10 + [1] * 10
    probs = np.array(
        [[0.9, 0.1]] * 10  # 10 negatives, all correct
        + [[0.6, 0.4]] * 5  # 5 positives, argmax says negative (false neg)
        + [[0.3, 0.7]] * 5,  # 5 positives, argmax says positive (true pos)
        dtype=np.float64,
    )
    result = binary_clinical_metrics(
        gt_labels=gt,
        probs=probs,
        label_tuple=("benign", "malignant"),
        positive_labels=frozenset({"malignant"}),
    )

    assert result.sensitivity == 0.5
    assert result.specificity == 1.0
    assert result.accuracy == 0.75
    assert result.auc is not None
    assert result.auc > 0.8  # probability column still separates well


# --- edge: positive_labels mismatch --------------------------------------


def test_positive_label_not_in_label_tuple_raises() -> None:
    """Typo'd registry entry: fail-loud rather than silently scoring
    against zero classes.
    """
    gt, probs = _three_class_probs(n_per_class=3, correct_prob=0.7)
    with pytest.raises(ValueError, match="suspiciousss"):
        binary_clinical_metrics(
            gt_labels=gt,
            probs=probs,
            label_tuple=("benign", "malignant", "normal"),
            positive_labels=frozenset({"suspiciousss"}),
        )


# --- edge: degenerate single-class eval sets -----------------------------


def test_all_positive_eval_set_specificity_and_auc_none() -> None:
    """Eval set with only positives: no true negatives possible, so
    specificity is undefined; sklearn cannot compute AUC on one class.
    """
    gt = [1] * 10
    probs = np.array([[0.2, 0.8]] * 10, dtype=np.float64)
    result = binary_clinical_metrics(
        gt_labels=gt,
        probs=probs,
        label_tuple=("benign", "malignant"),
        positive_labels=frozenset({"malignant"}),
    )

    assert result.sensitivity == 1.0
    assert result.specificity is None
    assert result.auc is None
    assert result.n_total == 10
    assert result.n_positive == 10


def test_all_negative_eval_set_sensitivity_zero_auc_none() -> None:
    """Eval set with only negatives: no positives means sensitivity is
    defined as 0 (no TP possible) and AUC is undefined.
    """
    gt = [0] * 10
    probs = np.array([[0.8, 0.2]] * 10, dtype=np.float64)
    result = binary_clinical_metrics(
        gt_labels=gt,
        probs=probs,
        label_tuple=("benign", "malignant"),
        positive_labels=frozenset({"malignant"}),
    )

    assert result.sensitivity == 0.0
    assert result.specificity == 1.0
    assert result.auc is None
    assert result.n_total == 10
    assert result.n_positive == 0


# --- multi-class positive set --------------------------------------------


def test_multi_class_positive_sums_probability_columns() -> None:
    """positive_labels={"malignant", "suspicious"} should sum the two
    columns into ``prob_positive`` for AUC + threshold mode. Verifies the
    probability-sum path without depending on argmax behavior.
    """
    # 4 samples: benign / benign / malignant / suspicious. With explicit
    # threshold=0.5 the summed positive probability for the two
    # malignant-or-suspicious rows clears the threshold and the benign
    # rows don't, giving a clean classification.
    gt = [0, 0, 1, 2]
    probs = np.array(
        [
            [0.7, 0.2, 0.1],  # benign: pos sum = 0.3
            [0.6, 0.3, 0.1],  # benign: pos sum = 0.4
            [0.2, 0.7, 0.1],  # malignant: pos sum = 0.8
            [0.1, 0.3, 0.6],  # suspicious: pos sum = 0.9
        ],
        dtype=np.float64,
    )
    result = binary_clinical_metrics(
        gt_labels=gt,
        probs=probs,
        label_tuple=("benign", "malignant", "suspicious"),
        positive_labels=frozenset({"malignant", "suspicious"}),
        threshold=0.5,
    )

    assert result.sensitivity == 1.0
    assert result.specificity == 1.0
    assert result.accuracy == 1.0
    assert result.n_positive == 2


# --- explicit threshold direction ----------------------------------------


def test_threshold_direction_high_lowers_sensitivity() -> None:
    """Higher threshold → fewer predicted positives → lower sensitivity,
    higher specificity. This is the textbook ROC tradeoff; verifying it
    here pins the threshold path against accidental sign flips.
    """
    # 6 positives, 4 negatives, with positive-class probabilities
    # spanning 0.4 to 0.9 for positives and 0.1 to 0.45 for negatives.
    gt = [1] * 6 + [0] * 4
    probs = np.array(
        [
            [0.6, 0.4],  # pos, weak
            [0.5, 0.5],  # pos, borderline
            [0.4, 0.6],  # pos, modest
            [0.3, 0.7],  # pos, strong
            [0.2, 0.8],  # pos, very strong
            [0.1, 0.9],  # pos, very strong
            [0.55, 0.45],  # neg, borderline
            [0.7, 0.3],  # neg, clear
            [0.8, 0.2],  # neg, clear
            [0.9, 0.1],  # neg, clear
        ],
        dtype=np.float64,
    )

    low_t = binary_clinical_metrics(
        gt_labels=gt,
        probs=probs,
        label_tuple=("benign", "malignant"),
        positive_labels=frozenset({"malignant"}),
        threshold=0.3,
    )
    high_t = binary_clinical_metrics(
        gt_labels=gt,
        probs=probs,
        label_tuple=("benign", "malignant"),
        positive_labels=frozenset({"malignant"}),
        threshold=0.7,
    )

    # Low threshold: every positive clears, plus the borderline negative
    # → sensitivity=1.0, specificity drops.
    assert low_t.sensitivity == 1.0
    assert low_t.specificity is not None
    # High threshold: only the three strongest positives clear → sensitivity
    # drops, specificity rises to 1.0.
    assert high_t.sensitivity < low_t.sensitivity
    assert high_t.specificity is not None
    assert high_t.specificity > low_t.specificity


# --- AUC bounds ----------------------------------------------------------


def test_auc_within_unit_interval_for_random_separation() -> None:
    """AUC must always land in ``[0, 1]`` when computed. Sanity check
    against any future refactor that returns raw scores instead of
    bounded probabilities.
    """
    rng = np.random.default_rng(42)
    n = 50
    gt = ([0] * (n // 2)) + ([1] * (n // 2))
    # Slight positive-class signal so AUC is non-degenerate but not 1.
    pos_probs = rng.uniform(0.45, 0.85, size=n // 2)
    neg_probs = rng.uniform(0.15, 0.55, size=n // 2)
    probs_pos_col = np.concatenate([neg_probs, pos_probs])
    probs = np.stack([1.0 - probs_pos_col, probs_pos_col], axis=1)

    result = binary_clinical_metrics(
        gt_labels=gt,
        probs=probs,
        label_tuple=("benign", "malignant"),
        positive_labels=frozenset({"malignant"}),
    )

    assert result.auc is not None
    assert 0.0 <= result.auc <= 1.0
