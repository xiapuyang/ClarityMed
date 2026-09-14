"""Fail-fast key validator on ``Task`` construction.

``feasibility_aware_score`` and ``floors.deficits`` both read
``breakdown.get(name, 0.0)`` — a typo in a ``ModelSpec``'s floors or
composite_weights would silently treat the missing metric as ``0``,
making a trial infeasible-forever (bad floor) or contributing zero
to the composite (bad weight) without any error. The validator on
``Task.__init__`` converts that silent miss into a loud ``ValueError``
at spec-module import.
"""

from __future__ import annotations

import pytest

from claritymed.ingest.vision.forge.tasks.base import FloorBundle
from claritymed.ingest.vision.forge.tasks.classification import (
    ClassificationTask,
)
from claritymed.ingest.vision.forge.tasks.cls_segmentation import (
    ClassificationSegmentationTask,
)


def _cls_floors(**overrides) -> FloorBundle:
    """Default cls floor bundle with optional per-phase overrides."""
    default = {"cancer_recall": 0.65, "accuracy": 0.65}
    return FloorBundle(
        search=overrides.get("search", default),
        train=overrides.get("train", default),
        deploy=overrides.get("deploy", default),
    )


def _cls_seg_floors(**overrides) -> FloorBundle:
    """Default cls+seg floor bundle with optional per-phase overrides."""
    default = {"malignant_recall": 0.65, "accuracy": 0.65, "dice": 0.40}
    return FloorBundle(
        search=overrides.get("search", default),
        train=overrides.get("train", default),
        deploy=overrides.get("deploy", default),
    )


# --- happy paths -------------------------------------------------------


def test_classification_task_constructs_with_consistent_keys() -> None:
    """The 7 cls models in-tree all pass these constraints — this asserts
    the validator doesn't reject them by accident."""
    ClassificationTask(
        critical_labels=("cancer",),
        critical_metric_name="cancer_recall",
        composite_weights={"cancer_recall": 0.7, "accuracy": 0.3},
        floors=_cls_floors(),
    )


def test_cls_seg_task_allows_asymmetric_floors_vs_composite() -> None:
    """BUSI ships ``accuracy`` in floors but not composite — must not trip.

    The asymmetry is intentional (accuracy as a sanity gate, recall+dice
    as the optimisation signal); validator only rejects keys that aren't
    in the breakdown at all.
    """
    ClassificationSegmentationTask(
        critical_labels=("malignant",),
        critical_metric_name="malignant_recall",
        composite_weights={"malignant_recall": 0.6, "dice": 0.4},  # no accuracy
        floors=_cls_seg_floors(),  # has accuracy
    )


# --- floors typos ------------------------------------------------------


def test_floor_typo_in_critical_metric_fails_loud() -> None:
    """A common typo on the spec's critical metric name."""
    bad = {"cancer_recal": 0.65, "accuracy": 0.65}  # missing trailing 'l'
    with pytest.raises(ValueError, match="floors.search"):
        ClassificationTask(
            critical_labels=("cancer",),
            critical_metric_name="cancer_recall",
            composite_weights={"cancer_recall": 0.7, "accuracy": 0.3},
            floors=_cls_floors(search=bad),
        )


def test_floor_referencing_metric_not_emitted_by_task_fails_loud() -> None:
    """Cls tasks don't emit ``dice`` — a stray dice floor is dead weight."""
    bad = {"cancer_recall": 0.65, "accuracy": 0.65, "dice": 0.5}
    with pytest.raises(ValueError, match="dice"):
        ClassificationTask(
            critical_labels=("cancer",),
            critical_metric_name="cancer_recall",
            composite_weights={"cancer_recall": 0.7, "accuracy": 0.3},
            floors=_cls_floors(deploy=bad),
        )


# --- composite_weights typos -------------------------------------------


def test_composite_weight_typo_fails_loud() -> None:
    """Bad composite key would silently zero the metric's contribution."""
    with pytest.raises(ValueError, match="composite_weights"):
        ClassificationSegmentationTask(
            critical_labels=("malignant",),
            critical_metric_name="malignant_recall",
            composite_weights={"malignant_recal": 0.6, "dice": 0.4},  # typo
            floors=_cls_seg_floors(),
        )


def test_composite_weight_unknown_metric_fails_loud_in_cls_seg() -> None:
    """Even a real-looking metric name that the task doesn't emit fails."""
    with pytest.raises(ValueError, match="precision"):
        ClassificationSegmentationTask(
            critical_labels=("malignant",),
            critical_metric_name="malignant_recall",
            composite_weights={"malignant_recall": 0.5, "precision": 0.5},
            floors=_cls_seg_floors(),
        )


# --- breakdown shape declaration --------------------------------------


def test_breakdown_metric_keys_for_cls_task() -> None:
    task = ClassificationTask(
        critical_labels=("cancer",),
        critical_metric_name="cancer_recall",
        composite_weights={"cancer_recall": 0.7, "accuracy": 0.3},
        floors=_cls_floors(),
    )
    assert task.breakdown_metric_keys == frozenset({"cancer_recall", "accuracy"})


def test_breakdown_metric_keys_for_cls_seg_task() -> None:
    task = ClassificationSegmentationTask(
        critical_labels=("malignant",),
        critical_metric_name="malignant_recall",
        composite_weights={"malignant_recall": 0.6, "dice": 0.4},
        floors=_cls_seg_floors(),
    )
    assert task.breakdown_metric_keys == frozenset(
        {"malignant_recall", "accuracy", "dice"}
    )
