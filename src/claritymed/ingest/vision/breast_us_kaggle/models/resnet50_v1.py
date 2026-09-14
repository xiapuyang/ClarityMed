"""``RESNET50_V1`` — breast_us_kaggle 2-class classifier ModelSpec.

Floor rationale:

* ``malignant_recall`` floor = 0.88. Tighter than BUSI's 0.85 because
  this dataset has substantially more positive examples; "missing
  malignant" on a 2-class problem should be a higher bar than on
  3-class.
* ``accuracy`` floor = 0.85. Same as BUSI — 2-class is the easiest of
  the three vision datasets, but the augmented-upstream samples
  introduce some noise that keeps the bar from going higher.

No ``dice`` floor — the upstream archive doesn't ship masks.

Composite weights: ``malignant_recall`` (0.65) + ``accuracy`` (0.35) —
recall-weighted because the medical bar is "don't miss the cancer".
"""

from __future__ import annotations

from claritymed.ingest.vision.breast_us_kaggle.dataset_spec import (
    BREAST_US_KAGGLE_DATASET,
)
from claritymed.ingest.vision.forge.spec import (
    BoolChoice,
    Categorical,
    LogUniform,
    ModelSpec,
    Uniform,
)
from claritymed.ingest.vision.forge.tasks.base import FloorBundle
from claritymed.ingest.vision.forge.tasks.classification import ClassificationTask


_BREAST_US_KAGGLE_FLOORS = FloorBundle(
    search={"malignant_recall": 0.70, "accuracy": 0.65},
    train={"malignant_recall": 0.83, "accuracy": 0.80},
    deploy={"malignant_recall": 0.88, "accuracy": 0.85},
)

_BREAST_US_KAGGLE_COMPOSITE_WEIGHTS = {"malignant_recall": 0.8, "accuracy": 0.2}


RESNET50_V1 = ModelSpec(
    dataset=BREAST_US_KAGGLE_DATASET,
    model_id="breast_us_kaggle_resnet50_v1",
    model_version="v1.0.0",
    framework="pytorch",
    task=ClassificationTask(
        critical_labels=("malignant",),
        critical_metric_name="malignant_recall",
        composite_weights=_BREAST_US_KAGGLE_COMPOSITE_WEIGHTS,
        floors=_BREAST_US_KAGGLE_FLOORS,
    ),
    hparam_space={
        "backbone": Categorical(("resnet50", "efficientnet_b0", "efficientnet_b3")),
        "lr": LogUniform(1e-5, 5e-3),
        "weight_decay": LogUniform(1e-6, 1e-3),
        # Already 1:1 balanced — Optuna will likely settle on ``none``,
        # but the option is here for consistency with the other
        # classification specs.
        "class_weight": Categorical(("none", "inverse_freq", "sqrt_inv_freq")),
    },
    inference_space={
        "temperature": LogUniform(0.5, 3.0),
        "critical_threshold": Uniform(0.20, 0.65),
        "confidence_low_max": Uniform(0.40, 0.75),
        "confidence_medium_max": Uniform(0.55, 0.95),
        "tta_default": BoolChoice(),
    },
)


__all__ = ["RESNET50_V1"]
