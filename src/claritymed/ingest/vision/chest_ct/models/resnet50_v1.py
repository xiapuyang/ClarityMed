"""``RESNET50_V1`` — chest CT 4-class classifier ModelSpec.

Floor rationale:

* ``cancer_recall`` floor = 0.85. ``cancer_recall`` is computed over
  the union of the three malignant labels vs ``normal``; the medical
  bar is "don't miss the cancer" regardless of which subtype it is.
* ``accuracy`` floor = 0.80. Lower than BUSI's 0.85 because 4-class
  is intrinsically harder than 3-class (random baseline 0.25 vs 0.33).
  A floor that's too tight here would gate out trainable models.

No ``dice`` floor — the upstream dataset doesn't ship masks, so the
classification task has nothing segmentation-based to score.

Composite weights: ``cancer_recall`` (0.6) + ``accuracy`` (0.4). The
secondary metric is accuracy rather than dice because dice is N/A.
"""

from __future__ import annotations

from claritymed.ingest.vision.chest_ct.dataset_spec import CHEST_CT_DATASET
from claritymed.ingest.vision.forge.spec import (
    BoolChoice,
    Categorical,
    LogUniform,
    ModelSpec,
    Uniform,
)
from claritymed.ingest.vision.forge.tasks.base import FloorBundle
from claritymed.ingest.vision.forge.tasks.classification import ClassificationTask


# Per-phase floors. Same graduating-bar shape as BUSI (search shows
# signs of life; train within tune's reach of deploy; tune == deploy).
_CHEST_CT_FLOORS = FloorBundle(
    search={"cancer_recall": 0.65, "accuracy": 0.55},
    train={"cancer_recall": 0.80, "accuracy": 0.75},
    deploy={"cancer_recall": 0.85, "accuracy": 0.80},
)

_CHEST_CT_COMPOSITE_WEIGHTS = {"cancer_recall": 0.6, "accuracy": 0.4}


# Three torchvision backbones — ResNet-50 is the default name; the
# Optuna search picks among the three. EfficientNet-B3 is the biggest
# and usually wins on CT; it stays in the search rather than as a
# default so the per-trial budget keeps it honest.
RESNET50_V1 = ModelSpec(
    dataset=CHEST_CT_DATASET,
    model_id="lung_chest_ct_resnet50_v1",
    model_version="v1.0.0",
    framework="pytorch",
    task=ClassificationTask(
        critical_labels=(
            "adenocarcinoma",
            "large_cell_carcinoma",
            "squamous_cell_carcinoma",
        ),
        critical_metric_name="cancer_recall",
        composite_weights=_CHEST_CT_COMPOSITE_WEIGHTS,
        floors=_CHEST_CT_FLOORS,
    ),
    hparam_space={
        "backbone": Categorical(("resnet50", "efficientnet_b0", "efficientnet_b3")),
        "lr": LogUniform(1e-5, 5e-3),
        # Weight decay range is mild — large values shrink ImageNet
        # priors and undo most of the pretrain gain on a small dataset.
        "weight_decay": LogUniform(1e-6, 1e-3),
        # Mild 1.8:1 train imbalance, with ``large_cell_carcinoma`` as
        # the smallest cancer subtype — plain CE leaves its recall at
        # ~0.72 vs 0.90+ on the rest. Optuna picks per trial.
        "class_weight": Categorical(("none", "inverse_freq", "sqrt_inv_freq")),
    },
    inference_space={
        "temperature": LogUniform(0.5, 3.0),
        "critical_threshold": Uniform(0.20, 0.70),
        "confidence_low_max": Uniform(0.40, 0.75),
        "confidence_medium_max": Uniform(0.55, 0.95),
        "tta_default": BoolChoice(),
    },
)


__all__ = ["RESNET50_V1"]
