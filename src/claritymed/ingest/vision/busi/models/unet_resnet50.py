"""``UNET_RESNET50`` — the BUSI U-Net + classifier ModelSpec.

Floors + composite weights match the historical busi/scoring.py
constants. ``UNET_RESNET50.task`` is reusable across diseases that
share the cls+seg shape; just instantiate
:class:`ClassificationSegmentationTask` with disease-specific floors.

Hparam space — three backbones, log-uniform LR, uniform
``seg_loss_weight``. Inference space — temperature, critical-class
threshold (mapped to ``malignant`` in the manifest), seg mask cutoff,
two confidence-tier boundaries, TTA default toggle.
"""

from __future__ import annotations

from claritymed.ingest.vision.busi.dataset_spec import BUSI_DATASET
from claritymed.ingest.vision.forge.spec import (
    BoolChoice,
    Categorical,
    LogUniform,
    ModelSpec,
    Uniform,
)
from claritymed.ingest.vision.forge.tasks.base import FloorBundle
from claritymed.ingest.vision.forge.tasks.cls_segmentation import (
    ClassificationSegmentationTask,
)


# Per-phase floors. Rationale per phase mirrors the BUSI scoring.py
# this replaces — search is "shows signs of life", train is "within
# tune's reach of deploy", tune == deploy is the medical bar.
_BUSI_FLOORS = FloorBundle(
    search={"malignant_recall": 0.65, "accuracy": 0.65, "dice": 0.40},
    train={"malignant_recall": 0.80, "accuracy": 0.80, "dice": 0.62},
    deploy={"malignant_recall": 0.85, "accuracy": 0.85, "dice": 0.70},
)

# Composite: recall + dice (the medical bar + the seg signal). Accuracy
# is gated by floors but not in the composite — historical decision
# documented in the original busi/scoring.py.
_BUSI_COMPOSITE_WEIGHTS = {"malignant_recall": 0.6, "dice": 0.4}


UNET_RESNET50 = ModelSpec(
    dataset=BUSI_DATASET,
    model_id="breast_busi_unet_v1",
    model_version="v1.0.0",
    framework="pytorch",
    task=ClassificationSegmentationTask(
        critical_labels=("malignant",),
        critical_metric_name="malignant_recall",
        composite_weights=_BUSI_COMPOSITE_WEIGHTS,
        floors=_BUSI_FLOORS,
    ),
    hparam_space={
        "backbone": Categorical(("resnet50", "efficientnet_b0", "custom_unet")),
        "lr": LogUniform(1e-5, 5e-3),
        "seg_loss_weight": Uniform(0.1, 2.0),
    },
    inference_space={
        "temperature": LogUniform(0.5, 3.0),
        "critical_threshold": Uniform(0.20, 0.70),
        "seg_threshold": Uniform(0.20, 0.80),
        "confidence_low_max": Uniform(0.40, 0.75),
        "confidence_medium_max": Uniform(0.55, 0.95),
        "tta_default": BoolChoice(),
    },
)


__all__ = ["UNET_RESNET50"]
