"""``RESNET50_V1`` — chest X-ray pneumonia 2-class classifier ModelSpec.

The whole point of this model is to be a **drift probe**: train it on
Kermany (pediatric, Guangzhou hospital, 5,856 images), then evaluate it
on RSNA (adult, multi-center, ~30k images) to quantify how much a
pediatric pneumonia classifier degrades on adult anatomy. The
specifically-tuned-for-Kermany hyperparameters are a feature, not a bug.
A second model variant tuned for adult chest X-rays would be a separate
ModelSpec when that becomes interesting.

Floor rationale:

* ``pneumonia_recall`` floor = 0.85. The medical bar is "don't miss the
  pneumonia"; in pediatric data this is achievable with even a basic
  ImageNet-pretrained ResNet given Kermany's clean labels.
* ``accuracy`` floor = 0.80. Two-class so random baseline is 0.50; a
  floor of 0.80 means the model has to be visibly better than coin-flip.
  Loose enough to admit reasonable models, tight enough to gate out
  garbage.

No ``dice`` floor — Kermany ships no masks. Mirror of chest_ct's
classification-only setup.

Composite weights: ``pneumonia_recall`` (0.7) + ``accuracy`` (0.3). The
recall weight is higher than chest_ct's (0.6) because pneumonia is a
single critical class (no per-subtype trade-off to balance), so the
composite can lean harder on it.
"""

from __future__ import annotations

from claritymed.ingest.vision.chest_xray_pneumonia.dataset_spec import (
    CHEST_XRAY_PNEUMONIA_DATASET,
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

# Per-phase floors. Search bar is "shows signs of life on pneumonia
# class"; train approaches deploy; deploy == the deployable bar.
_CHEST_XRAY_PNEUMONIA_FLOORS = FloorBundle(
    search={"pneumonia_recall": 0.70, "accuracy": 0.65},
    train={"pneumonia_recall": 0.80, "accuracy": 0.75},
    deploy={"pneumonia_recall": 0.85, "accuracy": 0.80},
)

_CHEST_XRAY_PNEUMONIA_COMPOSITE_WEIGHTS = {
    "pneumonia_recall": 0.7,
    "accuracy": 0.3,
}


# Single backbone for now: ResNet-50 with ImageNet pretrain. The drift
# experiment doesn't need architecture sweep — adding efficientnet_b0
# variants would multiply the search budget without changing the drift
# story.
RESNET50_V1 = ModelSpec(
    dataset=CHEST_XRAY_PNEUMONIA_DATASET,
    model_id="chest_xray_pneumonia_resnet50_v1",
    model_version="v1.0.0",
    framework="pytorch",
    task=ClassificationTask(
        critical_labels=("pneumonia",),
        critical_metric_name="pneumonia_recall",
        composite_weights=_CHEST_XRAY_PNEUMONIA_COMPOSITE_WEIGHTS,
        floors=_CHEST_XRAY_PNEUMONIA_FLOORS,
    ),
    hparam_space={
        "backbone": Categorical(("resnet50",)),
        "lr": LogUniform(1e-5, 5e-3),
        # Mild weight decay range — large values shrink ImageNet priors
        # and undo most of the pretrain gain on a smallish dataset.
        "weight_decay": LogUniform(1e-6, 1e-3),
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
