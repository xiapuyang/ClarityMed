"""``RESNET50_V1`` — lung histopathology 3-class classifier ModelSpec.

Floor rationale:

* ``cancer_recall`` floor = 0.92. Computed over the union of the two
  malignant labels (``adenocarcinoma``, ``squamous_cell_carcinoma``)
  vs the healthy ``normal`` baseline. LC25000 is a relatively clean
  patch-level dataset (5000 images per class, well-stained, no class
  imbalance), so the floor sits above skin_lesion (0.85) and matches
  the chest_ct CT bar's quality.
* ``accuracy`` floor = 0.88. 3-class with balanced classes lets a
  trainable model clear this comfortably.

No ``dice`` floor — the upstream archive doesn't ship masks.

Composite weights: ``cancer_recall`` (0.70) + ``accuracy`` (0.30). The
recall weight is the highest among the lung-cancer ingest modules
because histopathology is most often the *final* line of evidence
(post-biopsy) — missing a malignant patch here is closer to the
medical worst-case than missing a screening CT finding.
"""

from __future__ import annotations

from claritymed.ingest.vision.forge.spec import (
    BoolChoice,
    Categorical,
    LogUniform,
    ModelSpec,
    Uniform,
)
from claritymed.ingest.vision.forge.tasks.base import FloorBundle
from claritymed.ingest.vision.forge.tasks.classification import ClassificationTask
from claritymed.ingest.vision.lung_histopath.dataset_spec import LUNG_HISTOPATH_DATASET


_LUNG_HISTOPATH_FLOORS = FloorBundle(
    search={"cancer_recall": 0.75, "accuracy": 0.60},
    train={"cancer_recall": 0.88, "accuracy": 0.78},
    deploy={"cancer_recall": 0.92, "accuracy": 0.88},
)

_LUNG_HISTOPATH_COMPOSITE_WEIGHTS = {"cancer_recall": 0.70, "accuracy": 0.30}


RESNET50_V1 = ModelSpec(
    dataset=LUNG_HISTOPATH_DATASET,
    model_id="lung_histopath_resnet50_v1",
    model_version="v1.0.0",
    framework="pytorch",
    task=ClassificationTask(
        critical_labels=(
            "adenocarcinoma",
            "squamous_cell_carcinoma",
        ),
        critical_metric_name="cancer_recall",
        composite_weights=_LUNG_HISTOPATH_COMPOSITE_WEIGHTS,
        floors=_LUNG_HISTOPATH_FLOORS,
    ),
    hparam_space={
        "backbone": Categorical(("resnet50", "efficientnet_b0", "efficientnet_b3")),
        "lr": LogUniform(1e-5, 5e-3),
        "weight_decay": LogUniform(1e-6, 1e-3),
        # Already 1:1:1 balanced — Optuna will likely settle on
        # ``none``, but the option is here for consistency with the
        # other classification specs.
        "class_weight": Categorical(("none", "inverse_freq", "sqrt_inv_freq")),
    },
    inference_space={
        "temperature": LogUniform(0.5, 3.0),
        "critical_threshold": Uniform(0.15, 0.60),
        "confidence_low_max": Uniform(0.30, 0.70),
        "confidence_medium_max": Uniform(0.50, 0.90),
        "tta_default": BoolChoice(),
    },
)


__all__ = ["RESNET50_V1"]
