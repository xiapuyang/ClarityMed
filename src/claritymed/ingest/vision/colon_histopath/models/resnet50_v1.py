"""``RESNET50_V1`` — colon histopathology 2-class classifier ModelSpec.

Floor rationale:

* ``cancer_recall`` floor = 0.93. Computed over ``adenocarcinoma``
  vs ``normal``. 2-class with balanced classes (LC25000 ships 5000
  images per class) sits at the cleanest end of the vision-floor
  range; tighter floors are warranted because the medical bar for
  pathology — final-line-of-evidence — is the strictest in the
  project.
* ``accuracy`` floor = 0.90. The malignant / normal split is
  visually discriminative on H&E staining; a trainable model
  clears this with margin.

No ``dice`` floor — the upstream archive doesn't ship masks.

Composite weights: ``cancer_recall`` (0.70) + ``accuracy`` (0.30).
Matches lung_histopath's weighting; both modules share the "final
line of evidence" framing.
"""

from __future__ import annotations

from claritymed.ingest.vision.colon_histopath.dataset_spec import (
    COLON_HISTOPATH_DATASET,
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


_COLON_HISTOPATH_FLOORS = FloorBundle(
    search={"cancer_recall": 0.80, "accuracy": 0.70},
    train={"cancer_recall": 0.90, "accuracy": 0.82},
    deploy={"cancer_recall": 0.93, "accuracy": 0.90},
)

_COLON_HISTOPATH_COMPOSITE_WEIGHTS = {"cancer_recall": 0.70, "accuracy": 0.30}


RESNET50_V1 = ModelSpec(
    dataset=COLON_HISTOPATH_DATASET,
    model_id="colon_histopath_resnet50_v1",
    model_version="v1.0.0",
    framework="pytorch",
    task=ClassificationTask(
        critical_labels=("adenocarcinoma",),
        critical_metric_name="cancer_recall",
        composite_weights=_COLON_HISTOPATH_COMPOSITE_WEIGHTS,
        floors=_COLON_HISTOPATH_FLOORS,
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
        "critical_threshold": Uniform(0.15, 0.60),
        "confidence_low_max": Uniform(0.30, 0.70),
        "confidence_medium_max": Uniform(0.50, 0.90),
        "tta_default": BoolChoice(),
    },
)


__all__ = ["RESNET50_V1"]
