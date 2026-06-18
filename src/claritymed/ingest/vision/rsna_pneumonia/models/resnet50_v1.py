"""``RESNET50_V1`` — RSNA Pneumonia 2-class classifier ModelSpec.

The adult-population counterpart to Kermany's same-named spec. Same
2-class task (``normal`` / ``pneumonia``), same architecture, different
training source: RSNA's ~30k multi-center adult chest X-rays (image-
level labels derived by collapsing bbox rows: any bbox → pneumonia).

Pairs with Kermany's
``claritymed.ingest.vision.chest_xray_pneumonia.models.resnet50_v1:RESNET50_V1``
to form a symmetric drift pair — train one, eval on the other to
quantify pediatric ↔ adult distribution shift in both directions. Both
specs share ``disease_id="chest_xray_pneumonia"``, so the forge
artifact root, ``LATEST.jsonl``, and MLflow experiment are shared; the
``model_id`` is the discriminator (``rsna_pneumonia_resnet50_v1`` vs
``chest_xray_pneumonia_resnet50_v1``).

Floor rationale:

* Floors mirror Kermany's (``pneumonia_recall ≥ 0.85``,
  ``accuracy ≥ 0.80`` at deploy). They are an **inherited starting
  point**, not empirically validated against RSNA — RSNA has noisier
  labels (bbox-derived → some normals may carry subclinical findings)
  and higher inter-acquisition variance (multi-center, mixed AP/PA,
  mixed portable/standing). Expect the first real runs to land near
  or below the deploy floor; adjust the floor (not the model) once
  there's a baseline number to anchor on, with the bar being "the
  best honest model we can train" rather than "matches Kermany".
* Class imbalance is real (~20% positive vs Kermany's ~73%). If the
  search-phase recall floor blocks every trial, that's the signal
  to add class-weighted loss to the hparam space — not to lower the
  floor.

Composite weights: same as Kermany (``pneumonia_recall`` 0.7 +
``accuracy`` 0.3). Pneumonia is a single critical class on both
sources; the composite shape stays.

Single backbone (``resnet50``) to keep the search budget aligned with
Kermany's. Add ``efficientnet_b0`` etc. as a separate ModelSpec file
(``efficientnet_b0_v1.py``) if RSNA-specific architecture search ever
becomes interesting — don't mutate this one.
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
from claritymed.ingest.vision.rsna_pneumonia.dataset_spec import (
    RSNA_PNEUMONIA_DATASET,
)

_RSNA_PNEUMONIA_FLOORS = FloorBundle(
    search={"pneumonia_recall": 0.70, "accuracy": 0.65},
    train={"pneumonia_recall": 0.80, "accuracy": 0.75},
    deploy={"pneumonia_recall": 0.85, "accuracy": 0.80},
)

_RSNA_PNEUMONIA_COMPOSITE_WEIGHTS = {
    "pneumonia_recall": 0.7,
    "accuracy": 0.3,
}


RESNET50_V1 = ModelSpec(
    dataset=RSNA_PNEUMONIA_DATASET,
    model_id="rsna_pneumonia_resnet50_v1",
    model_version="v1.0.0",
    framework="pytorch",
    task=ClassificationTask(
        critical_labels=("pneumonia",),
        critical_metric_name="pneumonia_recall",
        composite_weights=_RSNA_PNEUMONIA_COMPOSITE_WEIGHTS,
        floors=_RSNA_PNEUMONIA_FLOORS,
    ),
    hparam_space={
        "backbone": Categorical(("resnet50",)),
        "lr": LogUniform(1e-5, 5e-3),
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
