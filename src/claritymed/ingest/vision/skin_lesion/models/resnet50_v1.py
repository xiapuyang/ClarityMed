"""``RESNET50_V1`` — skin lesion 9-class classifier ModelSpec.

Floor rationale:

* ``cancer_recall`` floor = 0.85. Computed over the union of the four
  malignant labels (actinic_keratosis, basal_cell_carcinoma, melanoma,
  squamous_cell_carcinoma) vs the five benign labels. The medical bar
  is "don't miss the cancer" regardless of malignant subtype.
* ``accuracy`` floor = 0.65. 9-class is intrinsically harder than the
  3- and 4-class problems (random baseline 0.11 vs 0.33 / 0.25), and
  the upstream class distribution is heavily imbalanced (melanoma and
  nevus dominate). A tighter floor would gate out trainable models.

No ``dice`` floor — the upstream ISIC archive doesn't ship masks.

Composite weights: ``cancer_recall`` (0.65) + ``accuracy`` (0.35). The
recall weight is slightly higher than chest_ct's 0.60 because the
class imbalance amplifies how easily a model can hit high accuracy
while still missing rare malignant entries.
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
from claritymed.ingest.vision.skin_lesion.dataset_spec import SKIN_LESION_DATASET


_SKIN_LESION_FLOORS = FloorBundle(
    search={"cancer_recall": 0.60, "accuracy": 0.40},
    train={"cancer_recall": 0.80, "accuracy": 0.60},
    deploy={"cancer_recall": 0.85, "accuracy": 0.65},
)

_SKIN_LESION_COMPOSITE_WEIGHTS = {"cancer_recall": 0.65, "accuracy": 0.35}


RESNET50_V1 = ModelSpec(
    dataset=SKIN_LESION_DATASET,
    model_id="skin_isic_resnet50_v1",
    model_version="v1.0.0",
    framework="pytorch",
    task=ClassificationTask(
        critical_labels=(
            "actinic_keratosis",
            "basal_cell_carcinoma",
            "melanoma",
            "squamous_cell_carcinoma",
        ),
        critical_metric_name="cancer_recall",
        composite_weights=_SKIN_LESION_COMPOSITE_WEIGHTS,
        floors=_SKIN_LESION_FLOORS,
    ),
    hparam_space={
        "backbone": Categorical(("resnet50", "efficientnet_b0", "efficientnet_b3")),
        "lr": LogUniform(1e-5, 5e-3),
        # Weight decay range is mild — large values shrink ImageNet
        # priors and undo most of the pretrain gain on a small dataset.
        "weight_decay": LogUniform(1e-6, 1e-3),
        # 6:1 imbalance with seborrheic_keratosis (N=56 in train) at the
        # tail — plain CE gives 0% recall on it. ``inverse_freq`` /
        # ``sqrt_inv_freq`` let Optuna trade per-class recall against
        # overall accuracy.
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
