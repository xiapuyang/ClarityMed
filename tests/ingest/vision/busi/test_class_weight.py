"""Class-weight wiring for the BUSI cls+seg task.

BUSI is the only cls+seg ModelSpec; class-imbalance support was already
plumbed for the pure-classification specs (skin_lesion / chest_ct /
breast_us_kaggle / colon_histopath / lung_histopath / rsna_pneumonia /
chest_xray_pneumonia) but ``ClassificationSegmentationTask`` silently
dropped ``_class_weight_tensor``, and the BUSI spec never declared
``class_weight`` in ``hparam_space``. These tests guard both halves of
the wiring against regression.
"""

from __future__ import annotations

import torch

from claritymed.ingest.vision.busi.models.unet_resnet50 import UNET_RESNET50
from claritymed.ingest.vision.forge.tasks.cls_segmentation import (
    ClassificationSegmentationTask,
)


def _make_task() -> ClassificationSegmentationTask:
    return ClassificationSegmentationTask(
        critical_labels=("malignant",),
        critical_metric_name="malignant_recall",
        composite_weights={"malignant_recall": 0.6, "dice": 0.4},
        floors=UNET_RESNET50.task.floors,
    )


def _stub_batch():
    """3-class logits + masks shaped so the test reads loss numbers, not gradients."""
    torch.manual_seed(0)
    cls_logits = torch.randn(4, 3)
    seg_logits = torch.randn(4, 1, 8, 8)
    labels = torch.tensor([0, 1, 2, 1])  # benign, malignant, normal, malignant
    masks = (torch.rand(4, 1, 8, 8) > 0.5).float()
    return cls_logits, seg_logits, labels, masks


def test_compute_loss_falls_back_to_unweighted_ce_when_tensor_absent() -> None:
    """Empty ``hp`` → plain CE, no surprise weighting on legacy callsites."""
    task = _make_task()
    cls_logits, seg_logits, labels, masks = _stub_batch()

    loss = task.compute_loss(
        (cls_logits, seg_logits),
        {"labels": labels, "masks": masks},
        hp={},
    )
    expected_cls = torch.nn.functional.cross_entropy(cls_logits, labels)
    expected_seg = torch.nn.functional.binary_cross_entropy_with_logits(
        seg_logits, masks
    )
    # seg_loss_weight defaults to 1.0
    assert torch.isclose(loss, expected_cls + expected_seg)


def test_compute_loss_honors_class_weight_tensor_when_present() -> None:
    """``_class_weight_tensor`` in ``hp`` → CE applies it per-class.

    The whole point of the recent fix: cls+seg used to silently drop
    this tensor, leaving BUSI's 3-class imbalance unaddressed in the
    gradient. Asserting the loss matches a weighted CE confirms the
    tensor flows into the cls term (not just stored unused).
    """
    task = _make_task()
    cls_logits, seg_logits, labels, masks = _stub_batch()
    # Boost malignant (index 1) heavily so the weighted vs unweighted
    # losses are clearly different — a permissive tolerance would hide
    # a "weight kwarg silently ignored" regression.
    weight = torch.tensor([0.5, 3.0, 0.7])

    weighted_loss = task.compute_loss(
        (cls_logits, seg_logits),
        {"labels": labels, "masks": masks},
        hp={"_class_weight_tensor": weight},
    )
    expected_cls = torch.nn.functional.cross_entropy(cls_logits, labels, weight=weight)
    expected_seg = torch.nn.functional.binary_cross_entropy_with_logits(
        seg_logits, masks
    )
    assert torch.isclose(weighted_loss, expected_cls + expected_seg)

    # Sanity: weighted ≠ unweighted, otherwise the test would still pass
    # against the old buggy implementation.
    unweighted = task.compute_loss(
        (cls_logits, seg_logits),
        {"labels": labels, "masks": masks},
        hp={},
    )
    assert not torch.isclose(weighted_loss, unweighted)


def test_busi_spec_declares_class_weight_in_hparam_space() -> None:
    """BUSI must carry the same imbalance hparam as every other cls model.

    Skipping this caused malignant_recall to stall at ~0.6 because the
    framework never had a way to ask the optimizer for a weighted CE.
    """
    assert "class_weight" in UNET_RESNET50.hparam_space
    choices = UNET_RESNET50.hparam_space["class_weight"].choices  # type: ignore[attr-defined]
    assert set(choices) == {"none", "inverse_freq", "sqrt_inv_freq"}
