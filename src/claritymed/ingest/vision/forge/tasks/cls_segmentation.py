"""``ClassificationSegmentationTask`` — U-Net + cls head.

Lifted from the original ``busi_unet`` adapter + busi/train.py. The
model returns ``(cls_logits, seg_logits)`` and is trained jointly with
a cross-entropy + BCE composite loss. The tune phase additionally
optimizes a ``seg_threshold`` for mask binarization.

Backbones:

* ``custom_unet`` — small base=32 from-scratch baseline. Useful as a
  no-pretrain reference on boxes that can't fetch ImageNet weights.
* ``resnet50`` / ``efficientnet_b0`` — encoders inside an
  ``segmentation_models_pytorch`` U-Net with an auxiliary
  classification head.

Inference-param vocabulary (in addition to the cls task's set):

* ``seg_threshold`` — sigmoid cutoff for binarizing the soft
  segmentation mask. Affects dice + bbox + area_ratio.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from claritymed.core.vision.schemas import (
    ConfidenceThresholds,
    TunedInferenceParams,
)
from claritymed.ingest.vision.forge.common import argmax_with_threshold, softmax
from claritymed.ingest.vision.forge.tasks.base import FloorBundle, Task

logger = logging.getLogger(__name__)


@dataclass
class _ClsSegCache:
    """Per-image cls logits + seg probs cached for the tune phase."""

    labels: list[int]
    masks: Any  # numpy uint8 [N, H, W] — binary GT mask
    cls_logits_plain: Any
    cls_logits_tta: Any
    seg_probs_plain: Any
    seg_probs_tta: Any


class ClassificationSegmentationTask(Task):
    """Cls + binary segmentation training driver.

    Args:
        critical_labels: As in :class:`ClassificationTask`. For BUSI
            that's ``("malignant",)``.
        critical_metric_name: e.g. ``"malignant_recall"``.
        composite_weights: ``{metric_name: weight}``. BUSI's historical
            value: ``{"malignant_recall": 0.6, "dice": 0.4}``.
        floors: per-phase :class:`FloorBundle`. Each dict carries the
            three metrics the breakdown surfaces (critical recall,
            dice, accuracy).
    """

    name = "classification+segmentation"

    def __init__(
        self,
        *,
        critical_labels: tuple[str, ...],
        critical_metric_name: str,
        composite_weights: dict[str, float],
        floors: FloorBundle,
    ) -> None:
        self.critical_labels = critical_labels
        self.critical_metric_name = critical_metric_name
        self.composite_weights = dict(composite_weights)
        self.floors = floors
        self._validate_metric_keys()

    @property
    def breakdown_metric_keys(self) -> frozenset[str]:
        return frozenset({self.critical_metric_name, "accuracy", "dice"})

    # --- model factory --------------------------------------------------

    def build_model(self, *, backbone: str, num_classes: int, pretrained: bool):
        """Delegate to the shared builder so training + inference agree.

        Architecture code lives in
        :func:`claritymed.servers.vision.adapters.forge_torch.build_cls_seg_model`
        — both this method and the runtime adapter call it.
        """
        from claritymed.servers.vision.adapters.forge_torch import build_cls_seg_model

        return build_cls_seg_model(
            backbone=backbone, num_classes=num_classes, pretrained=pretrained
        )

    # --- batch + loss ---------------------------------------------------

    def unpack_batch(self, batch) -> tuple[Any, dict[str, Any]]:
        """Cls+seg dataset returns ``(imgs, masks, labels)`` tuples."""
        imgs, masks, labels = batch
        return imgs, {"labels": labels, "masks": masks}

    def compute_loss(self, outputs, targets: dict[str, Any], hp: dict[str, Any]):
        import torch.nn.functional as F

        cls_logits, seg_logits = outputs
        # ``_class_weight_tensor`` is stashed in hp by the framework when
        # the trial picked an inverse_freq / sqrt_inv_freq scheme on
        # ``class_weight``. Absent → plain unweighted CE (historical
        # default). Mirrors the read in ``ClassificationTask.compute_loss``.
        cls_loss = F.cross_entropy(
            cls_logits, targets["labels"], weight=hp.get("_class_weight_tensor")
        )
        seg_loss = F.binary_cross_entropy_with_logits(seg_logits, targets["masks"])
        return cls_loss + float(hp.get("seg_loss_weight", 1.0)) * seg_loss

    # --- per-epoch evaluation -----------------------------------------

    def evaluate(
        self,
        model,
        loader,
        device,
        *,
        labels: tuple[str, ...],
        hp: dict[str, Any],
    ) -> tuple[float, dict[str, float]]:
        import torch

        critical_idx = [labels.index(lbl) for lbl in self.critical_labels]
        seg_weight = float(hp.get("seg_loss_weight", 1.0))

        model.eval()
        tp = fn = correct = total = 0
        dice_sum = 0.0
        loss_sum = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in loader:
                imgs, targets = self.unpack_batch(batch)
                imgs = imgs.to(device)
                gt = targets["labels"].to(device)
                masks = targets["masks"].to(device)
                cls_logits, seg_logits = model(imgs)
                loss_sum += float(
                    self.compute_loss(
                        (cls_logits, seg_logits), {"labels": gt, "masks": masks}, hp
                    )
                )
                preds = cls_logits.argmax(dim=1)
                gt_is_crit = torch.zeros_like(gt, dtype=torch.bool)
                pr_is_crit = torch.zeros_like(preds, dtype=torch.bool)
                for idx in critical_idx:
                    gt_is_crit |= gt == idx
                    pr_is_crit |= preds == idx
                tp += int((gt_is_crit & pr_is_crit).sum())
                fn += int((gt_is_crit & ~pr_is_crit).sum())
                correct += int((preds == gt).sum())
                total += int(gt.numel())
                dice_sum += float(_dice_score_batch(seg_logits.sigmoid(), masks))
                n_batches += 1

        critical_recall = tp / max(tp + fn, 1)
        accuracy = correct / max(total, 1)
        dice = dice_sum / max(n_batches, 1)
        composite = (
            self.composite_weights.get(self.critical_metric_name, 0.0) * critical_recall
            + self.composite_weights.get("dice", 0.0) * dice
        )
        _ = seg_weight  # used inside compute_loss via hp
        return loss_sum / max(n_batches, 1), {
            self.critical_metric_name: critical_recall,
            "accuracy": accuracy,
            "dice": dice,
            "composite": composite,
        }

    # --- tune phase: cache + cache-eval --------------------------------

    def cache_outputs(self, model, loader, device) -> _ClsSegCache:
        import numpy as np
        import torch

        cls_plain: list[Any] = []
        cls_tta: list[Any] = []
        seg_plain: list[Any] = []
        seg_tta: list[Any] = []
        out_labels: list[int] = []
        masks_out: list[Any] = []

        with torch.no_grad():
            for batch in loader:
                imgs, targets = self.unpack_batch(batch)
                imgs = imgs.to(device)
                cls_logits, seg_logits = model(imgs)
                cls_plain.append(cls_logits.cpu().numpy())
                seg_plain.append(seg_logits.sigmoid().cpu().numpy()[:, 0])

                cls_logits_flip, seg_logits_flip = model(torch.flip(imgs, dims=[3]))
                cls_avg = (cls_logits + cls_logits_flip) / 2.0
                seg_avg = (
                    seg_logits.sigmoid()
                    + torch.flip(seg_logits_flip.sigmoid(), dims=[3])
                ) / 2.0
                cls_tta.append(cls_avg.cpu().numpy())
                seg_tta.append(seg_avg.cpu().numpy()[:, 0])

                out_labels.extend(int(lbl) for lbl in targets["labels"].tolist())
                masks_out.append(targets["masks"].cpu().numpy()[:, 0])

        return _ClsSegCache(
            labels=out_labels,
            masks=np.concatenate(masks_out, axis=0),
            cls_logits_plain=np.concatenate(cls_plain, axis=0),
            cls_logits_tta=np.concatenate(cls_tta, axis=0),
            seg_probs_plain=np.concatenate(seg_plain, axis=0),
            seg_probs_tta=np.concatenate(seg_tta, axis=0),
        )

    def evaluate_cache(
        self,
        cache: _ClsSegCache,
        params: dict[str, Any],
        *,
        labels: tuple[str, ...],
    ) -> dict[str, float]:
        import numpy as np

        critical_idx = [labels.index(lbl) for lbl in self.critical_labels]

        use_tta = bool(params["tta_default"])
        cls_logits = cache.cls_logits_tta if use_tta else cache.cls_logits_plain
        seg_probs = cache.seg_probs_tta if use_tta else cache.seg_probs_plain

        scaled = cls_logits / float(params["temperature"])
        probs = softmax(scaled)
        threshold = float(params["critical_threshold"])
        seg_threshold = float(params["seg_threshold"])

        preds = argmax_with_threshold(
            probs, critical_indices=critical_idx, threshold=threshold
        )
        gt = np.asarray(cache.labels)
        gt_is_crit = np.isin(gt, critical_idx)
        pr_is_crit = np.isin(preds, critical_idx)
        tp = int((gt_is_crit & pr_is_crit).sum())
        fn = int((gt_is_crit & ~pr_is_crit).sum())
        correct = int((preds == gt).sum())
        total = int(gt.shape[0])

        critical_recall = tp / max(tp + fn, 1)
        accuracy = correct / max(total, 1)
        pred_masks = (seg_probs > seg_threshold).astype(np.float32)
        dice = _mean_dice(pred_masks, cache.masks)
        composite = (
            self.composite_weights.get(self.critical_metric_name, 0.0) * critical_recall
            + self.composite_weights.get("dice", 0.0) * dice
        )
        return {
            self.critical_metric_name: critical_recall,
            "accuracy": accuracy,
            "dice": dice,
            "composite": composite,
        }

    # --- manifest tuned_inference block --------------------------------

    def build_tuned_inference_block(
        self,
        params: dict[str, Any],
        *,
        labels: tuple[str, ...],
    ) -> TunedInferenceParams:
        thresholds = {
            label: float(params["critical_threshold"]) for label in self.critical_labels
        }
        return TunedInferenceParams(
            temperature=float(params["temperature"]),
            classification_thresholds=thresholds,
            seg_threshold=float(params["seg_threshold"]),
            confidence_thresholds=ConfidenceThresholds(
                low_max=float(params["confidence_low_max"]),
                medium_max=float(params["confidence_medium_max"]),
            ),
            tta_default=bool(params["tta_default"]),
        )

    # --- smoke paths ---------------------------------------------------

    def smoke_breakdown(self) -> dict[str, float]:
        deploy = self.floors.deploy
        critical = self.critical_metric_name
        critical_val = deploy.get(critical, 0.85) + 0.02
        accuracy = deploy.get("accuracy", 0.85) + 0.02
        dice = deploy.get("dice", 0.70) + 0.02
        composite = (
            self.composite_weights.get(critical, 0.0) * critical_val
            + self.composite_weights.get("dice", 0.0) * dice
        )
        return {
            critical: critical_val,
            "accuracy": accuracy,
            "dice": dice,
            "composite": composite,
        }

    def smoke_tuned_params(self) -> dict[str, Any]:
        return {
            "temperature": 1.2,
            "critical_threshold": 0.45,
            "seg_threshold": 0.5,
            "confidence_low_max": 0.55,
            "confidence_medium_max": 0.80,
            "tta_default": True,
        }


# --- numpy helpers used only by this task ---------------------------------


def _dice_score_batch(pred, target, eps: float = 1e-6) -> float:
    """Per-batch dice — used in the per-epoch eval loop (torch tensors)."""
    pred_bin = (pred > 0.5).float()
    num = 2 * (pred_bin * target).sum()
    den = pred_bin.sum() + target.sum() + eps
    return float(num / den)


def _mean_dice(pred_masks, gt_masks, eps: float = 1e-6) -> float:
    """Mean per-image dice over the cache pass (numpy arrays).

    Empty-vs-empty (e.g. ``normal`` images) score as 1.0 so the
    aggregate isn't pulled down by classes without lesions.
    """
    import numpy as np

    intersection = (pred_masks * gt_masks).sum(axis=(1, 2))
    union = pred_masks.sum(axis=(1, 2)) + gt_masks.sum(axis=(1, 2))
    dice = (2.0 * intersection + eps) / (union + eps)
    return float(np.mean(dice))


__all__ = ["ClassificationSegmentationTask"]
