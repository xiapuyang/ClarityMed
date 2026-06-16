"""``ClassificationTask`` — pure classifier, no segmentation head.

Used by datasets where masks are unavailable or irrelevant. The
critical-recall metric ("don't miss the cancer") is computed over the
union of :attr:`critical_labels`; in chest CT that's the three cancer
subtypes vs ``normal``, giving one number per breakdown regardless of
which subtype is present.

Backbones (set per-ModelSpec via ``hparam_space["backbone"]``):

* ``resnet50`` — torchvision ResNet-50, IMAGENET1K_V2 pretrained.
* ``efficientnet_b0`` — torchvision EfficientNet-B0.
* ``efficientnet_b3`` — torchvision EfficientNet-B3 (bigger, slower,
  usually strongest on CT).

Inference-param vocabulary (``inference_space`` keys this Task
recognises):

* ``temperature`` — pre-softmax logit scaling.
* ``critical_threshold`` — broadcast over every label in
  :attr:`critical_labels` to populate
  ``manifest.tuned_inference.classification_thresholds``.
* ``confidence_low_max`` / ``confidence_medium_max`` — confidence-tier
  boundaries; constraint ``low_max < medium_max`` enforced upstream.
* ``tta_default`` — manifest flag + cache-pass selector.
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
class _ClsCache:
    """Per-image cls logits cached for the tune phase."""

    labels: list[int]  # ground-truth class index per image
    cls_logits_plain: Any  # numpy float32 [N, num_classes]
    cls_logits_tta: Any  # numpy float32 [N, num_classes]


class ClassificationTask(Task):
    """Cls-only training driver.

    Construct with the medical thresholds + composite weights this
    dataset cares about; the framework reads them off the instance.

    Args:
        critical_labels: Label names that count toward critical recall
            (e.g. the cancer subtypes). Recall is computed over the
            UNION of these vs everything else.
        critical_metric_name: Key under which critical recall is
            surfaced in the breakdown dict (e.g. ``"cancer_recall"``).
            Carried by the manifest's ``eval_metrics.json`` so the
            metric is named for humans.
        composite_weights: ``{metric_name: weight}`` for the composite
            score used by feasibility-aware scoring.
        floors: per-phase :class:`FloorBundle`. Each dict carries the
            metrics this Task surfaces in its breakdown.
    """

    name = "classification"

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

    # --- model factory --------------------------------------------------

    def build_model(self, *, backbone: str, num_classes: int, pretrained: bool):
        """Delegate to the shared builder so training + inference agree.

        Architecture code lives in
        :func:`claritymed.servers.vision.adapters.forge_torch.build_classifier_model`
        — both this method and the runtime adapter call it, so the
        forward contract can't drift.
        """
        from claritymed.servers.vision.adapters.forge_torch import (
            build_classifier_model,
        )

        return build_classifier_model(
            backbone=backbone, num_classes=num_classes, pretrained=pretrained
        )

    # --- batch + loss ---------------------------------------------------

    def unpack_batch(self, batch) -> tuple[Any, dict[str, Any]]:
        """Cls dataset returns ``(imgs, labels)`` tuples."""
        imgs, labels = batch
        return imgs, {"labels": labels}

    def compute_loss(self, outputs, targets: dict[str, Any], hp: dict[str, Any]):
        import torch.nn.functional as F

        return F.cross_entropy(outputs, targets["labels"])

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
        """Forward over ``loader``; return ``(mean_loss, breakdown)``."""
        import torch
        import torch.nn.functional as F

        critical_idx = {labels.index(lbl) for lbl in self.critical_labels}

        model.eval()
        correct = total = 0
        tp = fn = 0
        loss_sum = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in loader:
                imgs, targets = self.unpack_batch(batch)
                imgs = imgs.to(device)
                gt = targets["labels"].to(device)
                logits = model(imgs)
                loss_sum += float(F.cross_entropy(logits, gt))
                preds = logits.argmax(dim=1)
                # Critical recall: predicted-in-critical-set vs
                # actual-in-critical-set. Build the membership masks via
                # explicit OR — sum() on bool tensors silently upcasts
                # to int and reads less clearly.
                gt_is_crit = torch.zeros_like(gt, dtype=torch.bool)
                pr_is_crit = torch.zeros_like(preds, dtype=torch.bool)
                for idx in critical_idx:
                    gt_is_crit |= gt == idx
                    pr_is_crit |= preds == idx
                tp += int((gt_is_crit & pr_is_crit).sum())
                fn += int((gt_is_crit & ~pr_is_crit).sum())
                correct += int((preds == gt).sum())
                total += int(gt.numel())
                n_batches += 1

        critical_recall = tp / max(tp + fn, 1)
        accuracy = correct / max(total, 1)
        composite = (
            self.composite_weights.get(self.critical_metric_name, 0.0) * critical_recall
            + self.composite_weights.get("accuracy", 0.0) * accuracy
        )
        breakdown = {
            self.critical_metric_name: critical_recall,
            "accuracy": accuracy,
            "composite": composite,
        }
        return loss_sum / max(n_batches, 1), breakdown

    # --- tune phase: cache + cache-eval --------------------------------

    def cache_outputs(self, model, loader, device) -> _ClsCache:
        """Cache plain + horizontal-flip TTA logits over ``loader``."""
        import numpy as np
        import torch

        plain: list[Any] = []
        tta: list[Any] = []
        out_labels: list[int] = []
        with torch.no_grad():
            for batch in loader:
                imgs, targets = self.unpack_batch(batch)
                imgs = imgs.to(device)
                logits = model(imgs)
                plain.append(logits.cpu().numpy())
                logits_flip = model(torch.flip(imgs, dims=[3]))
                tta.append(((logits + logits_flip) / 2.0).cpu().numpy())
                out_labels.extend(int(lbl) for lbl in targets["labels"].tolist())
        return _ClsCache(
            labels=out_labels,
            cls_logits_plain=np.concatenate(plain, axis=0),
            cls_logits_tta=np.concatenate(tta, axis=0),
        )

    def evaluate_cache(
        self,
        cache: _ClsCache,
        params: dict[str, Any],
        *,
        labels: tuple[str, ...],
    ) -> dict[str, float]:
        """Pure-numpy breakdown given one tune-param trial."""
        import numpy as np

        critical_idx = [labels.index(lbl) for lbl in self.critical_labels]

        use_tta = bool(params["tta_default"])
        cls_logits = cache.cls_logits_tta if use_tta else cache.cls_logits_plain

        scaled = cls_logits / float(params["temperature"])
        probs = softmax(scaled)

        threshold = float(params["critical_threshold"])
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
        composite = (
            self.composite_weights.get(self.critical_metric_name, 0.0) * critical_recall
            + self.composite_weights.get("accuracy", 0.0) * accuracy
        )
        return {
            self.critical_metric_name: critical_recall,
            "accuracy": accuracy,
            "composite": composite,
        }

    # --- manifest tuned_inference block --------------------------------

    def build_tuned_inference_block(
        self,
        params: dict[str, Any],
        *,
        labels: tuple[str, ...],
    ) -> TunedInferenceParams:
        """Broadcast ``critical_threshold`` over every critical label."""
        thresholds = {
            label: float(params["critical_threshold"]) for label in self.critical_labels
        }
        return TunedInferenceParams(
            temperature=float(params["temperature"]),
            classification_thresholds=thresholds,
            seg_threshold=None,
            confidence_thresholds=ConfidenceThresholds(
                low_max=float(params["confidence_low_max"]),
                medium_max=float(params["confidence_medium_max"]),
            ),
            tta_default=bool(params["tta_default"]),
        )

    # --- smoke paths ---------------------------------------------------

    def smoke_breakdown(self) -> dict[str, float]:
        """Synthetic feasible breakdown: every deploy floor cleared."""
        deploy = self.floors.deploy
        critical = self.critical_metric_name
        # Lift each metric ~0.02 above its deploy floor so the smoke
        # run is unambiguously feasible regardless of float wobble.
        critical_val = deploy.get(critical, 0.85) + 0.02
        accuracy = deploy.get("accuracy", 0.80) + 0.02
        composite = (
            self.composite_weights.get(critical, 0.0) * critical_val
            + self.composite_weights.get("accuracy", 0.0) * accuracy
        )
        return {
            critical: critical_val,
            "accuracy": accuracy,
            "composite": composite,
        }

    def smoke_tuned_params(self) -> dict[str, Any]:
        return {
            "temperature": 1.2,
            "critical_threshold": 0.45,
            "confidence_low_max": 0.55,
            "confidence_medium_max": 0.80,
            "tta_default": True,
        }


__all__ = ["ClassificationTask"]
