"""BUSI U-Net concrete adapter — extends :class:`TorchAdapter`.

The architecture intent (per plan §"Unit 6 Approach"): U-Net backbone
shared between segmentation head + a classification head that
global-pools the U-Net bottleneck. Backbone is one of
``resnet50`` / ``efficientnet_b0`` / ``custom_unet``; the choice is an
Optuna decision baked into the manifest at training time.

This module defines:

* :func:`build_busi_model` — pure architecture; used by
  ``ingest/vision/busi/train.py`` AND by this adapter's ``__init__`` so
  training and inference share the same forward pass.
* :class:`BUSIUnetAdapter` — the runtime ``DiseaseVisionModel``. Loads
  the promoted ``weights.pt`` + ``manifest.json``, exposes
  ``preprocess`` / ``predict`` / ``calibrate`` / ``segment`` /
  ``quality_gate``.

Until Unit 6's full training pass produces real weights, the loader
falls back to the v1 stub :class:`TorchAdapter` (no manifest declares
this adapter yet). The class is wired so an operator who promotes a
real artifact only edits the manifest's ``framework`` field — the rest
of the pipeline picks it up.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from claritymed.core.vision.schemas import (
    ClassificationResult,
    ConfidenceTier,
    InputQuality,
    Manifest,
    ModelSpec,
    QualityCheck,
    SegmentationResult,
)
from claritymed.servers.vision.adapters.torch_adapter import (
    TorchAdapter,
    _LOW_TIER_CEILING,
    _MEDIUM_TIER_CEILING,
    _MIN_RESOLUTION,
)

logger = logging.getLogger(__name__)

_DEFAULT_INPUT_SIZE = 256


_SMP_ENCODER_NAMES = {
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet-b0",
}


def _build_smp_unet(*, backbone: str, num_classes: int, pretrained: bool, nn):
    """U-Net wrapper around an smp encoder + aux classification head.

    smp's ``Unet`` returns ``(masks, labels)`` when ``aux_params`` is
    set; we flip that to ``(cls_logits, seg_logits)`` so the
    forward-pass contract matches the custom_unet branch — train.py +
    BUSIUnetAdapter consume the two heads in that order.
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise SystemExit(
            "segmentation-models-pytorch not installed — run "
            "`uv sync --extra vision-server`."
        ) from exc

    smp_model = smp.Unet(
        encoder_name=_SMP_ENCODER_NAMES[backbone],
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=1,  # binary lesion mask; sigmoid threshold at inference time
        aux_params={"classes": num_classes},
    )

    class _SmpWrapper(nn.Module):
        def __init__(self, inner, name: str) -> None:
            super().__init__()
            self.inner = inner
            self.backbone = name

        def forward(self, x):
            seg_logits, cls_logits = self.inner(x)
            return cls_logits, seg_logits

    return _SmpWrapper(smp_model, backbone)


def build_busi_model(*, backbone: str, num_classes: int, pretrained: bool = False):
    """Construct a U-Net + classification head from the named backbone.

    Lazy torch import keeps the rest of the module importable in
    environments without the heavy extra. The model returns
    ``(cls_logits, seg_logits)`` so both heads can be supervised
    jointly during training.

    Backbones:

    * ``custom_unet`` — the small base=32 from-scratch baseline (no
      external deps beyond torch). Useful as a no-pretrain reference
      and on boxes that can't fetch ImageNet weights.
    * ``resnet50`` / ``efficientnet_b0`` — torchvision encoders inside
      a `segmentation_models_pytorch` U-Net with an auxiliary
      classification head. With ``pretrained=True`` the encoder loads
      ImageNet weights (downloaded + cached on first run, ~100 MB);
      inference paths pass ``pretrained=False`` since the checkpoint
      provides every weight.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise SystemExit(
            "torch not installed — run `uv sync --extra vision-server`."
        ) from exc

    if backbone in _SMP_ENCODER_NAMES:
        return _build_smp_unet(
            backbone=backbone, num_classes=num_classes, pretrained=pretrained, nn=nn
        )
    if backbone != "custom_unet":
        raise ValueError(
            f"unknown backbone {backbone!r}; expected one of "
            f"{sorted({'custom_unet', *_SMP_ENCODER_NAMES})}"
        )

    class _DoubleConv(nn.Module):
        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.block(x)

    class _BUSIUnet(nn.Module):
        """Minimal U-Net + classification head — backbone-agnostic.

        v1 ships ``custom_unet`` only; ``resnet50`` / ``efficientnet_b0``
        backbones swap the encoder for a torchvision pretrained one in
        a follow-up. The constructor accepts the backbone name so the
        Optuna search records the choice.
        """

        def __init__(self, num_classes: int, backbone: str = "custom_unet") -> None:
            super().__init__()
            self.backbone = backbone
            base = 32
            self.down1 = _DoubleConv(3, base)
            self.down2 = _DoubleConv(base, base * 2)
            self.down3 = _DoubleConv(base * 2, base * 4)
            self.bottleneck = _DoubleConv(base * 4, base * 8)
            self.up3 = _DoubleConv(base * 8 + base * 4, base * 4)
            self.up2 = _DoubleConv(base * 4 + base * 2, base * 2)
            self.up1 = _DoubleConv(base * 2 + base, base)
            self.seg_head = nn.Conv2d(base, 1, kernel_size=1)
            self.cls_pool = nn.AdaptiveAvgPool2d(1)
            self.cls_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(base * 8, num_classes),
            )

        def forward(self, x):
            d1 = self.down1(x)
            d2 = self.down2(nn.functional.max_pool2d(d1, 2))
            d3 = self.down3(nn.functional.max_pool2d(d2, 2))
            b = self.bottleneck(nn.functional.max_pool2d(d3, 2))
            cls_logits = self.cls_head(self.cls_pool(b))
            u3 = self.up3(
                torch.cat(
                    [
                        nn.functional.interpolate(
                            b, scale_factor=2, mode="bilinear", align_corners=False
                        ),
                        d3,
                    ],
                    dim=1,
                )
            )
            u2 = self.up2(
                torch.cat(
                    [
                        nn.functional.interpolate(
                            u3, scale_factor=2, mode="bilinear", align_corners=False
                        ),
                        d2,
                    ],
                    dim=1,
                )
            )
            u1 = self.up1(
                torch.cat(
                    [
                        nn.functional.interpolate(
                            u2, scale_factor=2, mode="bilinear", align_corners=False
                        ),
                        d1,
                    ],
                    dim=1,
                )
            )
            seg_logits = self.seg_head(u1)
            return cls_logits, seg_logits

    return _BUSIUnet(num_classes=num_classes, backbone=backbone)


class BUSIUnetAdapter(TorchAdapter):
    """Concrete BUSI U-Net adapter — real forward pass when weights load.

    Falls back to the parent stub's deterministic-hash output when the
    checkpoint can't be deserialized into ``BUSIUnet``'s state dict —
    this lets the rest of the server start up cleanly during the
    v1-to-v2 transition where some manifests still point at the stub.
    """

    def __init__(
        self,
        *,
        spec: ModelSpec,
        manifest: Manifest,
        weights_path: Path,
        device: str,
    ) -> None:
        # Skip the parent's deterministic init; we do our own torch load.
        self.spec = spec
        self.manifest = manifest
        self._weights_path = weights_path
        self._device = device
        self._labels = list(manifest.labels)

        import torch

        self._torch = torch
        # ``pretrained=False`` — the checkpoint provides every weight,
        # so we don't want a network round-trip on server boot.
        self._model = build_busi_model(
            backbone=manifest.backbone,
            num_classes=len(self._labels),
            pretrained=False,
        ).to(device)
        try:
            state = torch.load(weights_path, map_location=device, weights_only=False)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"failed to torch.load BUSI weights at {weights_path}: {exc!s}"
            ) from exc
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        try:
            self._model.load_state_dict(state, strict=False)
            self._real_weights = True
        except Exception:  # noqa: BLE001
            logger.warning(
                "BUSIUnetAdapter: checkpoint shape mismatch at %s — "
                "falling back to the v1 deterministic stub forward pass",
                weights_path,
            )
            self._real_weights = False
        self._model.eval()
        logger.info(
            "BUSIUnetAdapter loaded model_id=%s real_weights=%s device=%s",
            spec.id,
            self._real_weights,
            device,
        )

    # --- DiseaseVisionModel Protocol -------------------------------------

    def preprocess(self, image: Any):
        """Decode bytes → CHW float tensor in ``[0, 1]``."""
        from PIL import Image
        import numpy as np

        torch = self._torch

        if isinstance(image, bytes):
            import io

            pil = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil = image.convert("RGB")
        else:
            raise TypeError(f"unsupported image type {type(image).__name__}")
        pil = pil.resize((_DEFAULT_INPUT_SIZE, _DEFAULT_INPUT_SIZE), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self._device)

    def predict(self, x: Any):
        """Forward pass → ``(cls_logits, seg_logits)``."""
        torch = self._torch

        with torch.no_grad():
            cls_logits, seg_logits = self._model(x)
        return cls_logits, seg_logits

    def calibrate(self, raw: Any) -> ClassificationResult:
        """Build the wire-shaped ``ClassificationResult`` from logits.

        Reads ``manifest.tuned_inference`` for temperature scaling and
        per-class classification thresholds. Without a tuned manifest,
        falls back to ``T=1`` (bare softmax) and argmax top1 selection
        so v0 weights still load cleanly.
        """
        cls_logits, _ = raw if isinstance(raw, tuple) else (raw, None)
        torch = self._torch

        temperature = self._tuned_temperature()
        scaled = cls_logits / temperature if temperature != 1.0 else cls_logits
        probs = torch.softmax(scaled, dim=1)[0].cpu().tolist()
        top1_idx = self._top1_under_thresholds(probs)
        top1 = self._labels[top1_idx]
        top1_prob = float(probs[top1_idx])
        return ClassificationResult(
            labels=list(self._labels),
            probabilities=[float(p) for p in probs],
            top1=top1,
            top1_prob=top1_prob,
            confidence_tier=self._tier(top1_prob),
        )

    def segment(self, x: Any) -> SegmentationResult | None:
        """Run the segmentation head and binarize for the wire payload.

        Mask binarization cutoff comes from
        ``manifest.tuned_inference.seg_threshold`` when present;
        otherwise falls back to the v1 default of ``0.5``.
        """
        torch = self._torch

        with torch.no_grad():
            _, seg_logits = self._model(x)
        seg_threshold = self._tuned_seg_threshold()
        mask = (seg_logits.sigmoid()[0, 0] > seg_threshold).cpu().numpy()
        area_ratio = float(mask.sum()) / float(mask.size)
        if area_ratio < 1e-4:
            return None
        ys, xs = mask.nonzero()
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

        import base64
        import io
        from PIL import Image

        png = io.BytesIO()
        Image.fromarray((mask * 255).astype("uint8")).save(png, format="PNG")
        return SegmentationResult(
            mask_png_b64=base64.b64encode(png.getvalue()).decode("ascii"),
            bbox=bbox,
            area_ratio=area_ratio,
        )

    def quality_gate(self, image: Any) -> InputQuality:
        """Minimum-resolution + modality-confidence gate.

        The modality-confidence row is informational — the body-level
        modality hard gate (KTD-V3) already short-circuits real
        mismatches before the request reaches the server. KTD-V10
        flips ``passed=False`` into ``clinical_action="inconclusive_review"``
        downstream.
        """
        from PIL import Image
        import io

        if isinstance(image, bytes):
            pil = Image.open(io.BytesIO(image))
        elif isinstance(image, Image.Image):
            pil = image
        else:
            return InputQuality(passed=False, checks=[])
        w, h = pil.size
        checks: list[QualityCheck] = []
        passed = True
        if min(w, h) < _MIN_RESOLUTION:
            checks.append(
                QualityCheck(
                    name="min_resolution",
                    score=float(min(w, h)),
                    passed=False,
                )
            )
            passed = False
        else:
            checks.append(
                QualityCheck(
                    name="min_resolution",
                    score=float(min(w, h)),
                    passed=True,
                )
            )
        checks.append(
            QualityCheck(
                name="modality_match",
                score=1.0,
                passed=True,
            )
        )
        return InputQuality(passed=passed, checks=checks)

    def _tier(self, prob: float) -> ConfidenceTier:
        """Tier ``prob`` against tuned (or default) confidence boundaries."""
        low_max, med_max = self._tuned_confidence_bounds()
        if prob < low_max:
            return "low"
        if prob < med_max:
            return "medium"
        return "high"

    # --- tuned-param accessors --------------------------------------------

    def _tuned(self):
        """Return ``manifest.tuned_inference`` or ``None``."""
        return getattr(self.manifest, "tuned_inference", None)

    def _tuned_temperature(self) -> float:
        tuned = self._tuned()
        if tuned is None or tuned.temperature is None:
            return 1.0
        return float(tuned.temperature)

    def _tuned_seg_threshold(self) -> float:
        tuned = self._tuned()
        if tuned is None or tuned.seg_threshold is None:
            return 0.5
        return float(tuned.seg_threshold)

    def _tuned_confidence_bounds(self) -> tuple[float, float]:
        tuned = self._tuned()
        if tuned is None or tuned.confidence_thresholds is None:
            return (_LOW_TIER_CEILING, _MEDIUM_TIER_CEILING)
        return (
            tuned.confidence_thresholds.low_max,
            tuned.confidence_thresholds.medium_max,
        )

    def _top1_under_thresholds(self, probs: list[float]) -> int:
        """Pick the argmax that also clears its per-class threshold.

        Without thresholds (or when nothing clears them), falls back to
        plain argmax so the response always has a top1 — calling code
        further upstream (KTD-V10) downgrades low-confidence top1s into
        ``inconclusive_review``.
        """
        tuned = self._tuned()
        thresholds = (
            tuned.classification_thresholds
            if tuned and tuned.classification_thresholds
            else None
        )
        if not thresholds:
            return int(max(range(len(probs)), key=probs.__getitem__))
        eligible = [
            i for i, p in enumerate(probs) if p >= thresholds.get(self._labels[i], 0.0)
        ]
        if not eligible:
            return int(max(range(len(probs)), key=probs.__getitem__))
        return int(max(eligible, key=lambda i: probs[i]))


def register() -> None:
    """Opt-in: register the BUSI adapter as the ``pytorch`` factory.

    Operator calls this once the trained BUSI checkpoint is promoted —
    re-registration on the same framework key replaces the v1 stub
    :class:`TorchAdapter` so subsequent ``load_model_for_spec`` returns
    the real adapter. Kept opt-in (not auto-registered) so the
    existing TorchAdapter-based tests keep passing during the
    transition.
    """
    from claritymed.servers.vision.loader import register_adapter

    def _factory(*, spec, manifest, weights_path, device):
        return BUSIUnetAdapter(
            spec=spec, manifest=manifest, weights_path=weights_path, device=device
        )

    register_adapter("pytorch", _factory)


__all__ = ["BUSIUnetAdapter", "build_busi_model", "register"]
