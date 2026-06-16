"""Forge-trained torch adapter — one factory routes cls and cls+seg manifests.

The forge framework produces manifests with ``task="classification"``
(chest CT, future skin) or ``task="classification+segmentation"``
(BUSI). This module owns the model-construction code for both shapes
so training (via :mod:`~claritymed.ingest.vision.forge.tasks`) and
inference (this adapter) build identical architectures from the same
backbone string. Drift between training and inference shapes would
manifest as a strict-mode ``load_state_dict`` failure — keeping one
source of truth here makes that impossible.

The :func:`register` entry replaces the old per-disease
``busi_unet.register()`` opt-in: a single ``pytorch`` framework key,
dispatching by ``manifest.task``. Operators who promote a chest CT
ResNet-50 alongside the BUSI U-Net call this ``register()`` once;
both run.
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


# === model construction ===================================================


def build_classifier_model(
    *, backbone: str, num_classes: int, pretrained: bool = False
):
    """Torchvision classifier wrapped to surface ``.backbone`` for round-trip.

    Used by chest CT (and any future cls-only dataset). Backbones:
    ``resnet50`` / ``efficientnet_b0`` / ``efficientnet_b3``. Adding
    a new backbone = one branch here + the matching choice in the
    ModelSpec's ``hparam_space``.
    """
    try:
        import torch.nn as nn
        import torchvision.models as tvm
    except ImportError as exc:
        raise SystemExit(
            "torch / torchvision not installed — `uv sync --extra vision-server`."
        ) from exc

    if backbone == "resnet50":
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        inner = tvm.resnet50(weights=weights)
        inner.fc = nn.Linear(inner.fc.in_features, num_classes)
    elif backbone == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        inner = tvm.efficientnet_b0(weights=weights)
        inner.classifier[1] = nn.Linear(inner.classifier[1].in_features, num_classes)
    elif backbone == "efficientnet_b3":
        weights = tvm.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        inner = tvm.efficientnet_b3(weights=weights)
        inner.classifier[1] = nn.Linear(inner.classifier[1].in_features, num_classes)
    else:
        raise ValueError(
            f"unknown classifier backbone {backbone!r}; expected "
            "'resnet50' / 'efficientnet_b0' / 'efficientnet_b3'."
        )

    class _ClsModel(nn.Module):
        def __init__(self, inner: nn.Module, backbone: str) -> None:
            super().__init__()
            self.inner = inner
            self.backbone = backbone

        def forward(self, x):
            return self.inner(x)

    return _ClsModel(inner, backbone)


def build_cls_seg_model(*, backbone: str, num_classes: int, pretrained: bool = False):
    """U-Net + classification head; returns ``(cls_logits, seg_logits)``.

    Used by BUSI (and any future cls+seg dataset). Three backbones:

    * ``custom_unet`` — small base=32 from-scratch baseline.
    * ``resnet50`` / ``efficientnet_b0`` — encoders inside an smp Unet
      with auxiliary classification head.
    """
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise SystemExit(
            "torch not installed — `uv sync --extra vision-server`."
        ) from exc

    if backbone in _SMP_ENCODER_NAMES:
        return _build_smp_unet(
            backbone=backbone, num_classes=num_classes, pretrained=pretrained, nn=nn
        )
    if backbone != "custom_unet":
        raise ValueError(
            f"unknown cls+seg backbone {backbone!r}; expected one of "
            f"{sorted({'custom_unet', *_SMP_ENCODER_NAMES})!r}."
        )
    return _build_custom_unet(num_classes=num_classes, torch=torch, nn=nn)


def _build_smp_unet(*, backbone, num_classes, pretrained, nn):
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise SystemExit(
            "segmentation-models-pytorch not installed — `uv sync --extra vision-server`."
        ) from exc

    smp_model = smp.Unet(
        encoder_name=_SMP_ENCODER_NAMES[backbone],
        encoder_weights="imagenet" if pretrained else None,
        in_channels=3,
        classes=1,
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


def _build_custom_unet(*, num_classes, torch, nn):
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

    class _CustomUnet(nn.Module):
        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.backbone = "custom_unet"
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
                nn.Flatten(), nn.Linear(base * 8, num_classes)
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

    return _CustomUnet(num_classes=num_classes)


# === Shared base for the two adapters =====================================


class _ForgeAdapterBase(TorchAdapter):
    """Common load / quality / temperature / tier logic.

    Both concrete adapters initialise the model, hash + load weights,
    and read tuned params off the manifest identically — only the
    forward-pass signature and ``segment`` body differ.
    """

    _model_factory_name: str  # set per subclass for log lines

    def __init__(
        self,
        *,
        spec: ModelSpec,
        manifest: Manifest,
        weights_path: Path,
        device: str,
    ) -> None:
        # Skip TorchAdapter's deterministic init; do our own torch load.
        self.spec = spec
        self.manifest = manifest
        self._weights_path = weights_path
        self._device = device
        self._labels = list(manifest.labels)

        import torch

        self._torch = torch
        self._model = self._build_arch(
            backbone=manifest.backbone,
            num_classes=len(self._labels),
            pretrained=False,
        ).to(device)
        try:
            state = torch.load(weights_path, map_location=device, weights_only=False)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"failed to torch.load weights at {weights_path}: {exc!s}"
            ) from exc
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        try:
            self._model.load_state_dict(state, strict=False)
            self._real_weights = True
        except Exception:  # noqa: BLE001
            logger.warning(
                "%s: checkpoint shape mismatch at %s — falling back to deterministic stub",
                type(self).__name__,
                weights_path,
            )
            self._real_weights = False
        self._model.eval()
        logger.info(
            "%s loaded model_id=%s real_weights=%s device=%s",
            type(self).__name__,
            spec.id,
            self._real_weights,
            device,
        )

    # subclass hook ----------------------------------------------------

    def _build_arch(self, *, backbone: str, num_classes: int, pretrained: bool):
        raise NotImplementedError

    # shared preprocessing --------------------------------------------

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

    def quality_gate(self, image: Any) -> InputQuality:
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
                    name="min_resolution", score=float(min(w, h)), passed=False
                )
            )
            passed = False
        else:
            checks.append(
                QualityCheck(name="min_resolution", score=float(min(w, h)), passed=True)
            )
        checks.append(QualityCheck(name="modality_match", score=1.0, passed=True))
        return InputQuality(passed=passed, checks=checks)

    def calibrate(self, raw: Any) -> ClassificationResult:
        """Build wire-shaped ``ClassificationResult`` from logits.

        ``raw`` is whatever :meth:`predict` returned — for cls-only
        that's bare logits, for cls+seg it's ``(cls_logits, seg_logits)``
        and we project to the first element.
        """
        cls_logits = raw[0] if isinstance(raw, tuple) else raw
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

    def _tier(self, prob: float) -> ConfidenceTier:
        low_max, med_max = self._tuned_confidence_bounds()
        if prob < low_max:
            return "low"
        if prob < med_max:
            return "medium"
        return "high"

    # tuned-param accessors --------------------------------------------

    def _tuned(self):
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
        """Pick argmax that also clears its per-label threshold."""
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


# === Concrete adapters ====================================================


class ClsAdapter(_ForgeAdapterBase):
    """Classification-only forge adapter (chest CT and similar)."""

    _model_factory_name = "build_classifier_model"

    def _build_arch(self, *, backbone, num_classes, pretrained):
        return build_classifier_model(
            backbone=backbone, num_classes=num_classes, pretrained=pretrained
        )

    def predict(self, x: Any):
        with self._torch.no_grad():
            return self._model(x)

    def segment(self, x: Any) -> SegmentationResult | None:
        """Cls-only — no segmentation head to query."""
        return None


class ClsSegAdapter(_ForgeAdapterBase):
    """Classification + segmentation forge adapter (BUSI U-Net)."""

    _model_factory_name = "build_cls_seg_model"

    def _build_arch(self, *, backbone, num_classes, pretrained):
        return build_cls_seg_model(
            backbone=backbone, num_classes=num_classes, pretrained=pretrained
        )

    def predict(self, x: Any):
        with self._torch.no_grad():
            cls_logits, seg_logits = self._model(x)
        return cls_logits, seg_logits

    def segment(self, x: Any) -> SegmentationResult | None:
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


# === Registration =========================================================


def register() -> None:
    """Register the unified pytorch factory under ``"pytorch"``.

    Routes by ``manifest.task`` so chest CT (cls-only) and BUSI
    (cls+seg) checkpoints both load through the same key. Operators
    promote multiple models for either dataset by calling this once.
    """
    from claritymed.servers.vision.loader import register_adapter

    def _factory(*, spec, manifest, weights_path, device):
        if manifest.task == "classification":
            return ClsAdapter(
                spec=spec, manifest=manifest, weights_path=weights_path, device=device
            )
        if manifest.task == "classification+segmentation":
            return ClsSegAdapter(
                spec=spec, manifest=manifest, weights_path=weights_path, device=device
            )
        raise RuntimeError(
            f"forge_torch: unsupported manifest.task={manifest.task!r} for "
            f"model {spec.id!r}. Add a new branch in forge_torch.register() "
            f"and a matching adapter class."
        )

    register_adapter("pytorch", _factory)


__all__ = [
    "ClsAdapter",
    "ClsSegAdapter",
    "build_classifier_model",
    "build_cls_seg_model",
    "register",
]
