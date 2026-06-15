"""PyTorch adapter — v1 scaffold for the BUSI U-Net (real subclass in Unit 6).

The v1 deliverable is the *integration* surface: the adapter satisfies the
``DiseaseVisionModel`` Protocol so the FastAPI app + loader can be exercised
end-to-end before the Unit 6 training pipeline produces a real BUSI
checkpoint. Tests load a tiny random-init ``.pt`` and exercise the wire
contract; production training will subclass / replace ``TorchAdapter.predict``
with the real U-Net forward pass.

The shape of the canned implementation:

* ``preprocess(image_bytes)`` decodes via Pillow and converts to RGB
  (a real adapter would normalize + resize; v1 returns the PIL image and
  defers the tensor conversion to whatever ``predict`` the subclass
  provides).
* ``predict`` returns deterministic-from-input probabilities. The
  output is keyed off a hash of the input bytes so two test calls with
  the same image return the same scoreboard — tests can assert on a
  specific top1 label without standing up a real model.
* ``calibrate`` is identity-pass for now (temperature-scaling lands when
  the real BUSI head ships); ``segment`` returns ``None`` (segmentation
  is opt-in per ``DetectOptions.return_segmentation``); ``quality_gate``
  applies a minimum-resolution check.

Heavy ``torch`` import is lazy inside ``__init__`` so the rest of the
vision-server package — including ``loader.py`` — stays importable
when the ``vision-server`` extra isn't synced. The ``torch.load`` call
is what verifies the .pt is a real checkpoint (catches corrupted files
that the sha256 chain would also catch, but with a clearer error
attribution).
"""

from __future__ import annotations

import hashlib
import io
import logging
from dataclasses import dataclass
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
from claritymed.servers.vision.loader import register_adapter

logger = logging.getLogger(__name__)

# Confidence-tier cut points. The probabilities the v1 stub emits are
# bounded to [0.0, 1.0] but the real BUSI head will also emit values in
# this range, so the same tiering rule applies before and after Unit 6.
# Keeping the cuts here (not in YAML) makes the rule easy to find when
# the calibration story lands.
_LOW_TIER_CEILING = 0.55
_MEDIUM_TIER_CEILING = 0.80

# Quality-gate minimum side length. BiomedCLIP-tagged ultrasound images
# from the BUSI dataset are typically ≥256 px on the short side; below
# 64 px the U-Net's first downsample produces single-pixel features and
# the network's outputs become meaningless. KTD-V10 translates a failed
# quality_gate into clinical_action="inconclusive_review" downstream.
_MIN_RESOLUTION = 64


@dataclass
class _LoadedCheckpoint:
    """Thin record of what torch.load actually returned, for logging."""

    keys: tuple[str, ...]
    device: str


class TorchAdapter:
    """v1 PyTorch adapter — deterministic stub satisfying ``DiseaseVisionModel``.

    Constructed by the loader's framework registry; subclasses (e.g.
    Unit 6's BUSI U-Net) override ``predict`` / ``segment`` / ``calibrate``
    to plug a real forward pass while keeping the integration code that
    wraps them.
    """

    def __init__(
        self,
        *,
        spec: ModelSpec,
        manifest: Manifest,
        weights_path: Path,
        device: str,
    ) -> None:
        self.spec = spec
        self.manifest = manifest
        self._weights_path = weights_path
        self._device = device
        self._labels: list[str] = list(manifest.labels)
        # Lazy torch import — keeps `from claritymed.servers.vision.loader
        # import register_adapter` cheap for callers that just need the
        # registry side-effect.
        import torch

        # torch.load on a tiny random .pt is fast (~ms) and verifies that
        # the checkpoint isn't corrupted (sha256 chain caught the digest
        # but not the format). Real adapters keep the loaded state on
        # ``self`` so ``predict`` can use it; v1 only records that the
        # load worked.
        state = torch.load(weights_path, map_location=device, weights_only=False)
        if not isinstance(state, dict):
            raise RuntimeError(
                f"torch.load({weights_path}) returned {type(state).__name__}, "
                "expected dict — checkpoint format mismatch"
            )
        self._checkpoint = _LoadedCheckpoint(keys=tuple(state.keys()), device=device)
        logger.info(
            "TorchAdapter loaded model_id=%s framework=%s keys=%s",
            spec.id,
            manifest.framework,
            self._checkpoint.keys,
        )

    # --- DiseaseVisionModel Protocol --------------------------------------

    def preprocess(self, image: Any) -> Any:
        """Decode image bytes → a PIL ``Image`` in RGB.

        The real Unit 6 adapter will resize + normalize into a tensor.
        v1 returns the PIL image so the deterministic stub ``predict``
        can hash it directly; the Protocol type is ``Any`` so this
        passes type-checking.
        """
        from PIL import Image

        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, bytes):
            return Image.open(io.BytesIO(image)).convert("RGB")
        raise TypeError(
            f"TorchAdapter.preprocess expected bytes or PIL.Image, "
            f"got {type(image).__name__}"
        )

    def predict(self, x: Any) -> ClassificationResult:
        """Deterministic stub classifier — top1 chosen by image-hash digest.

        Real Unit 6 adapter replaces this with the U-Net forward + the
        classification head. Until then the stub emits a stable scoreboard
        per image so the wire contract round-trips through the FastAPI
        app deterministically.

        The probabilities are biased toward the chosen top1 (~0.7) with
        the remaining mass split among the other labels, so
        ``confidence_tier`` lands at ``"medium"`` for the canned path —
        which exercises the KTD-V10 override neither way (it only fires
        on ``"low"``). Tests that need to force ``low`` pass an image
        whose hash makes the gap small, or use the ``InputQuality``
        override path instead.
        """
        from PIL import Image

        if not isinstance(x, Image.Image):
            raise TypeError(
                f"TorchAdapter.predict expected PIL.Image, got {type(x).__name__}"
            )
        # Hash the raw RGB bytes; PIL's tobytes is a stable serialization.
        digest = hashlib.sha256(x.tobytes()).digest()
        top1_idx = digest[0] % len(self._labels)
        top1_label = self._labels[top1_idx]
        # Probabilities: top1 gets 0.7, the remaining 0.3 split evenly.
        # Deterministic, easy to assert on in tests.
        if len(self._labels) == 1:
            probs = [1.0]
        else:
            top1_p = 0.70
            other_p = (1.0 - top1_p) / (len(self._labels) - 1)
            probs = [
                top1_p if i == top1_idx else other_p for i in range(len(self._labels))
            ]
        return ClassificationResult(
            labels=list(self._labels),
            probabilities=probs,
            top1=top1_label,
            top1_prob=probs[top1_idx],
            confidence_tier=_confidence_tier(probs[top1_idx]),
        )

    def calibrate(self, raw: Any) -> dict[str, float]:
        """Identity pass-through in v1.

        Returned shape is ``{label: probability}`` — matches what a
        real temperature-scaling step would emit. The v1 stub assumes
        ``raw`` already came from ``predict`` (calibrated identically),
        so this just unpacks the probabilities. Real Unit 6 adapter
        runs the temperature-scaled softmax here.
        """
        if isinstance(raw, ClassificationResult):
            return dict(zip(raw.labels, raw.probabilities, strict=True))
        if isinstance(raw, dict):
            return {k: float(v) for k, v in raw.items()}
        raise TypeError(
            f"TorchAdapter.calibrate expected ClassificationResult or dict, "
            f"got {type(raw).__name__}"
        )

    def segment(self, x: Any) -> SegmentationResult | None:
        """v1 stub: no segmentation. Real BUSI adapter overrides.

        Returning ``None`` is the documented "task is classification
        only" answer (origin §6 Protocol); the FastAPI handler honors
        ``DetectOptions.return_segmentation=False`` regardless of what
        we return here.
        """
        return None

    def quality_gate(self, image: Any) -> InputQuality:
        """Minimum-resolution check.

        Real adapter would add modality-specific checks (ultrasound
        speckle ratio, JPEG compression artifact score). v1 fires on
        the smallest side being under ``_MIN_RESOLUTION`` — catches the
        obvious "user pasted a 32×32 thumbnail" failure mode.
        """
        from PIL import Image

        if not isinstance(image, Image.Image):
            # Be lenient — the FastAPI handler calls preprocess then
            # quality_gate, but the Protocol signature lets either
            # accept the same shape. Decoding here is a small cost
            # and avoids forcing the caller to thread the PIL image.
            image = self.preprocess(image)
        width, height = image.size
        min_side = min(width, height)
        passed = min_side >= _MIN_RESOLUTION
        return InputQuality(
            passed=passed,
            checks=[
                QualityCheck(
                    name="min_resolution",
                    score=float(min_side),
                    passed=passed,
                )
            ],
        )


def _confidence_tier(top1_prob: float) -> ConfidenceTier:
    """Map post-calibration top1 probability to the tier ladder.

    KTD-V10 keys clinical-action override on ``"low"``; the LLM-side
    reply prompt branches on the same value. Splitting the rule into a
    function (vs. inlining at every callsite) keeps the cut points in
    one place.
    """
    if top1_prob < _LOW_TIER_CEILING:
        return "low"
    if top1_prob < _MEDIUM_TIER_CEILING:
        return "medium"
    return "high"


def _factory(
    *, spec: ModelSpec, manifest: Manifest, weights_path: Path, device: str
) -> TorchAdapter:
    """Loader-registry factory. Trivial wrapper for symmetry with onnx_adapter."""
    return TorchAdapter(
        spec=spec, manifest=manifest, weights_path=weights_path, device=device
    )


register_adapter("pytorch", _factory)


__all__ = ["TorchAdapter"]
