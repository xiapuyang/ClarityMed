"""ONNX adapter — real inference via onnxruntime.

The session loads an ``.onnx`` checkpoint and exposes it through the
:class:`DiseaseVisionModel` Protocol so the FastAPI inference loop
treats ONNX-backed diseases identically to PyTorch-backed ones.

``onnxruntime`` is already a top-level dep (the PHI scrubber and
``rapidocr-onnxruntime`` both need it), so the ONNX path costs nothing
extra in install size. Session construction is ~tens of ms for a
typical ResNet-50; per-request inference matches torch's CPU
throughput within a few percent.

Input handling
--------------

* :meth:`OnnxAdapter.preprocess` decodes via Pillow, resizes to the
  model's expected H×W (read from the session's first input shape; if
  any spatial dim is dynamic we fall back to :data:`_DEFAULT_INPUT_SIZE`
  to match the forge default), applies ImageNet normalization, and
  produces a ``float32`` array in NCHW or NHWC depending on where the
  channel-3 dimension sits in the session's declared input shape.
* :meth:`OnnxAdapter.predict` runs the session, applies softmax when
  the raw output looks like logits (any value < 0 or > 1), and emits a
  :class:`ClassificationResult` keyed off the manifest's label order.
* :meth:`OnnxAdapter.segment` returns ``None`` even for
  ``task='classification+segmentation'`` manifests — the segmentation
  output head is wired in a follow-up; producing one without a
  validated decoder would silently emit garbage masks.
* :meth:`OnnxAdapter.calibrate` is identity pass-through. Temperature
  scaling lives in :class:`Manifest.tuned_inference` and is applied at
  the registry layer, not here.
* :meth:`OnnxAdapter.quality_gate` mirrors the torch adapter's
  minimum-resolution check — same cut, same semantics, so the
  KTD-V10 "inconclusive_review" override fires the same way regardless
  of which framework backs the model.

Heavy imports (``onnxruntime``, ``numpy``, ``PIL``) are deferred to
the constructor and method bodies. Keeps ``import
claritymed.servers.vision.loader`` cheap for callers that only need
the registry side-effect.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

from claritymed.core.vision.schemas import (
    ClassificationResult,
    ConfidenceTier,
    InputQuality,
    Manifest,
    ModelSpec,
    ObjectDetectionResult,
    QualityCheck,
    SegmentationResult,
    round_sig,
)
from claritymed.servers.vision.loader import register_adapter

logger = logging.getLogger(__name__)

# Confidence-tier cut points. Mirror :mod:`torch_adapter` exactly so the
# KTD-V10 clinical-action override fires the same way whether the
# disease's primary model is .pt or .onnx. Drifting the cuts here would
# silently change behavior for ONNX-backed diseases — keep in sync.
_LOW_TIER_CEILING = 0.55
_MEDIUM_TIER_CEILING = 0.80

# Quality-gate minimum side length. Same value as the torch adapter —
# the gate is about input sanity, not the framework choice.
_MIN_RESOLUTION = 64

# Spatial size fallback when the ONNX session declares a dynamic
# H or W dim. 256 matches :data:`forge_torch._DEFAULT_INPUT_SIZE`, so a
# forge-trained model exported to ONNX without baking the size in still
# preprocesses to the same shape its training pipeline used.
_DEFAULT_INPUT_SIZE = 256

# ImageNet normalization constants. The vast majority of pretrained
# classifiers (ResNet, EfficientNet, ConvNeXt, …) ship trained against
# these. Models trained with custom normalization will need their own
# adapter subclass — keep this here so the override surface is one
# attribute, not a fork of the whole preprocess method.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class OnnxAdapter:
    """ONNX Runtime adapter satisfying :class:`DiseaseVisionModel`.

    Constructed by the loader's framework registry. Subclasses may
    override :meth:`preprocess` to swap the normalization stats or
    :meth:`predict` to add post-processing — both keep the
    integration code that wraps them.
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

        # Lazy import — keeps the registry side-effect cheap when this
        # module is imported only for ``register_adapter``.
        import onnxruntime as ort  # noqa: PLC0415

        providers = _providers_for(device)
        self._session = ort.InferenceSession(str(weights_path), providers=providers)

        # The vast majority of image classifiers have one input + one
        # output. Multi-input models would need a manifest-driven
        # mapping; reject early rather than guess.
        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise RuntimeError(
                f"OnnxAdapter expects exactly one input, got {len(inputs)} for "
                f"model_id={spec.id!r}; multi-input ONNX models need a custom adapter."
            )
        self._input_name = inputs[0].name
        self._input_shape = tuple(inputs[0].shape)
        self._layout, self._target_hw = _detect_layout_and_size(self._input_shape)
        logger.info(
            "OnnxAdapter loaded model_id=%s providers=%s input_shape=%s layout=%s target_hw=%s",
            spec.id,
            self._session.get_providers(),
            self._input_shape,
            self._layout,
            self._target_hw,
        )

    # --- DiseaseVisionModel Protocol --------------------------------------

    def preprocess(self, image: Any) -> Any:
        """Decode → resize → normalize → float32 array in session layout.

        Returns a numpy array with a leading batch dim of 1. Callers
        that already hold a PIL image (e.g. tests skipping the bytes
        path) can pass it through directly.
        """
        from PIL import Image  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        if isinstance(image, bytes):
            pil = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil = image.convert("RGB")
        else:
            raise TypeError(
                f"OnnxAdapter.preprocess expected bytes or PIL.Image, "
                f"got {type(image).__name__}"
            )

        target_h, target_w = self._target_hw
        pil = pil.resize((target_w, target_h), Image.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0  # HWC, [0, 1]
        mean = np.asarray(_IMAGENET_MEAN, dtype=np.float32)
        std = np.asarray(_IMAGENET_STD, dtype=np.float32)
        arr = (arr - mean) / std
        if self._layout == "NCHW":
            arr = arr.transpose(2, 0, 1)  # HWC → CHW
        # else NHWC: leave as HWC
        return arr[np.newaxis, ...]  # add batch dim

    def predict(self, x: Any) -> ClassificationResult:
        """Run the session and emit a :class:`ClassificationResult`.

        Treats the *first* session output as the classification head.
        Multi-head models (e.g. cls + seg) expose both heads but this
        adapter only consumes the classifier — segmentation lands in a
        follow-up once a validated decoder shape is wired.
        """
        import numpy as np  # noqa: PLC0415

        if not isinstance(x, np.ndarray):
            raise TypeError(
                f"OnnxAdapter.predict expected numpy.ndarray, got {type(x).__name__}"
            )
        raw = self._session.run(None, {self._input_name: x})[0]
        # raw shape: (1, num_classes). Squeeze batch.
        scores = np.asarray(raw, dtype=np.float32).reshape(-1)
        if scores.shape[0] != len(self._labels):
            raise RuntimeError(
                f"ONNX output size {scores.shape[0]} != manifest label count "
                f"{len(self._labels)} for model_id={self.spec.id!r}"
            )
        probs = _maybe_softmax(scores)
        # Round at the producer so wire-side schemas don't have to. The
        # torch adapter does the same; keeping it here means the
        # observable probability values are identical between
        # frameworks for the same logits.
        probs_list = [round_sig(float(p)) for p in probs]
        top1_idx = int(np.argmax(probs))
        top1_prob = probs_list[top1_idx]
        return ClassificationResult(
            labels=list(self._labels),
            probabilities=probs_list,
            top1=self._labels[top1_idx],
            top1_prob=top1_prob,
            confidence_tier=_confidence_tier(top1_prob),
        )

    def calibrate(self, raw: Any) -> dict[str, float]:
        """Identity pass-through — matches the torch adapter's v1 behavior.

        Temperature scaling, when needed, is read from
        :class:`Manifest.tuned_inference` by the registry layer; this
        method only unpacks ``raw`` into the label→probability dict
        shape the downstream wire schema expects.
        """
        if isinstance(raw, ClassificationResult):
            return dict(zip(raw.labels, raw.probabilities, strict=True))
        if isinstance(raw, dict):
            return {k: float(v) for k, v in raw.items()}
        raise TypeError(
            f"OnnxAdapter.calibrate expected ClassificationResult or dict, "
            f"got {type(raw).__name__}"
        )

    def segment(self, x: Any) -> SegmentationResult | None:
        """v1 stub: no segmentation output.

        Returning ``None`` matches the torch adapter's v1 contract.
        Adding real segmentation support requires validating the
        second output head's spatial shape against the manifest's
        ``task`` field — deferred until a real cls+seg ONNX ships.
        """
        return None

    def detect_boxes(self, x: Any) -> ObjectDetectionResult | None:
        """Classification-only adapter; YOLO detection lives in :mod:`yolo_adapter`."""
        return None

    def quality_gate(self, image: Any) -> InputQuality:
        """Minimum-resolution check, mirrored from the torch adapter.

        Decodes if necessary so callers don't have to thread a PIL
        image through both ``preprocess`` and ``quality_gate``.
        """
        from PIL import Image  # noqa: PLC0415

        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError(
                f"OnnxAdapter.quality_gate expected bytes or PIL.Image, "
                f"got {type(image).__name__}"
            )
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


# --- helpers ---------------------------------------------------------------


def _providers_for(device: str) -> list[str]:
    """Pick the onnxruntime provider list for ``device`` with CPU fallback.

    ``onnxruntime`` silently skips unavailable providers if CPU is
    listed last, but a missing CUDA provider on a "cuda" request is
    something we want to *log* not just paper over — so we surface the
    available-providers list at info level when the requested one is
    missing.
    """
    import onnxruntime as ort  # noqa: PLC0415

    available = set(ort.get_available_providers())
    requested: list[str] = []
    if device.startswith("cuda") and "CUDAExecutionProvider" in available:
        requested.append("CUDAExecutionProvider")
    elif device == "mps" and "CoreMLExecutionProvider" in available:
        requested.append("CoreMLExecutionProvider")
    elif device.startswith(("cuda", "mps")) and not requested:
        logger.info(
            "onnxruntime: no accelerator provider for device=%r (available=%s); "
            "falling back to CPU",
            device,
            sorted(available),
        )
    requested.append("CPUExecutionProvider")
    return requested


def _detect_layout_and_size(
    input_shape: tuple[int | str | None, ...],
) -> tuple[str, tuple[int, int]]:
    """Read layout (NCHW / NHWC) and target H×W from a session input shape.

    Strategy: a four-dim shape with ``shape[1] == 3`` is NCHW; with
    ``shape[3] == 3`` is NHWC. Dynamic batch dim is fine. Dynamic
    spatial dims fall back to :data:`_DEFAULT_INPUT_SIZE`.

    Any other shape (single channel, multi-channel ≠ 3, 3D, 5D) raises
    — the adapter is positioned for RGB image classifiers and rejecting
    a mismatched checkpoint at load time gives a clearer error than a
    confusing inference-time crash.
    """
    if len(input_shape) != 4:
        raise RuntimeError(
            f"OnnxAdapter expects a 4D input shape (N, C, H, W) or "
            f"(N, H, W, C); got {input_shape}"
        )

    def _to_int(d: int | str | None) -> int | None:
        return d if isinstance(d, int) and d > 0 else None

    c_first = _to_int(input_shape[1])
    c_last = _to_int(input_shape[3])
    if c_first == 3:
        layout = "NCHW"
        h = _to_int(input_shape[2]) or _DEFAULT_INPUT_SIZE
        w = _to_int(input_shape[3]) or _DEFAULT_INPUT_SIZE
    elif c_last == 3:
        layout = "NHWC"
        h = _to_int(input_shape[1]) or _DEFAULT_INPUT_SIZE
        w = _to_int(input_shape[2]) or _DEFAULT_INPUT_SIZE
    else:
        raise RuntimeError(
            f"OnnxAdapter could not identify channel dim in input shape "
            f"{input_shape} — expected 3 at position 1 (NCHW) or position 3 (NHWC)"
        )
    return layout, (h, w)


def _maybe_softmax(scores: Any) -> Any:
    """Apply softmax when ``scores`` look like raw logits.

    Heuristic: any negative entry or any entry > 1 means it's not a
    probability vector yet. ONNX models exported from PyTorch
    typically emit logits (the cross-entropy loss bakes in the
    softmax); models exported from frameworks that explicitly add
    softmax in the graph already emit probabilities and we leave them
    alone.
    """
    import numpy as np  # noqa: PLC0415

    scores = np.asarray(scores, dtype=np.float32)
    if (scores < 0.0).any() or (scores > 1.0).any():
        # Numerically stable softmax.
        shifted = scores - scores.max()
        exp = np.exp(shifted)
        return exp / exp.sum()
    # Already probabilities — guard against a near-zero sum from a
    # broken export by normalizing.
    total = float(scores.sum())
    if total > 0:
        return scores / total
    return scores


def _confidence_tier(top1_prob: float) -> ConfidenceTier:
    """Map post-calibration top1 probability to the tier ladder.

    Identical cut points to the torch adapter so KTD-V10's
    ``clinical_action`` override behaves the same regardless of
    framework backing.
    """
    if top1_prob < _LOW_TIER_CEILING:
        return "low"
    if top1_prob < _MEDIUM_TIER_CEILING:
        return "medium"
    return "high"


def _factory(
    *, spec: ModelSpec, manifest: Manifest, weights_path: Path, device: str
) -> OnnxAdapter:
    """Loader-registry factory."""
    return OnnxAdapter(
        spec=spec, manifest=manifest, weights_path=weights_path, device=device
    )


register_adapter("onnx", _factory)


__all__ = ["OnnxAdapter"]
