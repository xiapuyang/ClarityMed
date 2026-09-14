"""YOLO adapter — object-detection via an ONNX-exported YOLO checkpoint.

Ultralytics-trained YOLOv5/v8/v11 detectors export cleanly to ONNX,
and ``onnxruntime`` is already a top-level dep (PHI scrubber +
``rapidocr-onnxruntime`` need it), so a YOLO checkpoint costs no extra
install footprint relative to the classification path.

The adapter loads the ONNX session once and exposes the model through
the :class:`DiseaseVisionModel` Protocol so the FastAPI inference loop
treats detection-backed diseases the same as classification ones. Two
data axes come out per request:

* :meth:`predict` returns a derived :class:`ClassificationResult`
  ("disease present at confidence X vs absent") so the existing reply
  prompt's ``cancer_status`` / ``clinical_action`` branching keeps
  working without a schema fork. The synthesized class is the highest-
  confidence box's label; ``top1_prob`` is that box's confidence.
* :meth:`detect_boxes` returns the full :class:`ObjectDetectionResult`
  — every kept box after NMS, normalized to ``[0, 1]`` in xyxy. The
  TUI / future detection-aware UI consumes this; the LLM-side reply
  prompt can also branch on box counts.

Wire layout and assumptions
---------------------------

* Output tensor: ``(1, 84, num_anchors)`` (YOLOv8 default) or
  ``(1, num_anchors, 5+num_classes)`` (YOLOv5). The adapter sniffs
  which one based on which axis matches ``5 + len(manifest.labels)``,
  with the v8 layout (channels-first) tried first because that's the
  current default. A model whose output shape matches neither convention
  raises at load time, not mid-request.
* Box decoding: cxcywh → xyxy, scaled back from the letterboxed
  network input to the original image's pixel space, then normalized
  by image width/height. Letterbox padding offsets are computed from
  the same ``_letterbox`` helper used at preprocess time so a YOLO
  trained at 640×640 still works on arbitrary aspect-ratio inputs
  without distortion.
* NMS: pure-numpy implementation keyed on confidence + IoU. Cheap for
  the box counts we actually expect (<= 300 candidates pre-NMS) and
  keeps the adapter free of an additional torch dep at runtime.

Subclasses can override the constants at the top of the file to swap
the input size, confidence floor, or NMS IoU threshold without
forking the whole adapter.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

from claritymed.core.vision.schemas import (
    ClassificationResult,
    ConfidenceTier,
    DetectionBox,
    InputQuality,
    Manifest,
    ModelSpec,
    ObjectDetectionResult,
    QualityCheck,
    SegmentationResult,
    round_sig,
)
from claritymed.servers.vision.adapters.torch_adapter import (
    _LOW_TIER_CEILING,
    _MEDIUM_TIER_CEILING,
    _MIN_RESOLUTION,
)
from claritymed.servers.vision.loader import register_adapter

logger = logging.getLogger(__name__)

# Letterbox target side. 640 is the YOLOv5/v8/v11 default. Square inputs
# match the training-time aug pipeline so detection accuracy isn't
# silently degraded by a different resize policy.
_DEFAULT_INPUT_SIZE = 640

# Confidence floor for keeping a box pre-NMS. 0.25 is the
# ``ultralytics`` default for ``predict()`` — high enough that obvious
# noise is gone, low enough that borderline-confidence findings still
# bubble up so the clinical_action override can route them to
# ``inconclusive_review``.
_DEFAULT_CONF_THRESH = 0.25

# IoU threshold for NMS. 0.45 is the ultralytics default. Lower values
# keep more overlapping boxes (good for densely packed findings, bad
# for noise); higher values merge more aggressively.
_DEFAULT_IOU_THRESH = 0.45

# Maximum boxes kept after NMS. 300 mirrors the ultralytics default.
# A real disease detector almost never returns this many; the cap is
# defensive against an exported model with a degenerate output that
# would otherwise stream thousands of low-confidence boxes through the
# audit log.
_DEFAULT_MAX_DET = 300


class YoloAdapter:
    """YOLO object-detection adapter satisfying :class:`DiseaseVisionModel`.

    Constructed by the loader's framework registry. Subclasses may
    override the class-level confidence / IoU / input-size constants
    to tune for a specific deployment without re-implementing
    preprocess / decode / NMS.
    """

    conf_thresh: float = _DEFAULT_CONF_THRESH
    iou_thresh: float = _DEFAULT_IOU_THRESH
    max_detections: int = _DEFAULT_MAX_DET
    input_size: int = _DEFAULT_INPUT_SIZE

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

        import onnxruntime as ort  # noqa: PLC0415

        providers = _providers_for(device)
        self._session = ort.InferenceSession(str(weights_path), providers=providers)

        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise RuntimeError(
                f"YoloAdapter expects exactly one input, got {len(inputs)} for "
                f"model_id={spec.id!r}; multi-input YOLO exports need a custom adapter."
            )
        self._input_name = inputs[0].name

        # Per-call image metadata stashed on the instance between
        # preprocess() and detect_boxes() so the decode step knows the
        # letterbox scale + padding. Keeping it on ``self`` (not threading
        # it through every method) matches the existing adapter pattern
        # where preprocess returns just the model-ready ndarray.
        self._last_letterbox: _LetterboxMeta | None = None
        # Cached final output of the last predict() pass — synthesized
        # from the same forward run that detect_boxes() consumes, so we
        # don't run the session twice.
        self._last_boxes: ObjectDetectionResult | None = None
        logger.info(
            "YoloAdapter loaded model_id=%s providers=%s labels=%d input_size=%d",
            spec.id,
            self._session.get_providers(),
            len(self._labels),
            self.input_size,
        )

    # --- DiseaseVisionModel Protocol --------------------------------------

    def preprocess(self, image: Any) -> Any:
        """Decode → letterbox → normalize → NCHW float32 batched array.

        Letterboxing (resize-with-aspect + pad) is the YOLO convention
        — squashing to ``input_size × input_size`` would distort boxes
        on non-square inputs. The padding metadata is stashed on
        ``self._last_letterbox`` so the decode step can undo it.
        """
        from PIL import Image  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415

        if isinstance(image, bytes):
            pil = Image.open(io.BytesIO(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            pil = image.convert("RGB")
        else:
            raise TypeError(
                f"YoloAdapter.preprocess expected bytes or PIL.Image, "
                f"got {type(image).__name__}"
            )

        orig_w, orig_h = pil.size
        padded, meta = _letterbox(pil, self.input_size)
        self._last_letterbox = meta._replace(orig_w=orig_w, orig_h=orig_h)
        arr = np.asarray(padded, dtype=np.float32) / 255.0  # HWC, [0, 1]
        arr = arr.transpose(2, 0, 1)  # HWC → CHW
        return arr[np.newaxis, ...]  # add batch dim

    def predict(self, x: Any) -> ClassificationResult:
        """Run detection + synthesize a presence/absence classification.

        The synthesized classification matches the existing reply-prompt
        contract: ``top1`` is the highest-confidence detected label,
        ``top1_prob`` is that box's confidence, ``probabilities`` is a
        per-label max over kept boxes (zero for labels with no box). A
        manifest with zero kept detections still emits a valid
        classification — ``top1`` falls back to ``labels[0]`` with
        ``top1_prob=0.0``, which the inference layer's KTD-V10 override
        keys on to produce ``clinical_action='inconclusive_review'``.
        """
        import numpy as np  # noqa: PLC0415

        if not isinstance(x, np.ndarray):
            raise TypeError(
                f"YoloAdapter.predict expected numpy.ndarray, got {type(x).__name__}"
            )
        raw = self._session.run(None, {self._input_name: x})[0]
        boxes = self._decode_and_nms(np.asarray(raw, dtype=np.float32))
        self._last_boxes = boxes

        per_label_max: dict[str, float] = {label: 0.0 for label in self._labels}
        for box in boxes.boxes:
            if box.confidence > per_label_max[box.label]:
                per_label_max[box.label] = box.confidence

        probs_list = [round_sig(per_label_max[label]) for label in self._labels]
        if any(p > 0 for p in probs_list):
            top1_idx = int(max(range(len(probs_list)), key=lambda i: probs_list[i]))
        else:
            top1_idx = 0
        top1_prob = probs_list[top1_idx]
        return ClassificationResult(
            labels=list(self._labels),
            probabilities=probs_list,
            top1=self._labels[top1_idx],
            top1_prob=top1_prob,
            confidence_tier=_confidence_tier(top1_prob),
        )

    def calibrate(self, raw: Any) -> dict[str, float]:
        """Identity pass-through, matching the classifier adapters' v1 contract."""
        if isinstance(raw, ClassificationResult):
            return dict(zip(raw.labels, raw.probabilities, strict=True))
        if isinstance(raw, dict):
            return {k: float(v) for k, v in raw.items()}
        raise TypeError(
            f"YoloAdapter.calibrate expected ClassificationResult or dict, "
            f"got {type(raw).__name__}"
        )

    def segment(self, x: Any) -> SegmentationResult | None:
        """YOLO-seg variants would override this; the detect-only flavor stubs to None."""
        return None

    def detect_boxes(self, x: Any) -> ObjectDetectionResult | None:
        """Return the cached boxes from the last :meth:`predict` call.

        Re-running the session here would double the wall-clock; the
        inference loop calls ``predict`` first, so by the time
        ``detect_boxes`` runs we already have the decoded boxes
        attached as ``self._last_boxes``. Callers invoking
        ``detect_boxes`` without a preceding ``predict`` get a fresh
        forward pass — useful for tests and detection-only callers
        that don't need the synthesized classification.
        """
        import numpy as np  # noqa: PLC0415

        if self._last_boxes is not None:
            cached = self._last_boxes
            self._last_boxes = None
            return cached
        if not isinstance(x, np.ndarray):
            raise TypeError(
                f"YoloAdapter.detect_boxes expected numpy.ndarray (or run after "
                f"predict), got {type(x).__name__}"
            )
        raw = self._session.run(None, {self._input_name: x})[0]
        return self._decode_and_nms(np.asarray(raw, dtype=np.float32))

    def quality_gate(self, image: Any) -> InputQuality:
        """Minimum-resolution check mirrored from the classifier adapters."""
        from PIL import Image  # noqa: PLC0415

        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError(
                f"YoloAdapter.quality_gate expected bytes or PIL.Image, "
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

    # --- decode + NMS -----------------------------------------------------

    def _decode_and_nms(self, raw: Any) -> ObjectDetectionResult:
        """YOLO raw tensor → :class:`ObjectDetectionResult` of kept boxes.

        Handles both layouts (channels-first v8 vs flat v5) by sniffing
        which axis matches ``5 + num_classes`` (v5) or ``4 + num_classes``
        (v8 without explicit objectness). The decode produces xyxy in
        network coords, undoes letterbox padding/scale to image-pixel
        coords, and finally normalizes by image width/height for the
        wire schema.
        """
        import numpy as np  # noqa: PLC0415

        if self._last_letterbox is None:
            raise RuntimeError(
                "YoloAdapter._decode_and_nms requires a prior preprocess() call "
                "so the letterbox metadata is known"
            )
        meta = self._last_letterbox
        nc = len(self._labels)

        # Sniff layout. Most common YOLOv8 export: (1, 4+nc, anchors).
        # YOLOv5: (1, anchors, 5+nc). We support both.
        squeezed = np.squeeze(raw, axis=0)
        if squeezed.shape[0] == 4 + nc:
            # v8 channels-first: transpose to (anchors, 4+nc), no objectness.
            preds = squeezed.transpose(1, 0)
            objectness = None
            class_scores = preds[:, 4 : 4 + nc]
            boxes_cxcywh = preds[:, :4]
        elif squeezed.shape[-1] == 5 + nc:
            # v5 flat: (anchors, 5+nc), with objectness at index 4.
            preds = squeezed
            objectness = preds[:, 4]
            class_scores = preds[:, 5 : 5 + nc]
            boxes_cxcywh = preds[:, :4]
        else:
            raise RuntimeError(
                f"YoloAdapter could not match output shape {squeezed.shape} to "
                f"either (4+{nc}, anchors) or (anchors, 5+{nc}); check the export."
            )

        # Per-anchor class confidence = (objectness ×) class probability.
        if objectness is not None:
            confidences = class_scores * objectness[:, np.newaxis]
        else:
            confidences = class_scores

        # Best class per anchor + its confidence.
        class_ids = np.argmax(confidences, axis=1)
        best_conf = confidences[np.arange(confidences.shape[0]), class_ids]

        keep_mask = best_conf >= self.conf_thresh
        if not keep_mask.any():
            return ObjectDetectionResult(boxes=[])

        boxes_xyxy_net = _cxcywh_to_xyxy(boxes_cxcywh[keep_mask])
        boxes_xyxy_orig = _undo_letterbox(boxes_xyxy_net, meta)
        kept_conf = best_conf[keep_mask]
        kept_class_ids = class_ids[keep_mask]

        nms_indices = _nms(boxes_xyxy_orig, kept_conf, self.iou_thresh)
        nms_indices = nms_indices[: self.max_detections]

        result_boxes: list[DetectionBox] = []
        for idx in nms_indices:
            x1, y1, x2, y2 = boxes_xyxy_orig[idx]
            x1_n = float(np.clip(x1 / meta.orig_w, 0.0, 1.0))
            y1_n = float(np.clip(y1 / meta.orig_h, 0.0, 1.0))
            x2_n = float(np.clip(x2 / meta.orig_w, 0.0, 1.0))
            y2_n = float(np.clip(y2 / meta.orig_h, 0.0, 1.0))
            if x2_n <= x1_n or y2_n <= y1_n:
                # Degenerate box after clip — drop. Better than emitting
                # a zero-area box that the schema would reject anyway.
                continue
            result_boxes.append(
                DetectionBox(
                    label=self._labels[int(kept_class_ids[idx])],
                    confidence=round_sig(float(kept_conf[idx])),
                    x1=round_sig(x1_n),
                    y1=round_sig(y1_n),
                    x2=round_sig(x2_n),
                    y2=round_sig(y2_n),
                )
            )
        return ObjectDetectionResult(boxes=result_boxes)


# --- helpers ---------------------------------------------------------------


def _providers_for(device: str) -> list[str]:
    """Pick the onnxruntime provider list for ``device`` with CPU fallback.

    Mirrors :func:`onnx_adapter._providers_for` — kept duplicated rather
    than imported to keep the YOLO adapter standalone (the classifier
    adapter could be removed without breaking this one).
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


class _LetterboxMeta:
    """Per-call letterbox metadata stashed between preprocess and decode.

    Implemented as a tiny dataclass-like type rather than a NamedTuple
    so we get a usable ``._replace`` without committing to NamedTuple's
    positional API everywhere it's read.
    """

    __slots__ = ("scale", "pad_x", "pad_y", "orig_w", "orig_h")

    def __init__(
        self,
        *,
        scale: float,
        pad_x: float,
        pad_y: float,
        orig_w: int = 0,
        orig_h: int = 0,
    ) -> None:
        self.scale = scale
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.orig_w = orig_w
        self.orig_h = orig_h

    def _replace(
        self,
        *,
        orig_w: int | None = None,
        orig_h: int | None = None,
    ) -> _LetterboxMeta:
        return _LetterboxMeta(
            scale=self.scale,
            pad_x=self.pad_x,
            pad_y=self.pad_y,
            orig_w=orig_w if orig_w is not None else self.orig_w,
            orig_h=orig_h if orig_h is not None else self.orig_h,
        )


def _letterbox(pil_image: Any, target: int) -> tuple[Any, _LetterboxMeta]:
    """Resize-with-aspect + pad to ``target × target``.

    Returns the padded image + metadata for the decode step. Pad fill
    is mid-gray (114, 114, 114) — the ultralytics default; using black
    would bias the network toward "background" in the padded strips.
    """
    from PIL import Image  # noqa: PLC0415

    w, h = pil_image.size
    scale = target / max(w, h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = pil_image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target, target), (114, 114, 114))
    pad_x = (target - new_w) / 2
    pad_y = (target - new_h) / 2
    canvas.paste(resized, (int(pad_x), int(pad_y)))
    return canvas, _LetterboxMeta(scale=scale, pad_x=pad_x, pad_y=pad_y)


def _cxcywh_to_xyxy(boxes_cxcywh: Any) -> Any:
    """Convert centered (cx, cy, w, h) boxes to corner (x1, y1, x2, y2)."""
    import numpy as np  # noqa: PLC0415

    cx, cy, w, h = (
        boxes_cxcywh[:, 0],
        boxes_cxcywh[:, 1],
        boxes_cxcywh[:, 2],
        boxes_cxcywh[:, 3],
    )
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def _undo_letterbox(boxes_xyxy_net: Any, meta: _LetterboxMeta) -> Any:
    """Map xyxy from letterboxed network coords back to original-image pixels."""

    out = boxes_xyxy_net.copy()
    out[:, 0] = (out[:, 0] - meta.pad_x) / meta.scale
    out[:, 1] = (out[:, 1] - meta.pad_y) / meta.scale
    out[:, 2] = (out[:, 2] - meta.pad_x) / meta.scale
    out[:, 3] = (out[:, 3] - meta.pad_y) / meta.scale
    return out


def _nms(boxes_xyxy: Any, scores: Any, iou_thresh: float) -> Any:
    """Pure-numpy NMS. Returns indices into ``boxes_xyxy`` of kept boxes.

    Greedy: pick the highest-scoring box, drop any remaining box whose
    IoU with it exceeds ``iou_thresh``, repeat. O(N²) worst case, but
    N is bounded by ``max_detections`` upstream so this is fine for
    the medical-imaging detection regime (single-digit findings per
    image, not crowd-scene throughput).
    """
    import numpy as np  # noqa: PLC0415

    if boxes_xyxy.shape[0] == 0:
        return np.array([], dtype=np.int64)
    x1 = boxes_xyxy[:, 0]
    y1 = boxes_xyxy[:, 1]
    x2 = boxes_xyxy[:, 2]
    y2 = boxes_xyxy[:, 3]
    areas = np.clip(x2 - x1, a_min=0, a_max=None) * np.clip(
        y2 - y1, a_min=0, a_max=None
    )
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, a_min=0, a_max=None) * np.clip(
            yy2 - yy1, a_min=0, a_max=None
        )
        union = areas[i] + areas[rest] - inter
        # Guard against degenerate zero-area boxes; treat them as
        # non-overlapping rather than producing a NaN that would crash
        # the comparison below.
        iou = np.where(union > 0, inter / union, 0.0)
        order = rest[iou <= iou_thresh]
    return np.array(keep, dtype=np.int64)


def _confidence_tier(top1_prob: float) -> ConfidenceTier:
    """Map post-NMS top1 confidence to the tier ladder.

    Identical cut points to the classifier adapters so KTD-V10's
    ``clinical_action`` override behaves the same regardless of model
    task (classification vs detection).
    """
    if top1_prob < _LOW_TIER_CEILING:
        return "low"
    if top1_prob < _MEDIUM_TIER_CEILING:
        return "medium"
    return "high"


def _factory(
    *, spec: ModelSpec, manifest: Manifest, weights_path: Path, device: str
) -> YoloAdapter:
    """Loader-registry factory."""
    return YoloAdapter(
        spec=spec, manifest=manifest, weights_path=weights_path, device=device
    )


register_adapter("ultralytics", _factory)


__all__ = ["YoloAdapter"]
