"""YOLO adapter end-to-end wire tests.

Mirrors :mod:`test_onnx_adapter` — exports a tiny torch module whose
output shape matches the YOLO convention, then runs it through the
adapter's full path (preprocess → predict → detect_boxes → calibrate
→ quality_gate → segment) so the wire contract is exercised against
an honest ONNX session, not mocks.

Two output layouts are covered:

* **YOLOv8** ``(1, 4+nc, anchors)`` — channels-first, no explicit
  objectness term.
* **YOLOv5** ``(1, anchors, 5+nc)`` — flat, objectness at index 4.

Decode + NMS + letterbox helpers also have their own targeted unit
tests so a regression there surfaces with a clearer failure than
"end-to-end test stopped producing boxes".
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from claritymed.core.vision.schemas import (
    ClassificationResult,
    DetectionBox,
    DiseaseVisionModel,
    InputQuality,
    Manifest,
    ObjectDetectionResult,
)
from claritymed.servers.vision.adapters.yolo_adapter import (
    YoloAdapter,
    _cxcywh_to_xyxy,
    _letterbox,
    _nms,
    _undo_letterbox,
)


# --- helpers ----------------------------------------------------------------


def _export_tiny_yolo_v8(
    path: Path,
    *,
    num_classes: int = 3,
    num_anchors: int = 32,
    high_conf_anchor_idx: int = 0,
    high_conf_class: int = 0,
    high_conf_box_cxcywh: tuple[float, float, float, float] = (
        320.0,
        320.0,
        80.0,
        80.0,
    ),
) -> None:
    """Export a YOLOv8-shaped fake detector to ``path`` as ONNX.

    The "model" ignores its input and emits a constant tensor of shape
    ``(1, 4 + num_classes, num_anchors)`` — one anchor carries a
    high-confidence prediction at ``high_conf_class`` with the given
    cxcywh box, the rest are sub-threshold noise. Lets the test assert
    on a specific top1 label + bbox without standing up a real YOLO.
    """
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    nc = num_classes
    per_anchor = 4 + nc
    output = torch.zeros(1, per_anchor, num_anchors, dtype=torch.float32)
    cx, cy, w, h = high_conf_box_cxcywh
    output[0, 0, high_conf_anchor_idx] = cx
    output[0, 1, high_conf_anchor_idx] = cy
    output[0, 2, high_conf_anchor_idx] = w
    output[0, 3, high_conf_anchor_idx] = h
    output[0, 4 + high_conf_class, high_conf_anchor_idx] = 0.92
    # Low-noise dummy fills for the other anchors so NMS / threshold
    # filtering have something to chew on.
    for i in range(1, num_anchors):
        output[0, 4, i] = 0.05

    class _ConstYoloV8(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("canned", output)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x is consumed only so the export records a real input
            # node; we ignore the values.
            return self.canned + (x.sum() * 0)

    dummy = torch.zeros(1, 3, 640, 640, dtype=torch.float32)
    torch.onnx.export(
        _ConstYoloV8().eval(),
        (dummy,),
        str(path),
        input_names=["images"],
        output_names=["output0"],
        opset_version=17,
        dynamo=False,
    )


def _export_tiny_yolo_v5(
    path: Path,
    *,
    num_classes: int = 3,
    num_anchors: int = 32,
    high_conf_anchor_idx: int = 0,
    high_conf_class: int = 1,
) -> None:
    """Export a YOLOv5-shaped fake detector — flat (1, anchors, 5+nc) layout."""
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    nc = num_classes
    per_anchor = 5 + nc
    output = torch.zeros(1, num_anchors, per_anchor, dtype=torch.float32)
    # cx, cy, w, h.
    output[0, high_conf_anchor_idx, 0] = 200.0
    output[0, high_conf_anchor_idx, 1] = 200.0
    output[0, high_conf_anchor_idx, 2] = 60.0
    output[0, high_conf_anchor_idx, 3] = 60.0
    # objectness.
    output[0, high_conf_anchor_idx, 4] = 0.95
    # class prob for the target class.
    output[0, high_conf_anchor_idx, 5 + high_conf_class] = 0.88

    class _ConstYoloV5(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("canned", output)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.canned + (x.sum() * 0)

    dummy = torch.zeros(1, 3, 640, 640, dtype=torch.float32)
    torch.onnx.export(
        _ConstYoloV5().eval(),
        (dummy,),
        str(path),
        input_names=["images"],
        output_names=["output0"],
        opset_version=17,
        dynamo=False,
    )


def _make_manifest(num_classes: int = 3) -> Manifest:
    """Build a YOLO manifest with the conftest's standard label set."""
    labels = ["benign", "malignant", "normal"][:num_classes]
    labels_meta = {
        "benign": {
            "description": "non-cancerous mass",
            "cancer_status": "benign",
            "clinical_action": "routine_followup",
        },
        "malignant": {
            "description": "cancerous lesion",
            "cancer_status": "malignant",
            "clinical_action": "urgent_specialist",
        },
        "normal": {
            "description": "no detectable lesion",
            "cancer_status": "normal",
            "clinical_action": "no_action",
        },
    }
    return Manifest(
        model_id="yolo_test_v1",
        model_version="v1.0.0",
        framework="ultralytics",
        accepted_modality="ultrasound",
        sha256_weights="b" * 64,
        task="detection",
        labels=labels,
        labels_meta={k: labels_meta[k] for k in labels},  # type: ignore[arg-type]
        cancer_class=True,
        cancer_status_mapping={k: labels_meta[k]["cancer_status"] for k in labels},  # type: ignore[arg-type]
        clinical_action_mapping={k: labels_meta[k]["clinical_action"] for k in labels},  # type: ignore[arg-type]
    )


def _make_spec():
    from claritymed.core.vision.schemas import ModelSpec  # noqa: PLC0415

    return ModelSpec(
        id="yolo_test_v1",
        disease_id="breast_cancer_ultrasound",
        server_id="local_default",
        framework="ultralytics",
        accepted_modality="ultrasound",
        weights_subpath="vision/breast_cancer_ultrasound/yolo_test_v1",
        manifest_sha256="b" * 64,
        expected_ms=800,
    )


def _png_bytes(*, size: tuple[int, int] = (640, 480), color: int = 128) -> bytes:
    from PIL import Image  # noqa: PLC0415

    img = Image.new("RGB", size, (color, color, color))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def tiny_yolo_v8(tmp_path: Path) -> Path:
    weights = tmp_path / "weights_v8.onnx"
    _export_tiny_yolo_v8(weights, num_classes=3)
    return weights


@pytest.fixture
def tiny_yolo_v5(tmp_path: Path) -> Path:
    weights = tmp_path / "weights_v5.onnx"
    _export_tiny_yolo_v5(weights, num_classes=3)
    return weights


@pytest.fixture
def adapter_v8(tiny_yolo_v8: Path) -> YoloAdapter:
    return YoloAdapter(
        spec=_make_spec(),
        manifest=_make_manifest(),
        weights_path=tiny_yolo_v8,
        device="cpu",
    )


@pytest.fixture
def adapter_v5(tiny_yolo_v5: Path) -> YoloAdapter:
    return YoloAdapter(
        spec=_make_spec(),
        manifest=_make_manifest(),
        weights_path=tiny_yolo_v5,
        device="cpu",
    )


# --- end-to-end tests -------------------------------------------------------


def test_satisfies_protocol(adapter_v8: YoloAdapter) -> None:
    """YOLO adapter must satisfy the runtime-checkable Protocol."""
    assert isinstance(adapter_v8, DiseaseVisionModel)


def test_v8_predict_returns_synthesized_classification(adapter_v8: YoloAdapter) -> None:
    """predict() synthesizes a classification keyed on the highest-conf box.

    The fake model planted a high-confidence box on class 0 ("benign")
    so top1 must be "benign" and top1_prob must reflect the planted
    score (≈0.92).
    """
    pre = adapter_v8.preprocess(_png_bytes())
    result = adapter_v8.predict(pre)
    assert isinstance(result, ClassificationResult)
    assert result.top1 == "benign"
    assert result.top1_prob > 0.85
    assert result.confidence_tier == "high"
    assert len(result.labels) == 3


def test_v8_detect_boxes_returns_one_box(adapter_v8: YoloAdapter) -> None:
    """detect_boxes() returns the cached result from the prior predict() call."""
    pre = adapter_v8.preprocess(_png_bytes())
    _ = adapter_v8.predict(pre)
    boxes = adapter_v8.detect_boxes(pre)
    assert isinstance(boxes, ObjectDetectionResult)
    assert len(boxes.boxes) == 1
    box = boxes.boxes[0]
    assert isinstance(box, DetectionBox)
    assert box.label == "benign"
    assert box.confidence > 0.85
    # Box was planted at (320, 320) center on a 640x640 letterbox.
    # On a 640x480 input, letterbox scale = 1.0 (640 fits exactly on the
    # wide axis), and the planted center maps near image-center.
    assert 0.0 <= box.x1 < box.x2 <= 1.0
    assert 0.0 <= box.y1 < box.y2 <= 1.0


def test_v8_detect_boxes_without_predict_runs_session(
    adapter_v8: YoloAdapter,
) -> None:
    """A direct detect_boxes() call (no predict first) still works.

    Useful for callers that only care about the boxes (the inference
    loop will still call predict first in production; this is a
    test-only path).
    """
    pre = adapter_v8.preprocess(_png_bytes())
    boxes = adapter_v8.detect_boxes(pre)
    assert isinstance(boxes, ObjectDetectionResult)
    assert len(boxes.boxes) == 1


def test_v5_layout_decodes(adapter_v5: YoloAdapter) -> None:
    """YOLOv5 ``(1, anchors, 5+nc)`` layout decodes through the same path.

    Planted: class 1 ("malignant"), high obj+class score, box at (200,200,60,60).
    """
    pre = adapter_v5.preprocess(_png_bytes())
    result = adapter_v5.predict(pre)
    assert isinstance(result, ClassificationResult)
    assert result.top1 == "malignant"
    # YOLOv5 confidence is obj * class_prob = 0.95 * 0.88 ≈ 0.836.
    assert result.top1_prob > 0.80


def test_predict_empty_detections_emits_inconclusive_classification(
    tmp_path: Path,
) -> None:
    """Zero kept boxes → top1=labels[0], top1_prob=0.0, tier=low (KTD-V10 trip)."""
    weights = tmp_path / "empty.onnx"
    # All anchors below 0.25 confidence => everything filtered.
    _export_tiny_yolo_v8(
        weights,
        num_classes=3,
        num_anchors=8,
        high_conf_class=0,
        high_conf_box_cxcywh=(320.0, 320.0, 80.0, 80.0),
    )
    # Patch the high-confidence anchor to be sub-threshold by lowering the
    # confidence floor for the test — keeps the test independent of the
    # canned model.
    adapter = YoloAdapter(
        spec=_make_spec(),
        manifest=_make_manifest(),
        weights_path=weights,
        device="cpu",
    )
    adapter.conf_thresh = 0.99  # nothing should clear this
    pre = adapter.preprocess(_png_bytes())
    result = adapter.predict(pre)
    assert result.top1_prob == 0.0
    assert result.confidence_tier == "low"
    assert result.top1 == "benign"  # falls back to labels[0]
    boxes = adapter.detect_boxes(pre)
    assert isinstance(boxes, ObjectDetectionResult)
    assert boxes.boxes == []


def test_preprocess_rejects_wrong_type(adapter_v8: YoloAdapter) -> None:
    with pytest.raises(TypeError, match="bytes or PIL.Image"):
        adapter_v8.preprocess(1234)  # type: ignore[arg-type]


def test_quality_gate_passes_on_normal_image(adapter_v8: YoloAdapter) -> None:
    quality = adapter_v8.quality_gate(_png_bytes(size=(128, 128)))
    assert isinstance(quality, InputQuality)
    assert quality.passed is True


def test_quality_gate_fails_on_tiny_image(adapter_v8: YoloAdapter) -> None:
    quality = adapter_v8.quality_gate(_png_bytes(size=(32, 32)))
    assert quality.passed is False
    assert quality.checks[0].score == 32.0


def test_segment_returns_none(adapter_v8: YoloAdapter) -> None:
    """Detection-only YOLO returns no mask; -seg variants would override."""
    pre = adapter_v8.preprocess(_png_bytes())
    assert adapter_v8.segment(pre) is None


def test_calibrate_passes_through(adapter_v8: YoloAdapter) -> None:
    pre = adapter_v8.preprocess(_png_bytes())
    result = adapter_v8.predict(pre)
    calibrated = adapter_v8.calibrate(result)
    assert set(calibrated.keys()) == set(result.labels)


def test_calibrate_rejects_wrong_type(adapter_v8: YoloAdapter) -> None:
    with pytest.raises(TypeError, match="ClassificationResult or dict"):
        adapter_v8.calibrate(0.5)  # type: ignore[arg-type]


# --- helper unit tests ------------------------------------------------------


def test_letterbox_preserves_aspect_ratio() -> None:
    from PIL import Image  # noqa: PLC0415

    pil = Image.new("RGB", (1280, 720), (0, 0, 0))
    padded, meta = _letterbox(pil, 640)
    assert padded.size == (640, 640)
    # 640/1280 = 0.5 so the padded image is 640x360 with 140-px vertical padding.
    assert meta.scale == 640 / 1280
    assert meta.pad_x == 0
    assert meta.pad_y == (640 - 360) / 2


def test_cxcywh_to_xyxy_roundtrip() -> None:
    import numpy as np  # noqa: PLC0415

    cxcywh = np.array([[100.0, 100.0, 40.0, 20.0]], dtype=np.float32)
    xyxy = _cxcywh_to_xyxy(cxcywh)
    assert xyxy[0].tolist() == [80.0, 90.0, 120.0, 110.0]


def test_undo_letterbox_inverts_scale_and_pad() -> None:
    import numpy as np  # noqa: PLC0415
    from claritymed.servers.vision.adapters.yolo_adapter import _LetterboxMeta

    # A 640-wide input that became 320 wide in the letterboxed image with
    # 160-px x padding on each side; vertical pad is zero.
    meta = _LetterboxMeta(scale=0.5, pad_x=160.0, pad_y=0.0, orig_w=640, orig_h=480)
    boxes_net = np.array([[200.0, 0.0, 440.0, 240.0]], dtype=np.float32)
    boxes_orig = _undo_letterbox(boxes_net, meta)
    # (200 - 160) / 0.5 = 80; (440 - 160) / 0.5 = 560.
    assert boxes_orig[0].tolist() == [80.0, 0.0, 560.0, 480.0]


def test_nms_drops_high_iou_overlap() -> None:
    """Two near-identical boxes — only the higher-scored survives."""
    import numpy as np  # noqa: PLC0415

    boxes = np.array(
        [
            [10.0, 10.0, 100.0, 100.0],
            [12.0, 12.0, 102.0, 102.0],  # ~95% IoU with the first
        ],
        dtype=np.float32,
    )
    scores = np.array([0.6, 0.9], dtype=np.float32)
    kept = _nms(boxes, scores, iou_thresh=0.5)
    assert kept.tolist() == [1]


def test_nms_keeps_disjoint_boxes() -> None:
    """Non-overlapping boxes both survive regardless of NMS."""
    import numpy as np  # noqa: PLC0415

    boxes = np.array(
        [
            [10.0, 10.0, 50.0, 50.0],
            [200.0, 200.0, 240.0, 240.0],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.6, 0.9], dtype=np.float32)
    kept = _nms(boxes, scores, iou_thresh=0.5)
    assert sorted(kept.tolist()) == [0, 1]


def test_nms_handles_empty_input() -> None:
    import numpy as np  # noqa: PLC0415

    boxes = np.zeros((0, 4), dtype=np.float32)
    scores = np.zeros((0,), dtype=np.float32)
    kept = _nms(boxes, scores, iou_thresh=0.5)
    assert kept.tolist() == []
