"""ONNX adapter end-to-end wire tests.

Generates a tiny real ONNX model via :func:`torch.onnx.export` so the
adapter's full path (session load → preprocess → predict → calibrate
→ quality_gate → segment) runs against an honest checkpoint. torch is
already a hard dependency, so the export is free in CI.

The synthetic model is a 2-layer conv classifier — small enough that
loading + inference both finish in single-digit milliseconds, but
real enough to exercise:

* NCHW input layout detection from session metadata
* Softmax-on-logits heuristic (model emits raw logits)
* Manifest label-count cross-check (3 classes, matching the
  conftest's default labels)
* Quality gate decoding raw bytes
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from claritymed.core.vision.schemas import (
    ClassificationResult,
    DiseaseVisionModel,
    InputQuality,
    Manifest,
)
from claritymed.servers.vision.adapters.onnx_adapter import (
    OnnxAdapter,
    _detect_layout_and_size,
    _maybe_softmax,
)


# --- helpers ----------------------------------------------------------------


def _export_tiny_classifier(path: Path, *, num_classes: int = 3) -> None:
    """Export a 2-conv + GAP + linear classifier to ``path`` as ONNX.

    Input shape baked in: (1, 3, 32, 32) — small enough for ms-level
    inference, big enough that the channel dim is unambiguous (NCHW).
    Random init means predictions are noise, but tests assert on the
    *shape* of the output (probabilities sum to 1, top1 is in the
    label set), not the *value*.
    """
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(3, 4, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(4, 4, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(4, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.body(x)

    model = _Tiny().eval()
    dummy = torch.zeros(1, 3, 32, 32, dtype=torch.float32)
    # ``dynamo=False`` keeps us on the legacy TorchScript exporter,
    # which doesn't require the optional ``onnxscript`` dependency.
    # The new dynamo-based path emits cleaner ONNX for ops introduced
    # post-2.0 but we don't need any of them for this 2-conv classifier.
    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes={"input": {0: "batch"}},
        dynamo=False,
    )


def _make_manifest(num_classes: int = 3) -> Manifest:
    """Build a minimal Manifest matching the default conftest labels.

    Labels and mappings match
    :data:`tests.servers.vision.conftest._DEFAULT_LABELS_META` so the
    adapter's ``manifest.labels`` axis lines up with what the
    ClassificationResult assertions expect.
    """
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
        model_id="onnx_test_v1",
        model_version="v1.0.0",
        framework="onnx",
        accepted_modality="ultrasound",
        sha256_weights="a" * 64,
        task="classification",
        labels=labels,
        labels_meta={k: labels_meta[k] for k in labels},  # type: ignore[arg-type]
        cancer_class=True,
        cancer_status_mapping={k: labels_meta[k]["cancer_status"] for k in labels},  # type: ignore[arg-type]
        clinical_action_mapping={k: labels_meta[k]["clinical_action"] for k in labels},  # type: ignore[arg-type]
    )


def _make_spec(num_classes: int = 3):
    from claritymed.core.vision.schemas import ModelSpec  # noqa: PLC0415

    return ModelSpec(
        id="onnx_test_v1",
        disease_id="breast_cancer_ultrasound",
        server_id="local_default",
        framework="onnx",
        accepted_modality="ultrasound",
        weights_subpath="vision/breast_cancer_ultrasound/onnx_test_v1",
        manifest_sha256="a" * 64,
        expected_ms=800,
    )


def _png_bytes(*, size: tuple[int, int] = (64, 64), color: int = 128) -> bytes:
    """Build a solid-color PNG so PIL decodes successfully."""
    from PIL import Image  # noqa: PLC0415

    img = Image.new("RGB", size, (color, color, color))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def tiny_onnx(tmp_path: Path) -> Path:
    weights = tmp_path / "weights.onnx"
    _export_tiny_classifier(weights, num_classes=3)
    return weights


@pytest.fixture
def adapter(tiny_onnx: Path) -> OnnxAdapter:
    return OnnxAdapter(
        spec=_make_spec(),
        manifest=_make_manifest(),
        weights_path=tiny_onnx,
        device="cpu",
    )


# --- tests ------------------------------------------------------------------


def test_satisfies_protocol(adapter: OnnxAdapter) -> None:
    """The adapter must satisfy the ``DiseaseVisionModel`` runtime check.

    ``DiseaseVisionModel`` is ``runtime_checkable``, so the loader's
    ``isinstance`` guard uses this exact path. Failing this means the
    server would reject the adapter at startup.
    """
    assert isinstance(adapter, DiseaseVisionModel)


def test_predict_returns_valid_classification_result(adapter: OnnxAdapter) -> None:
    """Probabilities sum to ~1.0 and top1 is a known label."""
    pre = adapter.preprocess(_png_bytes())
    result = adapter.predict(pre)
    assert isinstance(result, ClassificationResult)
    assert result.top1 in {"benign", "malignant", "normal"}
    assert len(result.probabilities) == 3
    assert abs(sum(result.probabilities) - 1.0) < 0.05  # rounding wiggle
    assert 0.0 <= result.top1_prob <= 1.0
    assert result.confidence_tier in {"low", "medium", "high"}


def test_preprocess_handles_bytes_and_pil(adapter: OnnxAdapter) -> None:
    """Both bytes and a pre-decoded PIL image flow through preprocess."""
    from PIL import Image  # noqa: PLC0415

    bytes_arr = adapter.preprocess(_png_bytes())
    pil_arr = adapter.preprocess(Image.new("RGB", (64, 64), (128, 128, 128)))
    assert bytes_arr.shape == pil_arr.shape
    # NCHW = (1, 3, H, W) for this model.
    assert bytes_arr.shape[0] == 1
    assert bytes_arr.shape[1] == 3


def test_preprocess_rejects_wrong_type(adapter: OnnxAdapter) -> None:
    with pytest.raises(TypeError, match="bytes or PIL.Image"):
        adapter.preprocess("not-an-image")  # type: ignore[arg-type]


def test_quality_gate_passes_on_normal_image(adapter: OnnxAdapter) -> None:
    quality = adapter.quality_gate(_png_bytes(size=(128, 128)))
    assert isinstance(quality, InputQuality)
    assert quality.passed is True
    assert quality.checks[0].name == "min_resolution"


def test_quality_gate_fails_on_tiny_image(adapter: OnnxAdapter) -> None:
    """Below the 64px min-side cut, quality_gate must fail.

    KTD-V10 keys ``clinical_action='inconclusive_review'`` off
    ``passed=False`` — regressing this would silently route bad inputs
    through normal classification.
    """
    quality = adapter.quality_gate(_png_bytes(size=(32, 32)))
    assert quality.passed is False
    assert quality.checks[0].score == 32.0


def test_segment_returns_none(adapter: OnnxAdapter) -> None:
    """v1 contract: segment is opt-out by default."""
    pre = adapter.preprocess(_png_bytes())
    assert adapter.segment(pre) is None


def test_calibrate_passes_through(adapter: OnnxAdapter) -> None:
    """calibrate is identity in v1 — round-trips the label/prob mapping."""
    pre = adapter.preprocess(_png_bytes())
    result = adapter.predict(pre)
    calibrated = adapter.calibrate(result)
    assert set(calibrated.keys()) == set(result.labels)
    assert all(0.0 <= v <= 1.0 for v in calibrated.values())


def test_calibrate_accepts_dict(adapter: OnnxAdapter) -> None:
    assert adapter.calibrate({"a": 0.5, "b": 0.5}) == {"a": 0.5, "b": 0.5}


def test_calibrate_rejects_wrong_type(adapter: OnnxAdapter) -> None:
    with pytest.raises(TypeError, match="ClassificationResult or dict"):
        adapter.calibrate(0.5)  # type: ignore[arg-type]


def test_label_count_mismatch_raises(tiny_onnx: Path) -> None:
    """A manifest with the wrong label count must fail at predict time.

    The check sits in predict (not __init__) because computing the
    output dim from session metadata alone is brittle — some ONNX
    exports declare dynamic output dims. predict catches the mismatch
    on the first real call instead of speculating.
    """
    bad_manifest = _make_manifest(num_classes=2)
    # Force a label-count mismatch: model has 3 outputs, manifest has 2.
    adapter = OnnxAdapter(
        spec=_make_spec(),
        manifest=bad_manifest,
        weights_path=tiny_onnx,
        device="cpu",
    )
    pre = adapter.preprocess(_png_bytes())
    with pytest.raises(RuntimeError, match="ONNX output size"):
        adapter.predict(pre)


# --- helper unit tests ------------------------------------------------------


def test_detect_layout_nchw_static() -> None:
    layout, hw = _detect_layout_and_size((1, 3, 224, 224))
    assert layout == "NCHW"
    assert hw == (224, 224)


def test_detect_layout_nhwc_static() -> None:
    layout, hw = _detect_layout_and_size((1, 224, 224, 3))
    assert layout == "NHWC"
    assert hw == (224, 224)


def test_detect_layout_dynamic_spatial_falls_back() -> None:
    """Dynamic H/W resolves to the forge default size, not zero."""
    layout, hw = _detect_layout_and_size(("batch", 3, "h", "w"))
    assert layout == "NCHW"
    assert hw == (256, 256)


def test_detect_layout_rejects_grayscale() -> None:
    """No channel-3 dim ⇒ raise. Better than silently mis-routing."""
    with pytest.raises(RuntimeError, match="channel dim"):
        _detect_layout_and_size((1, 1, 224, 224))


def test_detect_layout_rejects_3d() -> None:
    with pytest.raises(RuntimeError, match="4D input shape"):
        _detect_layout_and_size((3, 224, 224))


def test_maybe_softmax_on_logits() -> None:
    import numpy as np  # noqa: PLC0415

    probs = _maybe_softmax(np.array([2.0, -1.0, 0.5], dtype=np.float32))
    assert abs(float(probs.sum()) - 1.0) < 1e-5
    assert (probs >= 0).all()


def test_maybe_softmax_on_probabilities_leaves_unchanged() -> None:
    """A valid probability vector is left alone (sum normalized)."""
    import numpy as np  # noqa: PLC0415

    given = np.array([0.5, 0.3, 0.2], dtype=np.float32)
    out = _maybe_softmax(given)
    # Allowed: small rounding from normalization, but distribution shape preserved.
    assert abs(float(out.sum()) - 1.0) < 1e-5
    assert float(out[0]) > float(out[1]) > float(out[2])
