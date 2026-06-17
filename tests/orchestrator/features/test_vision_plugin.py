"""VisionFeature plugin — gate cascade + happy path + post_process.

Collaborators are mocked in-process: the tool body is exercised
directly with a ``SimpleNamespace`` deps satisfying the ``TurnState``
shape, the registry uses an in-memory ``MockTransport`` against the
canned vision-server fixtures, and ``SessionAttachments`` /
``BlobStore`` are seeded under a temp ``CLARITYMED_HOME``.

Coverage of the pydantic-ai shim itself lives in Unit 9 e2e — these
tests are wire-level checks against the deterministic gate cascade.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from claritymed.context import apply_context, reset_context
from claritymed.core.vision.registry import VisionRegistry
from claritymed.core.vision.schemas import (
    DiseaseSpec,
    ModelSpec,
    OcrReportConfig,
    ServerSpec,
    ToolConfig,
    VisionConfig,
)
from claritymed.orchestrator.features.vision_plugin import (
    TOOL_NAME,
    VisionFeature,
    _read_blob_bytes,
)
from claritymed.stores.blob_store import BlobStore
from claritymed.stores.session_attachments import SessionAttachments


@contextmanager
def _ctx(request_id: str, user_id: str, language: str):
    tokens = apply_context(request_id=request_id, user_id=user_id, language=language)
    try:
        yield
    finally:
        reset_context(tokens)


_USER_ID = "test"
_REQUEST_ID = "20260614000000ABCDEFAB"
_SESSION_ID = "20260614120000ABCDEFAB"


# --- fixtures --------------------------------------------------------------


def _vision_config(*, enabled: bool = True) -> VisionConfig:
    return VisionConfig(
        diseases=[
            DiseaseSpec(
                id="breast_cancer_ultrasound",
                enabled=enabled,
                primary_model_id="breast_busi_unet_v1",
                # flow is fallbacks-only; primary auto-prepended via effective_flow
                flow=[],
                cancer_class=True,
                intent_hints_i18n_key="vision.intent.breast_cancer_ultrasound",
            )
        ],
        servers=[
            ServerSpec(
                id="local_default",
                base_url="http://test.local",
                expected_ms=800,
            )
        ],
        models=[
            ModelSpec(
                id="breast_busi_unet_v1",
                disease_id="breast_cancer_ultrasound",
                server_id="local_default",
                framework="pytorch",
                accepted_modality="ultrasound",
                weights_subpath="vision/breast_cancer_ultrasound/breast_busi_unet_v1",
                manifest_sha256="a" * 64,
                expected_ms=800,
            )
        ],
        tool=ToolConfig(confirm_before_run=False),  # auto-confirm for tests
        ocr_report=OcrReportConfig(markers={"en": ["findings"], "zh": ["所见"]}),
    )


def _canned_detect_response(sha: str) -> dict:
    """Mirror servers/vision/wire::DetectResponse shape with high-conf result."""
    return {
        "request_id": _REQUEST_ID,
        "disease_id": "breast_cancer_ultrasound",
        "model_id": "breast_busi_unet_v1",
        "model_version": "v1.0.0",
        "elapsed_ms": 120,
        "input_quality": {"passed": True, "checks": []},
        "classification": {
            "labels": ["benign", "malignant", "normal"],
            "probabilities": [0.12, 0.81, 0.07],
            "top1": "malignant",
            "top1_prob": 0.81,
            "confidence_tier": "high",
        },
        "cancer_status": "malignant",
        "clinical_action": "urgent_specialist",
        "segmentation": None,
        "saliency_b64": None,
        "labels_meta": {
            "benign": {
                "description": "non-cancerous",
                "cancer_status": "benign",
                "clinical_action": "routine_followup",
            },
            "malignant": {
                "description": "suspicious for cancer",
                "cancer_status": "malignant",
                "clinical_action": "urgent_specialist",
            },
            "normal": {
                "description": "no lesion",
                "cancer_status": "normal",
                "clinical_action": "no_action",
            },
        },
        "warnings": [],
        "model_card_url": None,
    }


def _canned_catalog() -> dict:
    return {
        "models": [
            {
                "disease_id": "breast_cancer_ultrasound",
                "model_id": "breast_busi_unet_v1",
                "model_version": "v1.0.0",
                "task": "classification+segmentation",
                "labels": ["benign", "malignant", "normal"],
                "cancer_class": True,
                "accepted_modality": "ultrasound",
                "manifest_sha": "a" * 64,
                "expected_ms": 800,
                "supports_saliency": False,
                "supports_tta": True,
            }
        ]
    }


def _make_registry(handler) -> VisionRegistry:
    cfg = _vision_config()
    transport = httpx.MockTransport(handler)
    return VisionRegistry(cfg, transport=transport)


def _seed_attachment(home: Path, image_bytes: bytes) -> str:
    """Drop the image into the blob store + tag it for the session.

    Returns the sha256 of the seeded image.
    """
    blob_store = BlobStore(_USER_ID)
    sha = blob_store.store(image_bytes, "png")
    SessionAttachments(_USER_ID, _SESSION_ID).add(
        sha256=sha,
        filename="test.png",
        mime="image/png",
        size=len(image_bytes),
    )
    return sha


def _write_ocr_metadata(sha: str, **fields) -> None:
    BlobStore(_USER_ID).write_ocr_result(
        sha,
        status="done",
        kind="ocr",
        ext="png",
        provider="test",
        chain_tried=["test"],
        reason=None,
        text=fields.pop("ocr_text", "test ocr text\n" * 30),
        original_filename="test.png",
        **fields,
    )


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CLARITYMED_HOME", str(tmp_path))
    # Force module-level path caches to re-resolve through env var.
    from claritymed import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "DATA_DIR", tmp_path / "data")
    return tmp_path


# --- tests -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_name_and_mode():
    """Plugin advertises the stable tool name + tool mode."""
    cfg = _vision_config()
    feature = VisionFeature(
        config=cfg,
        registry=VisionRegistry(cfg),
        get_session_id=lambda: None,
    )
    assert TOOL_NAME == "detect_disease_from_image"
    assert feature.name == "vision"
    assert feature.mode == "tool"


@pytest.mark.asyncio
async def test_missing_attachment_returns_structured_error(home):
    """An unresolved image_sha returns ``kind=missing_attachment``."""
    cfg = _vision_config()
    feature = VisionFeature(
        config=cfg,
        registry=VisionRegistry(cfg),
        get_session_id=lambda: _SESSION_ID,
    )
    deps = SimpleNamespace(user_id=_USER_ID, language="en", prompt_channel=None)
    ctx = SimpleNamespace(deps=deps)
    with _ctx(request_id=_REQUEST_ID, user_id=_USER_ID, language="en"):
        result = await feature._detect(
            ctx, disease_id="breast_cancer_ultrasound", image_sha="f" * 64
        )
    assert result["kind"] == "missing_attachment"


@pytest.mark.asyncio
async def test_ocr_override_short_circuits(home):
    """ocr_has_report=true → OcrOverrideResult, no server traffic."""
    sha = _seed_attachment(home, b"image-bytes")
    _write_ocr_metadata(
        sha,
        modality="ultrasound",
        is_medical=True,
        ocr_has_report=True,
    )

    cfg = _vision_config()

    # Any HTTP would blow up the test — none should fire.
    def _no_traffic(req):
        raise AssertionError(f"unexpected HTTP {req.method} {req.url}")

    transport = httpx.MockTransport(_no_traffic)
    registry = VisionRegistry(cfg, transport=transport)
    feature = VisionFeature(
        config=cfg, registry=registry, get_session_id=lambda: _SESSION_ID
    )
    deps = SimpleNamespace(user_id=_USER_ID, language="en", prompt_channel=None)
    ctx = SimpleNamespace(deps=deps)
    with _ctx(request_id=_REQUEST_ID, user_id=_USER_ID, language="en"):
        result = await feature._detect(
            ctx, disease_id="breast_cancer_ultrasound", image_sha=sha
        )
    assert result["kind"] == "ocr_override"


@pytest.mark.asyncio
async def test_not_medical_short_circuits(home):
    """is_medical=False → NotMedicalResult, no server traffic."""
    sha = _seed_attachment(home, b"pet-photo-bytes")
    _write_ocr_metadata(
        sha,
        modality="photo",
        is_medical=False,
        ocr_has_report=False,
    )

    cfg = _vision_config()

    def _no_traffic(req):
        raise AssertionError(f"unexpected HTTP {req.method} {req.url}")

    transport = httpx.MockTransport(_no_traffic)
    registry = VisionRegistry(cfg, transport=transport)
    feature = VisionFeature(
        config=cfg, registry=registry, get_session_id=lambda: _SESSION_ID
    )
    deps = SimpleNamespace(user_id=_USER_ID, language="en", prompt_channel=None)
    ctx = SimpleNamespace(deps=deps)
    with _ctx(request_id=_REQUEST_ID, user_id=_USER_ID, language="en"):
        result = await feature._detect(
            ctx, disease_id="breast_cancer_ultrasound", image_sha=sha
        )
    assert result["kind"] == "not_medical"


@pytest.mark.asyncio
async def test_modality_mismatch_short_circuits(home):
    """Wrong modality → ModalityMismatchResult before HTTP fires."""
    sha = _seed_attachment(home, b"ct-bytes")
    _write_ocr_metadata(
        sha,
        modality="ct",  # model accepts ultrasound
        is_medical=True,
        ocr_has_report=False,
    )

    cfg = _vision_config()

    def _no_traffic(req):
        raise AssertionError(f"unexpected HTTP {req.method} {req.url}")

    transport = httpx.MockTransport(_no_traffic)
    registry = VisionRegistry(cfg, transport=transport)
    feature = VisionFeature(
        config=cfg, registry=registry, get_session_id=lambda: _SESSION_ID
    )
    deps = SimpleNamespace(user_id=_USER_ID, language="en", prompt_channel=None)
    ctx = SimpleNamespace(deps=deps)
    with _ctx(request_id=_REQUEST_ID, user_id=_USER_ID, language="en"):
        result = await feature._detect(
            ctx, disease_id="breast_cancer_ultrasound", image_sha=sha
        )
    assert result["kind"] == "modality_mismatch"
    assert result["model_accepts"] == "ultrasound"
    assert result["image_modality"] == "ct"


@pytest.mark.asyncio
async def test_same_modality_wrong_anatomy_is_not_caught(home):
    """KNOWN GAP canary — same-modality wrong-anatomy passes through.

    The four gates only screen ``not_medical`` → ``ocr_override`` →
    ``modality_mismatch``. They do not check whether the image's
    anatomy actually matches the claimed ``disease_id``. So an
    arbitrary ultrasound image (thyroid, abdominal, cardiac, …)
    submitted under ``breast_cancer_ultrasound`` clears all gates and
    the breast model runs on it. The model returns its canned
    verdict; nothing downstream notices the mismatch.

    This test fails (correctly) the day anatomy gating ships. When
    that happens, change the expected ``kind`` here to match the new
    short-circuit (e.g. ``anatomy_mismatch``).
    """
    # Bytes are arbitrary — in practice this would be a non-breast
    # ultrasound (thyroid, abdominal, etc). The system can't tell.
    sha = _seed_attachment(home, b"non-breast-ultrasound-bytes")
    _write_ocr_metadata(
        sha,
        modality="ultrasound",  # matches breast model's accepted_modality
        is_medical=True,
        ocr_has_report=False,
    )

    cfg = _vision_config()
    detect_calls: list[str] = []

    def _handler(req):
        if req.url.path == "/v1/detect":
            detect_calls.append(req.url.path)
            return httpx.Response(200, json=_canned_detect_response(sha))
        if req.url.path == "/v1/catalog":
            return httpx.Response(200, json=_canned_catalog())
        return httpx.Response(404)

    registry = VisionRegistry(cfg, transport=httpx.MockTransport(_handler))
    feature = VisionFeature(
        config=cfg, registry=registry, get_session_id=lambda: _SESSION_ID
    )
    deps = SimpleNamespace(user_id=_USER_ID, language="en", prompt_channel=None)
    ctx = SimpleNamespace(deps=deps)
    with _ctx(request_id=_REQUEST_ID, user_id=_USER_ID, language="en"):
        result = await feature._detect(
            ctx, disease_id="breast_cancer_ultrasound", image_sha=sha
        )

    assert len(detect_calls) == 1, (
        "vision server was NOT called — that would mean a new anatomy gate "
        "fired. Update this canary's expected kind to match."
    )
    assert result["kind"] == "detection"


@pytest.mark.asyncio
async def test_happy_path_returns_detection_payload(home):
    """Modality + is_medical + ocr clean → tool reaches server + transforms result."""
    image_bytes = b"\x89PNG\r\n\x1a\n" + b"ultrasound-image-bytes"
    sha = _seed_attachment(home, image_bytes)
    _write_ocr_metadata(
        sha,
        modality="ultrasound",
        is_medical=True,
        ocr_has_report=False,
    )

    cfg = _vision_config()

    def _handler(req):
        if req.url.path == "/v1/detect":
            return httpx.Response(200, json=_canned_detect_response(sha))
        if req.url.path == "/v1/catalog":
            return httpx.Response(200, json=_canned_catalog())
        return httpx.Response(404)

    registry = VisionRegistry(cfg, transport=httpx.MockTransport(_handler))
    feature = VisionFeature(
        config=cfg, registry=registry, get_session_id=lambda: _SESSION_ID
    )
    deps = SimpleNamespace(user_id=_USER_ID, language="en", prompt_channel=None)
    ctx = SimpleNamespace(deps=deps)
    with _ctx(request_id=_REQUEST_ID, user_id=_USER_ID, language="en"):
        result = await feature._detect(
            ctx, disease_id="breast_cancer_ultrasound", image_sha=sha
        )
    assert result["kind"] == "detection"
    assert result["top1"] == "malignant"  # localized via i18n; EN keeps the lowercase
    assert result["clinical_action"] == "urgent_specialist"
    assert result["cancer_status"] == "malignant"
    assert result["confidence_tier"] == "high"
    assert result["model_version"] == "v1.0.0"


@pytest.mark.asyncio
async def test_user_declined_confirm_modal_returns_user_declined(home):
    """confirm_before_run=True + user picks 'No' → UserDeclinedResult."""
    from claritymed.core.interaction.schemas import AskUserQuestionResult

    image_bytes = b"\x89PNG\r\n\x1a\nultrasound-bytes"
    sha = _seed_attachment(home, image_bytes)
    _write_ocr_metadata(
        sha,
        modality="ultrasound",
        is_medical=True,
        ocr_has_report=False,
    )

    cfg_base = _vision_config()
    cfg = cfg_base.model_copy(
        update={"tool": cfg_base.tool.model_copy(update={"confirm_before_run": True})}
    )

    def _no_traffic(req):
        raise AssertionError(f"unexpected HTTP {req.method} {req.url}")

    registry = VisionRegistry(cfg, transport=httpx.MockTransport(_no_traffic))
    feature = VisionFeature(
        config=cfg, registry=registry, get_session_id=lambda: _SESSION_ID
    )

    class _NoChannel:
        async def ask(self, payload):
            # Pick the "no" label by answer text.
            from claritymed.core.i18n.loader import t

            label = t("vision.confirm.breast_cancer_ultrasound.no_label", lang="en")
            q = payload.questions[0]
            return AskUserQuestionResult(answers={q.question: label})

    deps = SimpleNamespace(user_id=_USER_ID, language="en", prompt_channel=_NoChannel())
    ctx = SimpleNamespace(deps=deps)
    with _ctx(request_id=_REQUEST_ID, user_id=_USER_ID, language="en"):
        result = await feature._detect(
            ctx, disease_id="breast_cancer_ultrasound", image_sha=sha
        )
    assert result["kind"] == "user_declined"


@pytest.mark.asyncio
async def test_post_process_emits_audit_on_missing_keyword(home, monkeypatch):
    """urgent_specialist reply without specialist phrase → audit, text unchanged."""
    cfg = _vision_config()
    feature = VisionFeature(
        config=cfg, registry=VisionRegistry(cfg), get_session_id=lambda: None
    )
    # Seed the stash as if a tool call just landed.
    feature._stash[_REQUEST_ID] = {
        "clinical_action": "urgent_specialist",
        "disease_id": "breast_cancer_ultrasound",
    }
    captured = []

    def _fake_audit(kind, payload=None):
        captured.append((kind, payload))
        return None

    monkeypatch.setattr(
        "claritymed.orchestrator.features.vision_plugin.audit_event", _fake_audit
    )

    with _ctx(request_id=_REQUEST_ID, user_id=_USER_ID, language="en"):
        out = await feature.post_process(
            text="Looks like the model flagged something on this image.",
            tool_result={"kind": "detection"},
        )
    assert out == "Looks like the model flagged something on this image."
    assert any(k == "vision.specialist_keywords.missing" for k, _ in captured)


def test_read_blob_bytes_finds_content_file(home):
    """``_read_blob_bytes`` returns the raw bytes regardless of extension."""
    sha = _seed_attachment(home, b"hello-world")
    assert _read_blob_bytes(_USER_ID, sha) == b"hello-world"


# --- validator error-path tests -------------------------------------------


def test_validate_vision_prompts_raises_when_prompt_missing():
    """_validate_vision_prompts raises RuntimeError when any prompt is absent."""
    from unittest.mock import MagicMock

    from claritymed.orchestrator.features.vision_plugin import _validate_vision_prompts

    mock_registry = MagicMock()
    mock_registry.get.side_effect = Exception("no such prompt")
    with pytest.raises(RuntimeError, match="Missing vision prompts"):
        _validate_vision_prompts(mock_registry)


def test_validate_specialist_keywords_raises_on_blank_entry(monkeypatch):
    """A blank entry in the specialist keywords i18n list raises RuntimeError."""
    from claritymed.orchestrator.features import vision_plugin as _vp

    monkeypatch.setattr(
        _vp, "t_list", lambda key, lang: [""] if "specialist" in key else ["specialist"]
    )
    with pytest.raises(RuntimeError, match="blank entry"):
        _vp._validate_specialist_keywords()


def test_validate_specialist_keywords_raises_on_missing_entry(monkeypatch):
    """Missing entries (empty list) in specialist keywords raises RuntimeError."""
    from claritymed.orchestrator.features import vision_plugin as _vp

    monkeypatch.setattr(_vp, "t_list", lambda key, lang: [])
    with pytest.raises(RuntimeError, match="Missing vision specialist_keywords"):
        _vp._validate_specialist_keywords()


def test_build_tool_description_template_without_placeholder():
    """Template without {covered_diseases} is returned as-is."""
    from unittest.mock import MagicMock

    cfg = _vision_config()
    mock_registry = MagicMock()
    mock_registry.get.return_value = "Fixed description without placeholder."
    feature = VisionFeature.__new__(VisionFeature)
    feature._config = cfg
    feature._registry = VisionRegistry(cfg)
    feature._prompt_registry = mock_registry
    feature._stash = {}
    result = feature._build_tool_description()
    assert result == "Fixed description without placeholder."


def test_build_tool_description_no_diseases_enabled():
    """No enabled diseases produces the no-op placeholder in the description."""
    from unittest.mock import MagicMock

    cfg = _vision_config(enabled=False)
    mock_registry = MagicMock()
    mock_registry.get.return_value = "Diseases:\n{covered_diseases}"
    feature = VisionFeature.__new__(VisionFeature)
    feature._config = cfg
    feature._registry = VisionRegistry(cfg)
    feature._prompt_registry = mock_registry
    feature._stash = {}
    result = feature._build_tool_description()
    assert result is not None
    assert "no diseases enabled" in result
