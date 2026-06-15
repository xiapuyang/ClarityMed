"""In-process vision pipeline — gate cascade + registry + result transform.

What this covers above the wire-level unit tests:

* The full attachment → ``ocr.json`` → vision-tag → plugin gate cascade
  threading. Unit tests stub `meta` directly; here it is read from the
  same ``BlobStore.read_ocr_metadata`` the production pipeline uses.
* ``VisionRegistry.bootstrap`` against an in-process catalog (via
  ``httpx.MockTransport``) — proves the boot-time cross-check works.
* ``to_llm_payload`` round-trip from a synthetic ``RawDetection`` to a
  payload the LLM-side reply prompt could consume.

We deliberately do NOT spin up subprocess servers — Unit 9 e2e does
that. This integration layer fills the gap between "mock everything"
and "boot both servers + real LLM" so a regression in the wiring
shows up in CI without a real provider key.
"""

from __future__ import annotations

import logging
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
from claritymed.orchestrator.features.vision_plugin import VisionFeature
from claritymed.stores.blob_store import BlobStore
from claritymed.stores.session_attachments import SessionAttachments

logger = logging.getLogger(__name__)

USER_ID = "test"
SESSION_ID = "20260614integration00000"


def _vision_config() -> VisionConfig:
    return VisionConfig(
        diseases=[
            DiseaseSpec(
                id="breast_cancer_ultrasound",
                enabled=True,
                primary_model_id="breast_busi_unet_v1",
                flow=["breast_busi_unet_v1"],
                cancer_class=True,
                intent_hints_i18n_key="vision.intent.breast_cancer_ultrasound",
            )
        ],
        servers=[
            ServerSpec(id="local_default", base_url="http://test", expected_ms=800)
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
        tool=ToolConfig(confirm_before_run=False),
        ocr_report=OcrReportConfig(markers={"en": ["findings"], "zh": ["所见"]}),
    )


def _catalog() -> dict:
    return {
        "models": [
            {
                "disease_id": "breast_cancer_ultrasound",
                "model_id": "breast_busi_unet_v1",
                "model_version": "v1.0.0",
                "framework": "pytorch",
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


def _detect_response(sha: str, request_id: str) -> dict:
    return {
        "request_id": request_id,
        "disease_id": "breast_cancer_ultrasound",
        "model_id": "breast_busi_unet_v1",
        "model_version": "v1.0.0",
        "elapsed_ms": 120,
        "input_quality": {"passed": True, "checks": []},
        "classification": {
            "labels": ["benign", "malignant", "normal"],
            "probabilities": [0.10, 0.85, 0.05],
            "top1": "malignant",
            "top1_prob": 0.85,
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


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("CLARITYMED_HOME", str(tmp_path))
    from claritymed import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "DATA_DIR", tmp_path / "data")
    return tmp_path


@pytest.fixture
def _ctx():
    tokens = apply_context("20260614integ0000pipeline", USER_ID, "en")
    yield
    reset_context(tokens)


async def test_full_pipeline_threads_modality_through_to_payload(home, _ctx):
    """Attachment write → ocr.json → meta read → tool body → payload.

    Verifies the seam every layer touches — a regression in
    ``BlobStore.read_ocr_metadata``, ``write_ocr_result`` field schema,
    or the vision plugin's ``_read_vision_meta`` would surface here.
    """
    image_bytes = b"\x89PNG\r\n\x1a\nintegration"
    blob_store = BlobStore(USER_ID)
    sha = blob_store.store(image_bytes, "png")
    SessionAttachments(USER_ID, SESSION_ID).add(
        sha256=sha,
        filename="ultrasound.png",
        mime="image/png",
        size=len(image_bytes),
    )
    blob_store.write_ocr_result(
        sha,
        status="done",
        kind="ocr",
        ext="png",
        provider="test",
        chain_tried=["test"],
        reason=None,
        text="",
        original_filename="ultrasound.png",
        modality="ultrasound",
        modality_confidence=0.95,
        is_medical=True,
        ocr_has_report=False,
    )

    config = _vision_config()
    captured: list[str] = []

    def _handler(req: httpx.Request) -> httpx.Response:
        captured.append(req.url.path)
        if req.url.path == "/v1/catalog":
            return httpx.Response(200, json=_catalog())
        if req.url.path == "/v1/detect":
            body = req.content.decode()
            import json as _json

            request_id = _json.loads(body)["request_id"]
            return httpx.Response(200, json=_detect_response(sha, request_id))
        return httpx.Response(404)

    registry = VisionRegistry(config, transport=httpx.MockTransport(_handler))
    await registry.bootstrap()
    try:
        feature = VisionFeature(
            config=config,
            registry=registry,
            get_session_id=lambda: SESSION_ID,
        )
        deps = SimpleNamespace(user_id=USER_ID, language="en", prompt_channel=None)
        ctx = SimpleNamespace(deps=deps)
        result = await feature._detect(
            ctx, disease_id="breast_cancer_ultrasound", image_sha=sha
        )
    finally:
        await registry.aclose()

    assert result["kind"] == "detection"
    assert result["clinical_action"] == "urgent_specialist"
    assert "/v1/catalog" in captured  # bootstrap fired
    assert "/v1/detect" in captured  # full pipeline reached the server
    # Audit payload should have landed for this request.
    from claritymed import config as _cfg

    audit_dir = _cfg.CLARITYMED_HOME / "data" / "users" / USER_ID / "audit_payloads"
    assert audit_dir.exists(), f"audit_payloads dir missing at {audit_dir}"
    payloads = list(audit_dir.glob("*.json"))
    assert payloads, "no audit payload written for the detection"
