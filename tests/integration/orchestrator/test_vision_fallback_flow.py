"""Multi-model fallback flow — every branch of ``_run_fallback_flow``.

Covers the design promise spelled out in ``configs/vision.yaml`` (BUSI
primary, ``breast_us_kaggle_resnet50_v1`` fallback): when the primary
returns low confidence or is unreachable, the orchestrator falls
through to the next model in ``disease.effective_flow`` within
``total_budget_ms``. ``effective_flow`` is ``[primary_model_id, *flow]``
— the primary is auto-prepended; ``flow`` in YAML lists fallbacks only.

Real-server e2e can't deterministically force a low-confidence verdict
without a synthetic OOD fixture, so this layer pins each branch with
``httpx.MockTransport`` against the same ``VisionFeature._detect`` entry
point used by ``test_vision_full_pipeline.py``.

Branches asserted (one test per case):

1. Primary low → secondary high: returns secondary's high-conf payload;
   audit ``fallback_count == 2``; both models hit the server.
2. Primary 5xx → secondary high: server_unreachable for primary, then
   recovery on secondary; warnings line + ``fallback_count == 2``.
3. Both low → returns ``last_low`` payload; warnings explain the
   fall-through; audit ``fallback_count == 2``.
4. Both 5xx → ``no_usable_result`` kind with both unreachable warnings;
   audit ``outcome == "server_unreachable"`` when the first call raises
   AND flow has only one entry; here the second 5xx is just appended.
5. Budget exhausted before secondary → ``last_low`` returned with a
   ``skipping ...: remaining ... < expected ...`` warning; only the
   primary actually hit ``/v1/detect``.
"""

from __future__ import annotations

import json
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

USER_ID = "test"
SESSION_ID = "20260617integ00fallback00"
PRIMARY_ID = "breast_busi_unet_v1"
SECONDARY_ID = "breast_us_kaggle_resnet50_v1"
DISEASE_ID = "breast_cancer_ultrasound"
PRIMARY_SHA = "a" * 64
SECONDARY_SHA = "b" * 64


# --- config + canned responses ---------------------------------------------


def _vision_config(
    *,
    total_budget_ms: int = 20_000,
    secondary_expected_ms: int = 700,
) -> VisionConfig:
    """Two-model flow matching the production ``vision.yaml`` shape.

    ``secondary_expected_ms`` defaults to the production value (700ms);
    the budget-exhaustion test passes a high value alongside a tiny
    ``total_budget_ms`` so the safety-factor gate trips after the
    primary call.
    """
    return VisionConfig(
        diseases=[
            DiseaseSpec(
                id=DISEASE_ID,
                enabled=True,
                primary_model_id=PRIMARY_ID,
                # fallbacks-only; PRIMARY_ID is auto-prepended → effective_flow
                # is [PRIMARY_ID, SECONDARY_ID].
                flow=[SECONDARY_ID],
                cancer_class=True,
                intent_hints_i18n_key="vision.intent.breast_cancer_ultrasound",
            )
        ],
        servers=[
            ServerSpec(id="local_default", base_url="http://test", expected_ms=800)
        ],
        models=[
            ModelSpec(
                id=PRIMARY_ID,
                disease_id=DISEASE_ID,
                server_id="local_default",
                framework="pytorch",
                accepted_modality="ultrasound",
                weights_subpath=f"vision/{DISEASE_ID}/{PRIMARY_ID}",
                manifest_sha256=PRIMARY_SHA,
                expected_ms=800,
            ),
            ModelSpec(
                id=SECONDARY_ID,
                disease_id=DISEASE_ID,
                server_id="local_default",
                framework="pytorch",
                accepted_modality="ultrasound",
                weights_subpath=f"vision/{DISEASE_ID}/{SECONDARY_ID}",
                manifest_sha256=SECONDARY_SHA,
                expected_ms=secondary_expected_ms,
            ),
        ],
        tool=ToolConfig(
            confirm_before_run=False,
            total_budget_ms=total_budget_ms,
        ),
        ocr_report=OcrReportConfig(markers={"en": ["findings"], "zh": ["所见"]}),
    )


def _catalog() -> dict:
    """Two-model served set so ``VisionRegistry.bootstrap`` accepts both."""
    return {
        "models": [
            {
                "disease_id": DISEASE_ID,
                "model_id": PRIMARY_ID,
                "model_version": "v1.0.0",
                "framework": "pytorch",
                "task": "classification+segmentation",
                "labels": ["benign", "malignant", "normal"],
                "cancer_class": True,
                "accepted_modality": "ultrasound",
                "manifest_sha": PRIMARY_SHA,
                "expected_ms": 800,
                "supports_saliency": False,
                "supports_tta": True,
            },
            {
                "disease_id": DISEASE_ID,
                "model_id": SECONDARY_ID,
                "model_version": "v1.0.0",
                "framework": "pytorch",
                "task": "classification",
                "labels": ["benign", "malignant"],
                "cancer_class": True,
                "accepted_modality": "ultrasound",
                "manifest_sha": SECONDARY_SHA,
                "expected_ms": 700,
                "supports_saliency": False,
                "supports_tta": False,
            },
        ]
    }


def _detect_response(
    *, request_id: str, model_id: str, tier: str, top1: str, top1_prob: float
) -> dict:
    """Build a ``DetectResponse`` body with the requested confidence tier."""
    return {
        "request_id": request_id,
        "disease_id": DISEASE_ID,
        "model_id": model_id,
        "model_version": "v1.0.0",
        "elapsed_ms": 90,
        "input_quality": {"passed": True, "checks": []},
        "classification": {
            "labels": ["benign", "malignant", "normal"],
            "probabilities": [0.30, 0.35, 0.35]
            if tier == "low"
            else [0.05, 0.90, 0.05],
            "top1": top1,
            "top1_prob": top1_prob,
            "confidence_tier": tier,
        },
        "cancer_status": "malignant" if top1 == "malignant" else "benign",
        # Per KTD-V10, the server overrides clinical_action to
        # inconclusive_review when the tier is low so the LLM-facing
        # copy stays safe even when the orchestrator exhausts the flow.
        "clinical_action": "inconclusive_review"
        if tier == "low"
        else "urgent_specialist",
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


# --- fixtures --------------------------------------------------------------


@pytest.fixture()
def home(tmp_path, monkeypatch):
    """Pin ``CLARITYMED_HOME`` + ``DATA_DIR`` under a fresh tmp dir.

    The audit payload writer derives its output dir from these so each
    test gets a clean ``audit_payloads/`` slate.
    """
    monkeypatch.setenv("CLARITYMED_HOME", str(tmp_path))
    from claritymed import config as cfg_mod

    monkeypatch.setattr(cfg_mod, "DATA_DIR", tmp_path / "data")
    return tmp_path


@pytest.fixture
def _ctx():
    tokens = apply_context("20260617integ0000fallback", USER_ID, "en")
    yield
    reset_context(tokens)


def _seed_attachment() -> str:
    """Store one fake PNG + matching ``ocr.json`` for the active user."""
    image_bytes = b"\x89PNG\r\n\x1a\nfallback-integration"
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
    return sha


def _make_handler(
    *,
    responses: dict[str, httpx.Response | Exception],
    captured: list[tuple[str, str | None]],
) -> httpx.MockTransport:
    """Build a transport routing ``/v1/detect`` by request body ``model_id``.

    ``responses`` is keyed by model_id and may hold either a ready
    ``httpx.Response`` or an ``Exception`` to raise (lets the test
    simulate connection errors that surface as
    :class:`VisionServerUnreachableError`). ``captured`` accumulates
    ``(path, model_id)`` pairs so the test can assert the call order.
    """

    def _handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/catalog":
            captured.append((req.url.path, None))
            return httpx.Response(200, json=_catalog())
        if req.url.path == "/v1/detect":
            body = json.loads(req.content.decode())
            model_id = body.get("model_id")
            captured.append((req.url.path, model_id))
            outcome = responses.get(model_id)
            if outcome is None:
                return httpx.Response(404, json={"error": {"code": "unknown_model"}})
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return httpx.Response(404)

    return httpx.MockTransport(_handler)


async def _run(
    *,
    config: VisionConfig,
    transport: httpx.MockTransport,
    sha: str,
) -> dict:
    """Bootstrap the registry + invoke ``_detect`` once; close cleanly."""
    registry = VisionRegistry(config, transport=transport)
    await registry.bootstrap()
    try:
        feature = VisionFeature(
            config=config,
            registry=registry,
            get_session_id=lambda: SESSION_ID,
        )
        deps = SimpleNamespace(user_id=USER_ID, language="en", prompt_channel=None)
        ctx = SimpleNamespace(deps=deps)
        return await feature._detect(ctx, disease_id=DISEASE_ID, image_sha=sha)
    finally:
        await registry.aclose()


def _detect_calls(captured: list[tuple[str, str | None]]) -> list[str | None]:
    return [model_id for path, model_id in captured if path == "/v1/detect"]


def _audit_payloads(caplog) -> list[dict]:
    """Pull every ``claritymed.audit`` JSON line emitted during the run."""
    return [
        json.loads(rec.getMessage())
        for rec in caplog.records
        if rec.name == "claritymed.audit"
    ]


# --- tests -----------------------------------------------------------------


async def test_falls_through_to_secondary_when_primary_returns_low(
    home, _ctx, caplog
) -> None:
    """Primary low → secondary high: payload reflects the secondary's verdict."""
    sha = _seed_attachment()
    config = _vision_config()
    captured: list[tuple[str, str | None]] = []
    responses = {
        PRIMARY_ID: httpx.Response(
            200,
            json=_detect_response(
                request_id="primary",  # overwritten by the handler — not echoed back
                model_id=PRIMARY_ID,
                tier="low",
                top1="benign",
                top1_prob=0.36,
            ),
        ),
        SECONDARY_ID: httpx.Response(
            200,
            json=_detect_response(
                request_id="secondary",
                model_id=SECONDARY_ID,
                tier="high",
                top1="malignant",
                top1_prob=0.92,
            ),
        ),
    }
    transport = _make_handler(responses=responses, captured=captured)

    with caplog.at_level(logging.INFO, logger="claritymed.audit"):
        result = await _run(config=config, transport=transport, sha=sha)

    assert result["kind"] == "detection"
    # The payload's tool result echoes back the *secondary* model's verdict.
    assert result["clinical_action"] == "urgent_specialist"
    # Both models in flow were called, primary first.
    assert _detect_calls(captured) == [PRIMARY_ID, SECONDARY_ID]
    # vision_detection_event with the final verdict carries fallback_count=2.
    detection_events = [
        ev
        for ev in _audit_payloads(caplog)
        if ev["kind"] == "vision_detection_event"
        and "fallback_count" in ev["payload"]
        and ev["payload"].get("model_id") == SECONDARY_ID
    ]
    assert detection_events, "no terminal vision_detection_event captured"
    assert detection_events[-1]["payload"]["fallback_count"] == 2
    assert detection_events[-1]["payload"]["confidence_tier"] == "high"


async def test_falls_through_to_secondary_when_primary_5xx(home, _ctx, caplog) -> None:
    """Primary 503 → secondary high: secondary recovers; warning is recorded."""
    sha = _seed_attachment()
    config = _vision_config()
    captured: list[tuple[str, str | None]] = []
    responses = {
        PRIMARY_ID: httpx.Response(503, text="upstream busy"),
        SECONDARY_ID: httpx.Response(
            200,
            json=_detect_response(
                request_id="secondary",
                model_id=SECONDARY_ID,
                tier="high",
                top1="malignant",
                top1_prob=0.88,
            ),
        ),
    }
    transport = _make_handler(responses=responses, captured=captured)

    with caplog.at_level(logging.INFO, logger="claritymed.audit"):
        result = await _run(config=config, transport=transport, sha=sha)

    assert result["kind"] == "detection"
    assert result["clinical_action"] == "urgent_specialist"
    assert _detect_calls(captured) == [PRIMARY_ID, SECONDARY_ID]
    detection_events = [
        ev
        for ev in _audit_payloads(caplog)
        if ev["kind"] == "vision_detection_event"
        and ev["payload"].get("model_id") == SECONDARY_ID
    ]
    assert detection_events, "no terminal vision_detection_event captured"
    # Primary failure counted as one attempt that produced no usable result;
    # secondary success is attempt #2.
    assert detection_events[-1]["payload"]["fallback_count"] == 2


async def test_returns_last_low_when_every_model_low(home, _ctx, caplog) -> None:
    """Both low: orchestrator surfaces last_low with fall-through warnings."""
    sha = _seed_attachment()
    config = _vision_config()
    captured: list[tuple[str, str | None]] = []
    responses = {
        PRIMARY_ID: httpx.Response(
            200,
            json=_detect_response(
                request_id="primary",
                model_id=PRIMARY_ID,
                tier="low",
                top1="benign",
                top1_prob=0.36,
            ),
        ),
        SECONDARY_ID: httpx.Response(
            200,
            json=_detect_response(
                request_id="secondary",
                model_id=SECONDARY_ID,
                tier="low",
                top1="benign",
                top1_prob=0.34,
            ),
        ),
    }
    transport = _make_handler(responses=responses, captured=captured)

    with caplog.at_level(logging.INFO, logger="claritymed.audit"):
        result = await _run(config=config, transport=transport, sha=sha)

    # Per KTD-V10 the server already overrode clinical_action to
    # inconclusive_review on low-tier; the orchestrator hands that back
    # unchanged so the LLM-facing reply still has safe copy.
    assert result["kind"] == "detection"
    assert result["clinical_action"] == "inconclusive_review"
    assert _detect_calls(captured) == [PRIMARY_ID, SECONDARY_ID]
    # Warnings on the *last_low* payload narrate the fall-through.
    warnings = result.get("warnings", [])
    assert any(PRIMARY_ID in w and "confidence_tier=low" in w for w in warnings), (
        f"expected fall-through warning for primary in {warnings!r}"
    )


async def test_no_usable_result_when_every_model_unreachable(
    home, _ctx, caplog
) -> None:
    """Both 5xx: tool returns ``no_usable_result`` with both warnings.

    The dedicated ``server_unreachable`` outcome branch in
    ``_run_fallback_flow`` only re-raises when the flow is a single
    entry; with two entries here both 5xx responses are appended to
    ``warnings`` and the outer ``_NoUsableResult`` is raised, surfacing
    the ``no_usable_result`` kind to the LLM.
    """
    sha = _seed_attachment()
    config = _vision_config()
    captured: list[tuple[str, str | None]] = []
    responses = {
        PRIMARY_ID: httpx.Response(503, text="primary down"),
        SECONDARY_ID: httpx.Response(503, text="secondary down"),
    }
    transport = _make_handler(responses=responses, captured=captured)

    with caplog.at_level(logging.INFO, logger="claritymed.audit"):
        result = await _run(config=config, transport=transport, sha=sha)

    assert result["kind"] == "no_usable_result"
    assert result["fallback_count"] == 2
    warnings = result["warnings"]
    assert any(PRIMARY_ID in w and "server_unreachable" in w for w in warnings)
    assert any(SECONDARY_ID in w and "server_unreachable" in w for w in warnings)
    fallback_events = [
        ev
        for ev in _audit_payloads(caplog)
        if ev["kind"] == "vision_detection_event"
        and ev["payload"].get("phase") == "fallback"
    ]
    assert fallback_events, "expected a fallback-phase audit event"
    assert fallback_events[-1]["payload"]["outcome"] == "no_usable_result"


async def test_skips_secondary_when_budget_exhausted(home, _ctx, caplog) -> None:
    """Budget too small for secondary → skip with explanatory warning.

    Squeeze the total budget to the schema minimum (100ms) and inflate
    ``expected_ms`` on the secondary; ``fallback_safety_factor=1.5``
    means the gate fires when ``remaining < expected * 1.5``. The
    primary's mock response returns instantly so essentially the full
    100ms is still available — well below ``5000 * 1.5 = 7500``, so the
    secondary is skipped and the last_low payload (with the skip
    warning appended) is returned.
    """
    sha = _seed_attachment()
    config = _vision_config(total_budget_ms=100, secondary_expected_ms=5_000)
    captured: list[tuple[str, str | None]] = []
    responses = {
        PRIMARY_ID: httpx.Response(
            200,
            json=_detect_response(
                request_id="primary",
                model_id=PRIMARY_ID,
                tier="low",
                top1="benign",
                top1_prob=0.36,
            ),
        ),
        # If the secondary actually got called we'd see this in captured;
        # the test asserts it doesn't.
        SECONDARY_ID: httpx.Response(
            200,
            json=_detect_response(
                request_id="secondary",
                model_id=SECONDARY_ID,
                tier="high",
                top1="malignant",
                top1_prob=0.95,
            ),
        ),
    }
    transport = _make_handler(responses=responses, captured=captured)

    with caplog.at_level(logging.INFO, logger="claritymed.audit"):
        result = await _run(config=config, transport=transport, sha=sha)

    assert result["kind"] == "detection"
    # last_low returned — KTD-V10 has the server override the action to
    # inconclusive_review on low-tier.
    assert result["clinical_action"] == "inconclusive_review"
    # Crucial: secondary was *not* called.
    assert _detect_calls(captured) == [PRIMARY_ID]
    warnings = result.get("warnings", [])
    assert any(
        SECONDARY_ID in w and "skipping" in w and "remaining" in w for w in warnings
    ), f"expected budget-skip warning for secondary in {warnings!r}"
