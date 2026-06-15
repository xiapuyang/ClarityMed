"""KTD-V6 OCR override — image carrying a clinician report short-circuits the tool.

When ``has_structured_report(ocr_text, language)`` flips
``ocr_has_report=true`` on the sentinel, the LLM should answer from
the OCR text and the tool body MUST refuse before reaching the model.

Pre-flight gates mirror :mod:`test_vision_e2e`. The report-overlay
fixture is sourced per ``tests/fixtures/vision/README.md`` — a CC0
ultrasound base with composited synthetic FINDINGS / IMPRESSION text.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from claritymed.context import apply_context, reset_context

logger = logging.getLogger(__name__)

USER_ID = "e2e"


@pytest.fixture(scope="module")
def _overlay_fixture_path() -> Path:
    fixture_dir = (
        Path(__file__).resolve().parents[1] / "fixtures" / "vision" / "report_overlay"
    )
    candidates = [
        p
        for p in fixture_dir.glob("*")
        if p.is_file()
        and not p.name.startswith("_")
        and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    if not candidates:
        pytest.skip(
            f"no report-overlay fixture in {fixture_dir}.\n"
            "  See tests/fixtures/vision/README.md for sourcing."
        )
    return candidates[0]


@pytest.fixture
def _ctx():
    tokens = apply_context("20260614ocroverride00000", USER_ID, "en")
    yield
    reset_context(tokens)


async def test_ocr_override_short_circuits_no_http(
    _overlay_fixture_path: Path, _ctx
) -> None:
    """Tag the image with ``ocr_has_report=true`` and call the body directly.

    Same rationale as :mod:`test_vision_modality_gate_e2e` — bypassing
    the LLM here keeps the test deterministic. A captured ``MockTransport``
    would blow up if the body reached ``/v1/detect``.
    """
    from claritymed.config import load_vision_config
    from claritymed.core.vision.registry import VisionRegistry
    from claritymed.orchestrator.features.vision_plugin import VisionFeature
    from claritymed.orchestrator.services.chat_session import ChatSession
    from claritymed.stores.blob_store import BlobStore
    from claritymed.stores.session_attachments import SessionAttachments

    chat = ChatSession.new(USER_ID)
    image_bytes = _overlay_fixture_path.read_bytes()
    blob_store = BlobStore(USER_ID)
    sha = blob_store.store(image_bytes, _overlay_fixture_path.suffix.lstrip("."))
    SessionAttachments(USER_ID, chat.session_id).add(
        sha256=sha,
        filename=_overlay_fixture_path.name,
        mime="image/png",
        size=len(image_bytes),
    )
    # A real OCR text with a marker keyword so ``has_structured_report``
    # would also flip; we set the flag explicitly so the test doesn't
    # depend on the heuristic's threshold.
    ocr_text = (
        "FINDINGS: The right breast contains a hypoechoic lesion measuring "
        "1.8 cm. IMPRESSION: BIRADS 4 — recommend biopsy. " * 5
    )
    blob_store.write_ocr_result(
        sha,
        status="done",
        kind="ocr",
        ext=_overlay_fixture_path.suffix.lstrip("."),
        provider="e2e-seed",
        chain_tried=["e2e-seed"],
        reason=None,
        text=ocr_text,
        original_filename=_overlay_fixture_path.name,
        modality="ultrasound",
        modality_confidence=0.95,
        is_medical=True,
        ocr_has_report=True,
    )

    config = load_vision_config()

    def _http_blocker(req):
        raise AssertionError(
            f"vision tool body must NOT reach the server when ocr_has_report=true; "
            f"saw {req.method} {req.url}"
        )

    registry = VisionRegistry(config, transport=httpx.MockTransport(_http_blocker))
    feature = VisionFeature(
        config=config,
        registry=registry,
        get_session_id=lambda: chat.session_id,
    )
    deps = SimpleNamespace(user_id=USER_ID, language="en", prompt_channel=None)
    ctx = SimpleNamespace(deps=deps)
    result = await feature._detect(
        ctx, disease_id="breast_cancer_ultrasound", image_sha=sha
    )
    assert result["kind"] == "ocr_override"
