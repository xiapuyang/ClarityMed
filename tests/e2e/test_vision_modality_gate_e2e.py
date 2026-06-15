"""KTD-V3 modality hard refuse — wrong-modality image MUST NOT reach the server.

Mirrors :mod:`test_vision_e2e` structurally; the only difference is the
seeded image is tagged ``modality="ct"`` so the breast-ultrasound model
must be refused before any HTTP call to ``/v1/detect``.

Pre-flight gates match :mod:`test_vision_e2e`. The CT fixture comes
from ``tests/fixtures/vision/modality_mismatch/``; placeholders skip
the test cleanly.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
import pytest

from claritymed.context import apply_context, reset_context

logger = logging.getLogger(__name__)

USER_ID = "e2e"
VISION_BASE_URL = "http://127.0.0.1:8085"
MEDICAL_CLIP_BASE_URL = "http://127.0.0.1:8086"


@pytest.fixture(scope="module", autouse=True)
def _require_vision_servers() -> None:
    for url, name in (
        (VISION_BASE_URL, "vision-server"),
        (MEDICAL_CLIP_BASE_URL, "medical-clip-server"),
    ):
        try:
            resp = httpx.get(f"{url}/health", timeout=2.0)
        except httpx.HTTPError as exc:
            pytest.skip(f"{name} unreachable at {url}: {exc}")
        if resp.status_code != 200:
            pytest.skip(f"{name} /health returned {resp.status_code}")


@pytest.fixture(scope="module")
def _ct_fixture_path() -> Path:
    fixture_dir = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "vision"
        / "modality_mismatch"
    )
    candidates = [
        p
        for p in fixture_dir.glob("*")
        if p.is_file()
        and not p.name.startswith("_")
        and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".dcm"}
    ]
    if not candidates:
        pytest.skip(
            f"no CT / X-ray fixture in {fixture_dir}.\n"
            "  See tests/fixtures/vision/README.md for sourcing."
        )
    return candidates[0]


@pytest.fixture
def _ctx():
    tokens = apply_context("20260614modalitygate0000", USER_ID, "en")
    yield
    reset_context(tokens)


async def test_modality_mismatch_returns_structured_refuse_no_http(
    _ct_fixture_path: Path, _ctx
) -> None:
    """Tag a CT image as ``ct`` and call the tool body directly.

    We deliberately bypass the LLM here — the gate's correctness is
    deterministic, and the value of the e2e is proving the body
    short-circuits against the same store + meta the real pipeline
    feeds it. A captured ``httpx.MockTransport`` on the registry would
    blow up if the body somehow reached ``/v1/detect``.
    """
    from types import SimpleNamespace

    from claritymed.config import load_vision_config
    from claritymed.core.vision.registry import VisionRegistry
    from claritymed.orchestrator.features.vision_plugin import VisionFeature
    from claritymed.orchestrator.services.chat_session import ChatSession
    from claritymed.stores.blob_store import BlobStore
    from claritymed.stores.session_attachments import SessionAttachments

    chat = ChatSession.new(USER_ID)
    image_bytes = _ct_fixture_path.read_bytes()
    blob_store = BlobStore(USER_ID)
    sha = blob_store.store(image_bytes, _ct_fixture_path.suffix.lstrip("."))
    SessionAttachments(USER_ID, chat.session_id).add(
        sha256=sha,
        filename=_ct_fixture_path.name,
        mime="image/png",
        size=len(image_bytes),
    )
    blob_store.write_ocr_result(
        sha,
        status="done",
        kind="ocr",
        ext=_ct_fixture_path.suffix.lstrip("."),
        provider="e2e-seed",
        chain_tried=["e2e-seed"],
        reason=None,
        text="",
        original_filename=_ct_fixture_path.name,
        modality="ct",
        modality_confidence=0.9,
        is_medical=True,
        ocr_has_report=False,
    )

    config = load_vision_config()

    def _http_blocker(req):
        raise AssertionError(
            f"vision tool body must NOT reach the server on modality mismatch; "
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
    assert result["kind"] == "modality_mismatch"
    assert result["model_accepts"] == "ultrasound"
    assert result["image_modality"] == "ct"
