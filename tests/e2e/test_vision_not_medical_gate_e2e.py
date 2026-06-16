"""R6 not-medical hard refuse — non-medical image must not reach the server.

Mirrors :mod:`test_vision_modality_gate_e2e` structurally. The only
difference is the seeded image is tagged ``is_medical=False`` so the
vision tool body short-circuits with :class:`NotMedicalResult` before
any HTTP call to ``/v1/detect``.

This is the defense-in-depth layer that runs *inside* the tool body —
if the LLM is fooled by a strong prompt and tries to invoke the tool on
a non-medical image (cat photo, screenshot, etc.), the body intercepts.
The medical-clip server is the upstream classifier that produces
``is_medical=False`` at ingest; this e2e seeds the result directly so
the gate's correctness is deterministic and doesn't depend on the
classifier's per-image accuracy.

A real non-medical fixture under ``tests/fixtures/vision/non_medical/``
is preferred; when absent, we generate a synthetic non-medical PNG
in-process so the test still runs in any environment.
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


def _generate_non_medical_png(target: Path) -> Path:
    """Write a small synthetic non-medical PNG (text on white).

    The bytes don't need to *look* non-medical to the test — the
    ``is_medical`` flag is seeded explicitly on the OCR sentinel. We
    only need readable bytes the blob store will accept.
    """
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (320, 240), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 100), "weather forecast: sunny", fill="black")
    img.save(str(target), format="PNG")
    return target


@pytest.fixture(scope="module")
def _non_medical_fixture_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a non-medical image — prefer a real fixture, fall back to synthetic.

    Generating in-process keeps the test self-contained when the
    ``non_medical/`` fixture dir holds only a placeholder.
    """
    fixture_dir = (
        Path(__file__).resolve().parents[1] / "fixtures" / "vision" / "non_medical"
    )
    candidates = [
        p
        for p in fixture_dir.glob("*")
        if p.is_file()
        and not p.name.startswith("_")
        and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    if candidates:
        return candidates[0]
    tmp = tmp_path_factory.mktemp("non_medical")
    return _generate_non_medical_png(tmp / "synthetic_non_medical.png")


@pytest.fixture
def _ctx():
    tokens = apply_context("20260615notmedical000000", USER_ID, "en")
    yield
    reset_context(tokens)


async def test_not_medical_returns_structured_refuse(
    _non_medical_fixture_path: Path, _ctx
) -> None:
    """Tag a non-medical image as ``is_medical=False`` and call the tool body directly.

    The tool body's R6 gate short-circuits before any HTTP call to
    ``/v1/detect`` — even if the LLM was fooled into picking the
    vision tool, the body refuses.
    """
    from types import SimpleNamespace

    from claritymed.config import load_vision_config
    from claritymed.core.vision.registry import VisionRegistry
    from claritymed.orchestrator.features.vision_plugin import VisionFeature
    from claritymed.orchestrator.services.chat_session import ChatSession
    from claritymed.stores.blob_store import BlobStore
    from claritymed.stores.session_attachments import SessionAttachments

    chat = ChatSession.new(USER_ID)
    image_bytes = _non_medical_fixture_path.read_bytes()
    blob_store = BlobStore(USER_ID)
    sha = blob_store.store(image_bytes, _non_medical_fixture_path.suffix.lstrip("."))
    SessionAttachments(USER_ID, chat.session_id).add(
        sha256=sha,
        filename=_non_medical_fixture_path.name,
        mime="image/png",
        size=len(image_bytes),
    )
    blob_store.write_ocr_result(
        sha,
        status="done",
        kind="ocr",
        ext=_non_medical_fixture_path.suffix.lstrip("."),
        provider="e2e-seed",
        chain_tried=["e2e-seed"],
        reason=None,
        text="",
        original_filename=_non_medical_fixture_path.name,
        modality="photo",
        modality_confidence=0.9,
        is_medical=False,
        ocr_has_report=False,
    )

    config = load_vision_config()
    registry = VisionRegistry(config)
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
    assert result["kind"] == "not_medical", (
        f"expected non-medical short-circuit but tool body returned {result!r}"
    )
