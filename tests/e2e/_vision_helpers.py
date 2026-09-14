"""Shared helpers for vision e2e tests.

Three test files (``test_vision_e2e.py`` — BUSI, ``test_vision_chest_ct_e2e.py``,
``test_vision_skin_e2e.py``) share the same trial shape:

1. Drop a fixture image into the bench user's blob store + write a
   sentinel ``ocr.json`` that fixes the modality / is_medical /
   ocr_has_report fields. This isolates the LLM-decision axis from the
   medical-clip classifier (which has its own Unit 2 metric).
2. Build a full orchestrator stack (``AskService`` + ``ChatSession`` +
   ``VisionFeature``) keyed on the right provider.
3. Drive a single user turn through ``service.run()`` with an
   auto-answer channel that approves the confirm modal so the tool
   actually invokes.
4. Return the (channel, events, sha) tuple so the per-disease test can
   assert on call counts + content.

Each disease test contributes the small disease-specific bits — the
prompt text, the fixture subdir, the promoted-artifact path — through
function args. The file is named with a leading underscore so pytest's
test collector skips it; the helpers are imported, not collected.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest

from claritymed.core.interaction.schemas import (
    AskUserQuestionInput,
    AskUserQuestionResult,
)

logger = logging.getLogger(__name__)

USER_ID = "e2e"
MAX_ATTEMPTS = 3
PER_TURN_TIMEOUT_S = 240.0
VISION_BASE_URL = "http://127.0.0.1:8085"
MEDICAL_CLIP_BASE_URL = "http://127.0.0.1:8086"
TOOL_NAME = "detect_disease_from_image"


# --- service pre-flight -----------------------------------------------------


def require_vision_servers() -> None:
    """Skip the calling module when vision/medical-clip servers aren't up.

    Both servers must respond ``200`` and vision-server's catalog must
    advertise at least one loaded model — otherwise no inference path
    can resolve. Call from an autouse fixture in each test module.
    """
    for url, name in (
        (VISION_BASE_URL, "vision-server"),
        (MEDICAL_CLIP_BASE_URL, "medical-clip-server"),
    ):
        try:
            resp = httpx.get(f"{url}/health", timeout=2.0)
        except httpx.HTTPError as exc:
            pytest.skip(
                f"{name} unreachable at {url}: {exc}\n"
                f"  Start it with: uv run claritymed-{name}"
            )
        if resp.status_code != 200:
            pytest.skip(f"{name} /health returned {resp.status_code}")
    body = httpx.get(f"{VISION_BASE_URL}/health", timeout=2.0).json()
    if not body.get("models_loaded"):
        pytest.skip(
            "vision-server is up but has no models loaded "
            f"(/health: {body!r}).\n"
            "  Flip the disease's enabled flag in configs/vision.yaml,\n"
            "  set manifest_sha256 to match manifest.json, then restart."
        )


def require_promoted_artifact(disease_id: str, model_id: str) -> None:
    """Skip when the per-disease promoted artifact dir doesn't exist."""
    from claritymed import config as _cfg

    target = _cfg.CLARITYMED_HOME / "models" / "vision" / disease_id / model_id
    if not (target / "manifest.json").exists():
        pytest.skip(
            f"no promoted artifact for {disease_id}/{model_id} at "
            f"{target}/manifest.json.\n"
            "  Run docs/vision-model-workflow.md steps 1-5 to produce one."
        )


def pick_fixture_or_skip(fixture_subdir: str) -> Path:
    """Return the first real image in ``tests/fixtures/vision/<subdir>``."""
    fixture_dir = (
        Path(__file__).resolve().parents[1] / "fixtures" / "vision" / fixture_subdir
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
            f"no real fixture images in {fixture_dir}.\n"
            "  See tests/fixtures/vision/README.md for sourcing."
        )
    return candidates[0]


# --- channel ----------------------------------------------------------------


class AutoAnswerChannel:
    """Auto-answers the vision confirm modal with the first option.

    The first option is always the affirmative one ("Yes, run it" /
    "好，运行") by convention (see ``configs/i18n/<lang>/vision.yaml``).
    Disambig modals that surface get the same treatment so the test
    deterministically continues; the per-disease assertions only care
    about the confirm-modal count.
    """

    def __init__(self) -> None:
        self.calls: list[AskUserQuestionInput] = []

    async def ask(self, payload: AskUserQuestionInput) -> AskUserQuestionResult:
        self.calls.append(payload)
        answers: dict[str, str] = {}
        numeric_values: dict[str, float] = {}
        for q in payload.questions:
            if q.numeric is not None:
                numeric_values[q.question] = float(q.numeric.min)
                continue
            if not q.options:
                answers[q.question] = "Yes"
                continue
            answers[q.question] = q.options[0].label
        return AskUserQuestionResult(answers=answers, numeric_values=numeric_values)


# --- one full attempt --------------------------------------------------------


async def run_one_attempt(
    *,
    provider_id: str,
    fixture_path: Path,
    modality: str,
    prompt: str,
) -> tuple[AutoAnswerChannel, list[Any], str]:
    """Run one full attempt: seed → build stack → drive ``service.run()``.

    The prompt may include ``{sha8}`` as a placeholder; the eight-char
    image sha is substituted in before the turn fires (the placeholder
    is harmless if absent). Returns the channel (for modal-count
    assertions), the event list (for downstream filtering), and the
    full 64-char sha so the caller can correlate with audit logs.
    """
    from claritymed.config import load_vision_config
    from claritymed.core.llm.model import build_model
    from claritymed.core.rag.schemas import load_retrieval_config
    from claritymed.core.vision.registry import VisionRegistry
    from claritymed.orchestrator.features.vision_plugin import VisionFeature
    from claritymed.orchestrator.services import AskService
    from claritymed.orchestrator.services.chat_session import ChatSession
    from claritymed.stores.blob_store import BlobStore
    from claritymed.stores.models import resolve_provider
    from claritymed.stores.session_attachments import SessionAttachments

    provider = resolve_provider(override=provider_id)
    model = build_model(provider)
    chat = ChatSession.new(USER_ID)
    channel = AutoAnswerChannel()
    rag_mode = load_retrieval_config().rag.mode

    image_bytes = fixture_path.read_bytes()
    blob_store = BlobStore(USER_ID)
    sha = blob_store.store(image_bytes, fixture_path.suffix.lstrip("."))
    SessionAttachments(USER_ID, chat.session_id).add(
        sha256=sha,
        filename=fixture_path.name,
        mime="image/png",
        size=len(image_bytes),
    )
    blob_store.write_ocr_result(
        sha,
        status="done",
        kind="ocr",
        ext=fixture_path.suffix.lstrip("."),
        provider="e2e-seed",
        chain_tried=["e2e-seed"],
        reason=None,
        text="",
        original_filename=fixture_path.name,
        modality=modality,
        modality_confidence=0.95,
        is_medical=True,
        ocr_has_report=False,
    )

    vision_config = load_vision_config()
    registry = VisionRegistry(vision_config)
    await registry.bootstrap()

    def _make_vision():
        return VisionFeature(
            config=vision_config,
            registry=registry,
            get_session_id=lambda: chat.session_id,
        )

    service = AskService(
        model=model,
        chat_session=chat,
        provider_id=provider.id,
        model_name=provider.model,
        provider_config=provider,
        rag_mode=rag_mode,
        prompt_channel=channel,
        vision_factory=_make_vision,
    )
    events: list[Any] = []
    prompt_resolved = prompt.replace("{sha8}", sha[:8])
    try:
        async with asyncio.timeout(PER_TURN_TIMEOUT_S):
            async for ev in service.run(prompt_resolved, user_id=USER_ID):
                events.append(ev)
    except asyncio.TimeoutError:
        logger.warning("vision attempt timed out after %.0fs", PER_TURN_TIMEOUT_S)
    finally:
        await registry.aclose()
    return channel, events, sha


async def assert_tool_invoked(
    *,
    provider_id: str,
    fixture_path: Path,
    modality: str,
    prompt: str,
    label: str,
) -> None:
    """Run ``MAX_ATTEMPTS`` trials; pass if any one fires the confirm modal.

    Mirrors the retry loop used by the BUSI test — small local models
    occasionally drop a tool call. ``label`` is the disease name surfaced
    in the failure message so a multi-disease test run can tell which
    one regressed.
    """
    last_calls = 0
    for attempt in range(MAX_ATTEMPTS):
        channel, _events, _sha = await run_one_attempt(
            provider_id=provider_id,
            fixture_path=fixture_path,
            modality=modality,
            prompt=prompt,
        )
        last_calls = len(channel.calls)
        if last_calls >= 1:
            logger.info(
                "[vision e2e %s] attempt %d/%d: confirm modal fired (%d total calls)",
                label,
                attempt + 1,
                MAX_ATTEMPTS,
                last_calls,
            )
            return
        logger.info(
            "[vision e2e %s] attempt %d/%d: tool not invoked",
            label,
            attempt + 1,
            MAX_ATTEMPTS,
        )
    pytest.fail(
        f"{TOOL_NAME} ({label}): not invoked across {MAX_ATTEMPTS} attempts.\n"
        f"  last attempt modal calls: {last_calls}\n"
        "  The local LLM never reached the confirm modal — strengthen the\n"
        "  prompt or swap to a stronger provider via CLARITYMED_E2E_PROVIDERS."
    )
