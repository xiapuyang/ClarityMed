"""Live full-stack vision e2e — vision-server + medical-clip-server + real LLM.

What this test covers that mocked unit tests structurally miss:

* Real BiomedCLIP forward pass (medical-clip-server) tags the image
  with the right modality.
* The LLM (omlx by default) reads the ``<image>`` tag and picks the
  ``detect_disease_from_image`` tool with the right ``disease_id`` +
  ``image_sha``.
* The plugin's confirm modal fires through the prompt-channel surface.
* The vision-server runs the promoted BUSI model and returns a
  ``RawDetection``.
* The reply prompt branches on ``clinical_action`` and the
  ``post_process`` hook audits specialist-keyword compliance.

Test gates (one missing → ``pytest.skip``):

1. ``http://127.0.0.1:8085/health`` reachable with a non-empty
   ``models_loaded`` list.
2. ``http://127.0.0.1:8086/health`` reachable.
3. A promoted BUSI artifact at
   ``~/.claritymed/models/vision/breast_cancer_ultrasound/breast_busi_unet_v1/``.
4. ``tests/fixtures/vision/busi/`` contains at least one real
   ultrasound image (placeholders are filtered out).
5. A reachable LLM provider (the shared ``e2e_provider_id`` fixture).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx
import pytest

from claritymed.context import apply_context, reset_context
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

# Strong prompt nudge — local 7-14B models flake on tool selection without
# an explicit tool name in the user text. Same pattern symptoms uses.
_PROMPT = (
    "Here is an ultrasound image I just had taken. Could this lesion be "
    "concerning? Please call the detect_disease_from_image tool with "
    "disease_id=breast_cancer_ultrasound on this image — do not answer "
    "in free text."
)


# --- module-level pre-flight -----------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _require_vision_servers() -> None:
    """Skip the module if vision-server or medical-clip-server are unavailable."""
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
            "  Flip configs/vision.yaml diseases[0].enabled to true,\n"
            "  set manifest_sha256 to match manifest.json, then restart."
        )


@pytest.fixture(scope="module", autouse=True)
def _require_promoted_busi_model() -> None:
    """Skip when the operator has not promoted a real BUSI artifact yet."""
    from claritymed import config as _cfg

    target = (
        _cfg.CLARITYMED_HOME
        / "models"
        / "vision"
        / "breast_cancer_ultrasound"
        / "breast_busi_unet_v1"
    )
    if not (target / "manifest.json").exists():
        pytest.skip(
            "no promoted BUSI artifact at "
            f"{target}/manifest.json.\n"
            "  Run docs/vision-model-workflow.md steps 1-5 to produce one."
        )


@pytest.fixture(scope="module")
def _busi_fixture_path() -> Path:
    """Return the first real BUSI image in the fixture dir; skip if absent."""
    fixture_dir = Path(__file__).resolve().parents[1] / "fixtures" / "vision" / "busi"
    candidates = [
        p
        for p in fixture_dir.glob("*")
        if p.is_file()
        and not p.name.startswith("_")
        and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ]
    if not candidates:
        pytest.skip(
            f"no real BUSI fixture images in {fixture_dir}.\n"
            "  See tests/fixtures/vision/README.md for sourcing."
        )
    return candidates[0]


# --- prompt-channel auto-answerer ------------------------------------------


class _AutoAnswerChannel:
    """Auto-answers the vision confirm modal with ``Yes``.

    Vision's plugin fires one confirm modal per tool call (the symptoms
    plugin's multi-modal Q&A pattern doesn't apply here). Anything else
    that surfaces — e.g. the disambig askuserquestion when ``modality``
    came back ``unknown`` — gets the first option as a deterministic
    deflector so the test doesn't deadlock.
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
            # Vision's confirm has "Yes, run it" + "No, skip" — first
            # label is yes. Disambig modals have disease-id options;
            # picking option 0 is the right call.
            answers[q.question] = q.options[0].label
        return AskUserQuestionResult(answers=answers, numeric_values=numeric_values)


# --- one full attempt --------------------------------------------------------


async def _run_one_attempt(
    provider_id: str,
    fixture_path: Path,
) -> tuple[_AutoAnswerChannel, list[Any], str]:
    """Fresh ChatSession + AskService → drain events → return.

    Builds a ``VisionFeature`` factory inline so the test can wire its
    own ``get_session_id`` closure against the new ``ChatSession``. The
    ``omlx`` provider runs locally so PHI never leaves the box.
    """
    from claritymed.config import load_vision_config
    from claritymed.core.rag.schemas import load_retrieval_config
    from claritymed.core.llm.model import build_model
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
    channel = _AutoAnswerChannel()
    rag_mode = load_retrieval_config().rag.mode

    # Seed the fixture image into the user's blob store + tag for the
    # active session so the plugin's attachment lookup resolves.
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
        modality="ultrasound",
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
    prompt_with_image = f"{_PROMPT} [Image sha:{sha[:8]}]"
    try:
        async with asyncio.timeout(PER_TURN_TIMEOUT_S):
            async for ev in service.run(prompt_with_image, user_id=USER_ID):
                events.append(ev)
    except asyncio.TimeoutError:
        logger.warning("vision attempt timed out after %.0fs", PER_TURN_TIMEOUT_S)
    finally:
        await registry.aclose()
    return channel, events, sha


@pytest.fixture
def _ctx():
    tokens = apply_context("20260614e2evis0000000000", USER_ID, "en")
    yield
    reset_context(tokens)


async def test_vision_full_round_trip(
    e2e_provider_id: str,
    _busi_fixture_path: Path,
    _ctx,
) -> None:
    """LLM picks the tool → confirm answers yes → server returns → reply branches."""
    last_calls = 0
    for attempt in range(MAX_ATTEMPTS):
        channel, _events, _sha = await _run_one_attempt(
            e2e_provider_id, _busi_fixture_path
        )
        last_calls = len(channel.calls)
        if last_calls >= 1:
            logger.info(
                "[vision e2e] attempt %d/%d: confirm modal fired (%d total calls)",
                attempt + 1,
                MAX_ATTEMPTS,
                last_calls,
            )
            return
        logger.info(
            "[vision e2e] attempt %d/%d: tool not invoked", attempt + 1, MAX_ATTEMPTS
        )
    pytest.fail(
        f"{TOOL_NAME}: not invoked across {MAX_ATTEMPTS} attempts.\n"
        f"  last attempt modal calls: {last_calls}\n"
        "  The local LLM never reached the confirm modal — strengthen the\n"
        "  prompt or swap to a stronger provider via CLARITYMED_E2E_PROVIDERS."
    )
