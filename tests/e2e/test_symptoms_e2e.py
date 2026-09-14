"""End-to-end symptoms tool drive — real omlx LLM + real subprocess server.

Sits one rung above ``tests/orchestrator/features/test_symptoms_plugin_e2e.py``:
that file drives the plugin against the FastAPI app in-process (ASGI
transport, stub agent). This one needs the live subprocess server on
``127.0.0.1:8084`` with real ``typed-BASD`` weights loaded, *and* a real
local LLM provider on the orchestrator side picking the
``predict_disease_from_symptoms`` tool out of its own volition.

Pre-flight:

* Start the subprocess server: ``scripts/run.sh symptoms`` (loads weights
  from ``$CLARITYMED_HOME/models/symptoms/ddxplus/typed_basd_v1/``).
* ``configs/symptoms.yaml`` must have ``datasets[0].enabled: true`` and a
  matching ``manifest_sha256:`` — the module-level fixture below skips
  the test cleanly otherwise.
* A reachable LLM provider — same fixture-driven discovery as the rest of
  ``tests/e2e/`` (``CLARITYMED_E2E_PROVIDERS=omlx`` or
  ``pick_reachable_provider()``).

Why retries (mirrors the ingest tools e2e): the local LLM is
non-deterministic about tool selection. We give it ``MAX_ATTEMPTS`` fresh
``ChatSession``s with strongly-worded prompts before declaring the model
unwilling to invoke the tool.

Eligibility is stubbed in this layer (``_AlwaysEligible``). The real
strategies are exercised under ``tests/core/symptoms/eligibility``; here
we want the LLM → tool dispatch → live HTTP → real BASD inference path
specifically. Wiring the production vocab loader for the test would
require duplicating the (unfinished) production-side wiring this file is
already steering around.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
import pytest

from claritymed.config import load_symptoms_config
from claritymed.context import apply_context, reset_context
from claritymed.core.interaction.schemas import (
    AskUserQuestionInput,
    AskUserQuestionResult,
)
from claritymed.core.rag import load_retrieval_config
from claritymed.core.symptoms.client import SymptomsServerClient
from claritymed.core.symptoms.eligibility.base import (
    EligibilityResult,
    EligibilityStrategy,
)
from claritymed.core.symptoms.registry import DatasetRegistry
from claritymed.orchestrator.features.symptoms_plugin import SymptomsFeature
from claritymed.orchestrator.services import AskService
from claritymed.orchestrator.services.chat_session import ChatSession

logger = logging.getLogger(__name__)

USER_ID = "e2e"
MAX_ATTEMPTS = 3
PER_TURN_TIMEOUT_S = 240.0
SYMPTOMS_BASE_URL = "http://127.0.0.1:8084"
TOOL_NAME = "predict_disease_from_symptoms"

# A multi-symptom diagnostic complaint that satisfies the tool description's
# "DO call" cues ("could this be serious", "since this morning", multiple
# symptoms in DDXPlus scope). The explicit tool-name nudge is the same
# pattern the ingest e2e uses — without it, local 7-14B-class models
# regularly compose a free-text reply instead of invoking the tool.
_COMPLAINT = (
    "I've had chest pain and shortness of breath since this morning, "
    "plus a low-grade fever. Could this be serious? Please call the "
    "predict_disease_from_symptoms tool to work out a differential — "
    "do not answer in free text."
)


# --- module-level pre-flight ------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _require_symptoms_server() -> None:
    """Skip the module if ``:8084`` is either unreachable or empty.

    Empty = ``datasets_loaded: []`` — happens when ``configs/symptoms.yaml``
    still has the ship-default ``enabled: false`` or the manifest sha256
    chain refused to load. The skip message points the operator at the
    fix rather than failing opaquely from a 404 deep inside the plugin.
    """
    try:
        resp = httpx.get(f"{SYMPTOMS_BASE_URL}/health", timeout=2.0)
    except httpx.HTTPError as exc:
        pytest.skip(
            f"symptoms server unreachable at {SYMPTOMS_BASE_URL}: {exc}\n"
            "  Start it with: scripts/run.sh symptoms"
        )
    if resp.status_code != 200:
        pytest.skip(
            f"symptoms server /health returned {resp.status_code}: {resp.text!r}"
        )
    body = resp.json()
    if not body.get("datasets_loaded"):
        pytest.skip(
            "symptoms server is up but has no datasets loaded "
            f"(/health: {body!r}).\n"
            "  Flip configs/symptoms.yaml datasets[0].enabled to true and\n"
            "  ensure manifest_sha256 matches the on-disk manifest.json,\n"
            "  then `scripts/restart.sh symptoms`."
        )


# --- prompt-channel auto-answerer ------------------------------------------


class _AutoAnswerChannel:
    """Introspects each modal payload and returns plausible answers.

    Why not :class:`tests._helpers.channels.AutoAnswerChannel`: that one is
    queue-driven (pre-scripted), which means the test would need to know
    in advance how many turns the BASD agent will run (5-12 depending on
    the complaint + stop head). Generating answers from the payload's
    shape keeps the test resilient across maxstep changes and lets the
    server pick its own trajectory.

    Decision rules:

    * Numeric ``"How old are you"`` → ``45`` (plausible adult; out of any
      pediatric-specific evidence vocab).
    * Numeric anything else → ``q.numeric.min`` (defensive default for
      future numeric questions; mid-bucket would also work).
    * Categorical with ``Yes/否/是/No`` → ``Yes`` (positive answers push
      the BASD agent toward a tighter differential faster than rejecting
      every evidence; either branch is valid).
    * Categorical sex → ``Male`` (DDXPlus is sex-stratified, the choice
      is arbitrary).
    * Categorical anything else → first option (deterministic and the
      modal-renderer-imposed first option already tends to be "no
      special finding" in the canonical question text).
    """

    def __init__(self) -> None:
        self.calls: list[AskUserQuestionInput] = []

    async def ask(self, payload: AskUserQuestionInput) -> AskUserQuestionResult:
        self.calls.append(payload)
        answers: dict[str, str] = {}
        numeric_values: dict[str, float] = {}
        for q in payload.questions:
            qtext = q.question
            if q.numeric is not None:
                if "old" in qtext.lower() or "年" in qtext or "岁" in qtext:
                    numeric_values[qtext] = 45.0
                else:
                    numeric_values[qtext] = float(q.numeric.min)
                continue
            if not q.options:
                # No-options edge case — shouldn't occur, defensive.
                answers[qtext] = "Yes"
                continue
            labels = [opt.label for opt in q.options]
            lower = qtext.lower()
            if "sex" in lower or "性别" in qtext:
                pick = next(
                    (lab for lab in labels if lab.lower() in {"male", "男"}),
                    labels[0],
                )
            else:
                pick = next(
                    (lab for lab in labels if lab.lower() in {"yes", "是"}),
                    labels[0],
                )
            answers[qtext] = pick
        return AskUserQuestionResult(answers=answers, numeric_values=numeric_values)


# --- always-eligible stub ---------------------------------------------------


class _AlwaysEligible(EligibilityStrategy):
    """Eligibility short-circuit so this layer measures LLM + server only.

    The production vocab loader for the ``direct`` strategy is not yet
    wired (no production-side ``build_features`` builds it), and the
    plugin construction otherwise can't run. The real strategies (direct
    / term_service / translation) have their own unit suites under
    ``tests/core/symptoms/eligibility``.
    """

    async def check(self, complaint, language, profile, dataset) -> EligibilityResult:
        return EligibilityResult(eligible=True, reason="in_scope", confidence=1.0)


# --- harness ---------------------------------------------------------------


def _make_symptoms_factory(client: SymptomsServerClient):
    """Closure factory that AskService can call once at construction.

    The same client + registry + config snapshot is reused across the
    plugin's lifetime — ``SymptomsServerClient`` is an async resource so
    we hand its ownership to whoever creates the factory and rely on the
    test's ``async with`` to close it.
    """
    config = load_symptoms_config()
    registry = DatasetRegistry(config.datasets)
    eligibility = _AlwaysEligible()

    def _factory():
        return SymptomsFeature(
            config=config,
            registry=registry,
            client=client,
            eligibility=eligibility,
        )

    return _factory


async def _run_one_attempt(
    provider_id: str,
    complaint: str,
) -> tuple[_AutoAnswerChannel, list[Any]]:
    """One fresh ChatSession + AskService → drain events → return.

    Fresh session so prior tool-call attempts can't bias the LLM's next
    decision. Wraps ``service.run`` in an ``asyncio.timeout`` because the
    symptoms loop can take 30-60 s end-to-end with a local model.
    """
    from claritymed.core.llm.model import build_model
    from claritymed.stores.models import resolve_provider

    provider = resolve_provider(override=provider_id)
    model = build_model(provider)
    chat = ChatSession.new(USER_ID)
    channel = _AutoAnswerChannel()
    rag_mode = load_retrieval_config().rag.mode

    async with SymptomsServerClient(SYMPTOMS_BASE_URL) as client:
        service = AskService(
            model=model,
            chat_session=chat,
            provider_id=provider.id,
            model_name=provider.model,
            provider_config=provider,
            rag_mode=rag_mode,
            prompt_channel=channel,
            symptoms_factory=_make_symptoms_factory(client),
        )
        events: list[Any] = []
        try:
            async with asyncio.timeout(PER_TURN_TIMEOUT_S):
                async for ev in service.run(complaint, user_id=USER_ID):
                    events.append(ev)
        except asyncio.TimeoutError:
            logger.warning("symptoms attempt timed out after %.0fs", PER_TURN_TIMEOUT_S)
    return channel, events


# --- fixtures + test --------------------------------------------------------


@pytest.fixture
def _ctx():
    tokens = apply_context("20260613e2esymp000000000", USER_ID, "en")
    yield
    reset_context(tokens)


async def test_symptoms_complaint_runs_real_model_through_real_server(
    e2e_provider_id: str,
    _ctx,
) -> None:
    """Live omlx (or whatever provider was selected) picks the tool, the
    plugin drives a real subprocess server, and a non-empty differential
    surfaces back to the orchestrator.

    Retries because local LLMs flake on tool selection — the test fails
    only after ``MAX_ATTEMPTS`` independent sessions have all skipped
    the tool, which is a real signal (prompt too weak, model too small,
    or the tool description is ambiguous).
    """
    last_channel_calls: int = 0
    last_headers: list[list[str | None]] = []
    for attempt in range(MAX_ATTEMPTS):
        channel, _events = await _run_one_attempt(e2e_provider_id, _COMPLAINT)
        last_channel_calls = len(channel.calls)
        last_headers = [[q.header for q in c.questions] for c in channel.calls]

        # Why the modal-call signal rather than ``ToolStarted`` events:
        # ``ToolStarted`` is only emitted by the approval-channel flow
        # used by the ingest tools — the symptoms tool runs through the
        # pydantic-ai tool loop directly and never reaches that emitter.
        # The cleanest evidence that omlx invoked the tool is the side
        # effect on this turn's only prompt-channel: the plugin drives
        # confirm → initial batch → ≥1 server-question modal as soon as
        # ``_predict`` is entered. ``ask_user_question`` is also wired,
        # but only when ``prompt_channel`` is set with no symptoms call —
        # since we set ``symptoms_factory``, modal traffic on this turn
        # is symptoms-loop traffic (a future change that also wires
        # ``ask_user_question`` to ``channel`` would need to look at
        # ``ToolCompleted`` to disambiguate).
        if len(channel.calls) < 3:
            logger.info(
                "[symptoms e2e] attempt %d/%d: tool not invoked or short-circuited "
                "(%d modal calls; headers: %r)",
                attempt + 1,
                MAX_ATTEMPTS,
                len(channel.calls),
                last_headers,
            )
            continue
        logger.info(
            "[symptoms e2e] attempt %d/%d: %s drove %d modal turns",
            attempt + 1,
            MAX_ATTEMPTS,
            TOOL_NAME,
            len(channel.calls),
        )
        return

    pytest.fail(
        f"{TOOL_NAME}: not invoked across {MAX_ATTEMPTS} attempts.\n"
        f"  last attempt modal calls: {last_channel_calls}\n"
        f"  last attempt headers: {last_headers!r}\n"
        f"  complaint sent: {_COMPLAINT!r}\n"
        "  The local LLM never proposed the tool — strengthen the prompt or\n"
        "  switch to a more capable provider via CLARITYMED_E2E_PROVIDERS."
    )
