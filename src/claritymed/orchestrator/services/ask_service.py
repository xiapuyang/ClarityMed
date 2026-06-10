"""Ask service: PHI scrub on input + drive the ask turn.

Owns the turn shape: PHI scrub, plugin composition (deterministic
pre-invoke text + tool registration), agent build, streaming, audit,
chat-session persistence. Per-feature behaviour (RAG, future vision /
symptoms) is delegated to ``FeaturePlugin`` instances built by
``core.features.build_features``; the service never branches on
"which feature" or "which mode" beyond that composition step.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from claritymed.core.events import (
    Done,
    Error,
    Event,
    LlmCallStarted,
    LlmFirstToken,
    TokenChunk,
)
from claritymed.core.features import TurnContext, build_features
from claritymed.core.observability.audit import audit_event
from claritymed.core.observability.latency import LatencyTrace, build_step_records
from claritymed.core.observability.latency import usage_dict as _usage_dict
from claritymed.core.observability.logging import get_access_logger
from claritymed.core.observability.tool_announce import detect_announcement
from claritymed.core.phi.guard import PhiGuard
from claritymed.orchestrator.agents import make_ask_agent
from claritymed.orchestrator.agents.ask_deps import AskDeps

if TYPE_CHECKING:
    from pydantic_ai.models import Model

    from claritymed.core.features import FeaturePlugin
    from claritymed.core.interaction.prompt_channel import PromptChannel
    from claritymed.core.rag.strategies.base import RagStrategy
    from claritymed.core.schemas import ProviderConfig
    from claritymed.core.schemas.retrieval import RetrievedChunk
    from claritymed.core.translation import TranslationProvider
    from claritymed.orchestrator.services.chat_session import ChatSession

logger = logging.getLogger(__name__)

_UNKNOWN = "?"

# Sliding-window history budget (bytes of serialized pydantic-ai messages).
# ~80 KB is roughly 20–25 K tokens — generous for chat history while
# leaving room for system prompt + tool schemas + completion in a 128 K
# context model. When exceeded, oldest request/response pairs are dropped
# from the front and a ``mode.ask.history_trimmed`` audit event records
# how many messages went.
HISTORY_BUDGET_BYTES = 80_000
# Floor on how many messages we always keep, regardless of size. Two
# pairs (4 messages) preserves enough recent context for the model to
# stay coherent even with a pathologically long single turn.
HISTORY_MIN_KEEP = 4


# Matches the exact splice ``_compose_prompt`` emits when an Evidence
# block is present. Anchoring on both the ``Evidence (cite by [n]):``
# header AND the ``\n\nQuestion:`` separator keeps this conservative —
# we only strip blocks the framework itself produced, never a user
# message that happens to mention the phrase.
_EVIDENCE_BLOCK_RE = re.compile(
    r"\n*Evidence \(cite by \[n\]\):.*?\n\nQuestion:\s*",
    re.DOTALL,
)
# Citation marker like ``[1]`` / ``[12]``. We only consider it
# out-of-range — never strip legitimate bracketed text — by matching
# digits only.
_CITATION_RE = re.compile(r"\[(\d+)\]")


def _strip_evidence_block(text: str) -> str:
    """Remove one inline ``Evidence (cite by [n]):`` block from a stored prompt.

    pydantic-ai's message history captures the *full* request body we
    sent — including any retrieval evidence ``RagFeature.pre_invoke``
    spliced in. Carrying that into the next turn teaches the LLM that
    prior turns' citation indices (``[1]``..``[N]``) are still valid,
    which is how ``[11]`` showed up in an answer whose current turn only
    surfaced 2 sources. Strip the splice; the assistant's reply (which
    still references the indices) stays — that's fine, the indices are
    now opaque to the model.
    """
    return _EVIDENCE_BLOCK_RE.sub("", text, count=1)


def _sanitize_history_for_llm(messages: list, *, scrub=None) -> list:
    """Return a copy of message history with Evidence blocks removed.

    Only touches ``UserPromptPart`` content strings; multimodal content
    lists are passed through untouched (no Evidence splice path exists
    for them today). ``ModelRequest`` / ``UserPromptPart`` are
    dataclasses in pydantic-ai, so we ``dataclasses.replace`` rather
    than ``model_copy``.

    When ``scrub`` is provided (callable ``str -> str``), every
    ``UserPromptPart`` string is scrubbed in addition to the Evidence
    strip. This is the cloud-turn defense against cross-turn PHI replay:
    local-turn prompts persisted into ``messages_json`` carry raw user
    input, and switching providers mid-session would otherwise leak that
    history to the cloud LLM unscrubbed. The scrub callable is supplied
    by the caller (typically ``PhiGuard.scrub_free_text`` lambda) so
    this helper stays decoupled from the guard.
    """
    import dataclasses

    from pydantic_ai.messages import ModelRequest, UserPromptPart

    out: list = []
    for m in messages:
        if not isinstance(m, ModelRequest):
            out.append(m)
            continue
        new_parts: list = []
        changed = False
        for p in m.parts:
            if isinstance(p, UserPromptPart) and isinstance(p.content, str):
                cleaned = _strip_evidence_block(p.content)
                if scrub is not None:
                    cleaned = scrub(cleaned)
                if cleaned != p.content:
                    new_parts.append(dataclasses.replace(p, content=cleaned))
                    changed = True
                    continue
            new_parts.append(p)
        out.append(dataclasses.replace(m, parts=new_parts) if changed else m)
    return out


def _trim_message_history(messages, budget: int):
    """Drop oldest pairs until the serialized history fits the budget.

    Returns ``(trimmed, dropped_count)``. Trim works on request/response
    pairs (2 messages at a time) because dropping a lone request leaves
    the next response unanchored, which pydantic-ai rejects.

    Cost: at most O(log N) full serializations because we use bisection
    on the drop count when the initial estimate overshoots, and a single
    final serialization to confirm. The previous implementation
    re-encoded the entire kept list after every pair drop, which was
    quadratic in message count for big overshoots.
    """
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    if not messages:
        return messages, 0
    encoded_full = ModelMessagesTypeAdapter.dump_json(messages)
    if len(encoded_full) <= budget:
        return messages, 0
    max_drop_pairs = (len(messages) - HISTORY_MIN_KEEP) // 2
    if max_drop_pairs <= 0:
        return messages, 0
    # Bisect on the number of pairs to drop. Invariant: lo always fits
    # within budget (or equals 0), hi never fits. We start with the
    # cheapest serializations (largest drops) so the common case of
    # 'one extra-long turn pushed us slightly over' still costs only
    # 1-2 dumps.
    # Invariant: lo never fits (precondition for the full slice); hi
    # either fits or is the unknown upper bound (max_drop_pairs + 1).
    # We want the smallest drop_pairs that fits.
    lo, hi = 0, max_drop_pairs + 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        candidate = messages[2 * mid :]
        if len(ModelMessagesTypeAdapter.dump_json(candidate)) <= budget:
            hi = mid
        else:
            lo = mid
    # Clamp when even the maximum allowed drop doesn't fit — we still
    # respect the HISTORY_MIN_KEEP floor and let the caller decide what
    # to do with an over-budget kept slice.
    drop_pairs = min(hi, max_drop_pairs)
    return messages[2 * drop_pairs :], drop_pairs * 2


class AskService:
    """Drive ask mode: scrub, compose features into a turn, stream.

    Boundaries the plugins do not own:

    * PHI scrubbing before any LLM call (cloud / local invariant).
    * Plugin composition — gather deterministic pre-invoke text +
      register tool-mode callables on the agent.
    * Streaming loop — merge LLM token stream with tool-emitted events.
    * Chat-session persistence (user turn before, assistant after).
    * Sources block injection just before ``Done``.
    * ``mode.ask`` audit row including per-feature modes.
    """

    def __init__(
        self,
        model: "Model",
        guard: PhiGuard | None = None,
        language: str = "en",
        chat_session: "ChatSession | None" = None,
        provider_id: str = _UNKNOWN,
        model_name: str = _UNKNOWN,
        *,
        strategy: "RagStrategy | None" = None,
        provider_config: "ProviderConfig | None" = None,
        user_whitelist: list[str] | None = None,
        translation_service: "TranslationProvider | None" = None,
        rag_mode: str = "tool",
        features: "list[FeaturePlugin] | None" = None,
        prompt_channel: "PromptChannel | None" = None,
    ) -> None:
        self._model = model
        self._guard = guard or PhiGuard.from_config()
        self._language = language
        self._chat_session = chat_session
        self._provider_id = provider_id
        self._model_name = model_name
        self._strategy = strategy
        self._provider_config = provider_config
        self._user_whitelist = user_whitelist
        self._translation_service = translation_service
        # Host-supplied interaction channel for the ``ask_user_question``
        # tool. None means the host is non-interactive — the tool body
        # falls back to a plain-text hint to the LLM rather than
        # blocking on a UI that does not exist.
        self._prompt_channel = prompt_channel
        # PromptRegistry walks every YAML in the store on construction.
        # When the channel is wired up we build the tool every turn, so
        # cache the registry once instead of paying disk + Pydantic
        # validation cost on each ``run()``.
        self._prompt_registry = None
        # Plugins are stateless w.r.t. turn data — built once at startup.
        # Test paths can inject a pre-built list to assert dispatch
        # without going through the factory.
        self._features = (
            features
            if features is not None
            else build_features(rag_mode=rag_mode, rag_strategy=strategy)
        )
        # Snapshot per-feature modes for the audit row; the LLM-facing
        # tool list is computed per-turn from ``as_tool``.
        self._feature_modes = {f.name: f.mode for f in self._features}
        self._last_chunks: list = []

    @property
    def last_chunks(self) -> list:
        """Retrieved chunks from the most recent run() call (for testing)."""
        return self._last_chunks

    async def run(self, user_input: str, user_id: str) -> AsyncIterator[Event]:
        from claritymed.context import (
            apply_context,
            new_request_id,
            request_id_ctx,
            reset_context,
        )

        rid = request_id_ctx.get() or new_request_id()
        tokens = apply_context(rid, user_id, self._language)
        try:
            async for ev in self._run_inner(user_input, user_id):
                yield ev
        finally:
            reset_context(tokens)

    async def _run_inner(self, user_input: str, user_id: str) -> AsyncIterator[Event]:
        from opentelemetry import trace as otel_trace

        tracer = otel_trace.get_tracer("claritymed.ask")
        with tracer.start_as_current_span("ask.request"):
            async for ev in self._run_scoped(user_input, user_id):
                yield ev

    async def _run_scoped(self, user_input: str, user_id: str) -> AsyncIterator[Event]:
        from claritymed.context import attach_session_baggage, detach_session_baggage

        # PHI scrub only for cloud-bound turns.
        # scrub_free_text loads the ONNX session on first call (can take
        # 10-60 s); running it in a thread keeps the event loop free.
        is_cloud = getattr(self._provider_config, "kind", None) == "cloud"
        if is_cloud:
            scrubbed, report = await asyncio.to_thread(
                self._guard.scrub_free_text, user_input
            )
            audit_event(
                "mode.ask.scrub",
                payload={
                    "user_id": user_id,
                    "rule_hits": report.rule_hits,
                    "model_hits": report.model_hits,
                    "model_failed": report.model_failed,
                    "text_len_before": report.text_len_before,
                    "text_len_after": report.text_len_after,
                },
            )
            if report.model_failed:
                # privacy_filter.enabled=true is the safety contract.
                # If the model layer fails for a cloud-bound turn, fail
                # loud instead of silently leaking PHI the regex layer
                # missed. The user sees an Error event; nothing reaches
                # the LLM.
                logger.error(
                    "privacy-filter model failed on cloud turn; refusing to "
                    "send unscrubbed text to %s",
                    self._provider_id,
                )
                yield Error(
                    error_type="scrub_unavailable",
                    message=(
                        "Privacy filter is configured but unavailable; "
                        "refusing to send unscrubbed text to the cloud "
                        "provider. Switch to a local provider or fix the "
                        "filter setup, then retry."
                    ),
                    retryable=False,
                )
                return
        else:
            scrubbed = user_input

        output_lang = self._language

        deps = AskDeps(
            strategy=self._strategy,
            user_id=user_id,
            user_whitelist=self._user_whitelist,
            provider_config=self._provider_config,
            language=output_lang,
            translation_service=self._translation_service,
            mode=self._feature_modes.get("rag", "tool"),
            prompt_channel=self._prompt_channel,
        )

        # User turn first so the JSONL timeline reflects send order.
        if self._chat_session is not None:
            try:
                self._chat_session.append_user(scrubbed)
            except Exception:  # noqa: BLE001
                logger.exception("failed to append user turn to chat session")
        message_history = (
            self._chat_session.message_history() if self._chat_session else None
        )
        if message_history:
            message_history, dropped = _trim_message_history(
                message_history, HISTORY_BUDGET_BYTES
            )
            if dropped:
                audit_event(
                    "mode.ask.history_trimmed",
                    payload={
                        "user_id": user_id,
                        "dropped_messages": dropped,
                        "kept_messages": len(message_history),
                        "budget_bytes": HISTORY_BUDGET_BYTES,
                    },
                )

        session_token = (
            attach_session_baggage(self._chat_session.session_id)
            if self._chat_session is not None
            else None
        )
        result: dict = {
            "final_text": "",
            "messages_json": None,
            "usage": None,
            "steps": [],
            "latency": None,
            "had_error": False,
        }
        finalized = False
        try:
            async for event in self._stream_turn(
                scrubbed, deps, message_history, user_id, result
            ):
                if isinstance(event, Done):
                    # Finalize while still inside the generator so _finalize_turn
                    # runs before the consumer closes us on Done.  Moving this
                    # after `yield event` would make it dead code because the TUI
                    # returns immediately when it receives Done.
                    if not result["had_error"]:
                        # Clamp before persist so future turns' history
                        # never carries out-of-range markers forward.
                        from claritymed.core.rag.retrieval_pipeline import (
                            deduplicate_chunks,
                        )

                        valid_max = len(deduplicate_chunks(deps.retrieved_chunks))
                        cleaned, offending = self._clamp_citations(
                            result["final_text"], valid_max
                        )
                        if offending:
                            audit_event(
                                "ask.citation.out_of_range",
                                payload={
                                    "user_id": user_id,
                                    "valid_max": valid_max,
                                    "offending": offending,
                                },
                            )
                            result["final_text"] = cleaned
                            # The user already saw the bad markers in the
                            # streamed text. Emit a TokenChunk that lists
                            # the dropped indices so the UI can render a
                            # short correction line under the answer,
                            # before the Sources block. Keep the message
                            # ASCII so it round-trips in any locale.
                            offending_str = ", ".join(f"[{n}]" for n in offending)
                            yield TokenChunk(
                                text=(
                                    "\n\n*Note: the markers "
                                    f"{offending_str} above point to sources "
                                    f"beyond the {valid_max} listed below "
                                    "and have been removed from the saved "
                                    "transcript.*"
                                )
                            )
                        self._finalize_turn(user_id, result, deps)
                        finalized = True
                    if deps.retrieved_chunks:
                        self._last_chunks = list(deps.retrieved_chunks)
                        yield TokenChunk(
                            text=self._format_sources(
                                deps.retrieved_chunks, lang=self._language
                            )
                        )
                        if os.environ.get("CLARITYMED_DEBUG"):
                            yield TokenChunk(
                                text=self._format_debug_collections(
                                    deps.retrieved_chunks
                                )
                            )
                yield event
        finally:
            # When the consumer cancels mid-stream (TUI Esc) or the
            # generator is GC'd without ever seeing Done, persist a
            # short cancelled-turn record so resume sees the question
            # paired with an empty assistant reply rather than a
            # dangling user message. Best-effort: the chat session may
            # already be closed.
            if not finalized and self._chat_session is not None:
                try:
                    self._chat_session.append_assistant(
                        text=result["final_text"],
                        messages_json=result["messages_json"] or b"",
                        model=self._model_name,
                        provider_id=self._provider_id,
                        usage=result["usage"],
                        latency=result["latency"],
                        steps=result["steps"],
                        cancelled=True,
                    )
                    audit_event(
                        "mode.cancelled",
                        payload={
                            "user_id": user_id,
                            "session_id": self._chat_session.session_id,
                            "had_error": bool(result["had_error"]),
                        },
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "failed to persist cancelled turn for user %s", user_id
                    )
            detach_session_baggage(session_token)

    async def _stream_turn(
        self,
        scrubbed: str,
        deps: AskDeps,
        message_history,
        user_id: str,
        result: dict,
    ) -> AsyncIterator[Event]:
        """Compose feature plugins into one turn and stream events.

        Deterministic features run pre-LLM and contribute prompt text;
        tool features register callables on the agent. The same loop
        handles both shapes — the only branch is whether the prompt
        gets a prepended evidence block.
        """
        from pydantic_ai import UsageLimits

        turn_ctx = TurnContext(scrubbed=scrubbed, deps=deps)

        # Deterministic pre-invoke: ordered concatenation so a future
        # plugin can rely on stable layout (e.g. vision findings always
        # above RAG evidence).
        pre_blocks: list[str] = []
        for feature in self._features:
            if feature.mode != "deterministic":
                continue
            try:
                text = await feature.pre_invoke(turn_ctx)
            except Exception as exc:  # noqa: BLE001
                logger.exception("pre_invoke failed for %s", feature.name)
                yield Error(
                    error_type="config_error",
                    message=f"{feature.name}.pre_invoke: {exc}",
                    retryable=False,
                )
                result["had_error"] = True
                return
            if text:
                pre_blocks.append(text)
        # Drain any events the pre-invoke steps queued (RetrievalPending
        # etc) so the UI sees them before LlmCallStarted.
        while not deps.event_queue.empty():
            try:
                yield deps.event_queue.get_nowait()
            except Exception:  # noqa: BLE001
                break

        pre_text = "\n\n".join(pre_blocks)
        prompt = f"{pre_text}\n\nQuestion: {scrubbed}" if pre_text else scrubbed

        tools: list = [t for f in self._features if (t := f.as_tool()) is not None]
        # Register ``ask_user_question`` only when a channel is wired up.
        # Without a channel the tool would always return the "unavailable"
        # hint, which wastes a turn and shows up as noise in the LLM's
        # tool list — better to omit it entirely for one-shot CLI / eval
        # runs.
        if self._prompt_channel is not None:
            from claritymed.core.interaction import build_ask_user_question_tool
            from claritymed.core.prompts.registry import PromptRegistry

            if self._prompt_registry is None:
                self._prompt_registry = PromptRegistry()
            tools.append(
                build_ask_user_question_tool(
                    self._prompt_registry,
                    language=self._language,
                )
            )
        any_tool = bool(tools)
        agent = make_ask_agent(self._model, language=self._language, tools=tools)

        out: asyncio.Queue[Event | None] = asyncio.Queue()
        st: dict = {
            "t_start": time.perf_counter(),
            "t_first_token": None,
            "t_end": 0.0,
        }

        async def _producer() -> None:
            logger.debug("_producer: START")
            await out.put(
                LlmCallStarted(
                    model_name=self._model_name,
                    provider_id=self._provider_id,
                )
            )
            audit_event(
                "llm.call.start",
                payload={
                    "user_id": user_id,
                    "provider_id": self._provider_id,
                    "model": self._model_name,
                },
            )
            get_access_logger().info(
                "llm.call.start model=%s provider=%s",
                self._model_name,
                self._provider_id,
            )
            # ``UsageLimits`` only matters when at least one tool is
            # registered; without tools the LLM cannot loop. Keep the
            # backstop on tool turns so a misbehaving model cannot run
            # the retrieval pipeline 50× per turn.
            # When the current turn is cloud, re-scrub every prior
            # ``UserPromptPart`` in the carried history. Local turns
            # persist raw user text into ``messages_json``; without this
            # the next cloud turn replays unscrubbed PHI from earlier
            # turns. Recomputed here so the producer is self-contained
            # rather than closing over a flag from the outer scope.
            producer_is_cloud = getattr(self._provider_config, "kind", None) == "cloud"
            history_scrub = None
            if producer_is_cloud and message_history:

                def _history_scrub(s: str) -> str:
                    scrubbed, _r = self._guard.scrub_free_text(s)
                    return scrubbed

                history_scrub = _history_scrub
            stream_kwargs: dict = {
                "deps": deps,
                "message_history": (
                    _sanitize_history_for_llm(message_history, scrub=history_scrub)
                    if message_history
                    else None
                ),
            }
            if any_tool:
                stream_kwargs["usage_limits"] = UsageLimits(request_limit=5)
            try:
                logger.debug("_producer: ENTER agent.run_stream")
                async with agent.run_stream(prompt, **stream_kwargs) as stream:
                    logger.debug(
                        "_producer: agent.run_stream context entered, iterating stream_text"
                    )
                    async for chunk in stream.stream_text(delta=True):
                        if chunk:
                            if st["t_first_token"] is None:
                                st["t_first_token"] = time.perf_counter()
                                ttft_ms = int(
                                    (st["t_first_token"] - st["t_start"]) * 1000
                                )
                                logger.debug(
                                    "_producer: FIRST TOKEN ttft=%dms", ttft_ms
                                )
                                await out.put(LlmFirstToken(ttft_ms=ttft_ms))
                            await out.put(TokenChunk(text=chunk))
                    result["final_text"] = await stream.get_output()
                    try:
                        result["messages_json"] = stream.all_messages_json()
                    except Exception:  # noqa: BLE001
                        logger.exception("failed to capture pydantic-ai messages")
                    try:
                        result["usage"] = stream.usage
                    except Exception:  # noqa: BLE001
                        logger.exception("failed to capture pydantic-ai usage")
                    try:
                        result["steps"] = build_step_records(
                            list(stream.new_messages())
                        )
                    except Exception:  # noqa: BLE001
                        logger.exception("failed to build per-step records")
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "_producer: agent.run_stream raised %s: %s", type(exc).__name__, exc
                )
                result["had_error"] = True
                await out.put(
                    Error(error_type="llm_error", message=str(exc), retryable=True)
                )
            finally:
                logger.debug("_producer: FINALLY (sentinel → out)")
                st["t_end"] = time.perf_counter()
                await out.put(None)

        async def _drain_tools() -> None:
            # Forward tool-emitted events from the deps queue concurrently
            # with the text stream so RetrievalPending / ToolStarted / etc
            # surface immediately rather than piling up behind the next
            # text chunk. The earlier version polled at 10Hz with
            # ``wait_for(get(), timeout=0.1)``, burning ~300 spurious
            # scheduler entries per 30 s call. A plain ``await get()``
            # plus the consumer's ``cancel()`` in finally is sufficient,
            # and any unexpected exception is re-raised so the consumer
            # loop sees it instead of silently dying.
            while True:
                try:
                    ev = await deps.event_queue.get()
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("event-queue drain raised; re-raising")
                    raise
                await out.put(ev)

        producer_task = asyncio.create_task(_producer())
        drain_task = asyncio.create_task(_drain_tools())

        try:
            while True:
                ev = await out.get()
                if ev is None:
                    break
                yield ev
            # Normal completion: flush any events still in the queues
            # so trailing telemetry (ToolCompleted, Steps, etc.) is not
            # lost. Done OUTSIDE the finally — yielding from a finally
            # block raises ``RuntimeError: async generator ignored
            # GeneratorExit`` when the consumer cancels (TUI Esc),
            # which previously crashed every cancelled turn.
            while not out.empty():
                try:
                    ev = out.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if ev is not None:
                    yield ev
            while not deps.event_queue.empty():
                try:
                    yield deps.event_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
        finally:
            # Cancellation cleanup only — never yield here. On cancel
            # we drop pending events: the consumer is gone, and these
            # events are non-critical telemetry (the audit row was
            # written in _finalize_turn before Done).
            producer_task.cancel()
            drain_task.cancel()
            await asyncio.gather(producer_task, drain_task, return_exceptions=True)

        t_end = st["t_end"]
        t_start = st["t_start"]
        t_first_token = st["t_first_token"]
        result["latency"] = LatencyTrace(
            total_ms=int((t_end - t_start) * 1000),
            ttft_ms=(
                int((t_first_token - t_start) * 1000)
                if t_first_token is not None
                else None
            ),
            completion_ms=(
                int((t_end - t_first_token) * 1000)
                if t_first_token is not None
                else None
            ),
        )
        if result["had_error"]:
            return
        yield Done(final=result["final_text"])

    def _finalize_turn(self, user_id: str, result: dict, deps: AskDeps) -> None:
        """Audit + chat-session persistence after the stream closes."""
        latency = result["latency"]
        if self._chat_session is not None and result["messages_json"] is not None:
            try:
                self._chat_session.append_assistant(
                    text=result["final_text"],
                    messages_json=result["messages_json"],
                    model=self._model_name,
                    provider_id=self._provider_id,
                    usage=result["usage"],
                    latency=latency,
                    steps=result["steps"],
                )
            except Exception:  # noqa: BLE001
                logger.exception("failed to append assistant turn to chat session")

        self._maybe_audit_announced_but_skipped(user_id, result, deps)

        payload: dict[str, object] = {
            "user_id": user_id,
            "answer_len": len(result["final_text"]),
            "model": self._model_name,
            "provider_id": self._provider_id,
            "latency_ms": latency.total_ms if latency else 0,
            "features": dict(self._feature_modes),
        }
        if latency and latency.ttft_ms is not None:
            payload["ttft_ms"] = latency.ttft_ms
        if latency and latency.completion_ms is not None:
            payload["completion_ms"] = latency.completion_ms
        if result["usage"] is not None:
            payload.update(_usage_dict(result["usage"]))
        if result["steps"]:
            payload["steps"] = result["steps"]
        if self._chat_session is not None:
            payload["session_id"] = self._chat_session.session_id
        if deps.tool_calls:
            payload["tool_calls"] = dict(deps.tool_calls)
        audit_event("mode.ask", payload=payload)
        ttft_str = (
            f" ttft={latency.ttft_ms}ms"
            if latency and latency.ttft_ms is not None
            else ""
        )
        tok_str = (
            f" tokens={result['usage'].total_tokens}"
            if result["usage"] is not None
            and getattr(result["usage"], "total_tokens", None)
            else ""
        )
        get_access_logger().info(
            "llm.call.done total=%dms%s%s model=%s",
            latency.total_ms if latency else 0,
            ttft_str,
            tok_str,
            self._model_name,
        )

    def _maybe_audit_announced_but_skipped(
        self, user_id: str, result: dict, deps: AskDeps
    ) -> None:
        """Emit ``mode.ask.tool_announced_but_skipped`` when warranted.

        Skipped if RAG is not in tool mode (deterministic always
        retrieves; agentic is disabled). Otherwise: regex the final
        text; if a match is present and the retrieve tool call count
        is still zero, emit one audit row. The matched snippet is NOT
        recorded — the LLM may have paraphrased user PHI back, and the
        signal we actually need (a count by provider/model) does not
        require the text. The pattern itself is captured for triage.
        """
        if self._feature_modes.get("rag") != "tool":
            return
        if deps.tool_calls.get("retrieve_medical_literature", 0) > 0:
            return
        snippet = detect_announcement(result["final_text"])
        if not snippet:
            return
        payload: dict[str, object] = {
            "user_id": user_id,
            "tool": "retrieve_medical_literature",
            "model": self._model_name,
            "provider_id": self._provider_id,
            "snippet_len": len(snippet),
        }
        if self._chat_session is not None:
            payload["session_id"] = self._chat_session.session_id
        audit_event("mode.ask.tool_announced_but_skipped", payload=payload)

    @staticmethod
    def _clamp_citations(text: str, max_n: int) -> tuple[str, list[int]]:
        """Strip ``[N]`` markers where ``N`` exceeds ``max_n``.

        Defense-in-depth against citation hallucination. ``max_n`` is the
        number of deduplicated Sources entries this turn — that's what
        the user sees, so any ``[N]`` above that is unmoored. Returns
        the cleaned text plus the sorted, deduplicated list of offending
        indices so the caller can audit them.

        We strip rather than substitute ``[?]`` because the stripped
        form reads cleanly when the user re-opens the conversation; an
        in-line ``[?]`` is just visual noise once the streaming UI has
        already rendered the original (incorrect) bracket.
        """
        bad: list[int] = []

        def _sub(m: "re.Match[str]") -> str:
            n = int(m.group(1))
            if n > max_n:
                bad.append(n)
                return ""
            return m.group(0)

        cleaned = _CITATION_RE.sub(_sub, text)
        return cleaned, sorted(set(bad))

    @staticmethod
    def _format_evidence(chunks: "list[RetrievedChunk]") -> str:
        from claritymed.core.rag.retrieval_pipeline import format_evidence

        return format_evidence(chunks)

    @staticmethod
    def _collection_label(chunk: "RetrievedChunk", lang: str | None) -> str:
        """Look up the user-facing label for a chunk's corpus.

        ``user_rag`` collapses to one shared label (e.g. ``My Library``)
        across every per-user store — ``user_rag_alice`` etc. is an
        internal id we never surface. ``system_rag`` chunks look up
        ``rag.collection.<collection_name>`` and fall back to the raw
        ``collection_name`` when no translation exists, so adding a new
        corpus needs only an i18n entry, never a code change.
        """
        from claritymed.core.i18n import t

        if chunk.source == "user_rag":
            key = "rag.collection.user_rag"
            label = t(key, lang=lang)
            return label if label != key else "My Library"
        name = chunk.collection_name or "source"
        key = f"rag.collection.{name}"
        label = t(key, lang=lang)
        return label if label != key else name

    @staticmethod
    def _format_sources(chunks: "list[RetrievedChunk]", lang: str | None = None) -> str:
        """Build the user-facing Sources section.

        Layout: ``[N] <title-or-uri> · <corpus-label>``. The corpus
        label comes from i18n (``configs/i18n/*.yaml`` under
        ``rag.collection.*``) so ``my library`` translates to ``我的
        资料库`` automatically and new corpora can be added without
        editing Python.

        This block is appended *after* ``_finalize_turn`` has persisted
        ``result["final_text"]``, so the corpus label never enters chat
        history and the LLM cannot mimic it on the next turn.
        """
        if not chunks:
            return ""
        from claritymed.core.rag.retrieval_pipeline import deduplicate_chunks

        unique = deduplicate_chunks(chunks)
        lines = ["\n\n**Sources:**"]
        for i, c in enumerate(unique, start=1):
            title = c.source_uri or c.doc_title
            corpus = AskService._collection_label(c, lang)
            display = f"{title} · {corpus}" if title else corpus
            lines.append(f"- [{i}] {display}")
        return "\n".join(lines)

    @staticmethod
    def _format_debug_collections(chunks: "list[RetrievedChunk]") -> str:
        """Markdown block listing the RAG-collection backing each Sources entry.

        Only emitted when ``CLARITYMED_DEBUG=1``. Indices match the
        Sources block (same dedup-by-doc_id, same first-seen order) so a
        reader can map ``[N]`` in the answer ↔ Sources ``[N]`` ↔ Debug
        ``[N]``. When more than one chunk of the same doc passed
        retrieval, the line reports the best score and a chunk count
        rather than spawning a second row — that information lives in
        the ``rag.retrieval`` audit row already.
        """
        lines = ["\n\n---\n**Debug — RAG Collections:**"]
        groups: dict[str, list] = {}
        order: list[str] = []
        for c in chunks:
            if c.doc_id not in groups:
                groups[c.doc_id] = []
                order.append(c.doc_id)
            groups[c.doc_id].append(c)
        for i, doc_id in enumerate(order, start=1):
            members = groups[doc_id]
            first = members[0]
            col = first.collection_name or "unknown"
            scores = [
                c.rerank_score if c.rerank_score is not None else c.score
                for c in members
            ]
            scores = [s for s in scores if s is not None]
            best = max(scores) if scores else None
            score_str = f"{best:.3f}" if best is not None else "—"
            n = len(members)
            chunk_suffix = f", {n} chunks" if n > 1 else ""
            doc = first.doc_id or "—"
            lines.append(f"- [{i}] `{col}` · `{doc}` (score {score_str}{chunk_suffix})")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _compose_prompt(scrubbed: str, evidence_block: str) -> str:
        if not evidence_block:
            return scrubbed
        return f"{evidence_block}\n\nQuestion: {scrubbed}"
