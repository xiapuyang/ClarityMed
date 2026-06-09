"""Ask service: PHI scrub on input + stream LLM tokens to the caller."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from claritymed.core.observability.audit import audit_event
from claritymed.core.observability.logging import get_access_logger
from claritymed.orchestrator import PhiGuard
from claritymed.orchestrator.agents import make_ask_agent
from claritymed.orchestrator.agents.ask_deps import AskDeps
from claritymed.orchestrator.services.chat_session import (
    LatencyTrace,
    _usage_dict,
    build_step_records,
)
from claritymed.orchestrator.services.events import (
    Done,
    Error,
    Event,
    LlmCallStarted,
    LlmFirstToken,
    TokenChunk,
)

if TYPE_CHECKING:
    from pydantic_ai.models import Model

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


def _trim_message_history(messages, budget: int):
    """Drop oldest pairs until the serialized history fits the budget.

    Returns ``(trimmed, dropped_count)``. Trim works on request/response
    pairs (2 messages at a time) because dropping a lone request leaves
    the next response unanchored, which pydantic-ai rejects.
    """
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    if not messages:
        return messages, 0
    encoded = ModelMessagesTypeAdapter.dump_json(messages)
    if len(encoded) <= budget:
        return messages, 0
    keep = list(messages)
    dropped = 0
    while len(keep) > HISTORY_MIN_KEEP:
        candidate = keep[2:]
        if not candidate:
            break
        keep = candidate
        dropped += 2
        if len(ModelMessagesTypeAdapter.dump_json(keep)) <= budget:
            break
    return keep, dropped


class AskService:
    """Drive ask mode: scrub user input, stream LLM tokens, finalize.

    The service owns three boundaries the agent does not:

    * PHI scrubbing before any LLM call (cloud / local invariant).
    * Token + latency accounting — captured from
      ``StreamedRunResult.usage()`` and a perf_counter span, then emitted
      both to the audit log (``mode.ask``) and to the chat session log
      (assistant event payload).
    * Chat session persistence — handing the prior ``message_history`` to
      ``Agent.run_stream`` for multi-turn context, then appending the new
      assistant turn back to the session JSONL.
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

        # Reuse the caller's request_id when one is already in context (TUI app
        # set it on the status bar; Typer's inject_context set it at CLI start).
        # Generating a fresh one here would silently desynchronise the status
        # bar and the audit log — the user would see one id, grep would find a
        # different one.
        rid = request_id_ctx.get() or new_request_id()
        tokens = apply_context(rid, user_id, self._language)
        try:
            async for ev in self._run_inner(user_input, user_id):
                yield ev
        finally:
            reset_context(tokens)

    async def _run_inner(self, user_input: str, user_id: str) -> AsyncIterator[Event]:
        from opentelemetry import trace as otel_trace

        # Open a request-scoped root span so both the scrub audit and the
        # final mode.ask audit pick up the same trace_id. When tracing is
        # off, get_tracer returns a no-op tracer and the span is invalid —
        # audit lines fall back to trace_id=null exactly as before.
        tracer = otel_trace.get_tracer("claritymed.ask")
        with tracer.start_as_current_span("ask.request"):
            async for ev in self._run_scoped(user_input, user_id):
                yield ev

    async def _run_scoped(self, user_input: str, user_id: str) -> AsyncIterator[Event]:
        from claritymed.context import attach_session_baggage, detach_session_baggage

        # Scrub PHI from the prompt only when the request will leave the local
        # machine.  Local providers run on-device and never transmit data, so
        # scrubbing degrades answer quality for no privacy gain.
        is_cloud = getattr(self._provider_config, "kind", None) == "cloud"
        if is_cloud:
            scrubbed, report = self._guard.scrub_free_text(user_input)
            audit_event(
                "mode.ask.scrub",
                payload={
                    "user_id": user_id,
                    "rule_hits": report.rule_hits,
                    "text_len_before": report.text_len_before,
                    "text_len_after": report.text_len_after,
                },
            )
        else:
            scrubbed = user_input

        output_lang = self._language

        # Build deps so the retrieve_medical_literature tool can access
        # the strategy, PHI state, and translation service, and so the
        # service can collect retrieved chunks for the Sources block after
        # the LLM finishes.
        # ``is_agentic`` is set by ``build_strategy`` on the
        # NaiveHybridStrategy returned for the agentic catalog entry; it
        # rides along on the strategy instance rather than the config so
        # the tool sees only the retrieval surface it needs.
        deps = AskDeps(
            strategy=self._strategy,
            user_id=user_id,
            user_whitelist=self._user_whitelist,
            provider_config=self._provider_config,
            language=output_lang,
            translation_service=self._translation_service,
            agentic=bool(getattr(self._strategy, "is_agentic", False)),
        )

        # Record the user turn first so the JSONL timeline reflects send
        # order, then pull the prior history for the LLM call. The chat
        # session stores the scrubbed user text (no evidence) so re-loading
        # a session does not re-inject yesterday's evidence.
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

        # Stamp ``claritymed.session_id`` baggage onto every span the LLM
        # call produces so Phoenix can group traces by conversation, not
        # just by request. Detach in the finally so a different session
        # cannot accidentally inherit this one's id on the next turn.
        session_token = (
            attach_session_baggage(self._chat_session.session_id)
            if self._chat_session is not None
            else None
        )
        try:
            async for event in self._run_with_agent(
                scrubbed, message_history, user_id, deps, output_lang=output_lang
            ):
                # Inject source and debug blocks just before Done so the TUI's
                # event loop processes them while the stream is still open.
                # Yielding after Done would be dropped — the TUI returns on Done.
                if isinstance(event, Done) and deps.retrieved_chunks:
                    self._last_chunks = list(deps.retrieved_chunks)
                    yield TokenChunk(text=self._format_sources(deps.retrieved_chunks))
                    if os.environ.get("CLARITYMED_DEBUG"):
                        yield TokenChunk(
                            text=self._format_debug_collections(deps.retrieved_chunks)
                        )
                yield event
        finally:
            detach_session_baggage(session_token)

    async def _run_with_agent(
        self,
        scrubbed: str,
        message_history,
        user_id: str,
        deps: AskDeps,
        output_lang: str | None = None,
    ) -> AsyncIterator[Event]:
        from pydantic_ai import UsageLimits

        agent = make_ask_agent(self._model, language=output_lang or self._language)

        # Merged output queue: both the stream producer and the concurrent tool
        # event drainer write here.  This lets RetrievalPending / ToolStarted /
        # etc. appear in the TUI immediately while stream_text is blocked
        # waiting for the tool to complete — without it, those events only
        # surface after the LLM emits its next text chunk.
        out: asyncio.Queue[Event | None] = asyncio.Queue()

        # Mutable state captured by the inner producer coroutine.
        _st: dict = {
            "messages_json": None,
            "usage": None,
            "steps": [],
            "final_text": "",
            "t_start": time.perf_counter(),
            "t_first_token": None,
            "t_end": 0.0,
            "had_error": False,
        }

        async def _producer() -> None:
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
            try:
                async with agent.run_stream(
                    scrubbed,
                    deps=deps,
                    message_history=message_history or None,
                    usage_limits=UsageLimits(request_limit=5),
                ) as stream:
                    async for chunk in stream.stream_text(delta=True):
                        if chunk:
                            if _st["t_first_token"] is None:
                                _st["t_first_token"] = time.perf_counter()
                                ttft_ms = int(
                                    (_st["t_first_token"] - _st["t_start"]) * 1000
                                )
                                await out.put(LlmFirstToken(ttft_ms=ttft_ms))
                            await out.put(TokenChunk(text=chunk))
                    _st["final_text"] = await stream.get_output()
                    try:
                        _st["messages_json"] = stream.all_messages_json()
                    except Exception:  # noqa: BLE001
                        logger.exception("failed to capture pydantic-ai messages")
                    try:
                        _st["usage"] = stream.usage
                    except Exception:  # noqa: BLE001
                        logger.exception("failed to capture pydantic-ai usage")
                    try:
                        _st["steps"] = build_step_records(list(stream.new_messages()))
                    except Exception:  # noqa: BLE001
                        logger.exception("failed to build per-step records")
            except Exception as exc:  # noqa: BLE001 — surface as event
                _st["had_error"] = True
                await out.put(
                    Error(
                        error_type="llm_error",
                        message=str(exc),
                        retryable=True,
                    )
                )
            finally:
                _st["t_end"] = time.perf_counter()
                await out.put(None)  # sentinel: producer done

        async def _drain_tools() -> None:
            # Forward tool events from deps.event_queue to out concurrently
            # with the text stream so they appear immediately in the TUI
            # rather than piling up until the next text chunk arrives.
            while True:
                try:
                    ev = await asyncio.wait_for(deps.event_queue.get(), timeout=0.1)
                    await out.put(ev)
                except asyncio.TimeoutError:
                    pass

        producer_task = asyncio.create_task(_producer())
        drain_task = asyncio.create_task(_drain_tools())

        try:
            while True:
                ev = await out.get()
                if ev is None:
                    break
                yield ev
        finally:
            producer_task.cancel()
            drain_task.cancel()
            await asyncio.gather(producer_task, drain_task, return_exceptions=True)
            # Flush events that drain_task put into out after the sentinel,
            # and any events still sitting in deps.event_queue.
            while not out.empty():
                try:
                    ev = out.get_nowait()
                    if ev is not None:
                        yield ev
                except asyncio.QueueEmpty:
                    break
            while not deps.event_queue.empty():
                try:
                    yield deps.event_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        if _st["had_error"]:
            return

        t_start = _st["t_start"]
        t_first_token = _st["t_first_token"]
        t_end = _st["t_end"]
        final_text = _st["final_text"]
        messages_json = _st["messages_json"]
        usage = _st["usage"]
        steps = _st["steps"]

        latency = LatencyTrace(
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

        if self._chat_session is not None and messages_json is not None:
            try:
                self._chat_session.append_assistant(
                    text=final_text,
                    messages_json=messages_json,
                    model=self._model_name,
                    provider_id=self._provider_id,
                    usage=usage,
                    latency=latency,
                    steps=steps,
                )
            except Exception:  # noqa: BLE001
                logger.exception("failed to append assistant turn to chat session")

        payload: dict[str, object] = {
            "user_id": user_id,
            "answer_len": len(final_text),
            "model": self._model_name,
            "provider_id": self._provider_id,
            "latency_ms": latency.total_ms,
            "agentic": deps.agentic,
        }
        if latency.ttft_ms is not None:
            payload["ttft_ms"] = latency.ttft_ms
        if latency.completion_ms is not None:
            payload["completion_ms"] = latency.completion_ms
        if usage is not None:
            payload.update(_usage_dict(usage))
        if steps:
            payload["steps"] = steps
        if self._chat_session is not None:
            payload["session_id"] = self._chat_session.session_id
        audit_event("mode.ask", payload=payload)
        ttft_str = f" ttft={latency.ttft_ms}ms" if latency.ttft_ms is not None else ""
        tok_str = (
            f" tokens={usage.total_tokens}"
            if usage is not None and getattr(usage, "total_tokens", None)
            else ""
        )
        get_access_logger().info(
            "llm.call.done total=%dms%s%s model=%s",
            latency.total_ms,
            ttft_str,
            tok_str,
            self._model_name,
        )
        yield Done(final=final_text)

    @staticmethod
    def _format_evidence(chunks: "list[RetrievedChunk]") -> str:
        from claritymed.orchestrator.tools.retrieve_medical_literature import (
            format_evidence,
        )

        return format_evidence(chunks)

    @staticmethod
    def _format_sources(chunks: "list[RetrievedChunk]") -> str:
        """Build an authoritative Sources section from chunk metadata.

        Uses source_uri (a real URL) when available. Falls back to doc_title
        (article title stored during ingest), then collection_name, then source type.
        The list mirrors the [N] numbering in the injected evidence block, so
        the LLM's inline citations resolve correctly.
        """
        if not chunks:
            return ""
        lines = ["\n\n**Sources:**"]
        for i, c in enumerate(chunks, start=1):
            src = c.source_uri or c.doc_title or c.collection_name or c.source
            lines.append(f"- [{i}] {src}")
        return "\n".join(lines)

    @staticmethod
    def _format_debug_collections(chunks: "list[RetrievedChunk]") -> str:
        """Append-only markdown block naming the RAG collection for each
        cited chunk. Emitted as a trailing TokenChunk only when
        CLARITYMED_DEBUG=1, so it never appears in production responses."""
        lines = ["\n\n---\n**Debug — RAG Collections:**"]
        for i, c in enumerate(chunks, start=1):
            col = c.collection_name or "unknown"
            effective_score = c.rerank_score if c.rerank_score is not None else c.score
            score = f"{effective_score:.3f}" if effective_score is not None else "—"
            doc = c.doc_id or "—"
            lines.append(f"- [{i}] `{col}` · `{doc}` (score {score})")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _compose_prompt(scrubbed: str, evidence_block: str) -> str:
        if not evidence_block:
            return scrubbed
        return f"{evidence_block}\n\nQuestion: {scrubbed}"
