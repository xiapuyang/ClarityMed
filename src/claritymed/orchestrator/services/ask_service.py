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

        deps = AskDeps(
            strategy=self._strategy,
            user_id=user_id,
            user_whitelist=self._user_whitelist,
            provider_config=self._provider_config,
            language=output_lang,
            translation_service=self._translation_service,
            mode=self._feature_modes.get("rag", "tool"),
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
        try:
            async for event in self._stream_turn(
                scrubbed, deps, message_history, user_id, result
            ):
                # Inject sources just before Done so the TUI's event loop
                # processes them while the stream is still open.
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

        if result["had_error"]:
            return
        self._finalize_turn(user_id, result, deps)

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

        tools = [t for f in self._features if (t := f.as_tool()) is not None]
        any_tool = bool(tools)
        agent = make_ask_agent(self._model, language=self._language, tools=tools)

        out: asyncio.Queue[Event | None] = asyncio.Queue()
        st: dict = {
            "t_start": time.perf_counter(),
            "t_first_token": None,
            "t_end": 0.0,
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
            # ``UsageLimits`` only matters when at least one tool is
            # registered; without tools the LLM cannot loop. Keep the
            # backstop on tool turns so a misbehaving model cannot run
            # the retrieval pipeline 50× per turn.
            stream_kwargs: dict = {
                "deps": deps,
                "message_history": message_history or None,
            }
            if any_tool:
                stream_kwargs["usage_limits"] = UsageLimits(request_limit=5)
            try:
                async with agent.run_stream(prompt, **stream_kwargs) as stream:
                    async for chunk in stream.stream_text(delta=True):
                        if chunk:
                            if st["t_first_token"] is None:
                                st["t_first_token"] = time.perf_counter()
                                ttft_ms = int(
                                    (st["t_first_token"] - st["t_start"]) * 1000
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
                result["had_error"] = True
                await out.put(
                    Error(error_type="llm_error", message=str(exc), retryable=True)
                )
            finally:
                st["t_end"] = time.perf_counter()
                await out.put(None)

        async def _drain_tools() -> None:
            # Forward tool-emitted events from the deps queue concurrently
            # with the text stream so RetrievalPending / ToolStarted / etc
            # surface immediately rather than piling up behind the next
            # text chunk.
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
        is still zero, emit one audit row with the matched snippet so
        a human can sanity-check and grow the pattern list.
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
            "snippet": snippet[:120],
        }
        if self._chat_session is not None:
            payload["session_id"] = self._chat_session.session_id
        audit_event("mode.ask.tool_announced_but_skipped", payload=payload)

    @staticmethod
    def _format_evidence(chunks: "list[RetrievedChunk]") -> str:
        from claritymed.core.rag.retrieval_pipeline import format_evidence

        return format_evidence(chunks)

    @staticmethod
    def _format_sources(chunks: "list[RetrievedChunk]") -> str:
        """Build an authoritative Sources section from chunk metadata.

        Uses source_uri (a real URL) when available. Falls back to doc_title
        (article title stored during ingest), then collection_name, then source type.
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
        cited chunk. Only emitted when ``CLARITYMED_DEBUG=1``."""
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
