"""Ask service: PHI scrub on input + stream LLM tokens to the caller."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from claritymed.core.observability.audit import audit_event
from claritymed.orchestrator import PhiGuard
from claritymed.orchestrator.agents import make_ask_agent
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
    RetrievalCompleted,
    RetrievalFiltered,
    RetrievalPending,
    RetrievalStarted,
    TokenChunk,
    ToolCompleted,
    ToolStarted,
)

if TYPE_CHECKING:
    from pydantic_ai.models import Model
    from pydantic_ai.usage import RunUsage

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

        # R18: scrub PHI from the prompt before any LLM call.
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

        # CLARITYMED_AUTO_LANGUAGE=1: detect output language from the input
        # rather than from the --lang setting. Useful when users mix languages
        # in a single session. Default: output follows the configured language.
        output_lang = self._language
        if os.environ.get("CLARITYMED_AUTO_LANGUAGE"):
            from claritymed.core.translation import detect_language

            output_lang = detect_language(scrubbed)

        # RAG retrieval (Unit 8): if a strategy is configured, fetch evidence
        # before calling the LLM. Cloud providers filter PHI chunks at the
        # Qdrant query layer (only_cloud_safe=True); local providers keep
        # everything so user-uploaded PHI can ground the answer.
        evidence_block = ""
        evidence_chunks: list = []
        async for ev in self._maybe_retrieve(scrubbed, user_id):
            if isinstance(ev, _EvidenceReady):
                evidence_block = ev.text
                evidence_chunks = ev.chunks
            else:
                yield ev
        prompt = self._compose_prompt(scrubbed, evidence_block)

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
                prompt, message_history, user_id, output_lang=output_lang
            ):
                # Inject source and debug blocks just before Done so the TUI's
                # event loop processes them while the stream is still open.
                # Yielding after Done would be dropped — the TUI returns on Done.
                if isinstance(event, Done) and evidence_chunks:
                    yield TokenChunk(text=self._format_sources(evidence_chunks))
                    if os.environ.get("CLARITYMED_DEBUG"):
                        yield TokenChunk(
                            text=self._format_debug_collections(evidence_chunks)
                        )
                yield event
        finally:
            detach_session_baggage(session_token)

    async def _run_with_agent(
        self,
        scrubbed: str,
        message_history,
        user_id: str,
        output_lang: str | None = None,
    ) -> AsyncIterator[Event]:
        agent = make_ask_agent(self._model, language=output_lang or self._language)
        messages_json: bytes | None = None
        usage: RunUsage | None = None
        steps: list[dict] = []
        final_text: str = ""
        t_start = time.perf_counter()
        t_first_token: float | None = None
        # Emit before agent.run_stream so the UI shows 'generating…' during
        # the LLM's TTFT window. On large local models (Qwen 35B on MLX) TTFT
        # can hit 3 minutes — without this event the assistant bubble looks
        # frozen since RAG completes in <1s and tokens don't start until much
        # later.
        yield LlmCallStarted(
            model_name=self._model_name,
            provider_id=self._provider_id,
        )
        audit_event(
            "llm.call.start",
            payload={
                "user_id": user_id,
                "provider_id": self._provider_id,
                "model": self._model_name,
            },
        )
        try:
            async with agent.run_stream(
                scrubbed, message_history=message_history or None
            ) as stream:
                async for chunk in stream.stream_text(delta=True):
                    if chunk:
                        if t_first_token is None:
                            t_first_token = time.perf_counter()
                            ttft_ms = int((t_first_token - t_start) * 1000)
                            yield LlmFirstToken(ttft_ms=ttft_ms)
                        yield TokenChunk(text=chunk)
                final_text = await stream.get_output()
                # Capture inside the ``with`` block — the stream goes out of
                # scope once it exits and the messages disappear with it.
                try:
                    messages_json = stream.all_messages_json()
                except Exception:  # noqa: BLE001
                    logger.exception("failed to capture pydantic-ai messages")
                try:
                    usage = stream.usage
                except Exception:  # noqa: BLE001
                    logger.exception("failed to capture pydantic-ai usage")
                try:
                    steps = build_step_records(list(stream.new_messages()))
                except Exception:  # noqa: BLE001
                    logger.exception("failed to build per-step records")
        except Exception as exc:  # noqa: BLE001 — surface as event
            yield Error(
                error_type="llm_error",
                message=str(exc),
                retryable=True,
            )
            return
        t_end = time.perf_counter()
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
        yield Done(final=final_text)

    # --- retrieval seam ------------------------------------------------

    def _collection_target_language(self) -> str | None:
        """Return the language to translate the query INTO for embedding.

        Looks at all cross-lingual system collections. When their language
        differs from the session language, embedding quality suffers —
        translating the query to the collection's language before embed
        closes most of that gap. Returns None when no mismatch exists
        (no translation needed) or when the config cannot be read.
        """
        try:
            from claritymed.core.rag.schemas import load_retrieval_config

            cfg = load_retrieval_config()
            mismatched = [
                c.language
                for c in cfg.system_rag.collections
                if c.cross_lingual and c.language != self._language
            ]
            if not mismatched:
                return None
            return max(set(mismatched), key=mismatched.count)
        except Exception:  # noqa: BLE001
            logger.warning("could not determine collection target language")
            return None

    async def _maybe_retrieve(
        self, scrubbed_query: str, user_id: str
    ) -> AsyncIterator[Event]:
        """Stream RetrievalStarted/Filtered/Completed events and stash the
        evidence_block on a private sentinel so the caller can splice it
        into the prompt without losing event ordering.
        """
        if self._strategy is None:
            return
        from claritymed.core.rag.strategies.base import RetrievalContext

        only_cloud_safe = self._is_cloud_provider()

        # Translate the query to the collection's native language before embedding.
        # BGE-M3 similarity is significantly lower for cross-lingual pairs;
        # translating closes that gap. Direction is derived from system collections
        # (e.g. statpearls_en language="en" + zh session → translate zh→en;
        # a hypothetical statpearls_zh would trigger the reverse). ctx.language
        # stays as the session language so CollectionRouter routing is unchanged.
        # Fires automatically when a TranslationService is configured and the
        # session language differs from any cross_lingual collection — no env var needed.
        embedding_query = scrubbed_query
        if self._translation_service:
            target_lang = self._collection_target_language()
            if target_lang:
                from claritymed.core.observability.steps import capture_steps

                with capture_steps() as translation_steps:
                    embedding_query = await self._translation_service.translate_query(
                        scrubbed_query,
                        target_lang=target_lang,  # type: ignore[arg-type]
                    )
                for rec in translation_steps:
                    yield ToolStarted(tool_name=rec.name, args_preview=rec.details)
                    yield ToolCompleted(
                        tool_name=rec.name,
                        duration_ms=rec.duration_ms,
                        summary=rec.summary if not rec.failed else "failed",
                    )

        ctx = RetrievalContext(
            query=embedding_query,
            user_id=user_id,
            language=self._language,  # type: ignore[arg-type]
            user_whitelist=self._user_whitelist,
            only_cloud_safe=only_cloud_safe,
        )
        # Emit the pending event *before* the await — the embed + search +
        # rerank pipeline is the longest stretch of any RAG turn (typically
        # 1-5s on local Qdrant). Without this signal the UI just sits on an
        # empty assistant bubble for the duration.
        yield RetrievalPending()
        try:
            bundle = await self._strategy.retrieve(ctx)
        except Exception as exc:  # noqa: BLE001 — surface as audit + event
            logger.exception("retrieval failed")
            audit_event(
                "rag.retrieval.failed",
                payload={"user_id": user_id, "error": str(exc)[:200]},
            )
            yield Error(
                error_type="retrieval_failed",
                message=str(exc),
                retryable=True,
            )
            return
        yield RetrievalStarted(
            active_collections=bundle.trace.active_collections,
            strategy=bundle.trace.strategy,
        )

        # Cloud-safe filter at the provider boundary. The Qdrant layer
        # already pre-filters when only_cloud_safe=True, but PhiGuard's
        # filter_chunks_for_provider is the canonical defense-in-depth check.
        chunks_pre = list(bundle.chunks)
        safe_chunks = self._filter_for_provider(chunks_pre)
        filtered = len(chunks_pre) - len(safe_chunks)
        if filtered > 0:
            yield RetrievalFiltered(
                total=len(chunks_pre),
                kept=len(safe_chunks),
                filtered_phi=filtered,
                reason="cloud_provider_phi_guard",
            )

        if bundle.trace.rerank_fallback:
            audit_event(
                "rag.rerank.fallback",
                payload={
                    "user_id": user_id,
                    "collections": bundle.trace.active_collections,
                },
            )

        yield RetrievalCompleted(
            num_chunks=len(safe_chunks),
            fallback_triggered=bundle.trace.fallback_triggered,
            rerank_fallback=bundle.trace.rerank_fallback,
            embed_ms=bundle.trace.embed_ms,
            search_ms=bundle.trace.search_ms,
            rerank_ms=bundle.trace.rerank_ms,
            parent_expand_ms=bundle.trace.parent_expand_ms,
        )

        audit_event(
            "rag.retrieval",
            payload={
                "user_id": user_id,
                "strategy": bundle.trace.strategy,
                "active_collections": bundle.trace.active_collections,
                "num_chunks": len(safe_chunks),
                "filtered_phi": filtered,
                "fallback_triggered": bundle.trace.fallback_triggered,
                "rerank_fallback": bundle.trace.rerank_fallback,
                # Timing breakdown — without this you can see num_chunks=0
                # but not whether embed, search, or rerank ate the budget.
                "embed_ms": bundle.trace.embed_ms,
                "search_ms": bundle.trace.search_ms,
                "rerank_ms": bundle.trace.rerank_ms,
                "parent_expand_ms": bundle.trace.parent_expand_ms,
            },
        )

        yield _EvidenceReady(
            text=self._format_evidence(safe_chunks), chunks=safe_chunks
        )

    def _is_cloud_provider(self) -> bool:
        if self._provider_config is None:
            return False
        return getattr(self._provider_config, "kind", None) == "cloud"

    def _filter_for_provider(
        self, chunks: "list[RetrievedChunk]"
    ) -> "list[RetrievedChunk]":
        if not self._is_cloud_provider():
            return chunks
        safe, _report = self._guard.filter_chunks_for_provider(
            chunks, provider_kind="cloud"
        )
        return safe

    @staticmethod
    def _format_evidence(chunks: "list[RetrievedChunk]") -> str:
        if not chunks:
            return ""
        lines = ["", "Evidence (cite by [n]):"]
        for i, c in enumerate(chunks, start=1):
            body = c.parent_text or c.text
            src = c.source_uri or c.collection_name or c.source
            lines.append(f"[{i}] ({src}) {body}")
        return "\n".join(lines)

    @staticmethod
    def _format_sources(chunks: "list[RetrievedChunk]") -> str:
        """Build an authoritative Sources section from chunk metadata.

        Uses source_uri when available so the user gets real URLs instead of
        LLM-invented descriptions. Falls back to collection_name then source type.
        The list mirrors the [N] numbering in the injected evidence block, so
        the LLM's inline citations resolve correctly.
        """
        if not chunks:
            return ""
        lines = ["\n\n**Sources:**"]
        for i, c in enumerate(chunks, start=1):
            src = c.source_uri or c.collection_name or c.source
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
            score = f"{c.score:.3f}" if c.score else "—"
            lines.append(f"- [{i}] `{col}` (score {score})")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _compose_prompt(scrubbed: str, evidence_block: str) -> str:
        if not evidence_block:
            return scrubbed
        return f"{evidence_block}\n\nQuestion: {scrubbed}"


class _EvidenceReady:
    """Private sentinel: carries the formatted evidence_block out of
    ``_maybe_retrieve`` without polluting the public ``Event`` union.

    ``chunks`` is also forwarded so the caller can emit a debug
    collections block when ``CLARITYMED_DEBUG`` is set — the LLM does
    not include collection metadata in its Sources section, so this
    must be appended by code after the stream finishes.
    """

    __slots__ = ("text", "chunks")

    def __init__(self, text: str, chunks: "list") -> None:
        self.text = text
        self.chunks = chunks
