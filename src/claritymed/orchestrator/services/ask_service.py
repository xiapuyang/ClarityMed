"""Ask service: PHI scrub on input + stream LLM tokens to the caller."""

from __future__ import annotations

import logging
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
    TokenChunk,
)

if TYPE_CHECKING:
    from pydantic_ai.models import Model
    from pydantic_ai.usage import RunUsage

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
    ) -> None:
        self._model = model
        self._guard = guard or PhiGuard.from_config()
        self._language = language
        self._chat_session = chat_session
        self._provider_id = provider_id
        self._model_name = model_name

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

        # Record the user turn first so the JSONL timeline reflects send
        # order, then pull the prior history for the LLM call.
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
            async for event in self._run_with_agent(scrubbed, message_history, user_id):
                yield event
        finally:
            detach_session_baggage(session_token)

    async def _run_with_agent(
        self,
        scrubbed: str,
        message_history,
        user_id: str,
    ) -> AsyncIterator[Event]:
        agent = make_ask_agent(self._model, language=self._language)
        messages_json: bytes | None = None
        usage: RunUsage | None = None
        steps: list[dict] = []
        final_text: str = ""
        t_start = time.perf_counter()
        t_first_token: float | None = None
        try:
            async with agent.run_stream(
                scrubbed, message_history=message_history or None
            ) as stream:
                async for chunk in stream.stream_text(delta=True):
                    if chunk:
                        if t_first_token is None:
                            t_first_token = time.perf_counter()
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
