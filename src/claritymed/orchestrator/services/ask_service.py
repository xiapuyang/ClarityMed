"""Ask service: PHI scrub on input + stream LLM tokens to the caller."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import logging

from claritymed.core.observability.audit import audit_event
from claritymed.orchestrator import PhiGuard
from claritymed.orchestrator.agents import make_ask_agent
from claritymed.orchestrator.services.events import (
    Done,
    Error,
    Event,
    TokenChunk,
)

if TYPE_CHECKING:
    from pydantic_ai.models import Model

    from claritymed.stores.chat_memory import ChatMemoryStore

logger = logging.getLogger(__name__)


class AskService:
    """Drive ask mode: scrub user input, stream LLM tokens, finalize.

    Retrieval (system_rag + user_rag) and context assembly land in the real
    text_rag plan; the Phase 1 stub feeds the LLM only the (scrubbed) user
    input. The streaming surface and the PHI scrubbing point are stable so
    the real plan does not change the service contract.
    """

    def __init__(
        self,
        model: "Model",
        guard: PhiGuard | None = None,
        language: str = "en",
        chat_memory: "ChatMemoryStore | None" = None,
    ) -> None:
        self._model = model
        self._guard = guard or PhiGuard.from_config()
        self._language = language
        self._chat_memory = chat_memory

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

        agent = make_ask_agent(self._model, language=self._language)
        messages_json: bytes | None = None
        try:
            async with agent.run_stream(scrubbed) as stream:
                async for chunk in stream.stream_text(delta=True):
                    if chunk:
                        yield TokenChunk(text=chunk)
                final_text = await stream.get_output()
                # Capture inside the ``with`` block — the stream goes out of
                # scope once it exits and the messages disappear with it.
                try:
                    messages_json = stream.all_messages_json()
                except Exception:  # noqa: BLE001
                    logger.exception("failed to capture pydantic-ai messages")
        except Exception as exc:  # noqa: BLE001 — surface as event
            yield Error(
                error_type="llm_error",
                message=str(exc),
                retryable=True,
            )
            return

        if messages_json is not None and self._chat_memory is not None:
            try:
                self._chat_memory.append_run_messages_json(messages_json)
            except Exception:  # noqa: BLE001
                logger.exception("failed to append chat messages to chat memory")

        audit_event(
            "mode.ask",
            payload={
                "user_id": user_id,
                "answer_len": len(final_text),
            },
        )
        yield Done(final=final_text)
