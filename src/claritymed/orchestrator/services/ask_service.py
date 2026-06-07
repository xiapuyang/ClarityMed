"""Ask service: PHI scrub on input + stream LLM tokens to the caller."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

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
    ) -> None:
        self._model = model
        self._guard = guard or PhiGuard.from_config()
        self._language = language

    async def run(self, user_input: str, user_id: str) -> AsyncIterator[Event]:
        from claritymed.context import apply_context, new_request_id, reset_context

        tokens = apply_context(new_request_id(), user_id, self._language)
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
        try:
            async with agent.run_stream(scrubbed) as stream:
                async for chunk in stream.stream_text(delta=True):
                    if chunk:
                        yield TokenChunk(text=chunk)
                final_text = await stream.get_output()
        except Exception as exc:  # noqa: BLE001 — surface as event
            yield Error(
                error_type="llm_error",
                message=str(exc),
                retryable=True,
            )
            return

        audit_event(
            "mode.ask",
            payload={
                "user_id": user_id,
                "answer_len": len(final_text),
            },
        )
        yield Done(final=final_text)
