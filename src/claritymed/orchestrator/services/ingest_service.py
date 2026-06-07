"""Ingest service: Phase 1 deterministic path (no LLM)."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from claritymed.context import (
    apply_context,
    new_request_id,
    request_id_ctx,
    reset_context,
)
from claritymed.core.observability.audit import audit_event
from claritymed.core.schemas.receipts import IngestReceipt, IngestRecord
from claritymed.orchestrator.agents import save_to_profile
from claritymed.orchestrator.services.events import (
    Done,
    Event,
    ToolCompleted,
    ToolStarted,
)


class IngestService:
    """Drive ingest mode end-to-end.

    Phase 1 implements the simplest viable path: parse the input as a
    ``key=value`` profile field and persist it. The OCR / lab pipeline /
    vision branches are wired by the real plans; this scaffolding shows
    the event flow and audit hooks the TUI relies on.
    """

    async def run(
        self,
        user_input: str,
        user_id: str,
        language: str = "en",
    ) -> AsyncIterator[Event]:
        rid = request_id_ctx.get() or new_request_id()
        tokens = apply_context(rid, user_id, language)
        try:
            async for ev in self._run_inner(user_input, user_id):
                yield ev
        finally:
            reset_context(tokens)

    async def _run_inner(self, user_input: str, user_id: str) -> AsyncIterator[Event]:
        if "=" not in user_input:
            yield ToolStarted(tool_name="save_to_profile", args_preview="(no kv)")
            yield Done(final=IngestReceipt(records=[], summary="no recognizable field"))
            return

        field, _, value = user_input.partition("=")
        field, value = field.strip(), value.strip()

        yield ToolStarted(tool_name="save_to_profile", args_preview=field)
        t0 = time.monotonic()
        record_id = save_to_profile(user_id, field, value)
        duration_ms = int((time.monotonic() - t0) * 1000)
        yield ToolCompleted(
            tool_name="save_to_profile",
            duration_ms=duration_ms,
            summary=f"saved {field}",
        )

        receipt = IngestReceipt(
            records=[IngestRecord(kind="profile", record_id=record_id)],
            summary=f"saved profile field {field}",
        )
        audit_event(
            "mode.ingest",
            payload={
                "user_id": user_id,
                "field": field,
                "record_id": record_id,
            },
        )
        yield Done(final=receipt)
