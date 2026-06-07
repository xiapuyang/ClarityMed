"""Rag service: Phase 1 deterministic ingestion into ``user_rag``."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from claritymed.core.observability.audit import audit_event
from claritymed.orchestrator.agents import embed_and_store
from claritymed.orchestrator.agents.rag_agent import chunk_document_stub
from claritymed.orchestrator.services.events import (
    Done,
    Event,
    ToolCompleted,
    ToolStarted,
)
from claritymed.stores.user_rag import UserRagStore


class RagService:
    """Drive rag mode end-to-end.

    The store is injected so tests can use an in-memory Qdrant and a stub
    embedder without touching the local filesystem or downloading a model.
    """

    def __init__(self, store: UserRagStore) -> None:
        self._store = store

    async def run(
        self,
        user_input: str,
        user_id: str,
        public: bool = False,
        language: str = "en",
    ) -> AsyncIterator[Event]:
        from claritymed.context import apply_context, new_request_id, reset_context

        tokens = apply_context(new_request_id(), user_id, language)
        try:
            async for ev in self._run_inner(user_input, user_id, public):
                yield ev
        finally:
            reset_context(tokens)

    async def _run_inner(
        self,
        user_input: str,
        user_id: str,
        public: bool,
    ) -> AsyncIterator[Event]:
        yield ToolStarted(tool_name="chunk_document_stub", args_preview="")
        chunks = chunk_document_stub(user_input)
        yield ToolCompleted(
            tool_name="chunk_document_stub",
            summary=f"{len(chunks)} chunks",
        )

        yield ToolStarted(tool_name="embed_and_store", args_preview=user_id)
        t0 = time.monotonic()
        receipt = embed_and_store(
            store=self._store,
            user_id=user_id,
            chunks=chunks,
            public=public,
        )
        duration_ms = int((time.monotonic() - t0) * 1000)
        yield ToolCompleted(
            tool_name="embed_and_store",
            duration_ms=duration_ms,
            summary=f"doc_id={receipt.doc_id} chunks={receipt.chunk_count}",
        )

        audit_event(
            "mode.rag",
            payload={
                "user_id": user_id,
                "doc_id": receipt.doc_id,
                "chunk_count": receipt.chunk_count,
                "public": public,
            },
        )
        yield Done(final=receipt)
