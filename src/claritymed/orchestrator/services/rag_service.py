"""Rag service: deterministic ingestion into ``user_rag``.

The chunking + embedding pipeline lives in ``stores/user_rag.py`` (Unit 8
of the RAG plan). RagService just drives the audit + event flow the
TUI / CLI listens to.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from claritymed.core.events import (
    Done,
    Event,
    ToolCompleted,
    ToolStarted,
)
from claritymed.core.observability.audit import audit_event
from claritymed.core.schemas.receipts import IngestionReceipt
from claritymed.stores.user_rag import UserRagStore, generate_doc_id


class RagService:
    """Drive rag mode end-to-end (deterministic ingest path)."""

    def __init__(self, store: UserRagStore) -> None:
        self._store = store

    async def run(
        self,
        user_input: str,
        user_id: str,
        public: bool = False,
        language: str = "en",
        source_uri: str | None = None,
    ) -> AsyncIterator[Event]:
        from claritymed.context import (
            apply_context,
            new_request_id,
            request_id_ctx,
            reset_context,
        )

        rid = request_id_ctx.get() or new_request_id()
        tokens = apply_context(rid, user_id, language)
        try:
            async for ev in self._run_inner(user_input, user_id, public, source_uri):
                yield ev
        finally:
            reset_context(tokens)

    async def _run_inner(
        self,
        user_input: str,
        user_id: str,
        public: bool,
        source_uri: str | None,
    ) -> AsyncIterator[Event]:
        yield ToolStarted(tool_name="embed_and_store", args_preview=user_id)
        t0 = time.monotonic()
        metadata = {"source_uri": source_uri} if source_uri else None
        doc_id = generate_doc_id()
        result = await self._store.add_document(
            user_id=user_id,
            doc_id=doc_id,
            text=user_input,
            metadata=metadata,
            public=public,
        )
        receipt = IngestionReceipt(
            doc_id=doc_id,
            chunk_count=result.written,
            skipped_chunk_count=result.skipped_chunks,
            # ``stub`` survives for the truly-empty case (no text /
            # chunker produced nothing); when every chunk dedup'd
            # against existing content the embedding pipeline ran fine,
            # so ``ok`` is the honest status.
            embedding_status="ok"
            if result.written or result.skipped_chunks
            else "stub",
            public=public,
        )
        duration_ms = int((time.monotonic() - t0) * 1000)
        summary = (
            f"doc_id={receipt.doc_id} chunks={receipt.chunk_count}"
            f" skipped={receipt.skipped_chunk_count}"
        )
        yield ToolCompleted(
            tool_name="embed_and_store",
            duration_ms=duration_ms,
            summary=summary,
        )

        audit_event(
            "mode.rag",
            payload={
                "user_id": user_id,
                "doc_id": receipt.doc_id,
                "chunk_count": receipt.chunk_count,
                "skipped_chunk_count": receipt.skipped_chunk_count,
                "public": public,
            },
        )
        yield Done(final=receipt)
