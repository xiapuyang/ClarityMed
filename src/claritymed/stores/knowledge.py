"""Shared knowledge store skeleton.

Read path is open to any authenticated user — Qdrant filter pins
``payload.user_id is None`` so a per-user record cannot leak in even if a
future ingestor mis-tags one. Write path (``upsert_chunk``, ``wipe_collection``)
goes through a template method that always calls ``require_admin()`` first,
so a third-party subclass cannot accidentally bypass the role guard.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from claritymed.errors import PhiViolationError
from claritymed.stores.account import require_admin

logger = logging.getLogger(__name__)


class KnowledgeChunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    payload: dict[str, Any] = Field(default_factory=dict)


class KnowledgeFilters(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    language: Optional[str] = None
    source_type: Optional[str] = None


class KnowledgeStore(ABC):
    """Public knowledge base interface (Qdrant-backed in the default impl).

    Search requires a ``user_id`` parameter for audit attribution, but the
    actual Qdrant filter pins ``payload.user_id is None`` — the shared layer
    must not return per-user data even if some future bug were to ingest it.
    """

    # --- read path (any user) ----------------------------------------

    def search(
        self,
        query: str,
        *,
        user_id: str,
        language: str,
        top_k: int = 10,
        filters: Optional[KnowledgeFilters] = None,
    ) -> list[KnowledgeChunk]:
        if not user_id:
            raise ValueError("KnowledgeStore.search requires user_id (for audit)")
        return self._search_unchecked(
            query,
            user_id=user_id,
            language=language,
            top_k=top_k,
            filters=filters,
        )

    @abstractmethod
    def _search_unchecked(
        self,
        query: str,
        *,
        user_id: str,
        language: str,
        top_k: int,
        filters: Optional[KnowledgeFilters],
    ) -> list[KnowledgeChunk]: ...

    # --- write path (admin only) ------------------------------------

    def upsert_chunk(self, chunk: KnowledgeChunk) -> None:
        require_admin()
        if chunk.payload.get("user_id") is not None:
            raise PhiViolationError(
                "knowledge chunk must not carry a user_id "
                "(shared layer is public knowledge only)"
            )
        self._upsert_chunk_unchecked(chunk)

    def wipe_collection(self) -> None:
        require_admin()
        from claritymed.core.observability.audit import audit_event

        audit_event("knowledge_wipe", payload={"collection": "knowledge_v1"})
        self._wipe_collection_unchecked()

    @abstractmethod
    def _upsert_chunk_unchecked(self, chunk: KnowledgeChunk) -> None: ...

    @abstractmethod
    def _wipe_collection_unchecked(self) -> None: ...


class QdrantKnowledgeStore(KnowledgeStore):
    """Default impl. Real Qdrant wiring is in the text_rag plan; this is a
    no-op stub that the test suite exercises through mocks."""

    def _search_unchecked(
        self,
        query: str,
        *,
        user_id: str,
        language: str,
        top_k: int,
        filters: Optional[KnowledgeFilters],
    ) -> list[KnowledgeChunk]:
        logger.warning(
            "QdrantKnowledgeStore.search is a v1 stub (returning []); user=%s",
            user_id,
        )
        return []

    def _upsert_chunk_unchecked(self, chunk: KnowledgeChunk) -> None:
        logger.warning("QdrantKnowledgeStore.upsert_chunk is a v1 stub")

    def _wipe_collection_unchecked(self) -> None:
        logger.warning("QdrantKnowledgeStore.wipe_collection is a v1 stub")
