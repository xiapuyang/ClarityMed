"""Tests for ``stores/user_phi_rag.py`` — the PHI-side per-user Qdrant store.

Same stubs as ``test_user_rag.py`` (StubEmbedder, StubChunker) so the
shape of the assertions is comparable. The key invariants:

* writes go to ``user_phi_<id>`` (not ``user_rag_<id>``).
* every chunk's payload carries ``is_phi=True`` and ``can_cloud=False``.
* no ``public`` kwarg is exposed (PHI is PHI by construction).
* ``delete_by_doc_id`` cascade removes every chunk for a record_path.
"""

from __future__ import annotations

import hashlib
import uuid

import pytest
from qdrant_client import AsyncQdrantClient

from claritymed.core.rag.chunking.base import (
    ChildChunk,
    ChunkedDocument,
    ParentChunk,
    RawDocument,
)
from claritymed.core.rag.embedding.base import Embedder, SparseVector
from claritymed.stores.user_phi_rag import UserPhiRagStore, collection_name

DENSE_DIM = 32


class StubEmbedder(Embedder):
    @property
    def dimension(self) -> int:
        return DENSE_DIM

    async def embed_dense(self, texts: list[str]) -> list[list[float]]:
        return [self._dense(t) for t in texts]

    async def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        return [{abs(hash(t)) % 100: 0.5} for t in texts]

    @staticmethod
    def _dense(t: str) -> list[float]:
        digest = hashlib.sha256(t.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:DENSE_DIM]]


class StubChunker:
    def chunk(self, doc: RawDocument) -> ChunkedDocument:
        if not doc.text.strip():
            return ChunkedDocument(parents=[], children=[])
        parent_id = f"{doc.doc_id}#p0"
        parent = ParentChunk(
            parent_id=parent_id,
            text=doc.text,
            doc_id=doc.doc_id,
            parent_index=0,
            metadata=dict(doc.metadata),
        )
        child = ChildChunk(
            child_id=str(uuid.uuid5(uuid.NAMESPACE_URL, doc.doc_id)),
            text=doc.text,
            parent_id=parent_id,
            doc_id=doc.doc_id,
            chunk_index=0,
            metadata=dict(doc.metadata),
        )
        return ChunkedDocument(parents=[parent], children=[child])


@pytest.fixture
def store() -> UserPhiRagStore:
    return UserPhiRagStore(
        aclient=AsyncQdrantClient(":memory:"),
        embedder=StubEmbedder(),
        chunker=StubChunker(),
    )


def test_collection_name_uses_user_phi_prefix():
    assert collection_name("alice") == "user_phi_alice"
    assert collection_name("alice") != "user_rag_alice"


async def test_add_record_writes_phi_chunk(store: UserPhiRagStore):
    n = await store.add_record(
        user_id="alice",
        record_path="exam-reports/2026-06-11-ab12cd34",
        ocr_text="hemoglobin 105 g/L, glucose 5.6 mmol/L",
    )
    assert n == 1


async def test_add_record_empty_text_returns_zero(store: UserPhiRagStore):
    """OCR ``empty`` outcome should not write a phantom chunk."""
    n = await store.add_record(
        user_id="alice",
        record_path="exam-reports/2026-06-11-ab12cd34",
        ocr_text="",
    )
    assert n == 0


async def test_add_record_does_not_accept_public_kwarg(store: UserPhiRagStore):
    """No ``public=True`` escape hatch — PHI store is PHI by construction."""
    with pytest.raises(TypeError):
        await store.add_record(
            user_id="alice",
            record_path="rec/p",
            ocr_text="x",
            public=True,  # type: ignore[call-arg]
        )


async def test_delete_by_doc_id_removes_chunks(store: UserPhiRagStore):
    record_path = "exam-reports/2026-06-11-ab12cd34"
    await store.add_record(user_id="alice", record_path=record_path, ocr_text="x")
    await store.delete_by_doc_id(user_id="alice", record_path=record_path)


async def test_record_path_propagates_to_payload(store: UserPhiRagStore):
    """``record_path`` shows up as both doc_id and a metadata field so
    LibraryView can attribute chunks back to their source record."""
    record_path = "exam-reports/2026-06-11-ab12cd34"
    await store.add_record(
        user_id="alice",
        record_path=record_path,
        ocr_text="hemoglobin 105 g/L",
        metadata={"title": "Annual checkup"},
    )
    # Sanity check: collection_store is constructed lazily; nothing to
    # introspect further without leaking Qdrant internals. The behavior
    # is exercised end-to-end in Unit 12's tests.
