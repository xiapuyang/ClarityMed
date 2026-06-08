"""Tests for ``stores/user_rag.py`` async ingest facade.

Migrated to bge-m3 / parent-child / async hybrid storage in Unit 8 of
the RAG plan. PHI scrub + per-user isolation invariants kept; the
chunker now produces parents + children instead of taking a pre-chunked
list.
"""

from __future__ import annotations

import hashlib
import importlib

import pytest
from qdrant_client import AsyncQdrantClient

from claritymed.core.rag.chunking.base import (
    ChildChunk,
    ChunkedDocument,
    ParentChunk,
    RawDocument,
)
from claritymed.core.rag.embedding.base import Embedder, SparseVector
from claritymed.orchestrator import PhiGuard
from claritymed.stores.user_rag import UserRagStore

DENSE_DIM = 32


class StubEmbedder(Embedder):
    """Deterministic 32-dim hash embedder. Same text -> same vectors."""

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
    """One ParentChunk + one ChildChunk per input — keeps tests simple
    and decouples them from upstream HierarchicalNodeParser behavior."""

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
            child_id=_uuid_for(doc.doc_id),
            text=doc.text,
            parent_id=parent_id,
            doc_id=doc.doc_id,
            chunk_index=0,
            metadata=dict(doc.metadata),
        )
        return ChunkedDocument(parents=[parent], children=[child])


def _uuid_for(seed: str) -> str:
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


@pytest.fixture
def store() -> UserRagStore:
    # Make sure config picks up the per-test tmp HOME.
    from claritymed import config as _cfg

    importlib.reload(_cfg)
    return UserRagStore(
        aclient=AsyncQdrantClient(":memory:"),
        embedder=StubEmbedder(),
        chunker=StubChunker(),
        guard=PhiGuard.from_config(),
    )


# --- ingest --------------------------------------------------------


async def test_add_document_round_trip(store):
    n = await store.add_document(
        user_id="alice",
        doc_id="doc1",
        text="diabetes is a chronic metabolic condition",
    )
    assert n == 1
    hits = await store.search("alice", "diabetes", top_k=2)
    assert len(hits) == 1
    assert "diabetes" in hits[0].text
    assert hits[0].source == "user_rag"
    assert hits[0].parent_text is not None


async def test_user_isolation_structural(store):
    await store.add_document("alice", "doc1", text="alice has a secret note")
    await store.add_document("bob", "doc1", text="bob has a different note")

    alice_hits = await store.search("alice", "note", top_k=5)
    bob_hits = await store.search("bob", "note", top_k=5)

    assert alice_hits and bob_hits
    assert all("alice" in h.text for h in alice_hits)
    assert all("bob" in h.text for h in bob_hits)


async def test_phi_scrub_on_add(store):
    await store.add_document(
        "alice",
        "report1",
        text="Patient contact 13800138000 email a@b.cn",
    )
    hits = await store.search("alice", "contact", top_k=5)
    assert len(hits) == 1
    text = hits[0].text
    assert "13800138000" not in text
    assert "a@b.cn" not in text
    assert "[REDACTED:PHONE]" in text
    assert "[REDACTED:EMAIL]" in text


async def test_public_doc_skips_scrub_and_flags_cloud_safe(store):
    await store.add_document(
        "alice",
        "paper1",
        text="Contact author a@b.cn for the dataset.",
        public=True,
    )
    hits = await store.search("alice", "dataset", top_k=5)
    assert len(hits) == 1
    assert "a@b.cn" in hits[0].text
    assert hits[0].is_phi is False
    assert hits[0].can_cloud is True


async def test_only_cloud_safe_filter(store):
    await store.add_document("alice", "private", text="my personal note")
    await store.add_document("alice", "paper", text="public paper", public=True)

    all_hits = await store.search("alice", "paper", top_k=5)
    cloud_hits = await store.search("alice", "paper", top_k=5, only_cloud_safe=True)

    assert any(not h.can_cloud for h in all_hits)
    assert all(h.can_cloud for h in cloud_hits)


async def test_empty_text_writes_nothing(store):
    n = await store.add_document("alice", "empty", text="")
    assert n == 0
    n = await store.add_document("alice", "empty2", text="   \n  ")
    assert n == 0


# --- search ---------------------------------------------------------


async def test_search_before_any_doc_returns_empty(store):
    hits = await store.search("ghost_user", "anything", top_k=5)
    assert hits == []


async def test_chunks_carry_user_rag_source_label(store):
    await store.add_document("alice", "doc1", text="something")
    hits = await store.search("alice", "something")
    assert hits[0].source == "user_rag"
    assert hits[0].collection_name == "user_rag_alice"


# --- delete / drop --------------------------------------------------


async def test_drop_user_removes_collection_and_parent_file(store, tmp_path):
    await store.add_document("alice", "doc1", text="alice note")
    assert await store.search("alice", "note") != []

    dropped = await store.drop_user("alice")
    assert dropped is True
    assert await store.search("alice", "note") == []
    # Idempotent
    assert await store.drop_user("alice") is False


async def test_drop_user_when_nothing_exists(store):
    assert await store.drop_user("never_existed") is False


async def test_delete_document_scrubs_qdrant_and_parents(store):
    await store.add_document("alice", "d1", text="first note about aspirin")
    await store.add_document("alice", "d2", text="second note about ibuprofen")
    await store.delete_document("alice", "d1")
    remaining = await store.search("alice", "note", top_k=5)
    assert all(h.doc_id != "d1" for h in remaining)
    assert any(h.doc_id == "d2" for h in remaining)


# --- migration -----------------------------------------------------


async def test_migrate_user_is_destructive_alias_of_drop(store):
    await store.add_document("alice", "d1", text="some note")
    await store.migrate_user("alice")
    assert await store.search("alice", "note") == []
