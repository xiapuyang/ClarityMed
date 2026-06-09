"""Tests for the rag agent module: deterministic ingest path post-Unit-8."""

from __future__ import annotations

import hashlib

import pytest
from qdrant_client import AsyncQdrantClient

from claritymed.core.rag.chunking.base import (
    ChildChunk,
    ChunkedDocument,
    ParentChunk,
    RawDocument,
)
from claritymed.core.rag.embedding.base import Embedder, SparseVector
from claritymed.core.phi.guard import PhiGuard
from claritymed.orchestrator.agents import (
    RAG_TOOL_NAMES,
    embed_and_store,
)
from claritymed.orchestrator.agents.rag_agent import fetch_web_link_stub, tag_phi
from claritymed.stores.user_rag import UserRagStore

DENSE_DIM = 32


class _StubEmbedder(Embedder):
    @property
    def dimension(self) -> int:
        return DENSE_DIM

    async def embed_dense(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            out.append([b / 255.0 for b in digest[:DENSE_DIM]])
        return out

    async def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        return [{abs(hash(t)) % 100: 0.5} for t in texts]


class _StubChunker:
    def chunk(self, doc: RawDocument) -> ChunkedDocument:
        if not doc.text.strip():
            return ChunkedDocument(parents=[], children=[])
        import uuid

        parent_id = f"{doc.doc_id}#p0"
        parent = ParentChunk(
            parent_id=parent_id,
            text=doc.text,
            doc_id=doc.doc_id,
            parent_index=0,
        )
        child = ChildChunk(
            child_id=str(uuid.uuid5(uuid.NAMESPACE_URL, doc.doc_id)),
            text=doc.text,
            parent_id=parent_id,
            doc_id=doc.doc_id,
            chunk_index=0,
        )
        return ChunkedDocument(parents=[parent], children=[child])


@pytest.fixture
def store() -> UserRagStore:
    return UserRagStore(
        aclient=AsyncQdrantClient(":memory:"),
        embedder=_StubEmbedder(),
        chunker=_StubChunker(),
        guard=PhiGuard.from_config(),
    )


def test_tool_names_match_modes_config():
    from claritymed import config as _cfg

    modes = _cfg.load_modes_config()
    assert set(modes.get("rag").tools) == set(RAG_TOOL_NAMES)


def test_fetch_web_link_stub_is_marked_stub():
    assert "[stub:" in fetch_web_link_stub("https://example.com")


def test_tag_phi_basic_heuristic():
    assert tag_phi("contact MRN 12345").get("is_phi") is True
    assert tag_phi("just a clinical note").get("is_phi") is False


async def test_embed_and_store_writes_to_user_rag(store):
    receipt = await embed_and_store(store=store, user_id="alice", text="hello world")
    assert receipt.chunk_count == 1
    assert receipt.doc_id.startswith("doc-")
    hits = await store.search("alice", "world", top_k=5)
    assert hits != []


async def test_embed_and_store_with_explicit_doc_id(store):
    receipt = await embed_and_store(
        store=store, user_id="alice", text="one", doc_id="my-doc-1"
    )
    assert receipt.doc_id == "my-doc-1"


async def test_embed_and_store_public_flag_passes_through(store):
    await embed_and_store(
        store=store, user_id="alice", text="public paper content", public=True
    )
    hits = await store.search("alice", "public", top_k=5)
    assert all(h.can_cloud for h in hits)


async def test_embed_and_store_empty_text_returns_empty_receipt(store):
    receipt = await embed_and_store(store=store, user_id="alice", text="")
    assert receipt.chunk_count == 0
    assert receipt.embedding_status == "stub"
