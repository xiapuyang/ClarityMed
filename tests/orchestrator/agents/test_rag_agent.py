"""Tests for the rag agent module: stubs + embed_and_store real path."""

from __future__ import annotations

import hashlib

import pytest
from qdrant_client import QdrantClient

from claritymed.orchestrator import PhiGuard
from claritymed.orchestrator.agents import (
    RAG_TOOL_NAMES,
    embed_and_store,
)
from claritymed.orchestrator.agents.rag_agent import (
    chunk_document_stub,
    fetch_web_link_stub,
    tag_phi,
)
from claritymed.stores.user_rag import UserRagStore


class _StubEmbedder:
    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in digest[: self.dimension]]

    @property
    def dimension(self) -> int:
        return 32


@pytest.fixture
def store():
    return UserRagStore(
        client=QdrantClient(":memory:"),
        embedder=_StubEmbedder(),
        guard=PhiGuard.from_config(),
    )


def test_tool_names_match_modes_config():
    from claritymed import config as _cfg

    modes = _cfg.load_modes_config()
    assert set(modes.get("rag").tools) == set(RAG_TOOL_NAMES)


def test_chunk_document_stub_splits_paragraphs():
    text = "para one\n\npara two\n\npara three"
    chunks = chunk_document_stub(text)
    assert len(chunks) == 3
    assert "para one" in chunks


def test_chunk_document_stub_single_chunk_when_no_breaks():
    chunks = chunk_document_stub("single block of text")
    assert len(chunks) == 1


def test_fetch_web_link_stub_is_marked_stub():
    assert "[stub:" in fetch_web_link_stub("https://example.com")


def test_tag_phi_basic_heuristic():
    assert tag_phi("contact MRN 12345").get("is_phi") is True
    assert tag_phi("just a clinical note").get("is_phi") is False


def test_embed_and_store_writes_to_user_rag(store):
    """The single non-stub real tool — exercises store integration."""
    receipt = embed_and_store(
        store=store,
        user_id="alice",
        chunks=["hello world", "another chunk"],
    )
    assert receipt.chunk_count == 2
    assert receipt.doc_id.startswith("doc-")
    # Verify chunks are actually searchable in the store.
    hits = store.search("alice", "world", top_k=5)
    assert hits != []


def test_embed_and_store_with_explicit_doc_id(store):
    receipt = embed_and_store(
        store=store,
        user_id="alice",
        chunks=["one"],
        doc_id="my-doc-1",
    )
    assert receipt.doc_id == "my-doc-1"


def test_embed_and_store_public_flag_passes_through(store):
    """public=True must reach add_document so the chunk is cloud-safe."""
    embed_and_store(
        store=store,
        user_id="alice",
        chunks=["public paper content"],
        public=True,
    )
    hits = store.search("alice", "public", top_k=5)
    assert all(h.can_cloud for h in hits)
