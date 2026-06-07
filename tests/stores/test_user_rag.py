"""Tests for ``stores/user_rag.py``: isolation, PHI scrub, CRUD."""

from __future__ import annotations

import hashlib
import importlib

import pytest
from qdrant_client import QdrantClient

from claritymed.orchestrator import PhiGuard
from claritymed.stores.user_rag import UserRagStore


class StubEmbedder:
    """Deterministic 32-dim hash-based embedder. Same text -> same vector."""

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # 32 bytes -> 32 floats in [0, 1)
        return [b / 255.0 for b in digest[: self.dimension]]

    @property
    def dimension(self) -> int:
        return 32


@pytest.fixture
def store():
    from claritymed import config as _cfg

    importlib.reload(_cfg)
    return UserRagStore(
        client=QdrantClient(":memory:"),
        embedder=StubEmbedder(),
        guard=PhiGuard.from_config(),
    )


def test_add_then_search_round_trip(store):
    n = store.add_document(
        user_id="alice",
        doc_id="doc1",
        chunks=["hello world", "diabetes is a chronic condition"],
    )
    assert n == 2
    hits = store.search("alice", "diabetes", top_k=2)
    assert len(hits) >= 1
    assert any("diabetes" in h.text for h in hits)


def test_user_isolation_structural(store):
    """Bob's search cannot see Alice's documents."""
    store.add_document("alice", "doc1", chunks=["alice has a secret note"])
    store.add_document("bob", "doc1", chunks=["bob has a different note"])

    alice_hits = store.search("alice", "note", top_k=5)
    bob_hits = store.search("bob", "note", top_k=5)

    assert all("alice" in h.text for h in alice_hits)
    assert all("bob" in h.text for h in bob_hits)


def test_phi_scrub_on_add(store):
    """User chunks have PII redacted before persistence."""
    store.add_document(
        "alice",
        "report1",
        chunks=["Patient contact 13800138000 email a@b.cn"],
    )
    hits = store.search("alice", "contact", top_k=5)
    assert len(hits) == 1
    text = hits[0].text
    assert "13800138000" not in text
    assert "a@b.cn" not in text
    assert "[REDACTED:PHONE]" in text
    assert "[REDACTED:EMAIL]" in text


def test_public_doc_skips_scrub(store):
    """``public=True`` (user-marked reference) keeps text intact and flags cloud-safe."""
    store.add_document(
        "alice",
        "paper1",
        chunks=["Contact author a@b.cn for the dataset."],
        public=True,
    )
    hits = store.search("alice", "dataset", top_k=5)
    assert len(hits) == 1
    assert "a@b.cn" in hits[0].text
    assert hits[0].is_phi is False
    assert hits[0].can_cloud is True


def test_only_cloud_safe_filter(store):
    store.add_document("alice", "private", chunks=["my personal note"])  # is_phi
    store.add_document("alice", "public_paper", chunks=["public paper"], public=True)

    all_hits = store.search("alice", "paper", top_k=5)
    cloud_hits = store.search("alice", "paper", top_k=5, only_cloud_safe=True)

    assert any(not h.can_cloud for h in all_hits)
    assert all(h.can_cloud for h in cloud_hits)


def test_search_before_ensure_collection_returns_empty(store):
    """Searching a user with no docs is harmless — empty list, no error."""
    hits = store.search("never_added_user", "anything", top_k=5)
    assert hits == []


def test_drop_user_removes_collection(store):
    store.add_document("alice", "doc1", chunks=["alice note"])
    assert store.search("alice", "note") != []

    dropped = store.drop_user("alice")
    assert dropped is True
    assert store.search("alice", "note") == []


def test_drop_user_idempotent(store):
    assert store.drop_user("never_existed") is False


def test_chunks_carry_user_rag_source_label(store):
    store.add_document("alice", "doc1", chunks=["something"])
    hits = store.search("alice", "something")
    assert hits[0].source == "user_rag"
