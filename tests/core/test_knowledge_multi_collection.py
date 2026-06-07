"""Tests for KnowledgeStore multi-collection extension + Account opt-in."""

from __future__ import annotations

from claritymed.core.schemas.account import Account
from claritymed.stores.knowledge import QdrantKnowledgeStore


def test_account_defaults_to_no_active_system_rag():
    """Backward compat: existing accounts have no active collections by default."""
    acc = Account(user_id="alice", display_name="Alice")
    assert acc.active_system_rag_collections == []


def test_account_accepts_active_collections():
    acc = Account(
        user_id="alice",
        display_name="Alice",
        active_system_rag_collections=["medcorp_en", "cmb_zh"],
    )
    assert "medcorp_en" in acc.active_system_rag_collections


def test_empty_active_collections_returns_empty(as_, trio):
    """Explicit [] short-circuits — no backend hit, no chunks."""
    bob = trio["users"][0]
    with as_(bob):
        result = QdrantKnowledgeStore().search(
            "anything",
            user_id=bob.user_id,
            language="en",
            active_collections=[],
        )
    assert result == []


def test_none_active_collections_preserves_legacy_behavior(as_, trio):
    """active_collections=None means 'whatever the impl defaults to'."""
    bob = trio["users"][0]
    with as_(bob):
        # stub returns [] regardless; the point is the call did not raise
        # and active_collections=None was accepted.
        QdrantKnowledgeStore().search(
            "anything",
            user_id=bob.user_id,
            language="en",
            active_collections=None,
        )


def test_active_collections_list_accepted(as_, trio):
    bob = trio["users"][0]
    with as_(bob):
        QdrantKnowledgeStore().search(
            "anything",
            user_id=bob.user_id,
            language="en",
            active_collections=["medcorp_en"],
        )


def test_retrieval_yaml_has_collections_section():
    """The collection metadata config should load and contain entries."""
    from claritymed import config as _cfg

    data = _cfg.load_yaml("retrieval.yaml")
    collections = data.get("system_rag", {}).get("collections", [])
    assert len(collections) >= 1
    assert all("name" in c for c in collections)
    assert all("language" in c for c in collections)
