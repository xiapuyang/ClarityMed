"""Tests for ``KnowledgeStore`` read/write split + admin guard."""

from __future__ import annotations

import pytest

from claritymed.errors import PermissionDeniedError, PhiViolationError
from claritymed.stores.knowledge import (
    KnowledgeChunk,
    QdrantKnowledgeStore,
)


def test_search_requires_user_id_kwarg(as_, trio):
    """Type-level enforcement: forgetting user_id is a TypeError."""
    with as_(trio["users"][0]):
        with pytest.raises(TypeError):
            QdrantKnowledgeStore().search("x", language="en")  # type: ignore[call-arg]


def test_search_rejects_empty_user_id(as_, trio):
    with as_(trio["users"][0]):
        with pytest.raises(ValueError):
            QdrantKnowledgeStore().search("x", user_id="", language="en")


def test_user_can_read(as_, trio):
    bob = trio["users"][0]
    with as_(bob):
        assert (
            QdrantKnowledgeStore().search(
                "headache", user_id=bob.user_id, language="en"
            )
            == []
        )


def test_user_cannot_wipe(as_, trio):
    with as_(trio["users"][0]):
        with pytest.raises(PermissionDeniedError):
            QdrantKnowledgeStore().wipe_collection()


def test_admin_can_wipe(as_, trio):
    with as_(trio["admin"]):
        QdrantKnowledgeStore().wipe_collection()  # no exception


def test_admin_can_upsert_chunk_without_user_id(as_, trio):
    chunk = KnowledgeChunk(
        chunk_id="c-1",
        text="public knowledge",
        payload={"language": "en", "source": "MedCorp"},
    )
    with as_(trio["admin"]):
        QdrantKnowledgeStore().upsert_chunk(chunk)


def test_admin_cannot_upsert_chunk_with_user_id(as_, trio):
    """The 'shared layer must not contain PHI' contract — admin or not."""
    chunk = KnowledgeChunk(
        chunk_id="c-2",
        text="alice's note",
        payload={"user_id": "alice"},
    )
    with as_(trio["admin"]):
        with pytest.raises(PhiViolationError):
            QdrantKnowledgeStore().upsert_chunk(chunk)


def test_user_cannot_upsert_at_all(as_, trio):
    chunk = KnowledgeChunk(chunk_id="c-3", text="x", payload={})
    with as_(trio["users"][0]):
        with pytest.raises(PermissionDeniedError):
            QdrantKnowledgeStore().upsert_chunk(chunk)


def test_read_path_does_not_invoke_admin_guard(as_, trio, tmp_path):
    """Reads must work for any user without emitting require_admin events."""
    bob = trio["users"][0]
    with as_(bob):
        QdrantKnowledgeStore().search("x", user_id=bob.user_id, language="en")
    audit_path = tmp_path / "logs" / "audit.log"
    if audit_path.exists():
        text = audit_path.read_text(encoding="utf-8")
        assert "require_admin_pass" not in text
        assert "require_admin_blocked" not in text
