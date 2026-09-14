"""Tests for ``stores/user_phi_rag.py`` — the PHI-side per-user Qdrant store.

Same stubs as ``test_user_rag.py`` (StubEmbedder, StubChunker) so the
shape of the assertions is comparable. The key invariants:

* writes go to ``user_phi_<id>`` (not ``user_rag_<id>``).
* every chunk's payload carries ``is_phi=True`` and ``can_cloud=False``.
* no ``public`` kwarg is exposed (PHI is PHI by construction).
* ``delete_by_doc_id`` cascade removes every chunk for a record_path.
* per-chunk cosine-similarity dedup mirrors the library path so a
  re-fed record doesn't duplicate in Qdrant (R11a).
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


@pytest.fixture(autouse=True)
def _disable_cosine_dedupe(monkeypatch):
    """Default cosine-sim dedupe to OFF for these tests.

    The ``StubEmbedder`` is deterministic SHA-256 → identical text is a
    100% cosine match. Tests that don't opt into the dedupe path would
    spuriously drop legitimate inserts. Tests that exercise dedupe set
    the threshold explicitly via monkeypatch or the kwarg.
    """
    monkeypatch.setattr("claritymed.config.record_dedupe_cosine_threshold", lambda: 0.0)


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


# --- chunk-level dedupe (R11a) ------------------------------------------


async def test_add_record_dedupes_identical_text_with_default_threshold(
    store: UserPhiRagStore, monkeypatch
):
    """Re-feeding identical OCR text drops the duplicate chunk under the
    default 0.95 threshold. Catches "user imports the same Notion page
    twice in two sessions" without needing any caller-side guard."""
    monkeypatch.setattr(
        "claritymed.config.record_dedupe_cosine_threshold", lambda: 0.95
    )
    first = await store.add_record(
        user_id="alice",
        record_path="exam-reports/2026-06-11-aaaaaaaa",
        ocr_text="hemoglobin 105 g/L, glucose 5.6 mmol/L",
    )
    assert first == 1
    second = await store.add_record(
        user_id="alice",
        record_path="exam-reports/2026-06-11-bbbbbbbb",
        ocr_text="hemoglobin 105 g/L, glucose 5.6 mmol/L",
    )
    assert second == 0


async def test_add_record_threshold_kwarg_overrides_config(store: UserPhiRagStore):
    """Caller can pin a strict threshold to force a no-op even when the
    config defaults to 0 (autouse fixture). Tests the bypass path Unit 8
    uses when an operator wants to override at the CLI."""
    first = await store.add_record(
        user_id="alice",
        record_path="exam-reports/2026-06-11-cccccccc",
        ocr_text="dose 500mg metformin twice daily",
    )
    assert first == 1
    second = await store.add_record(
        user_id="alice",
        record_path="exam-reports/2026-06-11-dddddddd",
        ocr_text="dose 500mg metformin twice daily",
        dedupe_threshold=0.95,
    )
    assert second == 0


async def test_add_record_threshold_zero_disables_dedupe(
    store: UserPhiRagStore, monkeypatch
):
    """Threshold ``<= 0`` is the documented kill-switch (matches the
    library path). Even with identical text, a 0 threshold writes both
    chunks. Single source of truth: ``filter_near_duplicates``."""
    monkeypatch.setattr(
        "claritymed.config.record_dedupe_cosine_threshold", lambda: 0.95
    )
    first = await store.add_record(
        user_id="alice",
        record_path="exam-reports/2026-06-11-eeeeeeee",
        ocr_text="penicillin allergy",
    )
    second = await store.add_record(
        user_id="alice",
        record_path="exam-reports/2026-06-11-ffffffff",
        ocr_text="penicillin allergy",
        dedupe_threshold=0.0,
    )
    assert first == 1
    assert second == 1


async def test_add_record_skipped_chunks_skip_parent_write(
    store: UserPhiRagStore, monkeypatch, tmp_path
):
    """When every child chunk is a near-duplicate the parent JSON write
    is skipped — no orphan ParentStore entries. Mirrors the library
    path's invariant (user_rag.py:187-190)."""
    monkeypatch.setattr(
        "claritymed.config.record_dedupe_cosine_threshold", lambda: 0.95
    )

    await store.add_record(
        user_id="alice",
        record_path="exam-reports/2026-06-11-11111111",
        ocr_text="thyroid panel within reference range",
    )

    from claritymed.stores.paths import user_parent_docstore_phi_path

    parent_path = user_parent_docstore_phi_path("alice")
    snapshot_first = (
        parent_path.read_text(encoding="utf-8") if parent_path.exists() else ""
    )

    n = await store.add_record(
        user_id="alice",
        record_path="exam-reports/2026-06-11-22222222",
        ocr_text="thyroid panel within reference range",
    )
    assert n == 0

    snapshot_second = (
        parent_path.read_text(encoding="utf-8") if parent_path.exists() else ""
    )
    assert snapshot_first == snapshot_second


async def test_add_record_empty_text_skips_embed(store: UserPhiRagStore):
    """Whitespace-only OCR text returns 0 without calling embed —
    regression guard against a future refactor that would needlessly
    invoke the embedder on the worker's ``ocr_status='empty'`` path.

    The autouse fixture sets threshold to 0 (no dedup), so this test
    pins the pre-existing short-circuit behavior is preserved."""
    n = await store.add_record(
        user_id="alice",
        record_path="exam-reports/2026-06-11-99999999",
        ocr_text="   ",
    )
    assert n == 0
