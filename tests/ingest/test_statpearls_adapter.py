"""Unit 9: StatPearls source adapter + ingest_corpus runner."""

from __future__ import annotations

import json

import pytest
from qdrant_client import AsyncQdrantClient

from claritymed.core.rag.chunking.base import (
    ChildChunk,
    ChunkedDocument,
    ParentChunk,
    RawDocument,
)
from claritymed.core.rag.embedding.base import Embedder, SparseVector
from claritymed.core.rag.parent_store import ParentStore
from claritymed.core.rag.qdrant_store import RagCollectionStore
from claritymed.ingest.corpus.base import ingest_corpus
from claritymed.ingest.corpus.statpearls import StatPearlsSource

DENSE_DIM = 4


class _StubEmbedder(Embedder):
    @property
    def dimension(self) -> int:
        return DENSE_DIM

    async def embed_dense(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t))] * DENSE_DIM for t in texts]

    async def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        return [{abs(hash(t)) % 100: 0.5} for t in texts]


class _StubChunker:
    """One parent + one child per doc (keeps test arithmetic predictable)."""

    def chunk(self, doc: RawDocument) -> ChunkedDocument:
        import uuid

        if not doc.text.strip():
            return ChunkedDocument(parents=[], children=[])
        pid = f"{doc.doc_id}#p0"
        return ChunkedDocument(
            parents=[
                ParentChunk(
                    parent_id=pid,
                    text=doc.text,
                    doc_id=doc.doc_id,
                    parent_index=0,
                    metadata=dict(doc.metadata),
                )
            ],
            children=[
                ChildChunk(
                    child_id=str(uuid.uuid5(uuid.NAMESPACE_URL, doc.doc_id)),
                    text=doc.text,
                    parent_id=pid,
                    doc_id=doc.doc_id,
                    chunk_index=0,
                    metadata=dict(doc.metadata),
                )
            ],
        )


# --- StatPearlsSource ---------------------------------------------------


def test_source_rejects_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError):
        StatPearlsSource(tmp_path / "does_not_exist")


def test_source_iterates_jsonl(tmp_path):
    f = tmp_path / "norm.jsonl"
    f.write_text(
        json.dumps({"doc_id": "NBK1", "title": "Aspirin", "text": "uses ..."})
        + "\n"
        + json.dumps({"doc_id": "NBK2", "title": "Ibuprofen", "text": "uses ..."})
        + "\n"
        + "not json\n",
        encoding="utf-8",
    )
    src = StatPearlsSource(tmp_path)
    docs = list(src.iter_raw_docs())
    assert len(docs) == 2
    assert docs[0].doc_id == "NBK1"
    assert "Aspirin" in docs[0].text
    assert docs[0].metadata["source_uri"].startswith("https://")


def test_source_skips_doc_without_text(tmp_path):
    f = tmp_path / "norm.jsonl"
    f.write_text(
        json.dumps({"doc_id": "NBK_EMPTY", "title": "Empty"}) + "\n",
        encoding="utf-8",
    )
    docs = list(StatPearlsSource(tmp_path).iter_raw_docs())
    assert docs == []


def test_source_parses_nxml(tmp_path):
    f = tmp_path / "NBK_X.nxml"
    f.write_text(
        "<article><front><article-meta><title-group>"
        "<article-title>Test Article</article-title></title-group></article-meta></front>"
        "<body><sec><p>First paragraph about diabetes.</p>"
        "<p>Second paragraph about treatment.</p></sec></body></article>",
        encoding="utf-8",
    )
    docs = list(StatPearlsSource(tmp_path).iter_raw_docs())
    assert len(docs) == 1
    assert docs[0].doc_id == "NBK_X"
    assert "Test Article" in docs[0].text
    assert "First paragraph" in docs[0].text
    assert "Second paragraph" in docs[0].text


def test_source_skips_bad_xml_file(tmp_path, caplog):
    (tmp_path / "broken.nxml").write_text("<<<not valid xml", encoding="utf-8")
    docs = list(StatPearlsSource(tmp_path).iter_raw_docs())
    assert docs == []


# --- ingest_corpus runner ---------------------------------------------


async def test_ingest_corpus_end_to_end(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "data.jsonl").write_text(
        json.dumps({"doc_id": "NBK1", "title": "A", "text": "first article body"})
        + "\n"
        + json.dumps({"doc_id": "NBK2", "title": "B", "text": "second article body"})
        + "\n",
        encoding="utf-8",
    )
    source = StatPearlsSource(raw)
    aclient = AsyncQdrantClient(":memory:")
    store = RagCollectionStore(aclient, "statpearls_en", DENSE_DIM)
    parent_store = ParentStore(tmp_path / "parent.json")

    stats = await ingest_corpus(
        source,
        chunker=_StubChunker(),
        embedder=_StubEmbedder(),
        store=store,
        parent_store=parent_store,
        limit=None,
        dry_run=False,
    )
    assert stats.docs_processed == 2
    assert stats.parents_written == 2
    assert stats.children_written == 2
    assert stats.docs_skipped == 0
    assert await store.count() == 2
    # Parent docstore persisted to disk.
    assert (tmp_path / "parent.json").exists()


async def test_ingest_corpus_dry_run_skips_qdrant(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "data.jsonl").write_text(
        json.dumps({"doc_id": "NBK1", "title": "A", "text": "body"}) + "\n",
        encoding="utf-8",
    )
    aclient = AsyncQdrantClient(":memory:")
    store = RagCollectionStore(aclient, "statpearls_en", DENSE_DIM)
    parent_store = ParentStore(tmp_path / "parent.json")

    stats = await ingest_corpus(
        StatPearlsSource(raw),
        chunker=_StubChunker(),
        embedder=_StubEmbedder(),
        store=store,
        parent_store=parent_store,
        dry_run=True,
    )
    assert stats.docs_processed == 1
    assert stats.parents_written == 1
    assert stats.children_written == 1
    # Dry run: nothing actually written to Qdrant.
    assert await store.count() == 0
    # No parent docstore persisted either.
    assert not (tmp_path / "parent.json").exists()


async def test_ingest_corpus_limit_caps_input(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "data.jsonl").write_text(
        "\n".join(
            json.dumps({"doc_id": f"NBK{i}", "title": "T", "text": "body"})
            for i in range(5)
        ),
        encoding="utf-8",
    )
    aclient = AsyncQdrantClient(":memory:")
    store = RagCollectionStore(aclient, "statpearls_en", DENSE_DIM)
    parent_store = ParentStore(tmp_path / "parent.json")

    stats = await ingest_corpus(
        StatPearlsSource(raw),
        chunker=_StubChunker(),
        embedder=_StubEmbedder(),
        store=store,
        parent_store=parent_store,
        limit=2,
    )
    assert stats.docs_processed == 2
    assert await store.count() == 2


async def test_ingest_corpus_skips_doc_with_no_chunks(tmp_path):
    """Chunker that produces nothing → counted as skipped, not crashed."""

    class _EmptyChunker:
        def chunk(self, doc: RawDocument) -> ChunkedDocument:
            return ChunkedDocument(parents=[], children=[])

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "data.jsonl").write_text(
        json.dumps({"doc_id": "NBK1", "title": "T", "text": "body"}),
        encoding="utf-8",
    )
    aclient = AsyncQdrantClient(":memory:")
    store = RagCollectionStore(aclient, "statpearls_en", DENSE_DIM)
    parent_store = ParentStore(tmp_path / "parent.json")

    stats = await ingest_corpus(
        StatPearlsSource(raw),
        chunker=_EmptyChunker(),
        embedder=_StubEmbedder(),
        store=store,
        parent_store=parent_store,
    )
    assert stats.docs_processed == 0
    assert stats.docs_skipped == 1


async def test_ingest_corpus_continues_when_one_doc_fails(tmp_path, caplog):
    """Chunker raising on one doc must not abort the whole run."""

    class _FlakyChunker:
        def __init__(self):
            self.calls = 0

        def chunk(self, doc: RawDocument) -> ChunkedDocument:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            import uuid

            pid = f"{doc.doc_id}#p0"
            return ChunkedDocument(
                parents=[
                    ParentChunk(
                        parent_id=pid,
                        text=doc.text,
                        doc_id=doc.doc_id,
                        parent_index=0,
                    )
                ],
                children=[
                    ChildChunk(
                        child_id=str(uuid.uuid5(uuid.NAMESPACE_URL, doc.doc_id)),
                        text=doc.text,
                        parent_id=pid,
                        doc_id=doc.doc_id,
                        chunk_index=0,
                    )
                ],
            )

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "data.jsonl").write_text(
        json.dumps({"doc_id": "NBK1", "title": "T", "text": "body1"})
        + "\n"
        + json.dumps({"doc_id": "NBK2", "title": "T", "text": "body2"}),
        encoding="utf-8",
    )
    aclient = AsyncQdrantClient(":memory:")
    store = RagCollectionStore(aclient, "statpearls_en", DENSE_DIM)
    parent_store = ParentStore(tmp_path / "parent.json")

    stats = await ingest_corpus(
        StatPearlsSource(raw),
        chunker=_FlakyChunker(),
        embedder=_StubEmbedder(),
        store=store,
        parent_store=parent_store,
    )
    # One doc skipped (the boom), one succeeded.
    assert stats.docs_processed == 1
    assert stats.docs_skipped == 1
    assert await store.count() == 1


# --- incremental / resume -------------------------------------------------


async def test_ingest_corpus_is_incremental_across_runs(tmp_path):
    """Second ingest of the same source must skip docs already in Qdrant.

    Drives the user-visible 'partial ingest can be safely re-run' contract.
    Without this, every interruption costs a full re-embed.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "data.jsonl").write_text(
        "\n".join(
            json.dumps({"doc_id": f"NBK{i}", "title": "T", "text": f"body {i}"})
            for i in range(3)
        ),
        encoding="utf-8",
    )
    aclient = AsyncQdrantClient(":memory:")
    store = RagCollectionStore(aclient, "statpearls_en", DENSE_DIM)
    parent_store = ParentStore(tmp_path / "parent.json")
    source = StatPearlsSource(raw)

    # First pass — all three docs are new.
    first = await ingest_corpus(
        source,
        chunker=_StubChunker(),
        embedder=_StubEmbedder(),
        store=store,
        parent_store=parent_store,
    )
    assert first.docs_processed == 3
    assert first.docs_resumed == 0

    # Second pass against the same source + store — every doc_id already
    # exists in Qdrant, so the resume probe must skip them all. Critical
    # property: docs_processed == 0 (no chunker / embedder calls), and
    # children_written == 0 (no Qdrant write).
    second = await ingest_corpus(
        source,
        chunker=_StubChunker(),
        embedder=_StubEmbedder(),
        store=store,
        parent_store=parent_store,
    )
    assert second.docs_processed == 0
    assert second.docs_resumed == 3
    assert second.children_written == 0


async def test_ingest_corpus_resumes_partial_run(tmp_path):
    """When only a subset is already ingested, resume skips those and
    processes the rest — the typical 'interrupted ingest, re-run later'
    case."""
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "data.jsonl").write_text(
        "\n".join(
            json.dumps({"doc_id": f"NBK{i}", "title": "T", "text": f"body {i}"})
            for i in range(5)
        ),
        encoding="utf-8",
    )
    aclient = AsyncQdrantClient(":memory:")
    store = RagCollectionStore(aclient, "statpearls_en", DENSE_DIM)
    parent_store = ParentStore(tmp_path / "parent.json")
    source = StatPearlsSource(raw)

    # First pass with limit=2 — only NBK0 and NBK1 are ingested.
    first = await ingest_corpus(
        source,
        chunker=_StubChunker(),
        embedder=_StubEmbedder(),
        store=store,
        parent_store=parent_store,
        limit=2,
    )
    assert first.docs_processed == 2
    assert first.docs_resumed == 0

    # Second pass without limit — should skip the first two, process the
    # remaining three.
    second = await ingest_corpus(
        source,
        chunker=_StubChunker(),
        embedder=_StubEmbedder(),
        store=store,
        parent_store=parent_store,
    )
    assert second.docs_processed == 3
    assert second.docs_resumed == 2
