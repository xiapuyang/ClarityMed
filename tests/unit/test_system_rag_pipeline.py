"""Unit tests for ``claritymed.ingest.system_rag``.

The pipeline module is the shared implementation behind both the CLI
script (``scripts/init_system_rag.py``) and the admin SPA endpoint
(``POST /admin/rag/collections/upsert``). These tests pin its pure
helpers (slug, yaml snippet, resolve_metadata) and one async path that
exercises ``ocr_files`` against a stubbed OCR provider. The fully-wired
``ingest_system_rag`` flow is left to manual smoke runs since it
depends on a live Qdrant + embedder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from claritymed.core.rag.chunking.base import RawDocument
from claritymed.ingest import system_rag as mod
from claritymed.ingest.system_rag import (
    SystemRagIngestRequest,
    _GenericSource,
    build_yaml_snippet,
    ocr_files,
    resolve_metadata,
    slugify_doc_id,
)


# --- slugify_doc_id -----------------------------------------------------


@pytest.mark.parametrize(
    "stem, expected",
    [
        ("Hello World", "hello-world"),
        ("Foo_Bar.v2", "foo-bar-v2"),
        ("ATS-IDSA Guidelines 2026", "ats-idsa-guidelines-2026"),
        ("foo___bar---baz", "foo-bar-baz"),
        ("", "doc"),
        ("---", "doc"),
        ("已确认", "doc"),  # non-ASCII collapses to fallback
        ("CAP/2007.pdf-rev", "cap-2007-pdf-rev"),
    ],
)
def test_slugify_doc_id(stem: str, expected: str) -> None:
    assert slugify_doc_id(stem) == expected


# --- build_yaml_snippet -------------------------------------------------


def _make_req(**overrides) -> SystemRagIngestRequest:
    defaults = dict(
        name="cap_en",
        files=[],
        topics=["pneumonia"],
        language="en",
        cross_lingual=False,
        authority_tier=2,
        license=None,
    )
    defaults.update(overrides)
    return SystemRagIngestRequest(**defaults)


def test_build_yaml_snippet_minimal() -> None:
    snippet = build_yaml_snippet(_make_req())
    assert "- name: cap_en" in snippet
    assert "language: en" in snippet
    assert "cross_lingual: false" in snippet
    assert "authority_tier: 2" in snippet
    assert "        - pneumonia" in snippet
    assert "license: null" in snippet
    # ``size_chunks`` was removed -- the live count comes from
    # ``corpora list`` querying Qdrant, not the snapshot in yaml.
    assert "size_chunks" not in snippet


def test_build_yaml_snippet_renders_multiple_topics_in_order() -> None:
    snippet = build_yaml_snippet(
        _make_req(topics=["pneumonia", "respiratory infections", "antibiotics"])
    )
    body = snippet.split("topics:")[1]
    assert body.split("\n")[1].strip() == "- pneumonia"
    assert body.split("\n")[2].strip() == "- respiratory infections"
    assert body.split("\n")[3].strip() == "- antibiotics"


def test_build_yaml_snippet_quotes_license_when_present() -> None:
    snippet = build_yaml_snippet(_make_req(license="ATS/IDSA (educational use)"))
    assert 'license: "ATS/IDSA (educational use)"' in snippet


def test_build_yaml_snippet_cross_lingual_true() -> None:
    snippet = build_yaml_snippet(
        _make_req(cross_lingual=True, language="zh", authority_tier=1)
    )
    assert "cross_lingual: true" in snippet
    assert "language: zh" in snippet
    assert "authority_tier: 1" in snippet


def test_build_yaml_snippet_includes_source_uri_prefix_when_set() -> None:
    snippet = build_yaml_snippet(
        _make_req(source_uri_prefix="https://www.ncbi.nlm.nih.gov/books/")
    )
    assert 'source_uri_prefix: "https://www.ncbi.nlm.nih.gov/books/"' in snippet


def test_build_yaml_snippet_emits_null_source_uri_prefix_by_default() -> None:
    snippet = build_yaml_snippet(_make_req())
    assert "source_uri_prefix: null" in snippet


def test_build_yaml_snippet_renders_empty_topics_inline() -> None:
    """Empty topics: ``topics: []`` so the YAML key isn't a null block."""
    snippet = build_yaml_snippet(_make_req(topics=[]))
    assert "      topics: []" in snippet
    assert "      topics:\n      disease_codes" not in snippet


# --- resolve_metadata ---------------------------------------------------


def _stub_retrieval_config(collections: list) -> argparse.Namespace:
    return argparse.Namespace(
        system_rag=argparse.Namespace(collections=list(collections))
    )


def _existing_entry(**overrides) -> argparse.Namespace:
    defaults = dict(
        name="cap_en",
        topics=("community-acquired pneumonia", "respiratory infections"),
        language="en",
        cross_lingual=True,
        authority_tier=1,
        license="ATS/IDSA",
        source_uri_prefix="https://www.atsjournals.org/",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_resolve_metadata_inherits_from_existing_yaml(monkeypatch) -> None:
    """Append: missing topic/language/tier/license are filled from yaml."""
    existing = _existing_entry()
    monkeypatch.setattr(
        mod, "load_retrieval_config", lambda: _stub_retrieval_config([existing])
    )

    req = SystemRagIngestRequest(name="cap_en", files=[])
    is_existing = resolve_metadata(req)

    assert is_existing is True
    assert req.topics == [
        "community-acquired pneumonia",
        "respiratory infections",
    ]
    assert req.language == "en"
    # ``cross_lingual`` defaults to False on the dataclass; on append we
    # treat False as "unspecified" so the yaml ``true`` survives.
    # Otherwise every append would silently flip cross-lingual off.
    assert req.cross_lingual is True
    assert req.authority_tier == 1
    assert req.license == "ATS/IDSA"
    assert req.source_uri_prefix == "https://www.atsjournals.org/"


def test_resolve_metadata_caller_values_win_over_existing(monkeypatch) -> None:
    """Explicit caller values trump yaml inheritance."""
    existing = _existing_entry()
    monkeypatch.setattr(
        mod, "load_retrieval_config", lambda: _stub_retrieval_config([existing])
    )

    req = SystemRagIngestRequest(
        name="cap_en",
        files=[],
        topics=["antibiotics"],
        language="zh",
        authority_tier=3,
        license="custom",
    )
    resolve_metadata(req)

    assert req.topics == ["antibiotics"]
    assert req.language == "zh"
    assert req.authority_tier == 3
    assert req.license == "custom"


def test_resolve_metadata_allows_empty_topic_for_new_collection(monkeypatch) -> None:
    """Empty topics is legitimate — centroid router scores by query vector."""
    monkeypatch.setattr(
        mod, "load_retrieval_config", lambda: _stub_retrieval_config([])
    )
    req = SystemRagIngestRequest(name="brand_new_en", files=[], topics=[])
    is_existing = resolve_metadata(req)

    assert is_existing is False
    assert req.topics == []
    assert req.language == "en"
    assert req.authority_tier == 2
    assert req.source_uri_prefix is None


def test_resolve_metadata_applies_historical_defaults_for_new_collection(
    monkeypatch,
) -> None:
    """New collection: language → 'en', tier → 2, prefix → None."""
    monkeypatch.setattr(
        mod, "load_retrieval_config", lambda: _stub_retrieval_config([])
    )
    req = SystemRagIngestRequest(name="brand_new_en", files=[], topics=["pneumonia"])
    resolve_metadata(req)

    assert req.language == "en"
    assert req.authority_tier == 2
    assert req.cross_lingual is False
    assert req.license is None
    assert req.source_uri_prefix is None


# --- _GenericSource -----------------------------------------------------


def test_generic_source_exposes_name_and_yields_in_order() -> None:
    docs = [
        RawDocument(doc_id="a", text="alpha"),
        RawDocument(doc_id="b", text="beta"),
        RawDocument(doc_id="c", text="gamma"),
    ]
    src = _GenericSource("cap_en", docs)
    assert src.name == "cap_en"
    assert [d.doc_id for d in src.iter_raw_docs()] == ["a", "b", "c"]


# --- ocr_files ----------------------------------------------------------


class _StubExtractResult:
    def __init__(self, text: str, provider_used: str = "pymupdf") -> None:
        self.text = text
        self.provider_used = provider_used


class _StubOcrProvider:
    def __init__(self, by_name: dict[str, str]) -> None:
        self._by_name = by_name
        self.calls: list[str] = []

    async def extract_text(self, path) -> _StubExtractResult:  # noqa: ANN001
        self.calls.append(path.name)
        text = self._by_name.get(path.name, "")
        return _StubExtractResult(text)


async def test_ocr_files_skips_missing_and_empty(tmp_path, monkeypatch) -> None:
    a = tmp_path / "a.pdf"
    a.write_bytes(b"\x25PDF-1.4 stub")
    b = tmp_path / "b.pdf"
    b.write_bytes(b"\x25PDF-1.4 stub")
    missing = tmp_path / "ghost.pdf"

    stub = _StubOcrProvider({"a.pdf": "first body text", "b.pdf": "   "})
    monkeypatch.setattr(mod, "make_ocr_provider", lambda: stub)
    monkeypatch.setattr(
        mod,
        "load_ocr_config",
        lambda: argparse.Namespace(text_extensions=[".txt", ".md"]),
    )

    docs = await ocr_files([a, b, missing], collection_name="cap_en", language="en")

    assert len(docs) == 1
    doc = docs[0]
    assert doc.doc_id == "a"
    assert doc.text == "first body text"
    assert doc.language == "en"
    assert doc.metadata["collection"] == "cap_en"
    assert doc.metadata["doc_title"] == "a"
    assert doc.metadata["ocr_provider"] == "pymupdf"
    assert stub.calls == ["a.pdf", "b.pdf"]


async def test_ocr_files_disambiguates_duplicate_slugs(tmp_path, monkeypatch) -> None:
    a = tmp_path / "ATS IDSA.pdf"
    a.write_bytes(b"x")
    b = tmp_path / "ats-idsa.pdf"
    b.write_bytes(b"x")
    c = tmp_path / "ats_idsa.pdf"
    c.write_bytes(b"x")

    stub = _StubOcrProvider(
        {"ATS IDSA.pdf": "v1", "ats-idsa.pdf": "v2", "ats_idsa.pdf": "v3"}
    )
    monkeypatch.setattr(mod, "make_ocr_provider", lambda: stub)
    monkeypatch.setattr(
        mod,
        "load_ocr_config",
        lambda: argparse.Namespace(text_extensions=[".txt", ".md"]),
    )

    docs = await ocr_files([a, b, c], collection_name="cap_en", language="en")
    assert [d.doc_id for d in docs] == ["ats-idsa", "ats-idsa-2", "ats-idsa-3"]


async def test_ocr_files_text_extension_bypasses_ocr(tmp_path, monkeypatch) -> None:
    note = tmp_path / "notes.md"
    note.write_text("# Pneumonia notes\n\nbody text", encoding="utf-8")

    stub = _StubOcrProvider({})  # OCR shouldn't be called for .md
    monkeypatch.setattr(mod, "make_ocr_provider", lambda: stub)
    monkeypatch.setattr(
        mod,
        "load_ocr_config",
        lambda: argparse.Namespace(text_extensions=[".txt", ".md"]),
    )

    docs = await ocr_files([note], collection_name="cap_en", language="en")
    assert len(docs) == 1
    assert "Pneumonia notes" in docs[0].text
    assert docs[0].metadata["ocr_provider"] == "text"
    assert stub.calls == []


async def test_ocr_files_calls_on_progress(tmp_path, monkeypatch) -> None:
    """Progress callback fires per file with the right phase labels."""
    a = tmp_path / "ok.pdf"
    a.write_bytes(b"x")
    missing = tmp_path / "nope.pdf"

    stub = _StubOcrProvider({"ok.pdf": "body"})
    monkeypatch.setattr(mod, "make_ocr_provider", lambda: stub)
    monkeypatch.setattr(
        mod,
        "load_ocr_config",
        lambda: argparse.Namespace(text_extensions=[".txt", ".md"]),
    )

    lines: list[str] = []
    await ocr_files(
        [a, missing],
        collection_name="cap_en",
        language="en",
        on_progress=lines.append,
    )
    assert any("extract: ok.pdf" in line for line in lines)
    assert any("skip (missing): nope.pdf" in line for line in lines)


# --- validate_name + _NoOpEmbedder -------------------------------------


def test_validate_name_accepts_canonical_name() -> None:
    """Normal name passes — no exception."""
    _make_req(name="cap_en").validate_name()


@pytest.mark.parametrize(
    "bad",
    [
        "CAP",  # uppercase
        "1cap",  # starts with digit
        "_cap",  # starts with underscore
        "cap-en",  # hyphen not allowed
        "cap.en",  # dot not allowed
        "a" * 65,  # exceeds 64-char limit
    ],
)
def test_validate_name_rejects_invalid_names(bad: str) -> None:
    with pytest.raises(ValueError) as ei:
        _make_req(name=bad).validate_name()
    # Pattern is surfaced in the error so the operator sees the rule
    # they violated, not just "rejected".
    assert "name must match" in str(ei.value)


def test_noop_embedder_advertises_fallback_dimension() -> None:
    """Dry-run stand-in carries the fallback dimension constant."""
    embedder = mod._NoOpEmbedder()
    assert embedder.dimension == mod._DENSE_DIM_FALLBACK


async def test_noop_embedder_returns_zero_vectors_and_empty_sparse() -> None:
    embedder = mod._NoOpEmbedder()
    dense = await embedder.embed_dense(["a", "b", "c"])
    sparse = await embedder.embed_sparse(["a", "b", "c"])
    assert len(dense) == 3
    assert all(len(v) == embedder.dimension for v in dense)
    assert all(all(x == 0.0 for x in v) for v in dense)
    assert sparse == [{}, {}, {}]


# --- ocr_files extraction-failure path ----------------------------------


class _RaisingOcrProvider:
    """Provider that raises on every extract — exercises the batch-survives path."""

    async def extract_text(self, path):  # noqa: ANN001
        raise RuntimeError(f"ocr broke on {path.name}")


async def test_ocr_files_skips_files_whose_extraction_raises(
    tmp_path, monkeypatch
) -> None:
    """One bad PDF must not kill the whole batch — failure path is logged + skipped."""
    a = tmp_path / "broken.pdf"
    a.write_bytes(b"\x25PDF garbage")
    monkeypatch.setattr(mod, "make_ocr_provider", lambda: _RaisingOcrProvider())
    monkeypatch.setattr(
        mod,
        "load_ocr_config",
        lambda: argparse.Namespace(text_extensions=[".txt", ".md"]),
    )

    lines: list[str] = []
    docs = await ocr_files(
        [a], collection_name="cap_en", language="en", on_progress=lines.append
    )
    assert docs == []
    # The exception message is surfaced verbatim so the operator can
    # pivot from the run log straight to the broken file.
    assert any("skip (ocr broke" in line and "broken.pdf" in line for line in lines)


# --- _refresh_centroid --------------------------------------------------


class _FakeAClient:
    """Stand-in async qdrant client — close() is the only method the pipeline
    actually calls in the cleanup path."""

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


async def test_refresh_centroid_happy_path(monkeypatch) -> None:
    """``maybe_refresh`` succeeds → returns True and emits a progress line."""

    async def _ok(aclient, name, store, force):  # noqa: ANN001
        assert force is True
        assert name == "cap_en"

    monkeypatch.setattr(mod, "maybe_refresh", _ok)

    lines: list[str] = []
    ok = await mod._refresh_centroid(_FakeAClient(), "cap_en", on_progress=lines.append)
    assert ok is True
    assert any("centroid refreshed: cap_en" in line for line in lines)


async def test_refresh_centroid_swallows_exception_returns_false(monkeypatch) -> None:
    """``maybe_refresh`` raises → returns False, error visible in progress."""

    async def _bad(aclient, name, store, force):  # noqa: ANN001
        raise RuntimeError("qdrant unreachable")

    monkeypatch.setattr(mod, "maybe_refresh", _bad)

    lines: list[str] = []
    ok = await mod._refresh_centroid(_FakeAClient(), "cap_en", on_progress=lines.append)
    assert ok is False
    assert any("centroid refresh failed" in line for line in lines)


# --- ingest_system_rag end-to-end ---------------------------------------


def _wire_pipeline(monkeypatch, *, stats, refresh_returns=True) -> _FakeAClient:
    """Mock every heavy IO dep ``ingest_system_rag`` depends on.

    Returns the fake aclient so the test can assert close() ran.
    """
    aclient = _FakeAClient()

    # Empty existing collections — req hits the "new collection" branch.
    monkeypatch.setattr(
        mod,
        "load_retrieval_config",
        lambda: argparse.Namespace(
            system_rag=argparse.Namespace(collections=[]),
            qdrant=argparse.Namespace(url="http://fake", api_key_env=None),
        ),
    )
    monkeypatch.setattr(mod, "build_qdrant_client", lambda url, api_key_env: aclient)

    class _FakeChunker:
        pass

    class _FakeEmbedder:
        dimension = 64

    monkeypatch.setattr(mod, "build_chunker", lambda: _FakeChunker())
    monkeypatch.setattr(mod, "build_embedder", lambda: _FakeEmbedder())

    class _FakeRagStore:
        def __init__(self, **kwargs) -> None:  # noqa: ANN003
            self.kwargs = kwargs

    class _FakeParentStore:
        def __init__(self, path) -> None:  # noqa: ANN001
            self.path = path

    monkeypatch.setattr(mod, "RagCollectionStore", _FakeRagStore)
    monkeypatch.setattr(mod, "ParentStore", _FakeParentStore)
    monkeypatch.setattr(mod, "shared_parent_docstore_path", lambda: "/tmp/parents.json")

    async def _ingest_corpus(*args, **kwargs):  # noqa: ANN002, ANN003
        return stats

    monkeypatch.setattr(mod, "ingest_corpus", _ingest_corpus)

    async def _refresh(aclient, name, on_progress=None):  # noqa: ANN001
        if on_progress:
            on_progress(
                f"centroid refreshed: {name}"
                if refresh_returns
                else "centroid refresh failed (forced)"
            )
        return refresh_returns

    monkeypatch.setattr(mod, "_refresh_centroid", _refresh)
    return aclient


def _make_pdf(tmp_path, name: str, content: bytes = b"%PDF-1.4 stub") -> Path:
    p = tmp_path / name
    p.write_bytes(content)
    return p


async def test_ingest_system_rag_raises_when_no_docs_survive_ocr(
    tmp_path, monkeypatch
) -> None:
    """All inputs missing → RuntimeError before chunk/embed/upsert wiring runs."""
    monkeypatch.setattr(
        mod,
        "load_retrieval_config",
        lambda: argparse.Namespace(
            system_rag=argparse.Namespace(collections=[]),
            qdrant=argparse.Namespace(url="http://x", api_key_env=None),
        ),
    )

    class _EmptyOcr:
        async def extract_text(self, path):  # noqa: ANN001
            return _StubExtractResult("")

    monkeypatch.setattr(mod, "make_ocr_provider", lambda: _EmptyOcr())
    monkeypatch.setattr(
        mod,
        "load_ocr_config",
        lambda: argparse.Namespace(text_extensions=[".txt", ".md"]),
    )

    req = SystemRagIngestRequest(
        name="cap_en", files=[_make_pdf(tmp_path, "empty.pdf")]
    )
    with pytest.raises(RuntimeError) as ei:
        await mod.ingest_system_rag(req)
    assert "no documents to ingest" in str(ei.value)


async def test_ingest_system_rag_dry_run_skips_centroid_refresh(
    tmp_path, monkeypatch
) -> None:
    """Dry-run path: pipeline runs through, centroid step is short-circuited."""
    from claritymed.ingest.corpus.base import IngestStats

    stats = IngestStats(
        source="cap_en",
        docs_processed=1,
        parents_written=1,
        children_written=5,
        docs_skipped=0,
    )
    aclient = _wire_pipeline(monkeypatch, stats=stats, refresh_returns=True)

    pdf = _make_pdf(tmp_path, "a.pdf")
    stub = _StubOcrProvider({"a.pdf": "body"})
    monkeypatch.setattr(mod, "make_ocr_provider", lambda: stub)
    monkeypatch.setattr(
        mod,
        "load_ocr_config",
        lambda: argparse.Namespace(text_extensions=[".txt", ".md"]),
    )

    req = SystemRagIngestRequest(name="cap_en", files=[pdf], dry_run=True)
    lines: list[str] = []
    result = await mod.ingest_system_rag(req, on_progress=lines.append)

    assert result.stats == stats
    assert result.is_new_collection is True
    # Dry run → centroid_refreshed must be False (we never even called it).
    assert result.centroid_refreshed is False
    assert aclient.closed is True  # ``finally`` always closes the qdrant aclient
    assert any("ingest: chunk + embed + upsert" in line for line in lines)
    assert "ingested: 1 docs" in "\n".join(lines)


async def test_ingest_system_rag_refreshes_centroid_when_children_written(
    tmp_path, monkeypatch
) -> None:
    """Real run with children_written > 0 → centroid refresh fires."""
    from claritymed.ingest.corpus.base import IngestStats

    stats = IngestStats(
        source="cap_en",
        docs_processed=1,
        parents_written=1,
        children_written=3,
        docs_skipped=0,
    )
    aclient = _wire_pipeline(monkeypatch, stats=stats, refresh_returns=True)

    pdf = _make_pdf(tmp_path, "doc.pdf")
    stub = _StubOcrProvider({"doc.pdf": "body"})
    monkeypatch.setattr(mod, "make_ocr_provider", lambda: stub)
    monkeypatch.setattr(
        mod,
        "load_ocr_config",
        lambda: argparse.Namespace(text_extensions=[".txt", ".md"]),
    )

    req = SystemRagIngestRequest(name="cap_en", files=[pdf])
    result = await mod.ingest_system_rag(req)

    assert result.centroid_refreshed is True
    assert aclient.closed is True


async def test_ingest_system_rag_skips_centroid_when_no_children_written(
    tmp_path, monkeypatch
) -> None:
    """children_written == 0 → no centroid refresh (skipped via the guard)."""
    from claritymed.ingest.corpus.base import IngestStats

    stats = IngestStats(
        source="cap_en",
        docs_processed=1,
        parents_written=1,
        children_written=0,
        docs_skipped=0,
    )
    aclient = _wire_pipeline(monkeypatch, stats=stats, refresh_returns=True)

    pdf = _make_pdf(tmp_path, "doc.pdf")
    stub = _StubOcrProvider({"doc.pdf": "body"})
    monkeypatch.setattr(mod, "make_ocr_provider", lambda: stub)
    monkeypatch.setattr(
        mod,
        "load_ocr_config",
        lambda: argparse.Namespace(text_extensions=[".txt", ".md"]),
    )

    req = SystemRagIngestRequest(name="cap_en", files=[pdf])
    result = await mod.ingest_system_rag(req)

    # No children → router stays on rule-based; centroid_refreshed stays False.
    assert result.centroid_refreshed is False
    assert aclient.closed is True


async def test_ingest_system_rag_honours_limit(tmp_path, monkeypatch) -> None:
    """``limit`` truncates the file list before OCR."""
    from claritymed.ingest.corpus.base import IngestStats

    stats = IngestStats(
        source="cap_en",
        docs_processed=1,
        parents_written=1,
        children_written=2,
        docs_skipped=0,
    )
    aclient = _wire_pipeline(monkeypatch, stats=stats, refresh_returns=True)

    a = _make_pdf(tmp_path, "a.pdf")
    b = _make_pdf(tmp_path, "b.pdf")
    c = _make_pdf(tmp_path, "c.pdf")
    stub = _StubOcrProvider({"a.pdf": "body-a", "b.pdf": "body-b", "c.pdf": "body-c"})
    monkeypatch.setattr(mod, "make_ocr_provider", lambda: stub)
    monkeypatch.setattr(
        mod,
        "load_ocr_config",
        lambda: argparse.Namespace(text_extensions=[".txt", ".md"]),
    )

    req = SystemRagIngestRequest(name="cap_en", files=[a, b, c], limit=2, dry_run=True)
    await mod.ingest_system_rag(req)

    # Only the first two paths should reach the OCR step.
    assert stub.calls == ["a.pdf", "b.pdf"]
    assert aclient.closed is True
