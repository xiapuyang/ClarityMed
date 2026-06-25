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
