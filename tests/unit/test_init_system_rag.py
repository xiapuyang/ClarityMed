"""Unit tests for ``scripts/init_system_rag.py``.

These cover the pure helpers (slug, argparse validation, YAML snippet)
plus one async path that exercises ``_ocr_files`` against a stubbed OCR
provider. The fully-wired ``_amain`` flow is left to manual smoke runs
since it depends on a live Qdrant + embedder.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

from claritymed.core.rag.chunking.base import RawDocument


def _load_script_module():
    """Load ``scripts/init_system_rag.py`` by path.

    ``scripts/`` is not a package on the import path (it ships as raw
    files alongside ``src/``), so the test loads the script via
    importlib instead of adjusting pythonpath project-wide.
    """
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "init_system_rag.py"
    spec = importlib.util.spec_from_file_location(
        "_init_system_rag_under_test", script_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_script_module()


# --- _slugify_doc_id ----------------------------------------------------


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
    assert mod._slugify_doc_id(stem) == expected


# --- _parse_args --------------------------------------------------------


def test_parse_args_minimal_happy_path() -> None:
    args = mod._parse_args(
        ["--name", "pneumonia_en", "--topic", "pneumonia", "a.pdf", "b.pdf"]
    )
    assert args.name == "pneumonia_en"
    assert args.topic == ["pneumonia"]
    assert [p.name for p in args.paths] == ["a.pdf", "b.pdf"]
    # ``--language`` / ``--authority-tier`` default to None at parse
    # time so ``_resolve_metadata`` can either inherit from the yaml on
    # append or apply the en/tier-2 defaults for a new collection.
    assert args.language is None
    assert args.authority_tier is None
    assert args.cross_lingual is False
    assert args.license is None
    assert args.dry_run is False
    assert args.dedupe_cosine_threshold == 0.0


def test_parse_args_collects_repeated_topics() -> None:
    args = mod._parse_args(
        [
            "--name",
            "cap_en",
            "--topic",
            "pneumonia",
            "--topic",
            "respiratory infections",
            "--topic",
            "antibiotics",
            "x.pdf",
        ]
    )
    assert args.topic == ["pneumonia", "respiratory infections", "antibiotics"]


def test_parse_args_full_optional_set() -> None:
    args = mod._parse_args(
        [
            "--name",
            "cap_en",
            "--topic",
            "pneumonia",
            "--language",
            "zh",
            "--cross-lingual",
            "--authority-tier",
            "1",
            "--license",
            "ATS/IDSA (educational)",
            "--dedupe-cosine-threshold",
            "0.92",
            "--dry-run",
            "x.pdf",
        ]
    )
    assert args.language == "zh"
    assert args.cross_lingual is True
    assert args.authority_tier == 1
    assert args.license == "ATS/IDSA (educational)"
    assert args.dedupe_cosine_threshold == pytest.approx(0.92)
    assert args.dry_run is True


def test_parse_args_allows_missing_topic() -> None:
    """``--topic`` is now optional at parse time -- inherited on append."""
    args = mod._parse_args(["--name", "cap_en", "a.pdf"])
    assert args.topic == []


def test_parse_args_rejects_invalid_name() -> None:
    # Starts with an uppercase letter — must match ^[a-z][a-z0-9_]+$.
    with pytest.raises(SystemExit):
        mod._parse_args(["--name", "BadName", "--topic", "x", "a.pdf"])


def test_parse_args_rejects_name_starting_with_digit() -> None:
    with pytest.raises(SystemExit):
        mod._parse_args(["--name", "2007_cap", "--topic", "x", "a.pdf"])


def test_parse_args_rejects_invalid_language() -> None:
    with pytest.raises(SystemExit):
        mod._parse_args(
            ["--name", "cap_en", "--topic", "x", "--language", "fr", "a.pdf"]
        )


def test_parse_args_rejects_invalid_authority_tier() -> None:
    with pytest.raises(SystemExit):
        mod._parse_args(
            [
                "--name",
                "cap_en",
                "--topic",
                "x",
                "--authority-tier",
                "5",
                "a.pdf",
            ]
        )


# --- _build_yaml_snippet ------------------------------------------------


def _make_ns(**overrides) -> argparse.Namespace:
    defaults = dict(
        name="cap_en",
        topic=["pneumonia"],
        language="en",
        cross_lingual=False,
        authority_tier=2,
        license=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_build_yaml_snippet_minimal() -> None:
    ns = _make_ns()
    snippet = mod._build_yaml_snippet(ns)
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
    ns = _make_ns(topic=["pneumonia", "respiratory infections", "antibiotics"])
    snippet = mod._build_yaml_snippet(ns)
    body = snippet.split("topics:")[1]
    # Topics indented 8 spaces under topics: block, order preserved.
    assert body.split("\n")[1].strip() == "- pneumonia"
    assert body.split("\n")[2].strip() == "- respiratory infections"
    assert body.split("\n")[3].strip() == "- antibiotics"


def test_build_yaml_snippet_quotes_license_when_present() -> None:
    ns = _make_ns(license="ATS/IDSA (educational use)")
    snippet = mod._build_yaml_snippet(ns)
    assert 'license: "ATS/IDSA (educational use)"' in snippet


def test_build_yaml_snippet_cross_lingual_true() -> None:
    ns = _make_ns(cross_lingual=True, language="zh", authority_tier=1)
    snippet = mod._build_yaml_snippet(ns)
    assert "cross_lingual: true" in snippet
    assert "language: zh" in snippet
    assert "authority_tier: 1" in snippet


def test_build_yaml_snippet_includes_source_uri_prefix_when_set() -> None:
    """``_resolve_metadata`` carries the yaml prefix forward on append."""
    ns = _make_ns(source_uri_prefix="https://www.ncbi.nlm.nih.gov/books/")
    snippet = mod._build_yaml_snippet(ns)
    assert 'source_uri_prefix: "https://www.ncbi.nlm.nih.gov/books/"' in snippet


def test_build_yaml_snippet_emits_null_source_uri_prefix_by_default() -> None:
    ns = _make_ns()  # no source_uri_prefix attribute on the namespace
    snippet = mod._build_yaml_snippet(ns)
    assert "source_uri_prefix: null" in snippet


def test_build_yaml_snippet_renders_empty_topics_inline() -> None:
    """Empty topics: ``topics: []`` so the YAML key isn't a null block."""
    ns = _make_ns(topic=[])
    snippet = mod._build_yaml_snippet(ns)
    assert "      topics: []" in snippet
    # Sanity: there should be no dangling ``topics:`` line followed by
    # a non-topic key on the next line (which would coerce to None).
    assert "      topics:\n      disease_codes" not in snippet


# --- _resolve_metadata --------------------------------------------------


def _stub_retrieval_config(collections: list) -> argparse.Namespace:
    """Build a fake retrieval config exposing only what _resolve_metadata reads.

    ``CollectionMetadata`` is a frozen pydantic model -- constructing
    real instances per test would couple this unit to schema changes
    unrelated to the resolver's behaviour. The resolver only touches
    ``.system_rag.collections[*].{name,topics,language,...}`` so a
    namespace stand-in is enough.
    """
    return argparse.Namespace(
        system_rag=argparse.Namespace(collections=list(collections))
    )


def _existing_entry(**overrides) -> argparse.Namespace:
    """Build a stand-in ``CollectionMetadata`` for resolver tests."""
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


def _fresh_args(**overrides) -> argparse.Namespace:
    """Mimic argparse output: all overridable fields default to ``None``."""
    defaults = dict(
        name="cap_en",
        topic=[],
        language=None,
        cross_lingual=False,
        authority_tier=None,
        license=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_resolve_metadata_inherits_from_existing_yaml(monkeypatch) -> None:
    """Append: missing topic/language/tier/license are filled from yaml."""
    existing = _existing_entry()
    monkeypatch.setattr(
        mod, "load_retrieval_config", lambda: _stub_retrieval_config([existing])
    )

    args = _fresh_args(name="cap_en")
    mod._resolve_metadata(args)

    assert args.topic == [
        "community-acquired pneumonia",
        "respiratory infections",
    ]
    assert args.language == "en"
    # ``--cross-lingual`` is store_true; False on the namespace must be
    # treated as "unspecified" so the yaml ``true`` survives. Otherwise
    # every append would flip cross-lingual collections off silently.
    assert args.cross_lingual is True
    assert args.authority_tier == 1
    assert args.license == "ATS/IDSA"
    assert args.source_uri_prefix == "https://www.atsjournals.org/"


def test_resolve_metadata_cli_overrides_existing_with_warning(
    monkeypatch, capsys
) -> None:
    """Explicit CLI flags win, but disagreement surfaces as a warning."""
    existing = _existing_entry()
    monkeypatch.setattr(
        mod, "load_retrieval_config", lambda: _stub_retrieval_config([existing])
    )

    args = _fresh_args(
        name="cap_en",
        topic=["antibiotics"],
        language="zh",
        authority_tier=3,
        license="custom",
    )
    mod._resolve_metadata(args)

    captured = capsys.readouterr().out
    assert args.topic == ["antibiotics"]
    assert args.language == "zh"
    assert args.authority_tier == 3
    assert args.license == "custom"
    # At least one yellow warning was emitted so the operator can spot
    # the divergence before pasting the snippet over the live config.
    assert "warning" in captured.lower()


def test_resolve_metadata_allows_empty_topic_for_new_collection(monkeypatch) -> None:
    """``--topic`` is optional even for a brand-new collection.

    The active ``centroid_classifier`` router scores by query-vector
    cosine, so an empty topics list isn't punished. The rule-based
    fallback also short-circuits to 1.0 when topics is empty, so the
    collection still survives routing while waiting for its centroid.
    """
    monkeypatch.setattr(
        mod, "load_retrieval_config", lambda: _stub_retrieval_config([])
    )
    args = _fresh_args(name="brand_new_en", topic=[])
    mod._resolve_metadata(args)

    assert args.topic == []
    assert args.language == "en"
    assert args.authority_tier == 2
    assert args.source_uri_prefix is None


def test_resolve_metadata_applies_historical_defaults_for_new_collection(
    monkeypatch,
) -> None:
    """New collection: language defaults to 'en', tier to 2, prefix to None."""
    monkeypatch.setattr(
        mod, "load_retrieval_config", lambda: _stub_retrieval_config([])
    )
    args = _fresh_args(name="brand_new_en", topic=["pneumonia"])
    mod._resolve_metadata(args)

    assert args.language == "en"
    assert args.authority_tier == 2
    assert args.cross_lingual is False
    assert args.license is None
    assert args.source_uri_prefix is None


# --- _GenericSource -----------------------------------------------------


def test_generic_source_exposes_name_and_yields_in_order() -> None:
    docs = [
        RawDocument(doc_id="a", text="alpha"),
        RawDocument(doc_id="b", text="beta"),
        RawDocument(doc_id="c", text="gamma"),
    ]
    src = mod._GenericSource("cap_en", docs)
    assert src.name == "cap_en"
    assert [d.doc_id for d in src.iter_raw_docs()] == ["a", "b", "c"]


# --- _ocr_files ---------------------------------------------------------


class _StubExtractResult:
    """Mimics ``ExtractResult`` enough for ``_extract_one``."""

    def __init__(self, text: str, provider_used: str = "pymupdf") -> None:
        self.text = text
        self.provider_used = provider_used


class _StubOcrProvider:
    """Returns a deterministic stub per filename so we can assert routing."""

    def __init__(self, by_name: dict[str, str]) -> None:
        self._by_name = by_name
        self.calls: list[str] = []

    async def extract_text(self, path) -> _StubExtractResult:  # noqa: ANN001
        self.calls.append(path.name)
        text = self._by_name.get(path.name, "")
        return _StubExtractResult(text)


async def test_ocr_files_skips_missing_and_empty(tmp_path, monkeypatch) -> None:
    """Missing files and empty-OCR files are skipped without crashing the batch."""
    a = tmp_path / "a.pdf"
    a.write_bytes(b"\x25PDF-1.4 stub")  # content irrelevant — stub OCR
    b = tmp_path / "b.pdf"
    b.write_bytes(b"\x25PDF-1.4 stub")
    missing = tmp_path / "ghost.pdf"

    stub = _StubOcrProvider({"a.pdf": "first body text", "b.pdf": "   "})
    monkeypatch.setattr(mod, "make_ocr_provider", lambda: stub)
    # Mock load_ocr_config so the test doesn't read a real YAML on disk.
    monkeypatch.setattr(
        mod,
        "load_ocr_config",
        lambda: argparse.Namespace(text_extensions=[".txt", ".md"]),
    )

    docs = await mod._ocr_files(
        [a, b, missing], collection_name="cap_en", language="en"
    )

    # Only ``a.pdf`` produced a doc; ``b.pdf`` was empty, ``ghost.pdf`` missing.
    assert len(docs) == 1
    doc = docs[0]
    assert doc.doc_id == "a"
    assert doc.text == "first body text"
    assert doc.language == "en"
    assert doc.metadata["collection"] == "cap_en"
    assert doc.metadata["doc_title"] == "a"
    assert doc.metadata["ocr_provider"] == "pymupdf"
    # OCR was invoked for the two real files but not the missing one.
    assert stub.calls == ["a.pdf", "b.pdf"]


async def test_ocr_files_disambiguates_duplicate_slugs(tmp_path, monkeypatch) -> None:
    """Two files whose stems slugify to the same id get suffixes (-2, -3, ...)."""
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

    docs = await mod._ocr_files([a, b, c], collection_name="cap_en", language="en")
    assert [d.doc_id for d in docs] == ["ats-idsa", "ats-idsa-2", "ats-idsa-3"]


async def test_ocr_files_text_extension_bypasses_ocr(tmp_path, monkeypatch) -> None:
    """``.txt`` / ``.md`` etc. read directly as utf-8 (matches /upload paste fast-path)."""
    note = tmp_path / "notes.md"
    note.write_text("# Pneumonia notes\n\nbody text", encoding="utf-8")

    stub = _StubOcrProvider({})  # OCR shouldn't be called for .md
    monkeypatch.setattr(mod, "make_ocr_provider", lambda: stub)
    monkeypatch.setattr(
        mod,
        "load_ocr_config",
        lambda: argparse.Namespace(text_extensions=[".txt", ".md"]),
    )

    docs = await mod._ocr_files([note], collection_name="cap_en", language="en")
    assert len(docs) == 1
    assert "Pneumonia notes" in docs[0].text
    assert docs[0].metadata["ocr_provider"] == "text"
    # OCR provider was never invoked for the .md path.
    assert stub.calls == []
