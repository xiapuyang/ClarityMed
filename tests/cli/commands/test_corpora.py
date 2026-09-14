"""Tests for ``claritymed rag corpora`` subcommands.

The four corpora commands (``list``, ``ingest``, ``refresh-centroid``,
``migrate-payload``) plus the TUI startup hook all funnel through the
same heavy IO surface — Qdrant client + retrieval config + a known
catalog of corpus sources. These tests pin the dispatch / error-path
behaviour without spinning a real Qdrant: the live ingest workhorse
``ingest_corpus`` is exercised by other suites; here we only verify the
CLI wiring (admin gate, unknown-corpus exit codes, no-op startup
branches, centroid refresh dispatch).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from claritymed.cli.commands import corpora as corpora_mod
from claritymed.cli.commands.corpora import (
    _NoOpEmbedder,
    corpora_app,
    refresh_system_centroids_on_startup,
)
from claritymed.stores.account import init_user

runner = CliRunner()


# --- shared fakes -------------------------------------------------------


class _FakeAClient:
    """Async-context Qdrant stand-in.

    The corpora commands hit four async methods: ``collection_exists``,
    ``count``, ``scroll``, ``set_payload``. Defaults make ``list`` and
    ``count``-using commands work cleanly; tests override per-method via
    monkeypatch when they want failure injection.
    """

    def __init__(
        self,
        *,
        exists: bool = True,
        count: int = 0,
        scroll_pages: list | None = None,
    ):
        self.exists_value = exists
        self.count_value = count
        self.scroll_pages = list(scroll_pages or [])
        self.set_payload_calls: list[dict] = []
        self.closed = False

    async def collection_exists(self, name):
        return self.exists_value

    async def count(self, name=None, *, exact=True, collection_name=None):
        return SimpleNamespace(count=self.count_value)

    async def scroll(self, **kwargs):  # noqa: ANN003
        # Each call pops one page; (results, next_offset) tuple.
        if not self.scroll_pages:
            return [], None
        return self.scroll_pages.pop(0)

    async def set_payload(self, **kwargs):  # noqa: ANN003
        self.set_payload_calls.append(kwargs)

    async def close(self):
        self.closed = True


def _stub_cfg(
    *,
    collections=(),
    qdrant_url="http://stub",
    api_key_env=None,
    rag_enabled=True,
    router_id="centroid_classifier",
):
    """Build a duck-typed retrieval config matching what corpora.py reads."""
    return SimpleNamespace(
        system_rag=SimpleNamespace(collections=list(collections)),
        qdrant=SimpleNamespace(url=qdrant_url, api_key_env=api_key_env),
        rag=SimpleNamespace(enabled=rag_enabled),
        router=SimpleNamespace(resolved=lambda: SimpleNamespace(id=router_id)),
    )


def _collection_meta(name: str, *, language="en", tier=2, topics=("pneumonia",)):
    return SimpleNamespace(
        name=name, language=language, authority_tier=tier, topics=list(topics)
    )


# --- corpora list -------------------------------------------------------


def test_corpora_list_renders_live_size_per_collection(monkeypatch):
    """Happy path: Qdrant returns a count → rendered verbatim in size=NN."""
    aclient = _FakeAClient(exists=True, count=42)
    monkeypatch.setattr(
        corpora_mod,
        "run_async",
        lambda coro: __import__("asyncio").get_event_loop().run_until_complete(coro)
        if False
        else _run_sync(coro),
    )
    # Patch the lazy imports inside the function body.
    import claritymed.core.rag.qdrant_store as _qs
    import claritymed.core.rag.schemas as _schemas

    monkeypatch.setattr(
        _schemas,
        "load_retrieval_config",
        lambda: _stub_cfg(collections=[_collection_meta("cap_en")]),
    )
    monkeypatch.setattr(_qs, "build_qdrant_client", lambda url, api_key_env: aclient)

    result = runner.invoke(corpora_app, ["list"])
    assert result.exit_code == 0, result.stdout
    assert "cap_en" in result.stdout
    assert "size=42" in result.stdout
    assert "lang=en" in result.stdout
    assert aclient.closed is True  # finally branch ran


def _run_sync(coro):
    """Drive a coroutine synchronously inside a test that already runs sync."""
    import asyncio

    return asyncio.new_event_loop().run_until_complete(coro)


def test_corpora_list_renders_question_mark_when_collection_missing(monkeypatch):
    aclient = _FakeAClient(exists=False, count=0)
    monkeypatch.setattr(corpora_mod, "run_async", _run_sync)
    import claritymed.core.rag.qdrant_store as _qs
    import claritymed.core.rag.schemas as _schemas

    monkeypatch.setattr(
        _schemas,
        "load_retrieval_config",
        lambda: _stub_cfg(collections=[_collection_meta("ghost")]),
    )
    monkeypatch.setattr(_qs, "build_qdrant_client", lambda url, api_key_env: aclient)

    result = runner.invoke(corpora_app, ["list"])
    assert result.exit_code == 0
    assert "size=?" in result.stdout


def test_corpora_list_swallows_per_collection_error(monkeypatch):
    """A flaky Qdrant call must NOT blank the whole listing — just the size."""

    class _AClientThatRaisesCount(_FakeAClient):
        async def count(self, *args, **kwargs):
            raise RuntimeError("count exploded")

    aclient = _AClientThatRaisesCount(exists=True)
    monkeypatch.setattr(corpora_mod, "run_async", _run_sync)
    import claritymed.core.rag.qdrant_store as _qs
    import claritymed.core.rag.schemas as _schemas

    monkeypatch.setattr(
        _schemas,
        "load_retrieval_config",
        lambda: _stub_cfg(collections=[_collection_meta("statpearls")]),
    )
    monkeypatch.setattr(_qs, "build_qdrant_client", lambda url, api_key_env: aclient)

    result = runner.invoke(corpora_app, ["list"])
    assert result.exit_code == 0
    assert "statpearls" in result.stdout
    assert "size=?" in result.stdout


def test_corpora_list_with_no_collections_short_circuits(monkeypatch):
    """No collections → no Qdrant client built (the guard clause skips it)."""
    built: list[Any] = []

    def _no_build(url, api_key_env):
        built.append((url, api_key_env))
        return _FakeAClient()

    monkeypatch.setattr(corpora_mod, "run_async", _run_sync)
    import claritymed.core.rag.qdrant_store as _qs
    import claritymed.core.rag.schemas as _schemas

    monkeypatch.setattr(_schemas, "load_retrieval_config", lambda: _stub_cfg())
    monkeypatch.setattr(_qs, "build_qdrant_client", _no_build)

    result = runner.invoke(corpora_app, ["list"])
    assert result.exit_code == 0
    assert built == []  # short-circuit before client construction


# --- corpora ingest unknown-corpus path ---------------------------------


def test_corpora_ingest_unknown_corpus_exits_with_2(monkeypatch):
    """``ingest foo`` → exits 2 with the available list, before touching Qdrant."""
    init_user("test")  # first user → admin
    result = runner.invoke(corpora_app, ["ingest", "foo", "--user", "test"])
    assert result.exit_code == 2
    combined = result.stdout + (result.stderr or "")
    assert "Unknown corpus: foo" in combined
    assert "statpearls" in combined and "textbooks" in combined


# --- corpora refresh-centroid -------------------------------------------


def test_refresh_centroid_unknown_corpus_exits_with_2(monkeypatch):
    init_user("test")
    result = runner.invoke(corpora_app, ["refresh-centroid", "foo", "--user", "test"])
    assert result.exit_code == 2
    assert "Unknown corpus" in result.stdout


def test_refresh_centroid_happy_path_prints_success(monkeypatch):
    init_user("test")
    # ``StatPearlsSource(root)`` validates the dir exists at construction;
    # pre-create it so the command can advance into the centroid path
    # without dragging a real raw corpus in.
    from claritymed.stores.paths import shared_knowledge_raw_dir

    (shared_knowledge_raw_dir() / "statpearls").mkdir(parents=True, exist_ok=True)
    aclient = _FakeAClient()
    monkeypatch.setattr(corpora_mod, "run_async", _run_sync)
    import claritymed.core.rag.qdrant_store as _qs
    import claritymed.core.rag.schemas as _schemas

    monkeypatch.setattr(_schemas, "load_retrieval_config", lambda: _stub_cfg())
    monkeypatch.setattr(_qs, "build_qdrant_client", lambda url, api_key_env: aclient)

    async def _ok(aclient, name, store, force):  # noqa: ANN001
        assert force is True  # CLI command forces a recompute

    import claritymed.core.rag.routing.centroid_store as _cs

    monkeypatch.setattr(_cs, "maybe_refresh", _ok)

    result = runner.invoke(
        corpora_app, ["refresh-centroid", "statpearls", "--user", "test"]
    )
    assert result.exit_code == 0, result.stdout
    assert "centroid refreshed" in result.stdout


def test_refresh_centroid_failure_exits_with_1(monkeypatch):
    init_user("test")
    from claritymed.stores.paths import shared_knowledge_raw_dir

    (shared_knowledge_raw_dir() / "statpearls").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(corpora_mod, "run_async", _run_sync)
    import claritymed.core.rag.qdrant_store as _qs
    import claritymed.core.rag.schemas as _schemas

    monkeypatch.setattr(_schemas, "load_retrieval_config", lambda: _stub_cfg())
    monkeypatch.setattr(
        _qs, "build_qdrant_client", lambda url, api_key_env: _FakeAClient()
    )

    async def _broken(aclient, name, store, force):  # noqa: ANN001
        raise RuntimeError("centroid recompute exploded")

    import claritymed.core.rag.routing.centroid_store as _cs

    monkeypatch.setattr(_cs, "maybe_refresh", _broken)

    result = runner.invoke(
        corpora_app, ["refresh-centroid", "statpearls", "--user", "test"]
    )
    assert result.exit_code == 1
    assert "failed" in result.stdout


# --- _refresh_centroid_for helper --------------------------------------


@pytest.mark.asyncio
async def test_refresh_centroid_for_helper_handles_no_change(monkeypatch, capsys):
    """``maybe_refresh`` → False means already up-to-date; helper says so."""
    import claritymed.core.rag.routing.centroid_store as _cs

    async def _no_change(aclient, name, store):  # noqa: ANN001
        return False

    monkeypatch.setattr(_cs, "maybe_refresh", _no_change)

    class _Console:
        def __init__(self):
            self.lines = []

        def print(self, text):
            self.lines.append(text)

    con = _Console()
    await corpora_mod._refresh_centroid_for(_FakeAClient(), "cap_en", con)
    assert any("up-to-date" in line and "cap_en" in line for line in con.lines)


@pytest.mark.asyncio
async def test_refresh_centroid_for_helper_handles_exception(monkeypatch):
    import claritymed.core.rag.routing.centroid_store as _cs

    async def _bad(aclient, name, store):  # noqa: ANN001
        raise RuntimeError("oops")

    monkeypatch.setattr(_cs, "maybe_refresh", _bad)

    class _Console:
        def __init__(self):
            self.lines = []

        def print(self, text):
            self.lines.append(text)

    con = _Console()
    await corpora_mod._refresh_centroid_for(_FakeAClient(), "x", con)
    assert any("failed" in line for line in con.lines)


# --- corpora migrate-payload --------------------------------------------


def test_migrate_payload_unknown_corpus_exits_with_2(monkeypatch):
    init_user("test")
    result = runner.invoke(
        corpora_app, ["migrate-payload", "textbooks", "--user", "test"]
    )
    assert result.exit_code == 2
    assert "Unknown corpus" in result.stdout


# --- refresh_system_centroids_on_startup --------------------------------


def test_startup_centroid_refresh_is_noop_when_rag_disabled(monkeypatch):
    import claritymed.core.rag as _rag

    monkeypatch.setattr(
        _rag, "load_retrieval_config", lambda: _stub_cfg(rag_enabled=False)
    )

    # No Qdrant build attempted — set a sentinel that would explode if called.
    def _no_build(url, api_key_env):
        raise AssertionError("should not build qdrant client when rag is disabled")

    import claritymed.core.rag.qdrant_store as _qs

    monkeypatch.setattr(_qs, "build_qdrant_client", _no_build)

    refresh_system_centroids_on_startup()  # must return without raising


def test_startup_centroid_refresh_is_noop_when_router_not_centroid(monkeypatch):
    import claritymed.core.rag as _rag
    import claritymed.core.rag.qdrant_store as _qs

    monkeypatch.setattr(
        _rag, "load_retrieval_config", lambda: _stub_cfg(router_id="rule_based")
    )

    def _no_build(url, api_key_env):
        raise AssertionError("non-centroid router → no startup refresh")

    monkeypatch.setattr(_qs, "build_qdrant_client", _no_build)

    refresh_system_centroids_on_startup()


def test_startup_centroid_refresh_is_noop_when_no_collections(monkeypatch):
    import claritymed.core.rag as _rag
    import claritymed.core.rag.qdrant_store as _qs

    monkeypatch.setattr(_rag, "load_retrieval_config", lambda: _stub_cfg())  # empty

    def _no_build(url, api_key_env):
        raise AssertionError("zero collections → nothing to refresh")

    monkeypatch.setattr(_qs, "build_qdrant_client", _no_build)

    refresh_system_centroids_on_startup()


def test_startup_centroid_refresh_runs_for_each_collection_and_swallows_failure(
    monkeypatch,
):
    """Two collections — one succeeds, one raises. The startup hook must
    finish without raising, calling ``maybe_refresh`` once per collection.
    """
    import claritymed.core.rag as _rag
    import claritymed.core.rag.qdrant_store as _qs
    import claritymed.core.rag.routing.centroid_store as _cs

    cfg = _stub_cfg(
        collections=[_collection_meta("cap_en"), _collection_meta("statpearls")]
    )
    monkeypatch.setattr(_rag, "load_retrieval_config", lambda: cfg)
    aclient = _FakeAClient()
    monkeypatch.setattr(_qs, "build_qdrant_client", lambda url, api_key_env: aclient)

    seen: list[str] = []

    async def _per_collection(aclient, name, store):  # noqa: ANN001
        seen.append(name)
        if name == "statpearls":
            raise RuntimeError("centroid math went sideways")
        return True

    monkeypatch.setattr(_cs, "maybe_refresh", _per_collection)
    monkeypatch.setattr(corpora_mod, "run_async", _run_sync)

    refresh_system_centroids_on_startup()  # must not raise
    assert set(seen) == {"cap_en", "statpearls"}


# --- _NoOpEmbedder ------------------------------------------------------


def test_noop_embedder_dimension_matches_dense_fallback():
    assert _NoOpEmbedder().dimension == 1024


async def test_noop_embedder_dense_and_sparse_return_zero_signal():
    embedder = _NoOpEmbedder()
    dense = await embedder.embed_dense(["a", "b"])
    sparse = await embedder.embed_sparse(["a", "b"])
    assert len(dense) == 2
    assert all(v == [0.0] * 1024 for v in dense)
    assert sparse == [{}, {}]


# --- corpora migrate-payload (body) ------------------------------------


def _statpearls_point(point_id: int, *, payload: dict):
    return SimpleNamespace(id=point_id, payload=dict(payload))


def test_migrate_payload_patches_source_uri_and_doc_title(monkeypatch):
    """End-to-end happy path: scan two scroll pages, patch payload fields,
    print the summary line at exit.
    """
    init_user("test")
    monkeypatch.setattr(corpora_mod, "asyncio", _AsyncioFacade())

    # Two scroll pages: first has two points (one needs source_uri rewrite,
    # one needs doc_title), second is empty (terminates the loop).
    page_one = (
        [
            _statpearls_point(
                1,
                payload={
                    "doc_id": "nbk-001",
                    "source_uri": "https://example.com/article-001",
                    "title": "first title",
                },
            ),
            _statpearls_point(
                2,
                payload={
                    "doc_id": "nbk-002",
                    "source_uri": "https://example.com/article-002",
                },
            ),
        ],
        "offset-2",
    )
    page_two = ([], None)

    aclient = _FakeAClient(count=2, scroll_pages=[page_one, page_two])

    import claritymed.core.rag.qdrant_store as _qs
    import claritymed.core.rag.schemas as _schemas

    monkeypatch.setattr(_schemas, "load_retrieval_config", lambda: _stub_cfg())
    monkeypatch.setattr(_qs, "build_qdrant_client", lambda url, api_key_env: aclient)

    result = runner.invoke(
        corpora_app, ["migrate-payload", "statpearls", "--user", "test"]
    )
    assert result.exit_code == 0, result.stdout

    # Both points triggered a set_payload call.
    assert len(aclient.set_payload_calls) == 2
    # The summary line is the final stdout signal — operators key off this.
    assert "scanned 2 points" in result.stdout
    assert "patched 2" in result.stdout


def test_migrate_payload_skips_points_already_well_formed(monkeypatch):
    """Points whose payload already has the right keys are NOT re-patched."""
    init_user("test")
    monkeypatch.setattr(corpora_mod, "asyncio", _AsyncioFacade())

    # Point has both fields already in canonical form — no patch needed.
    page = (
        [
            _statpearls_point(
                1,
                payload={
                    "doc_id": "nbk-001",
                    "source_uri": "https://www.ncbi.nlm.nih.gov/books/NBK-001",
                    "doc_title": "already set",
                    "title": "old title (ignored)",
                },
            )
        ],
        None,  # terminates after one page
    )

    aclient = _FakeAClient(count=1, scroll_pages=[page])

    import claritymed.core.rag.qdrant_store as _qs
    import claritymed.core.rag.schemas as _schemas

    monkeypatch.setattr(_schemas, "load_retrieval_config", lambda: _stub_cfg())
    monkeypatch.setattr(_qs, "build_qdrant_client", lambda url, api_key_env: aclient)

    result = runner.invoke(
        corpora_app, ["migrate-payload", "statpearls", "--user", "test"]
    )
    assert result.exit_code == 0, result.stdout
    assert aclient.set_payload_calls == []
    assert "patched 0" in result.stdout


class _AsyncioFacade:
    """Module-level ``asyncio`` rebind so the command's ``asyncio.run(_run())``
    drives our patched coroutine without spawning a real loop within tests
    that already manage their own.
    """

    def run(self, coro):
        return _run_sync(coro)


# --- corpora ingest dry-run --------------------------------------------


def _stub_source(*, name="statpearls_en", n_docs=3):
    """Build a minimal CorpusSource — only ``name`` and ``iter_raw_docs`` are
    read by the ingest command body.
    """
    from claritymed.core.rag.chunking.base import RawDocument

    docs = [
        RawDocument(doc_id=f"d{i}", text=f"body {i}", language="en")
        for i in range(n_docs)
    ]

    class _Src:
        def __init__(self):
            self.name = name

        def iter_raw_docs(self):
            return iter(docs)

    return _Src(), docs


def test_corpora_ingest_dry_run_drives_pipeline(monkeypatch):
    """Dry-run path: source-iter sizing + pipeline orchestration + summary line.

    Patches every heavy module dep, including ``StatPearlsSource`` so the
    test doesn't need a populated raw dir. Uses an embedder/chunker stub
    that ``ingest_corpus`` never inspects because we patch that too.
    """
    init_user("test")

    from claritymed.stores.paths import shared_knowledge_raw_dir

    (shared_knowledge_raw_dir() / "statpearls").mkdir(parents=True, exist_ok=True)

    src, _docs = _stub_source(n_docs=2)

    # Patch all the lazy imports inside corpora_ingest.
    import claritymed.core.rag as _rag
    import claritymed.core.rag.chunking.factory as _chunker_factory
    import claritymed.core.rag.embedding.factory as _embedder_factory
    import claritymed.core.rag.parent_store as _ps_mod
    import claritymed.core.rag.qdrant_store as _qs
    import claritymed.ingest.corpus.base as _ingest_base
    import claritymed.ingest.corpus.statpearls as _sp
    import claritymed.stores.paths as _paths

    monkeypatch.setattr(_sp, "StatPearlsSource", lambda root: src)
    monkeypatch.setattr(_rag, "load_retrieval_config", lambda: _stub_cfg())
    monkeypatch.setattr(_chunker_factory, "build_chunker", lambda: object())
    monkeypatch.setattr(_embedder_factory, "build_embedder", lambda: object())

    aclient = _FakeAClient()
    monkeypatch.setattr(_qs, "build_qdrant_client", lambda url, api_key_env: aclient)

    class _FakeRagStore:
        def __init__(self, **kwargs):  # noqa: ANN003
            pass

    class _FakeParentStore:
        def __init__(self, path):  # noqa: ANN001
            pass

    monkeypatch.setattr(_qs, "RagCollectionStore", _FakeRagStore)
    monkeypatch.setattr(_ps_mod, "ParentStore", _FakeParentStore)
    monkeypatch.setattr(
        _paths, "shared_parent_docstore_path", lambda: "/tmp/parents.json"
    )

    async def _fake_ingest(
        source, *, chunker, embedder, store, parent_store, limit, dry_run, on_doc
    ):  # noqa: ANN001
        assert dry_run is True
        assert limit is None
        # Trigger the progress callback to exercise its body.
        stats = SimpleNamespace(
            source=source.name,
            docs_processed=2,
            parents_written=4,
            children_written=8,
            docs_skipped=0,
            docs_resumed=0,
        )
        on_doc(stats)
        return stats

    monkeypatch.setattr(_ingest_base, "ingest_corpus", _fake_ingest)

    result = runner.invoke(
        corpora_app,
        ["ingest", "statpearls", "--user", "test", "--dry-run"],
    )
    assert result.exit_code == 0, result.stdout
    # Summary line carries the per-axis counts.
    assert "statpearls" in result.stdout
    assert "2 docs" in result.stdout
    assert "4 parents" in result.stdout
    assert "8 children" in result.stdout


def test_corpora_ingest_textbooks_dry_run_dispatches_to_textbooks_source(monkeypatch):
    """The ``textbooks`` branch picks the TextbooksSource, not StatPearls."""
    init_user("test")

    from claritymed.stores.paths import shared_knowledge_raw_dir

    (shared_knowledge_raw_dir() / "textbooks").mkdir(parents=True, exist_ok=True)

    src, _docs = _stub_source(name="textbooks_en", n_docs=1)

    import claritymed.core.rag as _rag
    import claritymed.core.rag.chunking.factory as _chunker_factory
    import claritymed.core.rag.embedding.factory as _embedder_factory
    import claritymed.core.rag.parent_store as _ps_mod
    import claritymed.core.rag.qdrant_store as _qs
    import claritymed.ingest.corpus.base as _ingest_base
    import claritymed.ingest.corpus.textbooks as _tx
    import claritymed.stores.paths as _paths

    monkeypatch.setattr(_tx, "TextbooksSource", lambda root: src)
    monkeypatch.setattr(_rag, "load_retrieval_config", lambda: _stub_cfg())
    monkeypatch.setattr(_chunker_factory, "build_chunker", lambda: object())
    monkeypatch.setattr(_embedder_factory, "build_embedder", lambda: object())

    aclient = _FakeAClient()
    monkeypatch.setattr(_qs, "build_qdrant_client", lambda url, api_key_env: aclient)

    class _FakeRagStore:
        def __init__(self, **kwargs):  # noqa: ANN003
            pass

    class _FakeParentStore:
        def __init__(self, path):  # noqa: ANN001
            pass

    monkeypatch.setattr(_qs, "RagCollectionStore", _FakeRagStore)
    monkeypatch.setattr(_ps_mod, "ParentStore", _FakeParentStore)
    monkeypatch.setattr(
        _paths, "shared_parent_docstore_path", lambda: "/tmp/parents.json"
    )

    async def _fake_ingest(source, **kwargs):  # noqa: ANN001, ANN003
        assert source.name == "textbooks_en"
        return SimpleNamespace(
            source="textbooks_en",
            docs_processed=1,
            parents_written=1,
            children_written=1,
            docs_skipped=0,
            docs_resumed=0,
        )

    monkeypatch.setattr(_ingest_base, "ingest_corpus", _fake_ingest)

    result = runner.invoke(
        corpora_app,
        ["ingest", "textbooks", "--user", "test", "--dry-run"],
    )
    assert result.exit_code == 0, result.stdout
    assert "textbooks_en" in result.stdout
