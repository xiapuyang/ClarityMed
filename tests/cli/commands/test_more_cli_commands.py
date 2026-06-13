"""Smoke + happy-path tests for the lower-coverage CLI subcommands.

These focus on:
- Argument parsing / help text
- Error branches (unknown corpus, missing log dir, no rules)
- Happy paths with mocked external IO (Phoenix, Qdrant, models)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from claritymed.cli.main import app

runner = CliRunner()


@pytest.fixture
def _alice(tmp_path, monkeypatch):
    """Initialize alice + point HOME at tmp."""
    monkeypatch.setenv("CLARITYMED_HOME", str(tmp_path))
    from claritymed.stores.account import init_user

    init_user("alice", display_name="Alice")
    return "alice"


# ===========================================================================
# rag subcommands: list / show / rm
# ===========================================================================


def _patch_user_rag(monkeypatch):
    """Replace make_user_rag_store with an in-memory stub everywhere it's used."""
    import hashlib

    from qdrant_client import AsyncQdrantClient

    from claritymed.core.rag.chunking.base import (
        ChildChunk,
        ChunkedDocument,
        ParentChunk,
        RawDocument,
    )
    from claritymed.core.rag.embedding.base import Embedder
    from claritymed.core.phi.guard import PhiGuard
    from claritymed.stores import user_rag as _ur

    class _StubEmbedder(Embedder):
        @property
        def dimension(self) -> int:
            return 32

        async def embed_dense(self, texts):
            return [
                [b / 255.0 for b in hashlib.sha256(t.encode()).digest()[:32]]
                for t in texts
            ]

        async def embed_sparse(self, texts):
            return [{abs(hash(t)) % 100: 0.5} for t in texts]

    class _StubChunker:
        def chunk(self, doc: RawDocument) -> ChunkedDocument:
            if not doc.text.strip():
                return ChunkedDocument(parents=[], children=[])
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

    shared = _ur.UserRagStore(
        aclient=AsyncQdrantClient(":memory:"),
        embedder=_StubEmbedder(),
        chunker=_StubChunker(),
        guard=PhiGuard.from_config(),
    )

    def _factory(user_id):
        return shared

    monkeypatch.setattr(_ur, "make_user_rag_store", _factory)
    import claritymed.cli.commands.rag as _cli_rag

    monkeypatch.setattr(_cli_rag, "make_user_rag_store", _factory)
    return shared


def test_rag_list_empty_prints_no_documents(_alice, monkeypatch):
    _patch_user_rag(monkeypatch)
    result = runner.invoke(app, ["rag", "list", "--user", "alice"])
    assert result.exit_code == 0, result.stdout
    assert "No documents" in result.stdout


def test_rag_list_and_show_after_add(_alice, tmp_path, monkeypatch):
    _patch_user_rag(monkeypatch)
    sample = tmp_path / "sample.txt"
    sample.write_text("renal function summary", encoding="utf-8")

    add = runner.invoke(app, ["rag", "add", str(sample), "--user", "alice", "--public"])
    assert add.exit_code == 0, add.stdout

    lst = runner.invoke(app, ["rag", "list", "--user", "alice"])
    assert lst.exit_code == 0
    assert "doc-" in lst.stdout  # generate_doc_id prefix

    # Show using the doc_id from add stdout.
    doc_id = None
    for tok in add.stdout.split():
        if tok.startswith("doc_id="):
            doc_id = tok.split("=", 1)[1]
            break
    assert doc_id is not None

    show = runner.invoke(app, ["rag", "show", doc_id, "--user", "alice"])
    assert show.exit_code == 0
    assert "renal function summary" in show.stdout


def test_rag_show_unknown_doc_prints_no_chunks(_alice, monkeypatch):
    _patch_user_rag(monkeypatch)
    result = runner.invoke(app, ["rag", "show", "nonexistent", "--user", "alice"])
    assert result.exit_code == 0
    assert "No chunks" in result.stdout


def test_rag_rm_prints_deleted(_alice, tmp_path, monkeypatch):
    _patch_user_rag(monkeypatch)
    sample = tmp_path / "sample.txt"
    sample.write_text("body", encoding="utf-8")
    add = runner.invoke(app, ["rag", "add", str(sample), "--user", "alice", "--public"])
    doc_id = next(
        tok.split("=", 1)[1] for tok in add.stdout.split() if tok.startswith("doc_id=")
    )
    result = runner.invoke(app, ["rag", "rm", doc_id, "--user", "alice"])
    assert result.exit_code == 0
    assert "Deleted" in result.stdout


def test_rag_add_duplicate_source_uri_prints_skipped(_alice, tmp_path, monkeypatch):
    """Force DuplicateDocumentError via store stub — in-memory Qdrant doesn't
    persist the payload index needed to detect duplicates organically."""
    _patch_user_rag(monkeypatch)
    sample = tmp_path / "sample.txt"
    sample.write_text("first body", encoding="utf-8")

    # Patch the service path's `service.run` to raise DuplicateDocumentError.
    from claritymed.errors import DuplicateDocumentError
    from claritymed.orchestrator.services import rag_service as _rag_svc

    async def _raise(*_a, **_kw):
        raise DuplicateDocumentError(
            source_uri=str(sample.resolve()), existing_doc_id="doc-existing"
        )
        yield  # pragma: no cover - never reached but makes function an async generator

    monkeypatch.setattr(_rag_svc.RagService, "run", _raise)
    result = runner.invoke(
        app, ["rag", "add", str(sample), "--user", "alice", "--public"]
    )
    assert result.exit_code == 0
    assert "already indexed" in result.stdout


# ===========================================================================
# corpora subcommands
# ===========================================================================


def test_corpora_list_prints_configured_collections(_alice):
    result = runner.invoke(app, ["rag", "corpora", "list"])
    assert result.exit_code == 0, result.stdout
    # configs/retrieval.yaml ships with at least one collection.
    assert "lang=" in result.stdout


def test_corpora_ingest_rejects_unknown_corpus(_alice):
    result = runner.invoke(
        app, ["rag", "corpora", "ingest", "no_such_corpus", "--user", "alice"]
    )
    assert result.exit_code == 2
    assert "Unknown corpus" in result.stdout


def test_corpora_refresh_centroid_rejects_unknown_corpus(_alice):
    result = runner.invoke(
        app,
        ["rag", "corpora", "refresh-centroid", "no_such_corpus", "--user", "alice"],
    )
    assert result.exit_code == 2
    assert "Unknown corpus" in result.stdout


def test_corpora_migrate_payload_rejects_unknown_corpus(_alice):
    result = runner.invoke(
        app, ["rag", "corpora", "migrate-payload", "frobnitz", "--user", "alice"]
    )
    assert result.exit_code == 2
    assert "Unknown corpus" in result.stdout


def test_refresh_system_centroids_on_startup_noop_when_rag_disabled(monkeypatch):
    """Helper called from `tui` should silently skip when rag.enabled=False."""
    from claritymed.cli.commands import corpora as _corpora

    class _Cfg:
        class _Rag:
            enabled = False

        rag = _Rag()

    # The function imports `load_retrieval_config` lazily from
    # ``claritymed.core.rag``, so patch the source binding.
    import claritymed.core.rag as _rag_pkg

    monkeypatch.setattr(_rag_pkg, "load_retrieval_config", lambda: _Cfg())
    # Must not raise and must not call any qdrant code.
    _corpora.refresh_system_centroids_on_startup()


def test_refresh_system_centroids_on_startup_noop_when_router_not_centroid(
    monkeypatch,
):
    from claritymed.cli.commands import corpora as _corpora

    class _Cfg:
        class _Rag:
            enabled = True

        rag = _Rag()

        class _Router:
            def resolved(self):
                class _R:
                    id = "rule_based"

                return _R()

        router = _Router()

    import claritymed.core.rag as _rag_pkg

    monkeypatch.setattr(_rag_pkg, "load_retrieval_config", lambda: _Cfg())
    _corpora.refresh_system_centroids_on_startup()  # no-op


def test_no_op_embedder_returns_zero_vectors():
    """The dry-run embedder exposes the right dimension and stub vectors."""
    from claritymed.cli.commands.corpora import _NoOpEmbedder

    e = _NoOpEmbedder()
    assert e.dimension == 1024

    import asyncio

    dense = asyncio.run(e.embed_dense(["a", "b"]))
    sparse = asyncio.run(e.embed_sparse(["a", "b"]))
    assert dense == [[0.0] * 1024, [0.0] * 1024]
    assert sparse == [{}, {}]


# ===========================================================================
# prompts subcommands — mock phoenix_sync
# ===========================================================================


def _make_sync_report(
    *, direction="push", dry_run=False, has_error=False, has_differ=False
):
    from types import SimpleNamespace

    entries = [
        SimpleNamespace(
            action="pushed" if direction == "push" else "pulled",
            prompt_name="ask",
            language="en",
            phoenix_name="claritymed_ask_en",
            detail="ok",
            unified_diff=[],
        )
    ]
    if has_differ:
        entries.append(
            SimpleNamespace(
                action="differs",
                prompt_name="ask",
                language="zh",
                phoenix_name="claritymed_ask_zh",
                detail="content differs",
                unified_diff=["- old", "+ new"],
            )
        )
    if has_error:
        entries.append(
            SimpleNamespace(
                action="error",
                prompt_name="ask",
                language="zh",
                phoenix_name="claritymed_ask_zh",
                detail="boom",
                unified_diff=[],
            )
        )
    return SimpleNamespace(
        entries=entries,
        changed=[e for e in entries if e.action in ("pushed", "pulled")],
        differs=[e for e in entries if e.action == "differs"],
        errors=[e for e in entries if e.action == "error"],
        direction=direction,
        dry_run=dry_run,
    )


def test_prompts_push_happy_path(monkeypatch):
    import claritymed.core.prompts.phoenix_sync as _ps

    monkeypatch.setattr(
        _ps, "push", lambda **kwargs: _make_sync_report(direction="push")
    )
    result = runner.invoke(app, ["prompts", "push", "--dry-run"])
    assert result.exit_code == 0, result.stdout
    assert "push" in result.stdout.lower()
    assert "changed=" in result.stdout


def test_prompts_push_with_errors_exits_nonzero(monkeypatch):
    import claritymed.core.prompts.phoenix_sync as _ps

    monkeypatch.setattr(
        _ps,
        "push",
        lambda **kwargs: _make_sync_report(direction="push", has_error=True),
    )
    result = runner.invoke(app, ["prompts", "push"])
    assert result.exit_code == 1


def test_prompts_push_handles_exception(monkeypatch):
    import claritymed.core.prompts.phoenix_sync as _ps

    def _boom(**_kwargs):
        raise RuntimeError("phoenix unreachable")

    monkeypatch.setattr(_ps, "push", _boom)
    result = runner.invoke(app, ["prompts", "push"])
    assert result.exit_code == 1


def test_prompts_pull_happy_path(monkeypatch):
    import claritymed.core.prompts.phoenix_sync as _ps

    monkeypatch.setattr(
        _ps, "pull", lambda **kwargs: _make_sync_report(direction="pull")
    )
    result = runner.invoke(app, ["prompts", "pull"])
    assert result.exit_code == 0
    assert "pull" in result.stdout.lower()


def test_prompts_pull_with_options(monkeypatch):
    import claritymed.core.prompts.phoenix_sync as _ps

    captured: dict = {}

    def _pull(**kwargs):
        captured.update(kwargs)
        return _make_sync_report(direction="pull")

    monkeypatch.setattr(_ps, "pull", _pull)
    result = runner.invoke(
        app,
        ["prompts", "pull", "ask", "--into-new-version", "--version-name", "v1.1"],
    )
    assert result.exit_code == 0
    assert captured["name"] == "ask"
    assert captured["into_new_version"] is True
    assert captured["new_version_name"] == "v1.1"


def test_prompts_pull_handles_exception(monkeypatch):
    import claritymed.core.prompts.phoenix_sync as _ps

    monkeypatch.setattr(
        _ps,
        "pull",
        lambda **_kw: (_ for _ in ()).throw(RuntimeError("phoenix down")),
    )
    result = runner.invoke(app, ["prompts", "pull"])
    assert result.exit_code == 1


def test_prompts_diff_clean_exits_zero(monkeypatch):
    import claritymed.core.prompts.phoenix_sync as _ps

    monkeypatch.setattr(
        _ps, "diff", lambda **kwargs: _make_sync_report(direction="diff")
    )
    result = runner.invoke(app, ["prompts", "diff"])
    # No diffs and no errors → exit 0.
    assert result.exit_code == 0


def test_prompts_diff_with_differences_exits_nonzero(monkeypatch):
    import claritymed.core.prompts.phoenix_sync as _ps

    monkeypatch.setattr(
        _ps,
        "diff",
        lambda **kwargs: _make_sync_report(direction="diff", has_differ=True),
    )
    result = runner.invoke(app, ["prompts", "diff"])
    assert result.exit_code == 1
    assert "differs=" in result.stdout


def test_prompts_diff_no_color(monkeypatch):
    import claritymed.core.prompts.phoenix_sync as _ps

    monkeypatch.setattr(
        _ps,
        "diff",
        lambda **kwargs: _make_sync_report(direction="diff", has_differ=True),
    )
    result = runner.invoke(app, ["prompts", "diff", "--no-color"])
    assert result.exit_code == 1
    # plain text diff lines should appear
    assert "- old" in result.stdout
    assert "+ new" in result.stdout


def test_prompts_diff_handles_exception(monkeypatch):
    import claritymed.core.prompts.phoenix_sync as _ps

    monkeypatch.setattr(
        _ps,
        "diff",
        lambda **_kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    result = runner.invoke(app, ["prompts", "diff"])
    assert result.exit_code == 1


# ===========================================================================
# audit grep
# ===========================================================================


def _write_audit_log(log_dir: Path, *, kinds=("mode.ask",)) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, kind in enumerate(kinds):
        lines.append(
            json.dumps(
                {
                    "kind": kind,
                    "payload": {"i": i},
                    "user_id": "alice",
                    "request_id": f"req-{i}",
                    "trace_id": "trace-xyz" if i % 2 == 0 else "trace-other",
                    "created_at": f"2026-06-09T10:0{i}:00Z",
                }
            )
        )
    (log_dir / "audit.log").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_audit_grep_no_log_dir_exits_1(tmp_path, monkeypatch):
    monkeypatch.setenv("CLARITYMED_LOG_DIR", str(tmp_path / "missing"))
    import importlib

    from claritymed import config as cfg_mod

    importlib.reload(cfg_mod)
    result = runner.invoke(app, ["audit", "grep"])
    assert result.exit_code == 1


def test_audit_grep_no_audit_files_exits_1(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    monkeypatch.setenv("CLARITYMED_LOG_DIR", str(log_dir))
    import importlib

    from claritymed import config as cfg_mod

    importlib.reload(cfg_mod)
    result = runner.invoke(app, ["audit", "grep"])
    assert result.exit_code == 1


def test_audit_grep_filters_by_kind(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    _write_audit_log(log_dir, kinds=("mode.ask", "mode.rag", "mode.ask"))
    monkeypatch.setenv("CLARITYMED_LOG_DIR", str(log_dir))
    import importlib

    from claritymed import config as cfg_mod

    importlib.reload(cfg_mod)
    result = runner.invoke(app, ["audit", "grep", "--kind", "mode.ask"])
    assert result.exit_code == 0
    # Both mode.ask lines present, mode.rag line absent.
    assert result.stdout.count("mode.ask") >= 2
    assert "mode.rag" not in result.stdout


def test_audit_grep_no_matches_exits_1(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    _write_audit_log(log_dir, kinds=("mode.ask",))
    monkeypatch.setenv("CLARITYMED_LOG_DIR", str(log_dir))
    import importlib

    from claritymed import config as cfg_mod

    importlib.reload(cfg_mod)
    result = runner.invoke(app, ["audit", "grep", "--kind", "no.such.kind"])
    assert result.exit_code == 1


def test_audit_grep_filters_by_request_and_user_and_trace(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    _write_audit_log(log_dir, kinds=("mode.ask", "mode.ask"))
    monkeypatch.setenv("CLARITYMED_LOG_DIR", str(log_dir))
    import importlib

    from claritymed import config as cfg_mod

    importlib.reload(cfg_mod)
    result = runner.invoke(
        app,
        [
            "audit",
            "grep",
            "--request-id",
            "req-0",
            "--user-id",
            "alice",
            "--trace-id",
            "trace-xyz",
        ],
    )
    assert result.exit_code == 0
    assert "req-0" in result.stdout


def test_audit_grep_json_output_and_limit(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    _write_audit_log(log_dir, kinds=("mode.ask",) * 5)
    monkeypatch.setenv("CLARITYMED_LOG_DIR", str(log_dir))
    import importlib

    from claritymed import config as cfg_mod

    importlib.reload(cfg_mod)
    result = runner.invoke(app, ["audit", "grep", "--json", "--limit", "2"])
    assert result.exit_code == 0
    # 2 matching lines emitted as JSON.
    lines = [ln for ln in result.stdout.strip().splitlines() if ln.startswith("{")]
    assert len(lines) == 2
    for ln in lines:
        json.loads(ln)


def test_audit_grep_filters_by_since_until(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    _write_audit_log(log_dir, kinds=("mode.ask",) * 3)
    monkeypatch.setenv("CLARITYMED_LOG_DIR", str(log_dir))
    import importlib

    from claritymed import config as cfg_mod

    importlib.reload(cfg_mod)
    # Filter to only the 10:01 entry.
    result = runner.invoke(
        app,
        [
            "audit",
            "grep",
            "--since",
            "2026-06-09T10:01:00Z",
            "--until",
            "2026-06-09T10:01:59Z",
        ],
    )
    assert result.exit_code == 0
    assert result.stdout.count("mode.ask") == 1


# ===========================================================================
# terminology summary
# ===========================================================================


def test_terminology_summary_missing_file_exits_1(tmp_path, monkeypatch):
    monkeypatch.setenv("CLARITYMED_HOME", str(tmp_path))
    result = runner.invoke(app, ["terminology", "summary"])
    assert result.exit_code == 1


def test_terminology_summary_with_data(tmp_path):
    f = tmp_path / "concepts.jsonl"
    f.write_text(
        json.dumps(
            {
                "type": "disease",
                "aliases": [
                    {"language": "en", "source": "umls"},
                    {"language": "zh", "source": "cmekg"},
                ],
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "drug",
                "aliases": [{"language": "en", "source": "umls"}],
            }
        )
        + "\n"
        + "{not json\n"
        + "\n",
        encoding="utf-8",
    )
    result = runner.invoke(app, ["terminology", "summary", "--path", str(f)])
    assert result.exit_code == 0
    assert "concepts" in result.stdout
    assert "By type" in result.stdout
    assert "By source" in result.stdout
    assert "By language" in result.stdout
    assert "disease" in result.stdout


# ===========================================================================
# ask — _maybe_build_strategy + ask --help
# ===========================================================================


def test_ask_maybe_build_strategy_returns_none_when_disabled(monkeypatch):
    """When rag.enabled=False, no strategy is built and no rag deps are imported."""
    from claritymed.cli.commands import ask as _ask_mod

    class _Cfg:
        class _Rag:
            enabled = False

        rag = _Rag()

    # The function imports `load_retrieval_config` lazily via from-import; patch
    # the binding on the original module.
    import claritymed.core.rag as _rag_pkg

    monkeypatch.setattr(_rag_pkg, "load_retrieval_config", lambda: _Cfg())
    assert _ask_mod._maybe_build_strategy(model=None) is None


# ===========================================================================
# tool — extra branches
# ===========================================================================


def test_tool_run_unknown_tool(_alice):
    result = runner.invoke(
        app,
        [
            "tool",
            "run",
            "no_such_tool",
            "{}",
            "--user",
            "alice",
        ],
    )
    assert result.exit_code == 1


def test_tool_run_invalid_json_args(_alice):
    result = runner.invoke(
        app,
        [
            "tool",
            "run",
            "save_allergy",
            "{not valid json",
            "--user",
            "alice",
        ],
    )
    assert result.exit_code == 1


def test_tool_rule_revoke_ambiguous_prefix(_alice, monkeypatch):
    """Two rules sharing a common id prefix → revoke must refuse to act."""
    import uuid

    from claritymed.stores.settings_store import SettingsStore

    # Force both new rule ids to share the same first 4 chars so a prefix
    # matches both. uuid4 stub returns hand-crafted hex strings.
    state = {"i": 0}
    fakes = [
        uuid.UUID("abcd0001-0000-0000-0000-000000000001"),
        uuid.UUID("abcd0002-0000-0000-0000-000000000002"),
    ]

    def fake_uuid4():
        v = fakes[state["i"]]
        state["i"] += 1
        return v

    monkeypatch.setattr(uuid, "uuid4", fake_uuid4)

    store = SettingsStore("alice")
    store.add_rule("save_allergy", {"a": 1})
    store.add_rule("save_allergy", {"b": 2})

    result = runner.invoke(app, ["tool", "rule-revoke", "abcd", "--user", "alice"])
    assert result.exit_code == 1


def test_tool_rule_list_with_rules(_alice):
    from claritymed.stores.settings_store import SettingsStore

    SettingsStore("alice").add_rule("save_allergy", {"substance": "peanut"})
    result = runner.invoke(app, ["tool", "rule-list", "--user", "alice"])
    assert result.exit_code == 0
    assert "save_allergy" in result.stdout


def test_tool_rule_revoke_happy_path(_alice):
    from claritymed.stores.settings_store import SettingsStore

    rule = SettingsStore("alice").add_rule("save_allergy", {"x": 1})
    result = runner.invoke(app, ["tool", "rule-revoke", rule.id, "--user", "alice"])
    assert result.exit_code == 0
    assert "revoked" in result.stdout


def test_tool_run_auto_approve_requires_non_tty(_alice, monkeypatch):
    """`_ensure_headless_or_die` errors when stdin is a tty."""
    import typer

    # Direct call ensures we exercise the isatty check independent of CliRunner.
    from claritymed.cli.commands.tool import _ensure_headless_or_die

    class _TtyStdin:
        def isatty(self):
            return True

    import sys

    monkeypatch.setattr(sys, "stdin", _TtyStdin())
    monkeypatch.setenv("CLARITYMED_HEADLESS", "1")
    with pytest.raises(typer.Exit):
        _ensure_headless_or_die()


def test_tool_run_auto_approve_requires_headless_env(_alice, monkeypatch):
    """`--auto-approve` errors when CLARITYMED_HEADLESS is not 1."""
    import sys

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.delenv("CLARITYMED_HEADLESS", raising=False)
    result = runner.invoke(
        app,
        [
            "tool",
            "run",
            "save_allergy",
            json.dumps(
                {"substance": "peanut", "severity": "mild", "source": "self_report"}
            ),
            "--user",
            "alice",
            "--auto-approve",
        ],
    )
    assert result.exit_code == 2


def test_tool_run_denied_by_settings_rule(_alice, monkeypatch):
    """A deny rule that matches the tool args blocks invocation."""
    from claritymed.stores.settings_store import SettingsStore

    SettingsStore("alice").add_rule(
        "save_allergy", {"substance": "peanut"}, action="deny"
    )

    import sys

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setenv("CLARITYMED_HEADLESS", "1")
    result = runner.invoke(
        app,
        [
            "tool",
            "run",
            "save_allergy",
            json.dumps(
                {"substance": "peanut", "severity": "mild", "source": "self_report"}
            ),
            "--user",
            "alice",
            "--auto-approve",
        ],
    )
    assert result.exit_code == 1


def test_tool_run_auto_approved_happy_path(_alice, monkeypatch):
    """Headless + auto-approve + clean settings → tool runs and JSON printed."""
    import sys

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setenv("CLARITYMED_HEADLESS", "1")
    result = runner.invoke(
        app,
        [
            "tool",
            "run",
            "save_allergy",
            json.dumps(
                {"substance": "peanut", "severity": "mild", "source": "self_report"}
            ),
            "--user",
            "alice",
            "--auto-approve",
        ],
    )
    assert result.exit_code == 0, result.stdout
    # The tool returns a dict; print_json yields a "{" line.
    assert "{" in result.stdout


def test_tool_run_tool_impl_raises(_alice, monkeypatch):
    """When the tool implementation raises, CLI exits 1 with a red error line."""
    import sys

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    monkeypatch.setenv("CLARITYMED_HEADLESS", "1")

    from claritymed.orchestrator.features import ingest_tools_plugin as _itp

    def _boom(*a, **kw):
        raise RuntimeError("internal error")

    # Replace the tool impl mid-test; restored by monkeypatch.
    monkeypatch.setitem(_itp.INGEST_TOOLS, "save_allergy", _boom)

    result = runner.invoke(
        app,
        [
            "tool",
            "run",
            "save_allergy",
            json.dumps(
                {"substance": "peanut", "severity": "mild", "source": "self_report"}
            ),
            "--user",
            "alice",
            "--auto-approve",
        ],
    )
    assert result.exit_code == 1


def test_ensure_headless_or_die_passes_with_headless_and_non_tty(monkeypatch):
    """Happy path: non-tty + CLARITYMED_HEADLESS=1 → no SystemExit."""
    from claritymed.cli.commands.tool import _ensure_headless_or_die

    class _PipeStdin:
        def isatty(self):
            return False

    import sys

    monkeypatch.setattr(sys, "stdin", _PipeStdin())
    monkeypatch.setenv("CLARITYMED_HEADLESS", "1")
    _ensure_headless_or_die()  # must not raise


# ===========================================================================
# common.prefetch_models
# ===========================================================================


def test_prefetch_models_noop_when_disabled(monkeypatch):
    from claritymed.cli import common as _common
    from claritymed.core.scrub.service import (
        PrivacyFilterConfig,
        ScrubConfig,
        ScrubService,
    )

    svc = ScrubService(ScrubConfig(privacy_filter=PrivacyFilterConfig(enabled=False)))
    monkeypatch.setattr(ScrubService, "from_config", classmethod(lambda cls: svc))
    _common.prefetch_models()  # must not raise


def test_prefetch_models_raises_systemexit_on_missing_deps(monkeypatch):
    from claritymed.cli import common as _common
    from claritymed.core.scrub.service import (
        PrivacyFilterConfig,
        ScrubConfig,
        ScrubService,
    )

    svc = ScrubService(ScrubConfig(privacy_filter=PrivacyFilterConfig(enabled=True)))
    monkeypatch.setattr(ScrubService, "from_config", classmethod(lambda cls: svc))

    def _missing(self):
        raise ImportError("onnxruntime missing")

    monkeypatch.setattr(ScrubService, "check_runtime_deps", _missing)

    with pytest.raises(SystemExit):
        _common.prefetch_models()


def test_prefetch_models_raises_systemexit_when_download_fails(monkeypatch):
    from claritymed.cli import common as _common
    from claritymed.core.scrub.service import (
        PrivacyFilterConfig,
        ScrubConfig,
        ScrubService,
    )

    svc = ScrubService(ScrubConfig(privacy_filter=PrivacyFilterConfig(enabled=True)))
    monkeypatch.setattr(ScrubService, "from_config", classmethod(lambda cls: svc))
    monkeypatch.setattr(ScrubService, "check_runtime_deps", lambda self: None)
    monkeypatch.setattr(ScrubService, "ensure_downloaded", lambda self: False)

    with pytest.raises(SystemExit):
        _common.prefetch_models()


def test_try_load_account_returns_none_for_unknown_user(tmp_path, monkeypatch):
    monkeypatch.setenv("CLARITYMED_HOME", str(tmp_path))
    from claritymed.cli.common import try_load_account

    assert try_load_account("ghost") is None


def test_try_load_account_returns_account_when_present(_alice):
    from claritymed.cli.common import try_load_account

    acc = try_load_account("alice")
    assert acc is not None
    assert acc.user_id == "alice"


def test_try_current_account_returns_none_outside_context():
    """Without an inject_context scope, current_account() raises → returns None."""
    from claritymed.cli.common import try_current_account

    assert try_current_account() is None


# ===========================================================================
# ask CLI — main flow with mocked AskService/build_model
# ===========================================================================


def test_ask_streams_token_chunks_and_finishes(_alice, monkeypatch):
    """Happy path: stream a TokenChunk and a Done — exit 0, text in stdout."""
    from types import SimpleNamespace

    from claritymed.cli.commands import ask as _ask_mod
    from claritymed.orchestrator.services import Done, TokenChunk
    from claritymed.orchestrator.services import ask_service as _svc_mod

    fake_provider = SimpleNamespace(id="local-fake", model="qwen3:14b", kind="local")
    monkeypatch.setattr(_ask_mod, "resolve_provider", lambda **kw: fake_provider)
    monkeypatch.setattr(_ask_mod, "build_model", lambda _p: object())
    monkeypatch.setattr(_ask_mod, "_maybe_build_strategy", lambda model: None)

    async def _fake_run(self, question, **kwargs):
        yield TokenChunk(text="answer")
        yield Done(final=SimpleNamespace(text="answer"))

    monkeypatch.setattr(_svc_mod.AskService, "run", _fake_run)

    # Translation provider factory is also imported lazily.
    import claritymed.core.translation as _tr

    monkeypatch.setattr(_tr, "make_translation_provider", lambda model, phi_kind: None)

    result = runner.invoke(app, ["ask", "what is glucose", "--user", "alice"])
    assert result.exit_code == 0, result.stdout
    assert "answer" in result.stdout


def test_ask_error_event_exits_1(_alice, monkeypatch):
    """An Error event during streaming → exit 1."""
    from types import SimpleNamespace

    from claritymed.cli.commands import ask as _ask_mod
    from claritymed.orchestrator.services import Error
    from claritymed.orchestrator.services import ask_service as _svc_mod

    fake_provider = SimpleNamespace(id="local-fake", model="qwen3:14b", kind="local")
    monkeypatch.setattr(_ask_mod, "resolve_provider", lambda **kw: fake_provider)
    monkeypatch.setattr(_ask_mod, "build_model", lambda _p: object())
    monkeypatch.setattr(_ask_mod, "_maybe_build_strategy", lambda model: None)
    import claritymed.core.translation as _tr

    monkeypatch.setattr(_tr, "make_translation_provider", lambda model, phi_kind: None)

    async def _fake_run(self, question, **kwargs):
        yield Error(error_type="OOPS", message="something broke")

    monkeypatch.setattr(_svc_mod.AskService, "run", _fake_run)

    result = runner.invoke(app, ["ask", "what is glucose", "--user", "alice"])
    assert result.exit_code == 1
