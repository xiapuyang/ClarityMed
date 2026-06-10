"""Smoke tests for the Typer CLI subcommands."""

from __future__ import annotations

from typer.testing import CliRunner

from claritymed.cli.main import app

runner = CliRunner()


def test_help_lists_three_modes():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.stdout
    assert "ask" in result.stdout
    assert "rag" in result.stdout


def test_ingest_profile_writes_field(tmp_path, monkeypatch):
    from claritymed.stores.account import init_user

    monkeypatch.setenv("CLARITYMED_HOME", str(tmp_path))
    init_user("alice")
    result = runner.invoke(
        app,
        ["ingest", "profile", "allergy=penicillin", "--user", "alice"],
    )
    assert result.exit_code == 0, result.stdout
    assert "saved" in result.stdout.lower()


def test_rag_add_with_file(tmp_path, monkeypatch):
    """Patch the embedder so the test does not download a fastembed model."""
    import hashlib

    from qdrant_client import AsyncQdrantClient

    from claritymed.core.rag.chunking.base import (
        ChildChunk,
        ChunkedDocument,
        ParentChunk,
        RawDocument,
    )
    from claritymed.core.rag.embedding.base import Embedder, SparseVector
    from claritymed.core.phi.guard import PhiGuard
    from claritymed.stores import user_rag as _ur

    class _StubEmbedder(Embedder):
        @property
        def dimension(self) -> int:
            return 32

        async def embed_dense(self, texts: list[str]) -> list[list[float]]:
            out = []
            for text in texts:
                digest = hashlib.sha256(text.encode()).digest()
                out.append([b / 255.0 for b in digest[: self.dimension]])
            return out

        async def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
            return [{abs(hash(t)) % 100: 0.5} for t in texts]

    class _StubChunker:
        def chunk(self, doc: RawDocument) -> ChunkedDocument:
            if not doc.text.strip():
                return ChunkedDocument(parents=[], children=[])
            import uuid

            parent_id = f"{doc.doc_id}#p0"
            parent = ParentChunk(
                parent_id=parent_id,
                text=doc.text,
                doc_id=doc.doc_id,
                parent_index=0,
            )
            child = ChildChunk(
                child_id=str(uuid.uuid5(uuid.NAMESPACE_URL, doc.doc_id)),
                text=doc.text,
                parent_id=parent_id,
                doc_id=doc.doc_id,
                chunk_index=0,
            )
            return ChunkedDocument(parents=[parent], children=[child])

    def _factory(_user_id: str):
        # Signature matches make_user_rag_store(user_id); the user_id is
        # ignored because the in-memory backend doesn't care about paths.
        return _ur.UserRagStore(
            aclient=AsyncQdrantClient(":memory:"),
            embedder=_StubEmbedder(),
            chunker=_StubChunker(),
            guard=PhiGuard.from_config(),
        )

    monkeypatch.setattr(_ur, "make_user_rag_store", _factory)
    # `make_user_rag_store` is imported by name into the rag subcommand
    # module; patch both bindings so the CLI invocation hits the stub.
    import claritymed.cli.commands.rag as _cli_rag

    monkeypatch.setattr(_cli_rag, "make_user_rag_store", _factory)
    monkeypatch.setenv("CLARITYMED_HOME", str(tmp_path))
    from claritymed.stores.account import init_user

    init_user("alice")
    sample = tmp_path / "sample.txt"
    sample.write_text("para one\n\npara two", encoding="utf-8")
    result = runner.invoke(
        app,
        ["rag", "add", str(sample), "--user", "alice", "--public"],
    )
    assert result.exit_code == 0, result.stdout
    assert "doc_id=" in result.stdout
    assert "chunks=" in result.stdout


def test_ask_help_does_not_error():
    """ask subcommand registers; the actual call requires a running LLM, so we
    only smoke-test that --help works and the command is wired."""
    result = runner.invoke(app, ["ask", "--help"])
    assert result.exit_code == 0
    assert "question" in result.stdout.lower()


def test_unknown_subcommand_returns_nonzero():
    result = runner.invoke(app, ["bogus-subcmd"])
    assert result.exit_code != 0


def test_tui_unknown_provider_fails_fast(tmp_path, monkeypatch):
    """`--provider <typo>` should exit non-zero before launching the TUI.

    Regression: prior behavior was to start the TUI, show the bad id in the
    status bar, then explode on the first user submission. The project rule
    is loud provider-resolution failures, not silent fallbacks.
    """
    monkeypatch.setenv("CLARITYMED_HOME", str(tmp_path))
    result = runner.invoke(
        app,
        ["tui", "--user", "alice", "--provider", "oMLX"],
    )
    assert result.exit_code != 0
    # Typer prints --help on Exit; mixed stdout/stderr — combined output is fine.
    combined = (result.stdout or "") + (result.stderr or "")
    assert "oMLX" in combined
    assert "models.yaml" in combined
