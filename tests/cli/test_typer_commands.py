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


def test_modes_command_lists_registry():
    result = runner.invoke(app, ["modes"])
    assert result.exit_code == 0
    assert "ingest" in result.stdout
    assert "ask" in result.stdout
    assert "rag" in result.stdout


def test_ingest_profile_writes_field(tmp_path, monkeypatch):
    monkeypatch.setenv("CLARITYMED_HOME", str(tmp_path))
    result = runner.invoke(
        app,
        ["ingest", "profile", "allergy=penicillin", "--user", "alice"],
    )
    assert result.exit_code == 0, result.stdout
    assert "saved" in result.stdout.lower()


def test_rag_add_with_file(tmp_path, monkeypatch):
    """Patch the embedder so the test does not download a fastembed model."""
    import hashlib

    from qdrant_client import QdrantClient

    from claritymed.orchestrator import PhiGuard
    from claritymed.stores import user_rag as _ur

    class _StubEmbedder:
        def embed(self, text: str) -> list[float]:
            digest = hashlib.sha256(text.encode()).digest()
            return [b / 255.0 for b in digest[: self.dimension]]

        @property
        def dimension(self) -> int:
            return 32

    def _factory(qdrant_path: str | None = None):
        return _ur.UserRagStore(
            client=QdrantClient(":memory:"),
            embedder=_StubEmbedder(),
            guard=PhiGuard.from_config(),
        )

    monkeypatch.setattr(
        _ur.UserRagStore,
        "from_defaults",
        classmethod(lambda cls, qdrant_path=None: _factory()),
    )
    monkeypatch.setenv("CLARITYMED_HOME", str(tmp_path))
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
