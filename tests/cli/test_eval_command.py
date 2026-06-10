"""Tests for the ``claritymed eval medqa`` Typer command."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from claritymed.cli.main import app
from claritymed.evals.protocol import RunResult

runner = CliRunner()


# ---------------------------------------------------------------------------
# Help / discoverability
# ---------------------------------------------------------------------------


def test_eval_help_lists_medqa_subcommand():
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    assert "medqa" in result.stdout


def test_eval_medqa_help_documents_with_rag():
    result = runner.invoke(app, ["eval", "medqa", "--help"])
    assert result.exit_code == 0
    assert "--with-rag" in result.stdout


# ---------------------------------------------------------------------------
# Happy path — mocked runner
# ---------------------------------------------------------------------------


def _stub_run_result(tmp_path: Path) -> RunResult:
    from datetime import datetime, timezone

    output = tmp_path / "ollama_medqa_x.jsonl"
    output.write_text("", encoding="utf-8")
    return RunResult(
        provider_id="ollama",
        task_id="medqa",
        n_questions=5,
        accuracy=0.6,
        output_path=output,
        started_at=datetime.now(timezone.utc),
        duration_s=1.2,
    )


def test_eval_medqa_invokes_runner(tmp_path, monkeypatch):
    captured: dict[str, Any] = {}

    class _StubRunner:
        def run(self, provider, task_id, limit):
            captured["provider_id"] = provider.id
            captured["task_id"] = task_id
            captured["limit"] = limit
            return _stub_run_result(tmp_path)

    monkeypatch.setattr(
        "claritymed.cli.eval.LmEvalRunner", lambda *a, **kw: _StubRunner()
    )
    result = runner.invoke(
        app, ["eval", "medqa", "--provider", "ollama", "--limit", "5"]
    )
    assert result.exit_code == 0, result.stdout
    assert captured == {"provider_id": "ollama", "task_id": "medqa", "limit": 5}


def test_eval_medqa_omits_provider_falls_back_to_default(tmp_path, monkeypatch):
    """No --provider → resolve_provider falls back to default ('ollama' in
    the repo's configs/models.yaml)."""
    captured: dict[str, Any] = {}

    class _StubRunner:
        def run(self, provider, task_id, limit):
            captured["provider_id"] = provider.id
            return _stub_run_result(tmp_path)

    monkeypatch.setattr(
        "claritymed.cli.eval.LmEvalRunner", lambda *a, **kw: _StubRunner()
    )
    result = runner.invoke(app, ["eval", "medqa", "--limit", "1"])
    assert result.exit_code == 0, result.stdout
    assert captured["provider_id"] == "ollama"  # repo default


# ---------------------------------------------------------------------------
# --with-rag is parked until Phase 2
# ---------------------------------------------------------------------------


def test_with_rag_exits_non_zero_until_phase_2(monkeypatch):
    # Even with a stub runner injected, --with-rag must short-circuit before
    # calling the runner.
    called = {"hit": False}

    class _Should_Not_Run:
        def run(self, *a, **kw):
            called["hit"] = True
            raise AssertionError("runner should not be invoked under --with-rag")

    monkeypatch.setattr(
        "claritymed.cli.eval.LmEvalRunner", lambda *a, **kw: _Should_Not_Run()
    )
    result = runner.invoke(
        app, ["eval", "medqa", "--provider", "ollama", "--with-rag", "--limit", "1"]
    )
    assert result.exit_code == 2
    assert called["hit"] is False


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_unknown_provider_exits_two(monkeypatch):
    class _Should_Not_Run:
        def run(self, *a, **kw):
            raise AssertionError("runner should not be invoked")

    monkeypatch.setattr(
        "claritymed.cli.eval.LmEvalRunner", lambda *a, **kw: _Should_Not_Run()
    )
    result = runner.invoke(
        app, ["eval", "medqa", "--provider", "does-not-exist", "--limit", "1"]
    )
    assert result.exit_code == 2
    assert "does-not-exist" in result.stdout or "does-not-exist" in (
        result.stderr or ""
    )


def test_negative_limit_rejected_by_typer():
    result = runner.invoke(
        app, ["eval", "medqa", "--provider", "ollama", "--limit", "-5"]
    )
    assert result.exit_code != 0
    # Typer renders the range error as part of usage output.
    assert "--limit" in (result.stdout + (result.stderr or ""))
