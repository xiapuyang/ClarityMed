"""Tests for the ``claritymed eval run <task>`` Typer command."""

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


def test_eval_help_lists_run_subcommand():
    result = runner.invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    assert "run" in result.stdout


def test_eval_run_help_documents_with_rag():
    result = runner.invoke(app, ["eval", "run", "--help"])
    assert result.exit_code == 0
    assert "--with-rag" in result.stdout
    assert "TASK_ID" in result.stdout.upper()


# ---------------------------------------------------------------------------
# Happy path — mocked runner
# ---------------------------------------------------------------------------


def _stub_run_result(tmp_path: Path, task_id: str = "medqa") -> RunResult:
    from datetime import datetime, timezone

    output = tmp_path / f"ollama_{task_id}_x.jsonl"
    output.write_text("", encoding="utf-8")
    return RunResult(
        provider_id="ollama",
        task_id=task_id,
        n_questions=5,
        accuracy=0.6,
        output_path=output,
        started_at=datetime.now(timezone.utc),
        duration_s=1.2,
    )


def test_eval_run_invokes_runner_with_task_id(tmp_path, monkeypatch):
    """Positional ``task_id`` is forwarded verbatim to the runner — the
    extensibility claim: new YAML + ``eval run <name>`` is enough."""
    captured: dict[str, Any] = {}

    class _StubRunner:
        def run(self, provider, task_id, limit):
            captured["provider_id"] = provider.id
            captured["task_id"] = task_id
            captured["limit"] = limit
            return _stub_run_result(tmp_path, task_id)

    monkeypatch.setattr(
        "claritymed.cli.commands.eval.LmEvalRunner", lambda *a, **kw: _StubRunner()
    )
    result = runner.invoke(
        app, ["eval", "run", "medqa", "--provider", "ollama", "--limit", "5"]
    )
    assert result.exit_code == 0, result.stdout
    assert captured == {"provider_id": "ollama", "task_id": "medqa", "limit": 5}


def test_eval_run_forwards_arbitrary_task_id(tmp_path, monkeypatch):
    """Any string is a valid ``task_id`` from the CLI's perspective —
    discoverability lives in the tasks/ directory, not in argparse."""
    captured: dict[str, Any] = {}

    class _StubRunner:
        def run(self, provider, task_id, limit):
            captured["task_id"] = task_id
            return _stub_run_result(tmp_path, task_id)

    monkeypatch.setattr(
        "claritymed.cli.commands.eval.LmEvalRunner", lambda *a, **kw: _StubRunner()
    )
    result = runner.invoke(
        app, ["eval", "run", "cmb_exam", "--provider", "ollama", "--limit", "1"]
    )
    assert result.exit_code == 0, result.stdout
    assert captured["task_id"] == "cmb_exam"


def test_eval_run_omits_provider_auto_picks_reachable(tmp_path, monkeypatch):
    """No --provider → auto-pick the first reachable local provider, same
    way the e2e fixture does. Mocked to return a specific provider for
    deterministic assertion."""
    from claritymed.core.schemas import ProviderConfig

    captured: dict[str, Any] = {}

    class _StubRunner:
        def run(self, provider, task_id, limit):
            captured["provider_id"] = provider.id
            return _stub_run_result(tmp_path)

    fake_provider = ProviderConfig.model_validate(
        {
            "id": "omlx",
            "kind": "local",
            "model": "Qwen3.6-35B-A3B-oQ4-mtp",
            "base_url": "http://127.0.0.1:8000/v1",
            "api_key_env": "OMLX_API_KEY",
        }
    )
    monkeypatch.setattr(
        "claritymed.cli.commands.eval.pick_reachable_provider", lambda: fake_provider
    )
    monkeypatch.setattr(
        "claritymed.cli.commands.eval.LmEvalRunner", lambda *a, **kw: _StubRunner()
    )
    result = runner.invoke(app, ["eval", "run", "medqa", "--limit", "1"])
    assert result.exit_code == 0, result.stdout
    assert captured["provider_id"] == "omlx"
    assert "auto-selected provider" in result.stdout


def test_eval_run_omits_provider_falls_back_to_catalog_default(tmp_path, monkeypatch):
    """No --provider AND no reachable provider → fall back to the catalog
    default (the LLM call will then fail loud at the wire, which is the
    expected behavior — we don't pretend to succeed)."""
    captured: dict[str, Any] = {}

    class _StubRunner:
        def run(self, provider, task_id, limit):
            captured["provider_id"] = provider.id
            return _stub_run_result(tmp_path)

    monkeypatch.setattr(
        "claritymed.cli.commands.eval.pick_reachable_provider", lambda: None
    )
    monkeypatch.setattr(
        "claritymed.cli.commands.eval.LmEvalRunner", lambda *a, **kw: _StubRunner()
    )
    result = runner.invoke(app, ["eval", "run", "medqa", "--limit", "1"])
    assert result.exit_code == 0, result.stdout
    assert captured["provider_id"] == "ollama"  # repo catalog default


# ---------------------------------------------------------------------------
# --with-rag (Phase 2) — swaps the LM adapter and tags the JSONL filename
# ---------------------------------------------------------------------------


def test_with_rag_swaps_adapter_and_tags_output(tmp_path, monkeypatch):
    """``--with-rag`` constructs the runner with a factory that yields a
    ``ClaritymedRagLM`` and ``run_tag="with-rag"`` so the JSONL filename
    is paired by ``eval delta``."""
    from claritymed.evals.lm.rag import ClaritymedRagLM

    captured: dict[str, Any] = {}

    class _StubRunner:
        def run(self, provider, task_id, limit):
            captured["provider_id"] = provider.id
            captured["task_id"] = task_id
            captured["limit"] = limit
            return _stub_run_result(tmp_path)

    def _record_factory(*args, **kwargs):
        captured["lm_factory"] = kwargs.get("lm_factory")
        captured["run_tag"] = kwargs.get("run_tag")
        return _StubRunner()

    monkeypatch.setattr("claritymed.cli.commands.eval.LmEvalRunner", _record_factory)
    # Block real adapter construction inside the factory closure so
    # invoking it (test below) doesn't touch the model layer.
    monkeypatch.setattr(
        "claritymed.evals.lm.rag.ClaritymedRagLM._build_service",
        lambda self: None,
    )
    monkeypatch.setattr("claritymed.evals.lm.rag._default_strategy", lambda _p: None)
    result = runner.invoke(
        app,
        ["eval", "run", "medqa", "--provider", "ollama", "--with-rag", "--limit", "1"],
    )
    assert result.exit_code == 0, result.stdout
    assert captured["run_tag"] == "with-rag"
    assert captured["provider_id"] == "ollama"

    # The factory should yield a ClaritymedRagLM with the default
    # ``deterministic`` rag_mode — caller didn't pass --rag-mode.
    from claritymed.core.schemas import ProviderConfig

    fake = ProviderConfig.model_validate(
        {
            "id": "anything",
            "kind": "local",
            "model": "stub",
            "base_url": "http://x/v1",
        }
    )
    lm = captured["lm_factory"](fake)
    assert isinstance(lm, ClaritymedRagLM)
    assert lm._rag_mode == "deterministic"


def test_with_rag_rag_mode_tool_passed_through(tmp_path, monkeypatch):
    """``--rag-mode tool`` flows into ``ClaritymedRagLM(rag_mode="tool")``
    so the operator can compare deterministic (always-retrieve) and
    tool-mode (LLM-decides) RAG arms."""
    from claritymed.evals.lm.rag import ClaritymedRagLM

    captured: dict[str, Any] = {}

    class _StubRunner:
        def run(self, provider, task_id, limit):
            return _stub_run_result(tmp_path)

    def _record_factory(*args, **kwargs):
        captured["lm_factory"] = kwargs.get("lm_factory")
        return _StubRunner()

    monkeypatch.setattr("claritymed.cli.commands.eval.LmEvalRunner", _record_factory)
    monkeypatch.setattr(
        "claritymed.evals.lm.rag.ClaritymedRagLM._build_service",
        lambda self: None,
    )
    monkeypatch.setattr("claritymed.evals.lm.rag._default_strategy", lambda _p: None)
    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "medqa",
            "--provider",
            "ollama",
            "--with-rag",
            "--rag-mode",
            "tool",
            "--limit",
            "1",
        ],
    )
    assert result.exit_code == 0, result.stdout

    from claritymed.core.schemas import ProviderConfig

    fake = ProviderConfig.model_validate(
        {
            "id": "anything",
            "kind": "local",
            "model": "stub",
            "base_url": "http://x/v1",
        }
    )
    lm = captured["lm_factory"](fake)
    assert isinstance(lm, ClaritymedRagLM)
    assert lm._rag_mode == "tool"


def test_invalid_rag_mode_rejected(tmp_path, monkeypatch):
    """``--rag-mode wonky`` exits non-zero with a clear error."""

    class _Should_Not_Run:
        def run(self, *a, **kw):
            raise AssertionError("runner should not be invoked on bad rag-mode")

    monkeypatch.setattr(
        "claritymed.cli.commands.eval.LmEvalRunner", lambda *a, **kw: _Should_Not_Run()
    )
    result = runner.invoke(
        app,
        [
            "eval",
            "run",
            "medqa",
            "--provider",
            "ollama",
            "--with-rag",
            "--rag-mode",
            "wonky",
            "--limit",
            "1",
        ],
    )
    assert result.exit_code == 2
    assert "rag-mode" in result.stdout or "rag-mode" in (result.stderr or "")


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_unknown_provider_exits_two(monkeypatch):
    class _Should_Not_Run:
        def run(self, *a, **kw):
            raise AssertionError("runner should not be invoked")

    monkeypatch.setattr(
        "claritymed.cli.commands.eval.LmEvalRunner", lambda *a, **kw: _Should_Not_Run()
    )
    result = runner.invoke(
        app,
        ["eval", "run", "medqa", "--provider", "does-not-exist", "--limit", "1"],
    )
    assert result.exit_code == 2
    assert "does-not-exist" in result.stdout or "does-not-exist" in (
        result.stderr or ""
    )


def test_negative_limit_rejected_by_typer():
    result = runner.invoke(
        app, ["eval", "run", "medqa", "--provider", "ollama", "--limit", "-5"]
    )
    assert result.exit_code != 0
    # Typer renders the range error as part of usage output.
    assert "--limit" in (result.stdout + (result.stderr or ""))
