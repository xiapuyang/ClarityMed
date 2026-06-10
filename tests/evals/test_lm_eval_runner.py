"""Tests for ``LmEvalRunner`` — JSONL writer + audit emission."""

from __future__ import annotations

import json
from typing import Any

import pytest

from claritymed.context import apply_context, new_request_id, reset_context
from claritymed.core.schemas import ProviderConfig
from claritymed.evals.runners.lm_eval_runner import LmEvalRunner

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _provider() -> ProviderConfig:
    return ProviderConfig.model_validate(
        {
            "id": "ollama",
            "kind": "local",
            "model": "qwen3:14b",
            "base_url": "http://127.0.0.1:11434/v1",
        }
    )


@pytest.fixture
def ctx():
    """Set request/user/language ContextVars for the test."""
    tokens = apply_context(new_request_id(), "eval", "en")
    try:
        yield
    finally:
        reset_context(tokens)


class _StubLM:
    """Looks like our ClaritymedBaselineLM enough for the writer."""

    def __init__(self, provider: ProviderConfig) -> None:
        self.provider_id = provider.id
        self.model_name = provider.model
        self.latencies_ms = [12.3, 45.6]
        self.latencies_ms_by_doc_id = {0: 12.3, 1: 45.6}


def _synthetic_samples() -> list[dict[str, Any]]:
    return [
        {
            "doc_id": 0,
            "doc": {"sent1": "Q1"},
            "target": "B",
            "arguments": [("Question: Q1\nAnswer:", {})],
            "resps": [["Answer: B"]],
            "filtered_resps": ["B"],
            "exact_match": 1.0,
            "filter": "extract_letter",
            "metrics": ["exact_match"],
        },
        {
            "doc_id": 1,
            "doc": {"sent1": "Q2"},
            "target": "C",
            "arguments": [("Question: Q2\nAnswer:", {})],
            "resps": [["A"]],
            "filtered_resps": ["A"],
            "exact_match": 0.0,
            "filter": "extract_letter",
            "metrics": ["exact_match"],
        },
    ]


def _synthetic_results(accuracy: float = 0.5) -> dict[str, Any]:
    return {
        "samples": {"medqa": _synthetic_samples()},
        "results": {
            "medqa": {
                "name": "medqa",
                "alias": "medqa",
                "sample_len": 2,
                "exact_match,extract_letter": accuracy,
                "exact_match_stderr,extract_letter": 0.0,
            }
        },
    }


def _runner(tmp_path) -> LmEvalRunner:
    return LmEvalRunner(output_dir=tmp_path, lm_factory=_StubLM)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_run_writes_one_jsonl_row_per_sample(tmp_path, ctx, monkeypatch, capsys):
    runner = _runner(tmp_path)
    monkeypatch.setattr(
        runner, "_simple_evaluate", lambda lm, t, lim: _synthetic_results(0.5)
    )

    result = runner.run(_provider(), "medqa", limit=2)

    assert result.n_questions == 2
    assert result.accuracy == 0.5
    assert result.task_id == "medqa"
    assert result.provider_id == "ollama"
    assert result.output_path.exists()

    rows = [
        json.loads(line)
        for line in result.output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 2

    # Field shape per the plan's R2.
    expected_keys = {
        "request_id",
        "task_id",
        "provider_id",
        "model_name",
        "question_idx",
        "question",
        "gold_letter",
        "model_completion",
        "extracted_letter",
        "correct",
        "latency_ms",
    }
    assert set(rows[0].keys()) == expected_keys

    assert rows[0]["question_idx"] == 0
    assert rows[0]["gold_letter"] == "B"
    assert rows[0]["extracted_letter"] == "B"
    assert rows[0]["correct"] is True
    assert rows[0]["model_completion"] == "Answer: B"
    assert rows[0]["latency_ms"] == 12.3

    assert rows[1]["correct"] is False
    assert rows[1]["extracted_letter"] == "A"
    assert rows[1]["latency_ms"] == 45.6

    captured = capsys.readouterr()
    assert "medqa" in captured.out
    assert "0.5000" in captured.out


def test_output_filename_follows_convention(tmp_path, ctx, monkeypatch):
    runner = _runner(tmp_path)
    monkeypatch.setattr(
        runner, "_simple_evaluate", lambda lm, t, lim: _synthetic_results(1.0)
    )
    result = runner.run(_provider(), "medqa", limit=2)
    name = result.output_path.name
    assert name.startswith("ollama_medqa_")
    assert name.endswith(".jsonl")


def test_audit_events_emitted(tmp_path, ctx, monkeypatch):
    """The audit logger has ``propagate=False`` and writes only to its
    own rotating file handler, so we capture the calls at the source by
    patching ``audit_event`` in the runner module."""
    runner = _runner(tmp_path)
    monkeypatch.setattr(
        runner, "_simple_evaluate", lambda lm, t, lim: _synthetic_results(0.6)
    )

    calls: list[tuple[str, dict[str, Any]]] = []

    def _record(kind, payload=None):
        calls.append((kind, payload or {}))

        class _Stub:
            pass

        return _Stub()

    monkeypatch.setattr("claritymed.evals.runners.lm_eval_runner.audit_event", _record)
    runner.run(_provider(), "medqa", limit=2)

    kinds = [k for k, _ in calls]
    assert "eval.run.started" in kinds
    assert "eval.run.completed" in kinds
    assert "eval.run.failed" not in kinds

    started_payload = next(p for k, p in calls if k == "eval.run.started")
    assert started_payload["provider_id"] == "ollama"
    assert started_payload["task_id"] == "medqa"
    assert started_payload["limit"] == 2

    completed_payload = next(p for k, p in calls if k == "eval.run.completed")
    assert completed_payload["n_questions"] == 2
    assert completed_payload["accuracy"] == 0.6
    assert "output_path" in completed_payload


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_limit_zero_yields_nan_accuracy(tmp_path, ctx, monkeypatch, capsys):
    runner = _runner(tmp_path)
    empty = {"samples": {"medqa": []}, "results": {"medqa": {}}}
    monkeypatch.setattr(runner, "_simple_evaluate", lambda lm, t, lim: empty)

    result = runner.run(_provider(), "medqa", limit=0)

    assert result.n_questions == 0
    # NaN == NaN is False; check via != self.
    assert result.accuracy != result.accuracy
    assert result.output_path.exists()
    assert result.output_path.read_text(encoding="utf-8") == ""


def test_output_dir_created_if_missing(tmp_path, ctx, monkeypatch):
    deep = tmp_path / "a" / "b" / "c"
    runner = LmEvalRunner(output_dir=deep, lm_factory=_StubLM)
    monkeypatch.setattr(
        runner, "_simple_evaluate", lambda lm, t, lim: _synthetic_results(1.0)
    )
    runner.run(_provider(), "medqa", limit=2)
    assert deep.exists()


def test_jsonl_row_uses_ordinal_when_doc_id_missing(tmp_path, ctx, monkeypatch):
    """Defensive: not all task types set ``doc_id``. The runner should
    still write a row with a sensible ``question_idx``."""
    runner = _runner(tmp_path)
    sample = {
        "doc_id": None,
        "target": "B",
        "arguments": [("Q?", {})],
        "resps": [["B"]],
        "filtered_resps": ["B"],
        "exact_match": 1.0,
    }
    fake_results = {
        "samples": {"medqa": [sample]},
        "results": {"medqa": {"exact_match,extract_letter": 1.0}},
    }
    monkeypatch.setattr(runner, "_simple_evaluate", lambda lm, t, lim: fake_results)

    result = runner.run(_provider(), "medqa", limit=1)
    row = json.loads(result.output_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["question_idx"] == 0
    assert row["latency_ms"] == 12.3  # fell back to positional list


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


def test_simple_evaluate_failure_emits_failed_audit_and_reraises(
    tmp_path, ctx, monkeypatch
):
    runner = _runner(tmp_path)

    def _boom(_lm, _task, _limit):
        raise RuntimeError("dataset offline")

    monkeypatch.setattr(runner, "_simple_evaluate", _boom)

    calls: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        "claritymed.evals.runners.lm_eval_runner.audit_event",
        lambda k, payload=None: calls.append((k, payload or {})) or None,
    )

    with pytest.raises(RuntimeError, match="dataset offline"):
        runner.run(_provider(), "medqa", limit=2)

    kinds = [k for k, _ in calls]
    assert "eval.run.started" in kinds
    assert "eval.run.failed" in kinds
    assert "eval.run.completed" not in kinds

    failed_payload = next(p for k, p in calls if k == "eval.run.failed")
    assert failed_payload["error_type"] == "RuntimeError"
    assert "dataset offline" in failed_payload["message"]


# ---------------------------------------------------------------------------
# Integration: real simple_evaluate against the medqa task with stub LM
# ---------------------------------------------------------------------------


def test_real_simple_evaluate_with_stub_lm(tmp_path, ctx):
    """End-to-end: real lm-eval-harness driving our medqa task YAML with a
    stub LM that returns 'A' to every question. Confirms (a) the task
    loads from our include_path, (b) the runner→adapter→harness wiring
    holds together, and (c) JSONL is written correctly."""
    from claritymed.evals.lm.baseline import ClaritymedBaselineLM  # noqa: F401

    class _ConstantStubAgent:
        def run_sync(self, *_a, **_kw):
            class _R:
                output = "A"

            return _R()

    class _ConstantStubLM(ClaritymedBaselineLM):
        def __init__(self, provider):
            # Skip Agent build entirely.
            super(ClaritymedBaselineLM, self).__init__()
            self.provider_id = provider.id
            self.model_name = provider.model
            self._agent = _ConstantStubAgent()
            self.latencies_ms = []
            self.latencies_ms_by_doc_id = {}

    runner = LmEvalRunner(output_dir=tmp_path, lm_factory=_ConstantStubLM)
    result = runner.run(_provider(), "medqa", limit=2)

    assert result.n_questions == 2
    rows = [
        json.loads(line)
        for line in result.output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert all(row["extracted_letter"] == "A" for row in rows)
    # Some rows may be correct (when gold is A); the framework reports the
    # aggregate honestly. We just assert the structural plumbing.
    assert all(isinstance(row["latency_ms"], float) for row in rows)
