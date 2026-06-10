"""Tests for ``evals.reporting.delta`` — baseline vs RAG comparison."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from claritymed.cli.main import app
from claritymed.evals.reporting.delta import (
    DeltaReportError,
    build_delta_report,
    find_latest_pair,
    render_delta_markdown,
    write_delta_markdown,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")
    return path


def _row(
    question_idx: int,
    gold: str,
    extracted: str,
    correct: bool,
) -> dict:
    return {
        "question_idx": question_idx,
        "gold_letter": gold,
        "extracted_letter": extracted,
        "correct": correct,
    }


def _synthetic_rows(
    n: int,
    *,
    correct_count: int,
    offset: int = 0,
) -> list[dict]:
    """Build n rows; first ``correct_count`` are correct, rest are wrong.

    ``offset`` shifts *which* rows are correct so two synthetic JSONLs can
    overlap, diverge, or strictly improve on one another by varying the
    correct-set's starting index.
    """
    rows: list[dict] = []
    for idx in range(n):
        # Rotate the correct-set window by `offset`.
        is_correct = ((idx - offset) % n) < correct_count
        rows.append(
            _row(
                question_idx=idx,
                gold="A",
                extracted="A" if is_correct else "B",
                correct=is_correct,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Happy paths — aggregation correctness
# ---------------------------------------------------------------------------


def test_strict_improvement_positive_delta(tmp_path):
    baseline = _write_jsonl(
        tmp_path / "ollama_medqa_20260101T000000Z.jsonl",
        _synthetic_rows(100, correct_count=60),
    )
    # RAG correct on the same first 60 + an extra 10 — strict improvement.
    rag = _write_jsonl(
        tmp_path / "ollama_medqa_with-rag_20260101T010000Z.jsonl",
        _synthetic_rows(100, correct_count=70),
    )

    report = build_delta_report(
        provider_id="ollama",
        task_id="medqa",
        baseline_path=baseline,
        rag_path=rag,
    )

    assert report.n_questions == 100
    assert report.baseline_accuracy == pytest.approx(0.60)
    assert report.rag_accuracy == pytest.approx(0.70)
    assert report.delta == pytest.approx(0.10)
    assert report.regression_count == 0
    assert report.gain_count == 10


def test_negative_delta_with_regressions(tmp_path):
    # Baseline correct on idx 0..59, wrong on 60..99.
    baseline = _write_jsonl(
        tmp_path / "ollama_medqa_20260101T000000Z.jsonl",
        _synthetic_rows(100, correct_count=60),
    )
    # Shift correct-set so RAG is right on idx 10..69 → first 10 baseline
    # winners regress; 10 baseline losers gain.
    rag = _write_jsonl(
        tmp_path / "ollama_medqa_with-rag_20260101T010000Z.jsonl",
        _synthetic_rows(100, correct_count=60, offset=10),
    )

    report = build_delta_report(
        provider_id="ollama",
        task_id="medqa",
        baseline_path=baseline,
        rag_path=rag,
    )

    assert report.baseline_accuracy == pytest.approx(0.60)
    assert report.rag_accuracy == pytest.approx(0.60)
    assert report.delta == pytest.approx(0.0)
    assert report.regression_count == 10
    assert report.gain_count == 10


def test_strict_regression_only(tmp_path):
    baseline = _write_jsonl(
        tmp_path / "ollama_medqa_20260101T000000Z.jsonl",
        _synthetic_rows(20, correct_count=20),
    )
    rag = _write_jsonl(
        tmp_path / "ollama_medqa_with-rag_20260101T010000Z.jsonl",
        _synthetic_rows(20, correct_count=10),
    )

    report = build_delta_report(
        provider_id="ollama",
        task_id="medqa",
        baseline_path=baseline,
        rag_path=rag,
    )

    assert report.regression_count == 10
    assert report.gain_count == 0
    assert report.delta == pytest.approx(-0.5)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_mismatched_row_counts_fails_loud(tmp_path):
    baseline = _write_jsonl(
        tmp_path / "ollama_medqa_20260101T000000Z.jsonl",
        _synthetic_rows(50, correct_count=30),
    )
    rag = _write_jsonl(
        tmp_path / "ollama_medqa_with-rag_20260101T010000Z.jsonl",
        _synthetic_rows(48, correct_count=30),
    )

    with pytest.raises(DeltaReportError, match="row count mismatch"):
        build_delta_report(
            provider_id="ollama",
            task_id="medqa",
            baseline_path=baseline,
            rag_path=rag,
        )


def test_empty_files_fail_loud(tmp_path):
    baseline = _write_jsonl(tmp_path / "ollama_medqa_a.jsonl", [])
    rag = _write_jsonl(tmp_path / "ollama_medqa_with-rag_b.jsonl", [])

    with pytest.raises(DeltaReportError, match="empty"):
        build_delta_report(
            provider_id="ollama",
            task_id="medqa",
            baseline_path=baseline,
            rag_path=rag,
        )


def test_missing_question_idx_in_rag(tmp_path):
    baseline = _write_jsonl(
        tmp_path / "ollama_medqa_a.jsonl",
        [_row(0, "A", "A", True), _row(1, "A", "A", True)],
    )
    # Same row count but different indices → can't join.
    rag = _write_jsonl(
        tmp_path / "ollama_medqa_with-rag_b.jsonl",
        [_row(0, "A", "A", True), _row(99, "A", "A", True)],
    )

    with pytest.raises(DeltaReportError, match="question_idx 1"):
        build_delta_report(
            provider_id="ollama",
            task_id="medqa",
            baseline_path=baseline,
            rag_path=rag,
        )


def test_find_latest_pair_picks_most_recent(tmp_path):
    old_baseline = _write_jsonl(
        tmp_path / "ollama_medqa_20260101T000000Z.jsonl",
        [_row(0, "A", "A", True)],
    )
    new_baseline = _write_jsonl(
        tmp_path / "ollama_medqa_20260101T020000Z.jsonl",
        [_row(0, "A", "A", True)],
    )
    rag = _write_jsonl(
        tmp_path / "ollama_medqa_with-rag_20260101T030000Z.jsonl",
        [_row(0, "A", "A", True)],
    )
    # Force distinct mtimes (Path.write_text + close happens fast).
    import os
    import time

    t0 = time.time()
    os.utime(old_baseline, (t0 - 100, t0 - 100))
    os.utime(new_baseline, (t0 - 50, t0 - 50))
    os.utime(rag, (t0, t0))

    baseline, rag_found = find_latest_pair(
        provider_id="ollama", task_id="medqa", results_dir=tmp_path
    )
    assert baseline == new_baseline
    assert rag_found == rag


def test_find_latest_pair_no_baseline_clear_message(tmp_path):
    _write_jsonl(
        tmp_path / "ollama_medqa_with-rag_a.jsonl",
        [_row(0, "A", "A", True)],
    )
    with pytest.raises(DeltaReportError, match="no baseline JSONL"):
        find_latest_pair(provider_id="ollama", task_id="medqa", results_dir=tmp_path)


def test_find_latest_pair_no_rag_clear_message(tmp_path):
    _write_jsonl(
        tmp_path / "ollama_medqa_a.jsonl",
        [_row(0, "A", "A", True)],
    )
    with pytest.raises(DeltaReportError, match="no with-rag JSONL"):
        find_latest_pair(provider_id="ollama", task_id="medqa", results_dir=tmp_path)


def test_find_latest_pair_missing_directory(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(DeltaReportError, match="does not exist"):
        find_latest_pair(provider_id="ollama", task_id="medqa", results_dir=missing)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def test_render_includes_header_and_accuracy_table(tmp_path):
    baseline = _write_jsonl(
        tmp_path / "ollama_medqa_a.jsonl",
        _synthetic_rows(10, correct_count=6),
    )
    rag = _write_jsonl(
        tmp_path / "ollama_medqa_with-rag_b.jsonl",
        _synthetic_rows(10, correct_count=7),
    )
    report = build_delta_report(
        provider_id="ollama",
        task_id="medqa",
        baseline_path=baseline,
        rag_path=rag,
    )
    md = render_delta_markdown(report)

    assert "Eval delta — medqa / ollama" in md
    assert "0.6000" in md  # baseline
    assert "0.7000" in md  # rag
    assert "+0.1000" in md  # delta
    assert "pp" in md


def test_regression_list_capped_with_overflow_sidecar(tmp_path):
    """30 regressions → inline table shows 20, sidecar JSONL holds all 30."""
    # Baseline gets every question right; RAG misses the first 30.
    baseline = _write_jsonl(
        tmp_path / "ollama_medqa_a.jsonl",
        _synthetic_rows(50, correct_count=50),
    )
    rag_rows = []
    for i in range(50):
        rag_rows.append(
            _row(
                question_idx=i,
                gold="A",
                extracted="A" if i >= 30 else "B",
                correct=i >= 30,
            )
        )
    rag = _write_jsonl(tmp_path / "ollama_medqa_with-rag_b.jsonl", rag_rows)

    report = build_delta_report(
        provider_id="ollama",
        task_id="medqa",
        baseline_path=baseline,
        rag_path=rag,
    )
    assert report.regression_count == 30

    md_path, sidecar = write_delta_markdown(report, output_dir=tmp_path / "out")
    assert md_path.exists()
    assert sidecar is not None and sidecar.exists()

    md_text = md_path.read_text(encoding="utf-8")
    assert "Regressions" in md_text
    assert "10 more regressions" in md_text  # 30 - 20 inline

    sidecar_lines = sidecar.read_text(encoding="utf-8").strip().splitlines()
    assert len(sidecar_lines) == 30


def test_no_regressions_means_no_sidecar(tmp_path):
    baseline = _write_jsonl(
        tmp_path / "ollama_medqa_a.jsonl",
        _synthetic_rows(20, correct_count=10),
    )
    rag = _write_jsonl(
        tmp_path / "ollama_medqa_with-rag_b.jsonl",
        _synthetic_rows(20, correct_count=15),
    )
    report = build_delta_report(
        provider_id="ollama",
        task_id="medqa",
        baseline_path=baseline,
        rag_path=rag,
    )
    md_path, sidecar = write_delta_markdown(report, output_dir=tmp_path / "out")
    assert md_path.exists()
    assert sidecar is None


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_eval_delta_cli_happy_path(tmp_path):
    baseline = _write_jsonl(
        tmp_path / "ollama_medqa_20260101T000000Z.jsonl",
        _synthetic_rows(10, correct_count=6),
    )
    _write_jsonl(
        tmp_path / "ollama_medqa_with-rag_20260101T010000Z.jsonl",
        _synthetic_rows(10, correct_count=7),
    )

    result = runner.invoke(
        app,
        [
            "eval",
            "delta",
            "--task",
            "medqa",
            "--provider",
            "ollama",
            "--results-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Eval delta — medqa / ollama" in result.stdout
    assert "+0.1000" in result.stdout
    # Markdown report persisted.
    assert any(p.suffix == ".md" and "delta" in p.name for p in tmp_path.iterdir())
    # Suppress unused warning — fixture path is used for path discovery only.
    _ = baseline


def test_eval_delta_cli_missing_pair_exits_two(tmp_path):
    # Empty results dir → no pair to find.
    result = runner.invoke(
        app,
        [
            "eval",
            "delta",
            "--task",
            "medqa",
            "--provider",
            "ollama",
            "--results-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    assert "no baseline JSONL" in result.stdout or "does not exist" in result.stdout


def test_eval_delta_cli_explicit_paths(tmp_path):
    baseline = _write_jsonl(
        tmp_path / "anywhere_baseline.jsonl",
        _synthetic_rows(5, correct_count=3),
    )
    rag = _write_jsonl(
        tmp_path / "anywhere_rag.jsonl",
        _synthetic_rows(5, correct_count=4),
    )

    result = runner.invoke(
        app,
        [
            "eval",
            "delta",
            "--task",
            "medqa",
            "--provider",
            "ollama",
            "--baseline",
            str(baseline),
            "--rag",
            str(rag),
            "--results-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Eval delta — medqa / ollama" in result.stdout
