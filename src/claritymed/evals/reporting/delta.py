"""Baseline-vs-RAG delta report — joins two JSONL files into one table.

Phase 2 deliverable: ``claritymed eval delta --task <id> --provider <id>``
takes the latest baseline and ``with-rag`` JSONL outputs from
``data/evals/results/`` for a ``(provider, task)`` pair and renders:

* aggregate accuracy on each arm,
* delta in percentage points,
* regression count (baseline correct → RAG wrong) and gain count
  (baseline wrong → RAG correct),
* a per-question regression list capped at 20 rows; overflow goes to a
  sibling JSONL so the table stays scannable.

The join key is ``question_idx`` — populated by the runner from
lm-eval-harness's ``doc_id``. Both arms iterate the same dataset split
in the same order, so the indices align by construction. Mismatched
question counts fail loud rather than partial-joining (silent partial
joins are how delta-reporting bugs hide).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Cap on the per-question regression list rendered inline. Longer lists
# spill into a JSONL sibling file so the markdown report stays scannable.
_REGRESSIONS_INLINE_LIMIT = 20

# Default output root, matches LmEvalRunner. Importing the runner
# constant would create an awkward dep direction; the path is short
# enough to repeat.
_DEFAULT_OUTPUT_DIR = Path("data") / "evals" / "results"


class DeltaReportError(RuntimeError):
    """Raised when a delta report cannot be produced (missing file, mismatched data)."""


@dataclass(frozen=True)
class _Row:
    """One per-question record loaded from a results JSONL."""

    question_idx: int
    gold_letter: str
    extracted_letter: str
    correct: bool


@dataclass(frozen=True)
class _Regression:
    """One question where baseline was correct but RAG was wrong."""

    question_idx: int
    gold_letter: str
    baseline_letter: str
    rag_letter: str


@dataclass(frozen=True)
class DeltaReport:
    """Aggregate + per-question view of a baseline-vs-RAG comparison."""

    provider_id: str
    task_id: str
    baseline_path: Path
    rag_path: Path
    n_questions: int
    baseline_accuracy: float
    rag_accuracy: float
    delta: float
    regression_count: int
    gain_count: int
    regressions: list[_Regression] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loaders / file discovery
# ---------------------------------------------------------------------------


def _load_jsonl(path: Path) -> list[_Row]:
    """Load a results JSONL into the minimal row shape the delta needs."""
    if not path.exists():
        raise DeltaReportError(f"JSONL not found: {path}")
    rows: list[_Row] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise DeltaReportError(
                    f"{path.name}:{lineno} — malformed JSONL: {exc.msg}"
                ) from exc
            rows.append(
                _Row(
                    question_idx=int(obj.get("question_idx", lineno - 1)),
                    gold_letter=str(obj.get("gold_letter", "")).strip(),
                    extracted_letter=str(obj.get("extracted_letter", "")).strip(),
                    correct=bool(obj.get("correct", False)),
                )
            )
    return rows


def find_latest_pair(
    *,
    provider_id: str,
    task_id: str,
    results_dir: Path | str = _DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Pick the most recent baseline + RAG JSONL for ``(provider, task)``.

    Selection is by mtime so re-running an arm always wins over older
    runs of the same shape. Raises ``DeltaReportError`` with an
    actionable message when either arm is missing.
    """
    root = Path(results_dir)
    if not root.exists():
        raise DeltaReportError(
            f"results directory {root} does not exist — "
            f"run `claritymed eval run {task_id} --provider {provider_id}` first."
        )

    baseline_pattern = f"{provider_id}_{task_id}_*.jsonl"
    candidates = sorted(root.glob(baseline_pattern), key=lambda p: p.stat().st_mtime)
    baselines = [p for p in candidates if "_with-rag_" not in p.name]
    rags = [p for p in candidates if "_with-rag_" in p.name]

    if not baselines:
        raise DeltaReportError(
            f"no baseline JSONL for ({provider_id}, {task_id}) under {root}. "
            f"Run `claritymed eval {task_id} --provider {provider_id}` first."
        )
    if not rags:
        raise DeltaReportError(
            f"no with-rag JSONL for ({provider_id}, {task_id}) under {root}. "
            f"Run `claritymed eval {task_id} --provider {provider_id} --with-rag` first."
        )
    return baselines[-1], rags[-1]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def build_delta_report(
    *,
    provider_id: str,
    task_id: str,
    baseline_path: Path,
    rag_path: Path,
) -> DeltaReport:
    """Load both JSONLs and compute the aggregate + regression view."""
    baseline_rows = _load_jsonl(baseline_path)
    rag_rows = _load_jsonl(rag_path)

    if len(baseline_rows) != len(rag_rows):
        raise DeltaReportError(
            f"row count mismatch: baseline={len(baseline_rows)}, "
            f"rag={len(rag_rows)} — were the runs against the same dataset "
            f"version and the same --limit?"
        )
    if not baseline_rows:
        raise DeltaReportError(
            "both JSONL files are empty — nothing to compare. Re-run without --limit 0."
        )

    rag_by_idx = {r.question_idx: r for r in rag_rows}
    baseline_correct = 0
    rag_correct = 0
    regressions: list[_Regression] = []
    gains = 0
    for b in baseline_rows:
        r = rag_by_idx.get(b.question_idx)
        if r is None:
            raise DeltaReportError(
                f"question_idx {b.question_idx} present in baseline but "
                f"missing from rag JSONL — runs likely targeted different "
                f"dataset slices."
            )
        if b.correct:
            baseline_correct += 1
        if r.correct:
            rag_correct += 1
        if b.correct and not r.correct:
            regressions.append(
                _Regression(
                    question_idx=b.question_idx,
                    gold_letter=b.gold_letter,
                    baseline_letter=b.extracted_letter,
                    rag_letter=r.extracted_letter,
                )
            )
        elif (not b.correct) and r.correct:
            gains += 1

    n = len(baseline_rows)
    baseline_acc = baseline_correct / n
    rag_acc = rag_correct / n

    return DeltaReport(
        provider_id=provider_id,
        task_id=task_id,
        baseline_path=baseline_path,
        rag_path=rag_path,
        n_questions=n,
        baseline_accuracy=baseline_acc,
        rag_accuracy=rag_acc,
        delta=rag_acc - baseline_acc,
        regression_count=len(regressions),
        gain_count=gains,
        regressions=regressions,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_delta_markdown(report: DeltaReport) -> str:
    """Render the report as markdown — what the CLI prints + persists."""
    delta_pp = report.delta * 100
    lines: list[str] = []
    lines.append(f"# Eval delta — {report.task_id} / {report.provider_id}")
    lines.append("")
    lines.append(f"- baseline JSONL: `{report.baseline_path}`")
    lines.append(f"- with-rag JSONL: `{report.rag_path}`")
    lines.append(f"- n_questions: **{report.n_questions}**")
    lines.append("")
    lines.append("| Arm | Accuracy |")
    lines.append("|---|---:|")
    lines.append(f"| baseline | {report.baseline_accuracy:.4f} |")
    lines.append(f"| with-rag | {report.rag_accuracy:.4f} |")
    lines.append(f"| **Δ** | **{report.delta:+.4f}** ({delta_pp:+.2f} pp) |")
    lines.append("")
    lines.append(f"- gains (baseline wrong → RAG right): **{report.gain_count}**")
    lines.append(
        f"- regressions (baseline right → RAG wrong): **{report.regression_count}**"
    )
    lines.append("")

    if report.regressions:
        capped = report.regressions[:_REGRESSIONS_INLINE_LIMIT]
        lines.append("## Regressions")
        lines.append("")
        lines.append("| question_idx | gold | baseline | with-rag |")
        lines.append("|---:|:---:|:---:|:---:|")
        for r in capped:
            lines.append(
                f"| {r.question_idx} | {r.gold_letter or '—'} | "
                f"{r.baseline_letter or '—'} | {r.rag_letter or '—'} |"
            )
        if len(report.regressions) > _REGRESSIONS_INLINE_LIMIT:
            extra = len(report.regressions) - _REGRESSIONS_INLINE_LIMIT
            lines.append("")
            lines.append(f"_…{extra} more regressions in the sidecar JSONL._")
    return "\n".join(lines).rstrip() + "\n"


def write_delta_markdown(
    report: DeltaReport,
    *,
    output_dir: Path | str = _DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path | None]:
    """Persist the markdown report; spill overflow regressions to JSONL.

    Returns ``(markdown_path, regressions_sidecar_path_or_None)``.
    The sidecar is only written when there are more regressions than the
    inline cap, so a tidy report doesn't leave orphan files behind.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{report.provider_id}_{report.task_id}_delta_{ts}"

    md_path = root / f"{stem}.md"
    md_path.write_text(render_delta_markdown(report), encoding="utf-8")

    sidecar: Path | None = None
    if len(report.regressions) > _REGRESSIONS_INLINE_LIMIT:
        sidecar = root / f"{stem}_regressions.jsonl"
        with sidecar.open("w", encoding="utf-8") as fh:
            for r in report.regressions:
                fh.write(
                    json.dumps(
                        {
                            "question_idx": r.question_idx,
                            "gold_letter": r.gold_letter,
                            "baseline_letter": r.baseline_letter,
                            "rag_letter": r.rag_letter,
                        },
                        ensure_ascii=False,
                    )
                )
                fh.write("\n")
    return md_path, sidecar


def report_audit_payload(report: DeltaReport) -> dict[str, Any]:
    """Compact audit payload for ``eval.delta.completed``."""
    return {
        "provider_id": report.provider_id,
        "task_id": report.task_id,
        "n_questions": report.n_questions,
        "baseline_accuracy": report.baseline_accuracy,
        "rag_accuracy": report.rag_accuracy,
        "delta": report.delta,
        "regression_count": report.regression_count,
        "gain_count": report.gain_count,
        "baseline_path": str(report.baseline_path),
        "rag_path": str(report.rag_path),
    }
