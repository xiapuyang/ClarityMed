"""Drive lm-evaluation-harness against the project's provider catalog.

One ``LmEvalRunner.run(...)`` call:

1. Emits ``eval.run.started`` audit.
2. Builds a ``ClaritymedBaselineLM`` (or whichever ``lm_factory`` the
   caller injects — Phase 2's ``ClaritymedRagLM`` slots in here).
3. Calls ``lm_eval.simple_evaluate`` with our project-local
   ``task_manager`` so the YAMLs under ``evals/tasks/`` are discovered.
4. Writes one JSONL row per question (``data/evals/results/<...>.jsonl``)
   joining harness output with per-call latency.
5. Emits ``eval.run.completed`` or, on exception, ``eval.run.failed``.
6. Returns a ``RunResult`` and prints a markdown summary to stdout.

The harness's ``simple_evaluate`` is synchronous and the project's
``audit_event`` reads ``ContextVars`` set by the CLI's ``inject_context``,
so the runner expects to be invoked from within an active context.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from claritymed.context import request_id_ctx
from claritymed.core.observability.audit import audit_event
from claritymed.evals.lm.baseline import ClaritymedBaselineLM
from claritymed.evals.protocol import RunResult

if TYPE_CHECKING:
    from lm_eval.api.model import LM

    from claritymed.core.schemas import ProviderConfig

# Project-local task YAMLs (medqa.yaml, ...). lm-eval-harness's
# ``TaskManager(include_path=...)`` scans this directory for additional
# task definitions on top of the bundled ones.
_TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks"

# Default output root for JSONL results, relative to the project root.
# Overridable per-run via ``LmEvalRunner(output_dir=...)``.
_DEFAULT_OUTPUT_DIR = Path("data") / "evals" / "results"


class LmEvalRunner:
    """``Benchmark`` implementation backed by lm-evaluation-harness."""

    def __init__(
        self,
        output_dir: Path | str | None = None,
        lm_factory: Callable[["ProviderConfig"], "LM"] | None = None,
    ) -> None:
        self.output_dir = Path(output_dir) if output_dir else _DEFAULT_OUTPUT_DIR
        self._lm_factory: Callable[["ProviderConfig"], "LM"] = (
            lm_factory or ClaritymedBaselineLM
        )

    def run(
        self,
        provider: "ProviderConfig",
        task_id: str,
        limit: int | None = None,
    ) -> RunResult:
        """Score ``provider`` on ``task_id``; persist + audit; return summary."""
        started_at = datetime.now(timezone.utc)
        audit_event(
            "eval.run.started",
            payload={
                "provider_id": provider.id,
                "model_name": provider.model,
                "task_id": task_id,
                "limit": limit,
            },
        )
        lm = self._lm_factory(provider)
        t0 = time.perf_counter()
        try:
            results = self._simple_evaluate(lm, task_id, limit)
        except Exception as exc:  # noqa: BLE001
            audit_event(
                "eval.run.failed",
                payload={
                    "provider_id": provider.id,
                    "task_id": task_id,
                    "error_type": exc.__class__.__name__,
                    "message": str(exc)[:500],
                    "traceback": traceback.format_exc()[-1500:],
                },
            )
            raise

        duration_s = time.perf_counter() - t0
        samples = results.get("samples", {}).get(task_id, []) or []
        accuracy = _extract_accuracy(results, task_id)

        output_path = self._write_jsonl(
            task_id=task_id,
            provider=provider,
            samples=samples,
            lm=lm,
            started_at=started_at,
        )

        result = RunResult(
            provider_id=provider.id,
            task_id=task_id,
            n_questions=len(samples),
            accuracy=accuracy,
            output_path=output_path,
            started_at=started_at,
            duration_s=duration_s,
        )
        audit_event(
            "eval.run.completed",
            payload={
                **{k: v for k, v in asdict(result).items() if k != "output_path"},
                "output_path": str(output_path),
                "started_at": result.started_at.isoformat(),
            },
        )
        _print_summary(result)
        return result

    # ------------------------------------------------------------------
    # Internals — split so tests can patch ``simple_evaluate`` cleanly.
    # ------------------------------------------------------------------

    def _simple_evaluate(
        self,
        lm: "LM",
        task_id: str,
        limit: int | None,
    ) -> dict[str, Any]:
        """Wrap ``lm_eval.simple_evaluate`` with our project-local TaskManager."""
        # lm-eval transitively imports transformers, which emits a "None of
        # PyTorch, TensorFlow >= 2.0, or Flax have been found" advisory on
        # import when no ML backend is installed. We don't need one — the
        # eval path talks to the configured provider via pydantic-ai, never
        # loads weights in-process. The advisory respects this env var.
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

        from lm_eval import simple_evaluate
        from lm_eval.tasks import TaskManager

        task_manager = TaskManager(include_path=str(_TASKS_DIR))
        return simple_evaluate(
            model=lm,
            tasks=[task_id],
            limit=limit,
            task_manager=task_manager,
            log_samples=True,
            write_out=False,
            apply_chat_template=False,
        )

    def _write_jsonl(
        self,
        *,
        task_id: str,
        provider: "ProviderConfig",
        samples: list[dict[str, Any]],
        lm: "LM",
        started_at: datetime,
    ) -> Path:
        """Write one JSONL row per sample. Returns the file path."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        ts = started_at.strftime("%Y%m%dT%H%M%SZ")
        path = self.output_dir / f"{provider.id}_{task_id}_{ts}.jsonl"
        rid = request_id_ctx.get() or "unknown"
        latency_by_doc = getattr(lm, "latencies_ms_by_doc_id", {}) or {}
        latency_list = getattr(lm, "latencies_ms", []) or []

        with path.open("w", encoding="utf-8") as fh:
            for ordinal, sample in enumerate(samples):
                row = _build_row(
                    sample=sample,
                    ordinal=ordinal,
                    request_id=rid,
                    provider_id=provider.id,
                    model_name=provider.model,
                    task_id=task_id,
                    latency_by_doc=latency_by_doc,
                    latency_list=latency_list,
                )
                fh.write(json.dumps(row, ensure_ascii=False))
                fh.write("\n")
        return path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_row(
    *,
    sample: dict[str, Any],
    ordinal: int,
    request_id: str,
    provider_id: str,
    model_name: str,
    task_id: str,
    latency_by_doc: dict[int, float],
    latency_list: list[float],
) -> dict[str, Any]:
    """Assemble one JSONL row from a harness sample dict."""
    doc_id = sample.get("doc_id")
    arguments = sample.get("arguments") or []
    question = arguments[0][0] if arguments else ""
    resps = sample.get("resps") or [[""]]
    raw_completion = resps[0][0] if resps and resps[0] else ""
    filtered = sample.get("filtered_resps") or [""]
    extracted = filtered[0] if filtered else ""
    exact_match = sample.get("exact_match", 0.0)
    correct = bool(exact_match >= 1.0)

    if doc_id is not None and doc_id in latency_by_doc:
        latency_ms: float | None = latency_by_doc[doc_id]
    elif ordinal < len(latency_list):
        latency_ms = latency_list[ordinal]
    else:
        latency_ms = None

    return {
        "request_id": request_id,
        "task_id": task_id,
        "provider_id": provider_id,
        "model_name": model_name,
        "question_idx": doc_id if doc_id is not None else ordinal,
        "question": question,
        "gold_letter": str(sample.get("target", "")).strip(),
        "model_completion": raw_completion,
        "extracted_letter": extracted,
        "correct": correct,
        "latency_ms": latency_ms,
    }


def _extract_accuracy(results: dict[str, Any], task_id: str) -> float:
    """Pull the exact_match scalar from ``results["results"][task]``.

    lm-eval keys metrics as ``"<metric>,<filter_name>"`` so we scan for
    any ``exact_match,...`` key. Returns NaN-like ``float("nan")`` when
    the metric is absent (e.g. ``limit=0`` runs).
    """
    per_task = results.get("results", {}).get(task_id, {}) or {}
    for key, value in per_task.items():
        if key.startswith("exact_match,") and not key.endswith("_stderr"):
            return float(value)
    return float("nan")


def _print_summary(result: RunResult) -> None:
    """Print a copy-pasteable markdown table to stdout."""
    acc = f"{result.accuracy:.4f}" if result.accuracy == result.accuracy else "n/a"
    print()
    print("| Task | Provider | N | Accuracy | Duration | Output |")
    print("|---|---|---:|---:|---:|---|")
    print(
        f"| {result.task_id} | {result.provider_id} | {result.n_questions} | "
        f"{acc} | {result.duration_s:.1f}s | {result.output_path} |"
    )
