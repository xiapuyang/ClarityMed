"""``claritymed eval ...`` sub-app — runs benchmarks against any provider.

Mirrors the shape of ``ask`` / ``ingest`` / ``rag``:

1. Wrap in ``inject_context(user_id="eval", ...)`` so audit rows and
   Phoenix span baggage carry the synthetic ``eval`` user — the runner
   is *not* acting as a real account, and tagging it consistently makes
   eval traffic trivially filterable in dashboards.
2. Resolve the provider through the same ``resolve_provider`` path the
   other commands use, so ``--provider`` flags behave identically.
3. Hand off to ``LmEvalRunner``; the runner prints its own summary table.

``--with-rag`` swaps the baseline adapter for ``ClaritymedRagLM`` and
tags the output filename with ``_with-rag`` so ``eval delta`` can join
the two runs by provider+task pair.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from claritymed.cli.entry import inject_context
from claritymed.core.observability.audit import audit_event
from claritymed.errors import UnknownProviderError
from claritymed.evals.lm.rag import ClaritymedRagLM
from claritymed.evals.reporting.delta import (
    DeltaReportError,
    build_delta_report,
    find_latest_pair,
    render_delta_markdown,
    report_audit_payload,
    write_delta_markdown,
)
from claritymed.evals.runners.lm_eval_runner import LmEvalRunner
from claritymed.stores.models import pick_reachable_provider, resolve_provider

eval_app = typer.Typer(
    name="eval",
    help="Run benchmarks (medqa, cmb_exam, medmcqa, pubmedqa) against any "
    "provider in models.yaml.",
    no_args_is_help=True,
)
_console = Console()


@eval_app.command("run")
def eval_run(
    task_id: str = typer.Argument(
        ...,
        help="Task id matching a YAML under src/claritymed/evals/tasks/ "
        "(e.g. medqa, cmb_exam, medmcqa, pubmedqa).",
    ),
    provider_id: str | None = typer.Option(
        None,
        "--provider",
        "-p",
        help="Catalog id (configs/models.yaml). Falls back to default.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=0,
        help="Cap questions for smoke tests. Default: full test split.",
    ),
    with_rag: bool = typer.Option(
        False,
        "--with-rag",
        help="Route through AskService (PHI guard + retrieval + LLM) instead of "
        "the bare model. Output filename gets a _with-rag suffix so "
        "`eval delta` can pair the runs.",
    ),
    rag_mode: str = typer.Option(
        "deterministic",
        "--rag-mode",
        case_sensitive=False,
        help=(
            "RAG mode used by --with-rag. 'deterministic' forces retrieval "
            "on every question (correct default for measuring retrieval "
            "value on MCQA). 'tool' hands the retrieval decision to the "
            "LLM (production agent behaviour; on short MCQA prompts the "
            "tool fires <30% of the time, collapsing the delta signal). "
            "Ignored without --with-rag."
        ),
    ),
    question_timeout: float | None = typer.Option(
        None,
        "--question-timeout",
        min=1.0,
        help=(
            "Per-question wall-clock cap in seconds for the --with-rag arm. "
            "Default (unset) uses ClaritymedRagLM's built-in 90.0s. Raise "
            "for hard multi-hop reasoning (medqa bio+micro combos can chew "
            "3000+ tokens > 90s on a local 35B model → empty completion "
            "counts as wrong); lower to fail fast on stuck sessions. "
            "Ignored without --with-rag."
        ),
    ),
) -> None:
    """Score the configured provider on TASK_ID.

    Adding a new benchmark is YAML-only: drop ``<task>.yaml`` (and an
    optional sibling ``utils_<task>.py`` for dataset preprocessing) into
    ``src/claritymed/evals/tasks/`` and invoke ``claritymed eval run <task>``.
    """
    arm = "with-rag" if with_rag else "baseline"
    if with_rag and rag_mode not in {"deterministic", "tool"}:
        raise typer.Exit(
            _emit_error(
                f"--rag-mode must be 'deterministic' or 'tool', got {rag_mode!r}."
            )
        )
    if question_timeout is not None and not with_rag:
        # Baseline arm bypasses ClaritymedRagLM entirely, so the flag has
        # no effect there. Warn instead of silently swallowing.
        _console.print(
            "[yellow]warning:[/yellow] --question-timeout requires --with-rag; "
            "the baseline arm ignores it."
        )
    command_label = f"eval.{task_id} provider={provider_id or 'default'} arm={arm}" + (
        f" rag_mode={rag_mode}" if with_rag else ""
    )
    with inject_context(
        user_id="eval",
        language="en",
        command=command_label,
    ):
        provider = _resolve_for_eval(provider_id)
        if with_rag:
            # Only forward question_timeout when the caller explicitly set it —
            # ClaritymedRagLM's own default is the single source of truth for
            # the 90 s baseline (see _DEFAULT_QUESTION_TIMEOUT_S in
            # evals/lm/rag.py). Passing None here would clobber that.
            rag_kwargs: dict = {"rag_mode": rag_mode}
            if question_timeout is not None:
                rag_kwargs["question_timeout_s"] = question_timeout
            runner = LmEvalRunner(
                lm_factory=lambda p: ClaritymedRagLM(p, **rag_kwargs),
                run_tag="with-rag",
            )
        else:
            runner = LmEvalRunner()
        runner.run(provider, task_id=task_id, limit=limit)


@eval_app.command("delta")
def eval_delta(
    task_id: str = typer.Option(
        ...,
        "--task",
        "-t",
        help="Task id (e.g. medqa) — used to pair the baseline and with-rag JSONLs.",
    ),
    provider_id: str = typer.Option(
        ...,
        "--provider",
        "-p",
        help="Catalog id whose runs are being compared.",
    ),
    baseline_path: Path | None = typer.Option(
        None,
        "--baseline",
        exists=True,
        readable=True,
        help="Explicit baseline JSONL. Default: latest run for (provider, task).",
    ),
    rag_path: Path | None = typer.Option(
        None,
        "--rag",
        exists=True,
        readable=True,
        help="Explicit with-rag JSONL. Default: latest run for (provider, task).",
    ),
    results_dir: Path = typer.Option(
        Path("data") / "evals" / "results",
        "--results-dir",
        help="Where to look for default JSONLs and where to drop the report.",
    ),
) -> None:
    """Compare baseline vs RAG runs for one (provider, task) pair.

    Picks the latest baseline + with-rag JSONL by mtime when explicit
    paths are omitted; renders a markdown table to stdout and persists
    the full report under ``data/evals/results/``.
    """
    command_label = f"eval.delta provider={provider_id} task={task_id}"
    with inject_context(
        user_id="eval",
        language="en",
        command=command_label,
    ):
        try:
            if baseline_path is None or rag_path is None:
                latest_baseline, latest_rag = find_latest_pair(
                    provider_id=provider_id,
                    task_id=task_id,
                    results_dir=results_dir,
                )
                baseline_path = baseline_path or latest_baseline
                rag_path = rag_path or latest_rag
            report = build_delta_report(
                provider_id=provider_id,
                task_id=task_id,
                baseline_path=baseline_path,
                rag_path=rag_path,
            )
        except DeltaReportError as exc:
            raise typer.Exit(_emit_error(str(exc))) from exc

        md_path, sidecar = write_delta_markdown(report, output_dir=results_dir)
        _console.print(render_delta_markdown(report))
        _console.print(f"[dim]report: {md_path}[/dim]")
        if sidecar is not None:
            _console.print(f"[dim]regressions overflow: {sidecar}[/dim]")
        audit_event(
            "eval.delta.completed",
            payload={
                **report_audit_payload(report),
                "report_path": str(md_path),
                "regressions_sidecar": str(sidecar) if sidecar else None,
            },
        )


def _resolve_for_eval(provider_id: str | None):
    """Pick the provider for an eval run.

    When the caller passes ``--provider``, honor it exactly (same path as
    every other CLI command — typos surface as ``UnknownProviderError``).

    When the caller omits ``--provider``, mirror the e2e fixture's
    auto-pick: probe the known local servers (omlx → ollama) and use the
    first one that's both reachable and credentialed. This stops the
    common surprise of a fresh shell defaulting to ``ollama`` when the
    operator's only running backend is ``omlx``. Falls back to the
    catalog default if no probed local provider is up — the run will
    fail loud at the LLM call site, same as before.
    """
    if provider_id is not None:
        try:
            return resolve_provider(override=provider_id)
        except UnknownProviderError as exc:
            raise typer.Exit(_emit_error(str(exc))) from exc

    reachable = pick_reachable_provider()
    if reachable is not None:
        _console.print(
            f"[dim]auto-selected provider [bold]{reachable.id}[/bold] "
            f"(model: {reachable.model}); pass --provider to override.[/dim]"
        )
        return reachable
    return resolve_provider(override=None)


def _emit_error(message: str) -> int:
    """Print a red error line to stderr and return exit code 2."""
    _console.print(f"[red]error[/red] {message}")
    return 2
