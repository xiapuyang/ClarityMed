"""``claritymed eval ...`` sub-app — runs benchmarks against any provider.

Mirrors the shape of ``ask`` / ``ingest`` / ``rag``:

1. Wrap in ``inject_context(user_id="eval", ...)`` so audit rows and
   Phoenix span baggage carry the synthetic ``eval`` user — the runner
   is *not* acting as a real account, and tagging it consistently makes
   eval traffic trivially filterable in dashboards.
2. Resolve the provider through the same ``resolve_provider`` path the
   other commands use, so ``--provider`` flags behave identically.
3. Hand off to ``LmEvalRunner``; the runner prints its own summary table.

Phase 2 (Unit 6) wires ``--with-rag`` to ``ClaritymedRagLM``. Until then
the flag parses and fails loud so callers know the feature is on the way.
"""

from __future__ import annotations

import typer
from rich.console import Console

from claritymed.cli.entry import inject_context
from claritymed.errors import UnknownProviderError
from claritymed.evals.runners.lm_eval_runner import LmEvalRunner
from claritymed.stores.models import resolve_provider

eval_app = typer.Typer(
    name="eval",
    help="Run benchmarks (MedQA, ...) against any provider in models.yaml.",
    no_args_is_help=True,
)
_console = Console()


@eval_app.command("medqa")
def eval_medqa(
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
        help="Cap questions for smoke tests. Default: full 1273-question test split.",
    ),
    with_rag: bool = typer.Option(
        False,
        "--with-rag",
        help="Phase 2: route through AskService. Currently raises NotImplementedError.",
    ),
) -> None:
    """Score the configured provider on MedQA-USMLE (4-option English MCQA)."""
    command_label = f"eval.medqa provider={provider_id or 'default'}"
    with inject_context(
        user_id="eval",
        language="en",
        command=command_label,
    ):
        if with_rag:
            raise typer.Exit(
                _emit_error(
                    "--with-rag arrives in Phase 2 (Unit 6 of the evals plan). "
                    "Run without --with-rag for the baseline arm."
                )
            )

        try:
            provider = resolve_provider(override=provider_id)
        except UnknownProviderError as exc:
            raise typer.Exit(_emit_error(str(exc))) from exc

        LmEvalRunner().run(provider, task_id="medqa", limit=limit)


def _emit_error(message: str) -> int:
    """Print a red error line to stderr and return exit code 2."""
    _console.print(f"[red]error[/red] {message}")
    return 2
