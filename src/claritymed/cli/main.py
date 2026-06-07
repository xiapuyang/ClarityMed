"""Typer CLI entry point: ``claritymed ask|ingest|rag``.

The CLI is the headless surface that mirrors the TUI's actions one-shot.
Both go through the same service layer — the CLI never touches an Agent
directly so any service-level change (audit, scrub, event schema) reaches
both surfaces with one edit.
"""

from __future__ import annotations

import asyncio
import sys

import typer
from rich.console import Console

from claritymed import config as _cfg
from claritymed.cli.entry import inject_context
from claritymed.core.llm.model import build_model
from claritymed.orchestrator.services import (
    AskService,
    Done,
    Error,
    IngestService,
    RagService,
    TokenChunk,
    ToolCompleted,
    ToolStarted,
)
from claritymed.stores.models import resolve_provider
from claritymed.stores.user_rag import UserRagStore

app = typer.Typer(
    name="claritymed",
    help="ClarityMed CLI — ingest / ask / rag (headless mirror of the TUI).",
    no_args_is_help=True,
)
console = Console()


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _run_async(coro):
    return asyncio.run(coro)


# --- ask ----------------------------------------------------------------


@app.command()
def ask(
    question: str = typer.Argument(..., help="The medical question to ask."),
    user: str | None = typer.Option(None, "--user", "-u"),
    language: str | None = typer.Option(None, "--lang", "-l"),
    provider_id: str | None = typer.Option(None, "--provider", "-p"),
) -> None:
    """Stream a grounded answer to ``question``."""

    async def _run() -> None:
        with inject_context(user_id=user, language=language) as (_, uid, lang):
            provider = resolve_provider(override=provider_id)
            model = build_model(provider)
            service = AskService(model=model, language=lang)

            async for event in service.run(question, user_id=uid):
                if isinstance(event, TokenChunk):
                    console.print(event.text, end="")
                elif isinstance(event, Error):
                    _stderr(f"\n[error] {event.error_type}: {event.message}")
                    raise typer.Exit(code=1)
                elif isinstance(event, Done):
                    console.print()  # newline after streaming text

    _run_async(_run())


# --- ingest -------------------------------------------------------------


ingest_app = typer.Typer(help="Ingest personal info, history, or reports.")
app.add_typer(ingest_app, name="ingest")


@ingest_app.command("profile")
def ingest_profile(
    field: str = typer.Argument(..., help="key=value, e.g. allergy=penicillin"),
    user: str | None = typer.Option(None, "--user", "-u"),
    language: str | None = typer.Option(None, "--lang", "-l"),
) -> None:
    """Save a single profile field."""

    async def _run() -> None:
        with inject_context(user_id=user, language=language) as (_, uid, _):
            service = IngestService()
            async for event in service.run(field, user_id=uid):
                if isinstance(event, ToolStarted):
                    console.print(f"[dim]→ {event.tool_name}[/dim]")
                elif isinstance(event, ToolCompleted):
                    console.print(f"[dim]✓ {event.tool_name}: {event.summary}[/dim]")
                elif isinstance(event, Done):
                    console.print(f"[green]{event.final.summary}[/green]")
                elif isinstance(event, Error):
                    _stderr(f"[error] {event.message}")
                    raise typer.Exit(code=1)

    _run_async(_run())


# --- rag ----------------------------------------------------------------


rag_app = typer.Typer(help="Manage reference materials (per-user RAG).")
app.add_typer(rag_app, name="rag")


@rag_app.command("add")
def rag_add(
    path: str = typer.Argument(..., help="Local file path to add."),
    user: str | None = typer.Option(None, "--user", "-u"),
    language: str | None = typer.Option(None, "--lang", "-l"),
    public: bool = typer.Option(
        False,
        "--public",
        help="Mark as public reference (skips PHI scrub, allows cloud).",
    ),
) -> None:
    """Add a document to the user's personal RAG store."""

    async def _run() -> None:
        with inject_context(user_id=user, language=language) as (_, uid, _):
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            store = UserRagStore.from_defaults()
            service = RagService(store=store)
            async for event in service.run(text, user_id=uid, public=public):
                if isinstance(event, ToolStarted):
                    console.print(f"[dim]→ {event.tool_name}[/dim]")
                elif isinstance(event, ToolCompleted):
                    console.print(f"[dim]✓ {event.tool_name}: {event.summary}[/dim]")
                elif isinstance(event, Done):
                    console.print(
                        f"[green]doc_id={event.final.doc_id} "
                        f"chunks={event.final.chunk_count}[/green]"
                    )
                elif isinstance(event, Error):
                    _stderr(f"[error] {event.message}")
                    raise typer.Exit(code=1)

    _run_async(_run())


# Convenience subcommand to list known modes — useful for shell completion.
@app.command()
def modes() -> None:
    """List configured interaction modes."""
    cfg = _cfg.load_modes_config()
    for name, mode in cfg.modes.items():
        console.print(
            f"[bold]{name}[/bold]  llm={mode.allow_llm_inference}  tools={len(mode.tools)}"
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
