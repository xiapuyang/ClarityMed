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
from claritymed.core.observability.logging import setup_logging
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
from claritymed.stores.models import load_models, resolve_provider
from claritymed.stores.user_rag import UserRagStore

app = typer.Typer(
    name="claritymed",
    help="ClarityMed CLI — ingest / ask / rag (headless mirror of the TUI).",
    no_args_is_help=True,
)
console = Console()

_BOOTSTRAPPED = False


def _bootstrap_once() -> None:
    """Idempotent CLI bootstrap: load ~/.claritymed/.env then init the three loggers.

    Called from every subcommand. The first call wins; subsequent calls in the
    same process are no-ops. Tests that need a fresh state can re-enter via
    ``setup_logging`` directly.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    _cfg.load_env_file()
    setup_logging(script_name="claritymed", console_level=None)
    _BOOTSTRAPPED = True


@app.callback()
def _cli_root() -> None:
    """Root callback — runs before every subcommand."""
    _bootstrap_once()


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
        from claritymed.orchestrator.services import ChatSession

        with inject_context(user_id=user, language=language) as (_, uid, lang):
            provider = resolve_provider(override=provider_id)
            model = build_model(provider)
            service = AskService(
                model=model,
                language=lang,
                chat_session=ChatSession.new(uid),
                provider_id=provider.id,
                model_name=provider.model,
            )

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


@app.command()
def tui(
    user: str | None = typer.Option(None, "--user", "-u"),
    language: str | None = typer.Option(None, "--lang", "-l"),
    provider_id: str | None = typer.Option(None, "--provider", "-p"),
) -> None:
    """Launch the Textual TUI."""
    from claritymed.cli.tui import ClarityMedApp
    from claritymed.errors import UnknownProviderError

    # Resolve the provider up front so a typo (`--provider oMLX`) fails
    # cleanly to stderr instead of opening the TUI and exploding on the
    # first submit. Matches the project rule: provider resolution is loud,
    # never silent.
    try:
        provider = resolve_provider(override=provider_id)
    except UnknownProviderError as exc:
        valid = ", ".join(p.id for p in load_models().providers)
        _stderr(f"[error] {exc}")
        _stderr(f"  valid provider ids: {valid}")
        raise typer.Exit(code=1) from exc

    ClarityMedApp(
        user_id=user,
        language=language,
        provider_id=provider.id,
    ).run()


prompts_app = typer.Typer(
    name="prompts",
    help="Sync YAML prompts with a Phoenix instance.",
    no_args_is_help=True,
)
app.add_typer(prompts_app, name="prompts")


def _print_sync_report(report) -> None:
    for entry in report.entries:
        marker = {
            "pushed": "[green]→[/green]",
            "pulled": "[green]←[/green]",
            "skipped": "[dim]·[/dim]",
            "missing": "[yellow]?[/yellow]",
            "error": "[red]✗[/red]",
        }.get(entry.action, "?")
        console.print(
            f"  {marker} {entry.prompt_name}({entry.language}) "
            f"-> {entry.phoenix_name}  {entry.detail}"
        )
    changed = len(report.changed)
    errors = len(report.errors)
    console.print(
        f"\n[bold]{report.direction}[/bold] "
        f"{'(dry-run) ' if report.dry_run else ''}"
        f"changed={changed} errors={errors}"
    )


@prompts_app.command("push")
def prompts_push(
    name: str | None = typer.Argument(None, help="Filter to one prompt name."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report changes without writing to Phoenix."
    ),
) -> None:
    """Push local YAML prompts (latest version) to Phoenix as the ``production`` tag."""
    from claritymed.core.prompts.phoenix_sync import push

    try:
        report = push(name=name, dry_run=dry_run)
    except Exception as exc:  # noqa: BLE001
        _stderr(f"[error] {exc}")
        raise typer.Exit(code=1) from exc
    _print_sync_report(report)
    if report.errors:
        raise typer.Exit(code=1)


@prompts_app.command("pull")
def prompts_pull(
    name: str | None = typer.Argument(None, help="Filter to one prompt name."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report changes without writing YAML."
    ),
    into_new_version: bool = typer.Option(
        False,
        "--into-new-version",
        help="Append a new YAML version instead of overwriting the latest in place.",
    ),
    new_version_name: str | None = typer.Option(
        None,
        "--version-name",
        help="Override the new version label (implies --into-new-version). "
        "Default auto-bumps to vN+1.",
    ),
) -> None:
    """Pull Phoenix ``production``-tagged prompts back into YAML.

    Default: overwrite the latest version in place (compact diff). Use
    ``--into-new-version`` to append a fresh version block instead, and
    ``--version-name v1.1`` to tag it explicitly.
    """
    from claritymed.core.prompts.phoenix_sync import pull

    try:
        report = pull(
            name=name,
            dry_run=dry_run,
            into_new_version=into_new_version,
            new_version_name=new_version_name,
        )
    except Exception as exc:  # noqa: BLE001
        _stderr(f"[error] {exc}")
        raise typer.Exit(code=1) from exc
    _print_sync_report(report)
    if report.errors:
        raise typer.Exit(code=1)


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
