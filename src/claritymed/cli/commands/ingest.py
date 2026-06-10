"""``claritymed ingest`` — save profile / history fields without an LLM.

One sub-app today (``profile``); future ``history`` / ``lab`` subcommands
land here too. Each subcommand delegates to ``IngestService`` so the
audit row and event shape stay identical to the TUI path.
"""

from __future__ import annotations

import logging
import typer

from claritymed.cli.common import console, run_async
from claritymed.cli.entry import inject_context
from claritymed.orchestrator.services import (
    Done,
    Error,
    IngestService,
    ToolCompleted,
    ToolStarted,
)

logger = logging.getLogger(__name__)

ingest_app = typer.Typer(help="Ingest personal info, history, or reports.")


@ingest_app.command("profile")
def ingest_profile(
    field: str = typer.Argument(..., help="key=value, e.g. allergy=penicillin"),
    user: str | None = typer.Option(None, "--user", "-u"),
    language: str | None = typer.Option(None, "--lang", "-l"),
) -> None:
    """Save a single profile field."""

    async def _run() -> None:
        with inject_context(
            user_id=user,
            language=language,
            command=f"ingest.profile {field!r}",
            check_user_exists=True,
        ) as (_, uid, _):
            service = IngestService()
            async for event in service.run(field, user_id=uid):
                if isinstance(event, ToolStarted):
                    console.print(f"[dim]→ {event.tool_name}[/dim]")
                elif isinstance(event, ToolCompleted):
                    console.print(f"[dim]✓ {event.tool_name}: {event.summary}[/dim]")
                elif isinstance(event, Done):
                    console.print(f"[green]{event.final.summary}[/green]")
                elif isinstance(event, Error):
                    logger.error("%s", event.message)
                    raise typer.Exit(code=1)

    run_async(_run())
