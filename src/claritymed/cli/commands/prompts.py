"""``claritymed prompts`` — sync YAML prompts with a Phoenix instance.

Three subcommands: ``push`` (YAML → Phoenix), ``pull`` (Phoenix → YAML),
``diff`` (read-only comparison, exits non-zero when entries differ so
this can gate CI). The runtime path never reads from Phoenix; this CLI
is the only direction that touches Phoenix's HTTP API.
"""

from __future__ import annotations

import logging
import typer

from claritymed.cli.common import console

logger = logging.getLogger(__name__)

prompts_app = typer.Typer(
    name="prompts",
    help="Sync YAML prompts with a Phoenix instance.",
    no_args_is_help=True,
)


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
        logger.error("%s", exc)
        raise typer.Exit(code=1) from exc
    _print_sync_report(report)
    if report.errors:
        raise typer.Exit(code=1)


@prompts_app.command("diff")
def prompts_diff(
    name: str | None = typer.Argument(None, help="Filter to one prompt name."),
    color: bool = typer.Option(
        True, "--color/--no-color", help="Colorize diff output."
    ),
) -> None:
    """Show a unified diff between local YAML and Phoenix production-tagged prompts.

    Same content shape as ``push`` / ``pull`` reads — one entry per
    ``(name, language)`` pair. Exit code is 0 when everything matches,
    1 when at least one entry differs or errors so this can gate CI.
    """
    from claritymed.core.prompts.phoenix_sync import diff

    try:
        report = diff(name=name)
    except Exception as exc:  # noqa: BLE001
        logger.error("%s", exc)
        raise typer.Exit(code=1) from exc

    for entry in report.entries:
        marker = {
            "same": "[dim]·[/dim]",
            "differs": "[yellow]Δ[/yellow]",
            "remote_missing": "[yellow]?[/yellow]",
            "error": "[red]✗[/red]",
        }.get(entry.action, "?")
        console.print(
            f"  {marker} {entry.prompt_name}({entry.language}) "
            f"-> {entry.phoenix_name}  {entry.detail}"
        )
        if entry.action == "differs":
            if color:
                from rich.syntax import Syntax

                console.print(
                    Syntax(
                        "\n".join(entry.unified_diff),
                        "diff",
                        theme="ansi_dark",
                        background_color="default",
                        word_wrap=True,
                    )
                )
            else:
                for line in entry.unified_diff:
                    console.print(line)

    differs = len(report.differs)
    errors = len(report.errors)
    console.print(f"\n[bold]diff[/bold] differs={differs} errors={errors}")
    if differs or errors:
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
        logger.error("%s", exc)
        raise typer.Exit(code=1) from exc
    _print_sync_report(report)
    if report.errors:
        raise typer.Exit(code=1)
