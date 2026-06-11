"""``claritymed tool`` — headless invocation of the v1 ingest tools.

Mirrors the LLM-callable surface. Two subcommands:

* ``claritymed tool <name> '{json_args}'`` — run one tool with JSON
  args. Without ``--auto-approve``, errors out under headless mode
  (no TUI to host the modal); with ``--auto-approve``, requires
  ``not isatty() AND CLARITYMED_HEADLESS=1`` (footgun guard, not a
  security boundary — the real defense is the SettingsStore deny
  rules which ``--auto-approve`` still honors).
* ``claritymed tool rule-list`` / ``rule-revoke <id>`` — manage the
  TTL'd allow rules from the shell.

This is the v1 minimum-shippable headless surface. The ingest
deprecation shim that maps the old ``claritymed ingest profile
allergy=penicillin`` syntax onto ``tool save_allergy ...`` is left
intentionally for follow-up — Unit 6's tools are already wired into
the ingest path through their pydantic args schemas, and the syntax
swap is a small Typer parameter rewrite.
"""

from __future__ import annotations

import json
import logging
import os
import sys

import typer
from rich.console import Console
from rich.table import Table

from claritymed.cli.entry import inject_context
from claritymed.core.observability.audit import audit_event
from claritymed.orchestrator.features.ingest_tools_plugin import INGEST_TOOLS
from claritymed.orchestrator.services.tool_dispatcher import ToolDispatcher
from claritymed.stores.settings_store import SettingsStore

logger = logging.getLogger(__name__)
console = Console()
err_console = Console(stderr=True)

tool_app = typer.Typer(help="Headless tool invocation (LLM-equivalent surface).")


def _ensure_headless_or_die() -> None:
    """Footgun guard. Documented in Unit 11: NOT a security boundary."""
    if sys.stdin.isatty():
        err_console.print(
            "[red]--auto-approve requires a non-interactive stdin "
            "(redirect from /dev/null or a pipe)[/red]"
        )
        raise typer.Exit(code=2)
    if os.environ.get("CLARITYMED_HEADLESS") != "1":
        err_console.print("[red]--auto-approve requires CLARITYMED_HEADLESS=1[/red]")
        raise typer.Exit(code=2)


@tool_app.command("run")
def tool_run(
    name: str = typer.Argument(..., help="Tool name (e.g. save_allergy)"),
    args_json: str = typer.Argument(..., help="JSON dict of args"),
    user: str | None = typer.Option(None, "--user", "-u"),
    language: str | None = typer.Option(None, "--lang", "-l"),
    auto_approve: bool = typer.Option(
        False,
        "--auto-approve",
        help="Skip the TUI approval modal. Requires non-TTY + "
        "CLARITYMED_HEADLESS=1. SettingsStore deny rules still honored.",
    ),
) -> None:
    """Run one ingest tool with JSON-encoded args."""
    impl = INGEST_TOOLS.get(name)
    if impl is None:
        err_console.print(f"[red]unknown tool: {name!r}[/red]")
        raise typer.Exit(code=1)
    try:
        args = json.loads(args_json)
    except json.JSONDecodeError as exc:
        err_console.print(f"[red]invalid JSON args: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    with inject_context(
        user_id=user,
        language=language,
        command=f"tool.run {name}",
    ):
        # Respect SettingsStore deny rules even under --auto-approve.
        from claritymed.context import user_id_ctx

        active_user = user_id_ctx.get()
        if active_user is not None:
            deny = SettingsStore(active_user).match_rule(name, args)
            if deny is not None and deny.action == "deny":
                err_console.print(f"[red]denied by settings rule {deny.id}[/red]")
                audit_event(
                    "tool.approval.denied",
                    {"tool_name": name, "rule_id": deny.id},
                )
                raise typer.Exit(code=1)

        if not auto_approve:
            err_console.print("[red]headless invocation requires --auto-approve[/red]")
            raise typer.Exit(code=1)
        _ensure_headless_or_die()
        audit_event(
            "tool.auto_approved",
            {"tool_name": name, "severity": "high"},
        )

        dispatcher = ToolDispatcher()
        try:
            result = impl(args, dispatcher=dispatcher)
        except Exception as exc:  # noqa: BLE001
            err_console.print(f"[red]tool failed: {exc}[/red]")
            raise typer.Exit(code=1) from exc
        console.print_json(data=result)


@tool_app.command("rule-list")
def rule_list(
    user: str | None = typer.Option(None, "--user", "-u"),
    language: str | None = typer.Option(None, "--lang", "-l"),
) -> None:
    """List active approval rules for the current user."""
    with inject_context(user_id=user, language=language, command="tool.rule-list"):
        from claritymed.context import user_id_ctx

        active_user = user_id_ctx.get()
        if active_user is None:
            err_console.print("[red]no user resolved[/red]")
            raise typer.Exit(code=1)
        rules = SettingsStore(active_user).list_rules()
        if not rules:
            console.print("[dim]no active rules[/dim]")
            return
        table = Table(title="Approval rules")
        for col in ("id", "tool", "action", "pattern", "ttl_days", "expires_at"):
            table.add_column(col)
        for r in rules:
            table.add_row(
                r.id[:8],
                r.tool,
                r.action,
                json.dumps(r.args_pattern, sort_keys=True),
                str(r.ttl_days),
                r.expires_at.isoformat(),
            )
        console.print(table)


@tool_app.command("rule-revoke")
def rule_revoke(
    rule_id: str = typer.Argument(..., help="Rule id (prefix accepted)"),
    user: str | None = typer.Option(None, "--user", "-u"),
    language: str | None = typer.Option(None, "--lang", "-l"),
) -> None:
    """Revoke an approval rule by id (or unique prefix)."""
    with inject_context(user_id=user, language=language, command="tool.rule-revoke"):
        from claritymed.context import user_id_ctx

        active_user = user_id_ctx.get()
        if active_user is None:
            err_console.print("[red]no user resolved[/red]")
            raise typer.Exit(code=1)
        store = SettingsStore(active_user)
        # Accept full id or unique prefix.
        candidates = [r.id for r in store.list_rules() if r.id.startswith(rule_id)]
        if not candidates:
            err_console.print(f"[red]no rule matches {rule_id!r}[/red]")
            raise typer.Exit(code=1)
        if len(candidates) > 1:
            err_console.print(
                f"[red]ambiguous prefix {rule_id!r}; "
                f"{len(candidates)} rules match[/red]"
            )
            raise typer.Exit(code=1)
        if store.revoke_rule(candidates[0]):
            console.print(f"[green]revoked {candidates[0][:8]}[/green]")
        else:
            err_console.print(f"[red]rule {candidates[0][:8]} not found[/red]")
            raise typer.Exit(code=1)
