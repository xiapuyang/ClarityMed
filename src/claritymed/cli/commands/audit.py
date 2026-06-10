"""``claritymed audit`` — inspect the structured audit log.

Three subcommands: ``grep`` (filter by trace/request/user/kind/window),
``list-rules`` (enumerate the registered analytics rules), and ``scan``
(stream the log through all or a chosen subset of rules and emit a
human-readable or JSON report).
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from claritymed import config as _cfg
from claritymed.cli.common import console, stderr

audit_app = typer.Typer(
    name="audit",
    help="Inspect the structured audit log.",
    no_args_is_help=True,
)


@audit_app.command("grep")
def audit_grep(
    trace_id: str | None = typer.Option(
        None, "--trace-id", help="Filter by OTel trace id."
    ),
    request_id: str | None = typer.Option(
        None, "--request-id", help="Filter by request id."
    ),
    user_id: str | None = typer.Option(None, "--user-id", help="Filter by user id."),
    kind: str | None = typer.Option(
        None, "--kind", help="Filter by audit kind, e.g. mode.ask."
    ),
    since: str | None = typer.Option(
        None, "--since", help="Lower bound on created_at (ISO 8601 prefix match)."
    ),
    until: str | None = typer.Option(
        None, "--until", help="Upper bound on created_at (ISO 8601 prefix match)."
    ),
    limit: int | None = typer.Option(None, "--limit", help="Stop after N matches."),
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Emit one JSON object per line (drops the formatter prefix).",
    ),
) -> None:
    """Grep audit.log* JSONL by trace / request / user / kind / time window.

    Reads every rotated ``audit.log*`` under ``CLARITYMED_LOG_DIR``, parses
    the JSON payload from each line, and prints lines whose payload matches
    every supplied filter. Files are walked in modification order so output
    is roughly chronological even across rotations.
    """
    log_dir = Path(_cfg.LOG_DIR)
    if not log_dir.exists():
        stderr(f"[error] log dir does not exist: {log_dir}")
        raise typer.Exit(code=1)
    files = sorted(log_dir.glob("audit.log*"), key=lambda p: p.stat().st_mtime)
    if not files:
        stderr(f"[error] no audit.log* files in {log_dir}")
        raise typer.Exit(code=1)

    matched = 0
    for fp in files:
        try:
            fh = fp.open(encoding="utf-8")
        except OSError as exc:
            stderr(f"[warn] cannot open {fp}: {exc}")
            continue
        with fh:
            for raw in fh:
                line = raw.rstrip("\n")
                idx = line.find("{")
                if idx < 0:
                    continue
                try:
                    event = json.loads(line[idx:])
                except json.JSONDecodeError:
                    continue
                if trace_id and event.get("trace_id") != trace_id:
                    continue
                if request_id and event.get("request_id") != request_id:
                    continue
                if user_id and event.get("user_id") != user_id:
                    continue
                if kind and event.get("kind") != kind:
                    continue
                created = event.get("created_at", "")
                if since and created < since:
                    continue
                if until and created > until:
                    continue
                if json_out:
                    print(json.dumps(event, ensure_ascii=False))
                else:
                    print(line)
                matched += 1
                if limit is not None and matched >= limit:
                    return
    if matched == 0:
        stderr("[info] no matches")
        raise typer.Exit(code=1)


@audit_app.command("list-rules")
def audit_list_rules() -> None:
    """List registered audit rules and their descriptions."""
    from claritymed.core.audit import list_rules

    for name, desc in list_rules():
        console.print(f"[bold]{name}[/]")
        console.print(f"  {desc}")


@audit_app.command("scan")
def audit_scan(
    rules: list[str] | None = typer.Option(
        None,
        "--rule",
        "-r",
        help="Rule name(s) to run; pass repeatedly. Default: every registered rule.",
    ),
    since: str | None = typer.Option(
        None, "--since", help="Lower bound on created_at (ISO 8601 prefix match)."
    ),
    until: str | None = typer.Option(
        None, "--until", help="Upper bound on created_at (ISO 8601 prefix match)."
    ),
    user_id: str | None = typer.Option(None, "--user-id", help="Filter by user id."),
    json_out: bool = typer.Option(
        False, "--json", help="Emit one JSON object per rule (machine-readable)."
    ),
) -> None:
    """Run audit rules over ``audit.log*`` and print findings.

    Designed for `cron`-style scheduled health checks. Each rule is a
    streaming aggregator that consumes events once; adding a new rule
    is one class + one factory entry — no CLI changes.

    Examples::

        # last 24h, all rules, human-readable
        claritymed audit scan --since $(date -u -v-1d +%Y-%m-%d)

        # one specific rule, machine-readable
        claritymed audit scan --rule tool_announced_but_skipped --json
    """
    from claritymed.core.audit import build_rules, read_audit_events

    try:
        rule_objs = build_rules(only=rules)
    except KeyError as exc:
        stderr(f"[error] {exc}")
        raise typer.Exit(code=2) from None

    try:
        event_iter = read_audit_events(since=since, until=until, user_id=user_id)
    except FileNotFoundError as exc:
        stderr(f"[error] {exc}")
        raise typer.Exit(code=1) from None

    consumed = 0
    for event in event_iter:
        consumed += 1
        for r in rule_objs:
            r.accept(event)

    reports = [r.report() for r in rule_objs]

    if json_out:
        # ``dataclasses.asdict`` would deep-copy; samples lists are
        # already plain dicts so a shallow translation is cheaper.
        out = [
            {
                "name": rep.name,
                "description": rep.description,
                "total_relevant": rep.total_relevant,
                "counts": rep.counts,
                "findings": rep.findings,
                "samples": rep.samples,
            }
            for rep in reports
        ]
        print(
            json.dumps(
                {"events_consumed": consumed, "rules": out},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if consumed == 0:
        stderr(f"[warn] no audit events in window since={since!r} until={until!r}")

    for rep in reports:
        console.print(f"\n[bold cyan]{rep.name}[/] — {rep.description}")
        console.print(f"  considered: {rep.total_relevant} event(s)")
        if not rep.counts and not rep.findings:
            console.print("  [dim]no findings[/dim]")
            continue
        if rep.counts:
            console.print("  counts:")
            for label, n in sorted(rep.counts.items(), key=lambda kv: -kv[1]):
                display = rep.rates.get(label, str(n))
                console.print(f"    {label}: {display}")
        if rep.findings:
            console.print("  findings:")
            for line in rep.findings:
                console.print(f"    • {line}")
        if rep.samples:
            console.print("  samples:")
            for s in rep.samples[:3]:
                snippet = s.get("snippet", "")
                model = s.get("model", "?")
                console.print(f"    [{model}] {snippet}")
