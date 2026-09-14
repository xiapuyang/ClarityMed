"""``claritymed audit`` — inspect the structured audit log.

Subcommands: ``grep`` (filter by trace/request/user/kind/window),
``list-rules`` (enumerate the registered analytics rules), ``scan``
(stream the log through all or a chosen subset of rules and emit a
human-readable or JSON report), and ``ocr-overrides`` (walk blob
sentinels and surface every uploaded image whose OCR text already
carried a clinician report — the KTD-V6 short-circuit's input set).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from claritymed import config as _cfg
from claritymed.cli.common import console

logger = logging.getLogger(__name__)

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
    user_id: str | None = typer.Option(
        None, "--user", "--user-id", "-u", help="Filter by user id."
    ),
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
        logger.error("log dir does not exist: %s", log_dir)
        raise typer.Exit(code=1)
    files = sorted(log_dir.glob("audit.log*"), key=lambda p: p.stat().st_mtime)
    if not files:
        logger.error("no audit.log* files in %s", log_dir)
        raise typer.Exit(code=1)

    matched = 0
    for fp in files:
        try:
            fh = fp.open(encoding="utf-8")
        except OSError as exc:
            logger.warning("cannot open %s: %s", fp, exc)
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
        # Zero matches is a valid outcome, not an error — exit 0 so cron
        # / agent callers don't conflate "clean window" with "command broke".
        logger.info("no matches")


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
    user_id: str | None = typer.Option(
        None, "--user", "--user-id", "-u", help="Filter by user id."
    ),
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
        logger.error("%s", exc)
        raise typer.Exit(code=2) from None

    try:
        event_iter = read_audit_events(since=since, until=until, user_id=user_id)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
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
                "rates": rep.rates,
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
        logger.warning("no audit events in window since=%r until=%r", since, until)

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


@audit_app.command("ocr-overrides")
def audit_ocr_overrides(
    user_id: str | None = typer.Option(
        None,
        "--user",
        "--user-id",
        "-u",
        help="Restrict to one user (default: walk every user under data/users/).",
    ),
    limit: int | None = typer.Option(None, "--limit", help="Stop after N matches."),
    with_text: bool = typer.Option(
        False,
        "--with-text",
        help="Also include the extracted OCR text under `ocr_text` (JSON mode only).",
    ),
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Emit JSONL — one record per blob. Suitable as benchmark seed input.",
    ),
) -> None:
    """List blobs whose ``ocr.json`` carries ``ocr_has_report=true``.

    Walks ``data/users/<id>/blobs/<sha[:2]>/<sha>/ocr.json`` and prints
    every sentinel where the OCR text already contained a clinician
    report (KTD-V6 → ``kind == "ocr_override"`` in the vision tool reply).
    Use ``--json`` to pipe matches into the vision benchmark as
    real-world OCR-override seeds.

    Reads only the sentinel by default; pass ``--with-text`` to also
    inline the extracted markdown (sibling ``ocr.md``) — heavier, useful
    when grading answer quality against the report content.
    """
    data_root = _cfg.DATA_DIR / "users"
    if not data_root.exists():
        logger.error("data root does not exist: %s", data_root)
        raise typer.Exit(code=1)

    user_dirs = (
        [data_root / user_id]
        if user_id
        else sorted(p for p in data_root.iterdir() if p.is_dir())
    )

    matched = 0
    for user_dir in user_dirs:
        blobs_dir = user_dir / "blobs"
        if not blobs_dir.is_dir():
            continue
        uid = user_dir.name
        for sentinel in sorted(blobs_dir.glob("*/*/ocr.json")):
            try:
                payload = json.loads(sentinel.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("skip %s: %s", sentinel, exc)
                continue
            if payload.get("ocr_has_report") is not True:
                continue
            blob_dir = sentinel.parent
            record = {
                "user_id": uid,
                "sha256": blob_dir.name,
                "blob_dir": str(blob_dir),
                "modality": payload.get("modality"),
                "is_medical": payload.get("is_medical"),
                "provider": payload.get("provider"),
                "chars": payload.get("chars"),
                "original_filename": payload.get("original_filename"),
            }
            if with_text:
                ocr_md = blob_dir / "ocr.md"
                try:
                    record["ocr_text"] = ocr_md.read_text(encoding="utf-8")
                except OSError:
                    record["ocr_text"] = None
            if json_out:
                print(json.dumps(record, ensure_ascii=False))
            else:
                console.print(
                    f"[cyan]{uid}[/]  {record['sha256'][:12]}…  "
                    f"modality={record['modality']!s:<10} "
                    f"chars={record['chars']!s:<5} "
                    f"file={record['original_filename'] or '?'}"
                )
            matched += 1
            if limit is not None and matched >= limit:
                return

    if matched == 0:
        # Zero matches is a valid outcome — exit 0 to keep scripts simple.
        logger.info("no blobs with ocr_has_report=true")
