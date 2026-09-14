"""``claritymed record …`` — template-driven record-import CLI surface.

Five subcommands:

* ``import-from-template`` — copy template into state dir, apply facts
  then cases via WAL pattern, emit streaming JSON events, clean up on
  success.
* ``import-status`` — collapse ``rows.jsonl`` into per-user summary.
* ``import-resume`` — re-run ``import-from-template`` against an
  existing state dir; reuses the same WAL pattern.
* ``ocr-extract`` — synchronous OCR for one file via the project's
  existing pipeline; BlobStore-cached so each attachment is OCR'd at
  most once across the skill + worker.
* ``health`` — preflight: data dir writable, ``phi_policy: local-only``,
  OCR provider builds, every chain entry ``is_local``, at least one
  user account exists.

Single-OCR guarantee: ``ocr-extract`` writes the sentinel; Unit 6's
case writer reads it to set ``Attachment.ocr_status`` and the lazy
``OcrWorker`` post-import sees the sentinel and skips. See plan
§OCR pipeline reuse.

The streaming JSON event shape is deliberately minimal so the skill
(Unit 9) can parse line-by-line and surface progress:
``{"event":"row_done","kind":"case","user_id":"alice","row_id":"...","slug":"..."}``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import signal
import sys
from pathlib import Path
from typing import Any, Optional

import typer

from claritymed.cli.entry import inject_context

logger = logging.getLogger(__name__)

# Event names emitted on stdout when --json is set. Documented here so
# the skill (Unit 9) and any future parser have a single source.
EVENT_ROW_DONE = "row_done"
EVENT_ROW_SKIPPED = "row_skipped"
EVENT_ROW_ERROR = "row_error"
EVENT_SUMMARY = "summary"
EVENT_FATAL = "fatal"

record_app = typer.Typer(
    name="record",
    help="Import medical records from a template directory built by the import-medical-record skill.",
    no_args_is_help=True,
)


# --- import-from-template ----------------------------------------------


@record_app.command("import-from-template")
def import_from_template_cmd(
    template_dir: Path = typer.Argument(
        ..., exists=True, file_okay=False, dir_okay=True, resolve_path=True
    ),
    user: Optional[str] = typer.Option(None, "--user", "-u"),
    lang: Optional[str] = typer.Option(None, "--lang", "-l"),
    import_id_override: Optional[str] = typer.Option(
        None, "--import-id", help="Override content-hash import_id (debugging)."
    ),
    dry_run: bool = typer.Option(False, "--dry-run"),
    keep_template: bool = typer.Option(False, "--keep-template"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Apply a template directory to the per-user stores."""
    try:
        with inject_context(
            user_id=user,
            language=lang,
            command="record.import-from-template",
            check_user_exists=True,
        ) as (_rid, uid, _lang):
            from claritymed.ingest.records.template_loader import load_template

            loaded = load_template(template_dir)
            import_id = import_id_override or loaded.import_id

            if dry_run:
                _emit_dry_run(loaded, import_id, json_output)
                return

            _run_import(
                uid=uid,
                loaded=loaded,
                import_id=import_id,
                source_template_dir=template_dir,
                keep_template=keep_template,
                json_output=json_output,
                copy_template=True,
            )
    except Exception as exc:
        _emit_fatal_and_exit(exc, json_output)


# --- import-resume ----------------------------------------------------


@record_app.command("import-resume")
def import_resume_cmd(
    import_id: str = typer.Argument(...),
    user: Optional[str] = typer.Option(None, "--user", "-u"),
    lang: Optional[str] = typer.Option(None, "--lang", "-l"),
    keep_template: bool = typer.Option(False, "--keep-template"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Resume a partially-completed import. Fails if the template
    directory has been cleaned up (import was already complete)."""
    try:
        with inject_context(
            user_id=user,
            language=lang,
            command="record.import-resume",
            check_user_exists=True,
        ) as (_rid, uid, _lang):
            from claritymed.ingest.records.state import ImportState
            from claritymed.ingest.records.template_loader import load_template

            state = ImportState(import_id)
            if not state.template_dir.exists():
                raise typer.BadParameter(
                    f"template directory for {import_id} no longer exists — "
                    "import may have already completed and been cleaned up. "
                    "Re-run import-from-template with the original source if "
                    "this was unintended."
                )
            loaded = load_template(state.template_dir)

            _run_import(
                uid=uid,
                loaded=loaded,
                import_id=import_id,
                source_template_dir=state.template_dir,
                keep_template=keep_template,
                json_output=json_output,
                copy_template=False,
            )
    except Exception as exc:
        _emit_fatal_and_exit(exc, json_output)


# --- import-status ----------------------------------------------------


@record_app.command("import-status")
def import_status_cmd(
    import_id: str = typer.Argument(...),
    user: Optional[str] = typer.Option(None, "--user", "-u"),
    lang: Optional[str] = typer.Option(None, "--lang", "-l"),
    verbose: bool = typer.Option(False, "--verbose"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Summarize one import's state. ``rows.jsonl`` is authoritative."""
    try:
        with inject_context(
            user_id=user,
            language=lang,
            command="record.import-status",
            check_user_exists=True,
        ):
            from claritymed.ingest.records.state import ImportState

            state = ImportState(import_id)
            rows = state.read_latest_rows()
            summary = _summarize_rows(rows, import_id)
            if json_output:
                summary["verbose_rows"] = (
                    [r.model_dump(mode="json") for r in rows.values()]
                    if verbose
                    else None
                )
                print(json.dumps(summary, ensure_ascii=False))
            else:
                _print_status_table(summary, rows, verbose)
    except Exception as exc:
        _emit_fatal_and_exit(exc, json_output)


# --- ocr-extract ------------------------------------------------------


@record_app.command("ocr-extract")
def ocr_extract_cmd(
    file: Path = typer.Argument(
        ..., exists=True, dir_okay=False, file_okay=True, resolve_path=True
    ),
    user: Optional[str] = typer.Option(None, "--user", "-u"),
    json_output: bool = typer.Option(True, "--json/--no-json"),
) -> None:
    """Synchronous OCR for one file, BlobStore-cached by sha256.

    Cloud-bypass: refuses to run unless ``configs/ocr.yaml::phi_policy``
    is ``"local-only"``. Identical gate as ``health``.
    """
    try:
        with inject_context(
            user_id=user,
            language=None,
            command="record.ocr-extract",
            check_user_exists=True,
        ) as (_rid, uid, _lang):
            from claritymed.core.schemas.ocr import load_ocr_config

            cfg = load_ocr_config()
            if cfg.phi_policy != "local-only":
                payload = {
                    "status": "failed",
                    "failed_reason": "phi_policy_not_local_only",
                    "error": (
                        "OCR refused: configs/ocr.yaml::phi_policy is "
                        f"{cfg.phi_policy!r}; expected 'local-only'."
                    ),
                }
                print(json.dumps(payload, ensure_ascii=False))
                raise typer.Exit(code=1)

            payload = asyncio.run(_run_ocr_extract(uid, file))
            print(json.dumps(payload, ensure_ascii=False))
            if payload.get("status") == "failed":
                raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as exc:
        _emit_fatal_and_exit(exc, json_output)


# --- health -----------------------------------------------------------


@record_app.command("health")
def health_cmd(
    json_output: bool = typer.Option(True, "--json/--no-json"),
) -> None:
    """Preflight service-availability check for the import skill.

    Verifies: (a) data dir writable, (b) ``phi_policy == "local-only"``,
    (c) OCR provider builds, (d) every chain entry is ``is_local``, (e)
    at least one resolvable user account exists.
    """
    checks, failed = _run_health_checks()
    payload: dict[str, Any] = {
        "ok": not failed,
        "checks": checks,
    }
    if failed:
        payload["failed"] = failed
    print(json.dumps(payload, ensure_ascii=False))
    if failed:
        raise typer.Exit(code=1)


# --- helpers: dry-run + status -----------------------------------------


def _emit_dry_run(loaded, import_id: str, json_output: bool) -> None:
    plan_per_user = {
        uid: {
            "cases": len(b.cases),
            "facts": {
                "profile": 1 if b.facts.profile is not None else 0,
                "allergies": len(b.facts.allergies),
                "conditions": len(b.facts.conditions),
                "medications": len(b.facts.medications),
            },
        }
        for uid, b in loaded.bundles.items()
    }
    payload = {
        "event": "dry_run",
        "import_id": import_id,
        "warnings": loaded.warnings,
        "plan": plan_per_user,
    }
    if json_output:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(f"import_id: {import_id}")
        for uid, p in plan_per_user.items():
            print(f"  {uid}: {p['cases']} cases, facts={p['facts']}")
        for w in loaded.warnings:
            print(f"  WARN: {w}")


def _summarize_rows(rows, import_id: str) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {}
    for row in rows.values():
        bucket = counts.setdefault(
            row.user_id,
            {"case": 0, "fact": 0, "done": 0, "skipped": 0, "error": 0, "pending": 0},
        )
        bucket[row.kind] += 1
        bucket[row.status if row.status != "done_recovered" else "done"] += 1
    return {
        "import_id": import_id,
        "user_counts": counts,
        "row_count": len(rows),
    }


def _print_status_table(summary, rows, verbose: bool) -> None:
    print(f"import_id: {summary['import_id']}  rows: {summary['row_count']}")
    for uid, c in summary["user_counts"].items():
        print(
            f"  {uid}: case={c['case']} fact={c['fact']} "
            f"done={c['done']} skipped={c['skipped']} "
            f"error={c['error']} pending={c['pending']}"
        )
    if verbose:
        for row in rows.values():
            if row.status == "done":
                continue
            print(
                f"    {row.user_id} {row.kind}/{row.row_id} "
                f"status={row.status} error={row.error}"
            )


# --- helpers: WAL orchestrator ----------------------------------------


def _run_import(
    *,
    uid: str,
    loaded,
    import_id: str,
    source_template_dir: Path,
    keep_template: bool,
    json_output: bool,
    copy_template: bool,
) -> None:
    """Shared orchestration for ``import-from-template`` + ``import-resume``."""
    from claritymed.ingest.records.case_writer import apply_case
    from claritymed.ingest.records.fact_writer import (
        apply_facts,
    )
    from claritymed.ingest.records.state import ImportState, RowRecord, now_utc

    state = ImportState(import_id)
    state.ensure_dir()

    if copy_template:
        _copy_template_into_state(source_template_dir, state.template_dir)

    interrupted: dict[str, Any] = {"row_id": None}
    previous_sigint = signal.getsignal(signal.SIGINT)

    def _sigint_handler(signum, frame):  # noqa: ARG001
        in_flight = interrupted["row_id"]
        if in_flight:
            state.append_row(
                RowRecord(
                    kind="case",
                    user_id=uid,
                    row_id=in_flight,
                    ts=now_utc(),
                    status="error",
                    error="interrupted",
                )
            )
        sys.stdout.flush()
        sys.exit(130)

    signal.signal(signal.SIGINT, _sigint_handler)

    try:
        existing = state.read_latest_rows()
        already_terminal = {
            rid
            for rid, r in existing.items()
            if r.status in {"done", "done_recovered", "skipped", "error"}
        }

        # Facts first (R3): a fact equivalence read sees an unmodified
        # profile.db while cases haven't started writing yet.
        for user_id, bundle in loaded.bundles.items():
            for fact_kind, _ in _iter_fact_rows(bundle.facts):
                pass  # placeholder — actual iteration via apply_facts below

            results = apply_facts(user_id, bundle.facts)
            for r in results:
                if r.row_id in already_terminal:
                    continue
                interrupted["row_id"] = r.row_id
                state.append_row(
                    RowRecord(
                        kind="fact",
                        user_id=user_id,
                        row_id=r.row_id,
                        ts=now_utc(),
                        status="pending",
                        fact_kind=r.fact_kind,
                    )
                )
                final_status = r.status
                state.append_row(
                    RowRecord(
                        kind="fact",
                        user_id=user_id,
                        row_id=r.row_id,
                        ts=now_utc(),
                        status=final_status,
                        fact_kind=r.fact_kind,
                        error=r.error_detail if final_status == "error" else None,
                        reason=r.error_detail if final_status == "skipped" else None,
                    )
                )
                interrupted["row_id"] = None
                _emit_row_event(
                    json_output,
                    "fact",
                    user_id,
                    r.row_id,
                    final_status,
                    slug=None,
                    error=r.error_detail,
                )
                already_terminal.add(r.row_id)

        # Cases.
        for user_id, bundle in loaded.bundles.items():
            for case in bundle.cases:
                row_id = f"case:{user_id}:{case.case_id}"
                if row_id in already_terminal:
                    continue

                interrupted["row_id"] = row_id
                state.append_row(
                    RowRecord(
                        kind="case",
                        user_id=user_id,
                        row_id=row_id,
                        ts=now_utc(),
                        status="pending",
                    )
                )

                result = apply_case(user_id, case)
                final_status: str
                reason: str | None = None

                if (
                    result.status == "skipped"
                    and result.error_detail == "already_imported"
                ):
                    # WAL distinction: a prior `pending` line in THIS rows.jsonl
                    # means this is a crash-recovered done, not a prior-run skip.
                    # The `pending` we just appended counts — so check if there's
                    # an EARLIER pending we need to distinguish from.
                    earlier_pending = _had_pending_before_this_run(
                        state, row_id, existing
                    )
                    if earlier_pending:
                        final_status = "done_recovered"
                        reason = "crash_recovered"
                    else:
                        final_status = "skipped"
                        reason = "already_imported"
                else:
                    final_status = result.status

                state.append_row(
                    RowRecord(
                        kind="case",
                        user_id=user_id,
                        row_id=row_id,
                        ts=now_utc(),
                        status=final_status,
                        slug=result.slug,
                        error=result.error_detail if final_status == "error" else None,
                        reason=reason,
                    )
                )
                interrupted["row_id"] = None
                _emit_row_event(
                    json_output,
                    "case",
                    user_id,
                    row_id,
                    final_status,
                    slug=result.slug,
                    error=result.error_detail if final_status == "error" else None,
                )
                already_terminal.add(row_id)

        latest = state.read_latest_rows()
        summary = _summarize_rows(latest, import_id)
        any_error = any(r.status == "error" for r in latest.values())

        # Template cleanup gate: keep alive if any error OR --keep-template.
        template_kept = keep_template or any_error
        if not template_kept:
            state.delete_template_dir()

        summary["template_retained"] = template_kept
        summary["any_error"] = any_error
        state.write_session(summary)

        if json_output:
            print(json.dumps({"event": EVENT_SUMMARY, **summary}, ensure_ascii=False))
        else:
            _print_status_table(summary, latest, verbose=False)
            if any_error:
                print(
                    f"\n  template retained at {state.template_dir}; "
                    f"run `claritymed record import-resume {import_id}` to retry errored rows."
                )
    finally:
        signal.signal(signal.SIGINT, previous_sigint)


def _iter_fact_rows(facts):
    """Reserved hook — kept so future iteration changes touch one spot."""
    if facts.profile is not None:
        yield "profile", None
    for a in facts.allergies:
        yield "allergy", a
    for c in facts.conditions:
        yield "condition", c
    for m in facts.medications:
        yield "medication", m


def _had_pending_before_this_run(state, row_id: str, existing_at_start) -> bool:
    """True iff a ``pending`` line existed BEFORE the current run started.

    The current run always appends a ``pending`` before the store call,
    so a literal ``has_pending_row`` check would over-report. The clean
    discriminator: was this row already in the latest-collapsed view at
    run start? If yes AND it was a ``pending``, this is crash recovery.
    """
    prior = existing_at_start.get(row_id)
    return prior is not None and prior.status == "pending"


def _emit_row_event(
    json_output: bool,
    kind: str,
    user_id: str,
    row_id: str,
    status: str,
    *,
    slug: str | None,
    error: str | None,
) -> None:
    event = {
        EVENT_ROW_DONE: EVENT_ROW_DONE,
        "done_recovered": EVENT_ROW_DONE,
        "skipped": EVENT_ROW_SKIPPED,
        "error": EVENT_ROW_ERROR,
    }.get(status, EVENT_ROW_DONE)
    payload = {
        "event": event,
        "kind": kind,
        "user_id": user_id,
        "row_id": row_id,
        "status": status,
        "slug": slug,
        "error": error,
    }
    if json_output:
        print(json.dumps(payload, ensure_ascii=False))


def _copy_template_into_state(source: Path, target: Path) -> None:
    """Copy each file from the skill's draft directory into the state
    template tree with mode 0700/0600. ``follow_symlinks=False`` —
    we've already rejected symlinks at loader level, this is defense
    in depth."""
    target.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(target, 0o700)
    except OSError:
        pass
    for entry in source.iterdir():
        if entry.is_dir():
            continue
        dest = target / entry.name
        shutil.copy2(entry, dest, follow_symlinks=False)
        try:
            os.chmod(dest, 0o600)
        except OSError:
            pass


# --- helpers: ocr-extract ---------------------------------------------


async def _run_ocr_extract(uid: str, file: Path) -> dict[str, Any]:
    """Wrap ``make_ocr_provider().extract_text`` with the BlobStore cache.

    Cache hit: read existing ``ocr.md`` + ``ocr.json``; emit
    ``cache_hit: true``; no OCR call.
    Cache miss: store the blob (CAS), run OCR, write the sentinel,
    emit ``cache_hit: false``.
    """
    from claritymed.core.ocr.base import OcrEmpty, OcrError
    from claritymed.core.ocr.factory import make_ocr_provider
    from claritymed.stores.blob_store import BlobStore

    content = file.read_bytes()
    sha = hashlib.sha256(content).hexdigest()
    blob = BlobStore(uid)

    if blob.ocr_done(sha):
        meta = blob.read_ocr_metadata(sha) or {}
        try:
            text = blob.read_extracted_text(sha)
        except FileNotFoundError:
            text = ""
        return {
            "status": meta.get("status", "ok"),
            "text": text,
            "sha256": sha,
            "cache_hit": True,
            "provider_used": meta.get("provider"),
            "chain_tried": meta.get("chain_tried", []),
            "modality": meta.get("modality"),
            "is_medical": meta.get("is_medical"),
        }

    ext = file.suffix.lstrip(".").lower() or "bin"
    blob.store(content, ext)

    provider = make_ocr_provider()
    target_path = file
    try:
        result = await provider.extract_text(target_path)
        status = "ok"
        text = result.text
        chain_tried = result.chain_tried
        provider_used = result.provider_used
        modality = result.modality
        is_medical = result.is_medical
        reason = None
        kind = "ocr"
    except OcrEmpty as exc:
        status = "empty"
        text = ""
        chain_tried = (
            getattr(exc.extraction, "chain_tried", []) if exc.extraction else []
        )
        provider_used = (
            getattr(exc.extraction, "provider_used", None) if exc.extraction else None
        )
        modality = None
        is_medical = None
        reason = str(exc)
        kind = "ocr"
    except OcrError as exc:
        status = "failed"
        text = ""
        chain_tried = []
        provider_used = None
        modality = None
        is_medical = None
        reason = str(exc)
        kind = "ocr"

    blob.write_ocr_result(
        sha,
        status=status,
        kind=kind,
        ext=ext,
        provider=provider_used,
        chain_tried=list(chain_tried or []),
        reason=reason,
        text=text,
        original_filename=file.name,
        modality=modality,
        is_medical=is_medical,
    )

    return {
        "status": status,
        "text": text,
        "sha256": sha,
        "cache_hit": False,
        "provider_used": provider_used,
        "chain_tried": list(chain_tried or []),
        "modality": modality,
        "is_medical": is_medical,
        "error": reason if status == "failed" else None,
    }


# --- helpers: health --------------------------------------------------


def _run_health_checks() -> tuple[list[dict], list[str]]:
    """Return ``(checks_log, failed_check_names)``."""
    checks: list[dict] = []
    failed: list[str] = []

    # 1. data dir writable
    try:
        from claritymed import config as _cfg

        _cfg.ensure_runtime_dirs()
        probe = _cfg.DATA_DIR / ".healthcheck_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        checks.append({"check": "data_dir_writable", "ok": True})
    except Exception as exc:
        checks.append({"check": "data_dir_writable", "ok": False, "error": repr(exc)})
        failed.append("data_dir_writable")

    # 2. phi_policy: local-only
    try:
        from claritymed.core.schemas.ocr import load_ocr_config

        cfg = load_ocr_config()
        if cfg.phi_policy == "local-only":
            checks.append({"check": "phi_policy_local_only", "ok": True})
        else:
            checks.append(
                {
                    "check": "phi_policy_local_only",
                    "ok": False,
                    "actual": cfg.phi_policy,
                }
            )
            failed.append("phi_policy_not_local_only")
    except Exception as exc:
        checks.append(
            {"check": "phi_policy_local_only", "ok": False, "error": repr(exc)}
        )
        failed.append("phi_policy_not_local_only")

    # 3. ocr provider builds + 4. every chain entry is_local
    try:
        from claritymed.core.ocr.factory import make_ocr_provider

        provider = make_ocr_provider()
        checks.append({"check": "ocr_provider_builds", "ok": True})
        try:
            non_local = _find_non_local_entries(provider)
            if not non_local:
                checks.append({"check": "all_chain_entries_local", "ok": True})
            else:
                checks.append(
                    {
                        "check": "all_chain_entries_local",
                        "ok": False,
                        "non_local": non_local,
                    }
                )
                failed.append("non_local_chain_entry")
        except Exception as exc:
            checks.append(
                {
                    "check": "all_chain_entries_local",
                    "ok": False,
                    "error": repr(exc),
                }
            )
            failed.append("chain_introspect_failed")
    except Exception as exc:
        checks.append({"check": "ocr_provider_builds", "ok": False, "error": repr(exc)})
        failed.append("ocr_provider_build_failed")

    # 5. at least one user account
    try:
        from claritymed.stores.paths import list_user_ids

        user_ids = list_user_ids()
        if user_ids:
            checks.append(
                {
                    "check": "at_least_one_user",
                    "ok": True,
                    "user_count": len(user_ids),
                }
            )
        else:
            checks.append({"check": "at_least_one_user", "ok": False})
            failed.append("no_user_account")
    except Exception as exc:
        checks.append({"check": "at_least_one_user", "ok": False, "error": repr(exc)})
        failed.append("no_user_account")

    return checks, failed


def _find_non_local_entries(provider) -> list[str]:
    """Walk the document + image chains and return non-local provider labels.

    ``RoutingOcrProvider`` already filters under ``phi_policy=local-only``
    — this is defense-in-depth so the health check fails loud if a
    non-local provider somehow survived the filter.
    """
    non_local: list[str] = []
    for chain_name in ("document_chain", "image_chain"):
        chain = getattr(provider, chain_name, None) or []
        for entry in chain:
            if getattr(entry, "is_local", True) is False:
                non_local.append(f"{chain_name}:{type(entry).__name__}")
    return non_local


# --- helpers: fatal -----------------------------------------------------


def _emit_fatal_and_exit(exc: BaseException, json_output: bool) -> None:
    payload = {"event": EVENT_FATAL, "error": repr(exc)}
    if json_output:
        print(json.dumps(payload, ensure_ascii=False))
    else:
        print(f"FATAL: {exc!r}", file=sys.stderr)
    raise typer.Exit(code=1)
