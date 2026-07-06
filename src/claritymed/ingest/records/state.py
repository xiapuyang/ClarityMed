"""Per-import state: ``rows.jsonl`` append + ``session.yaml`` snapshot.

State-machine rationale (mirror of plan §"State machine per row"):

* ``rows.jsonl`` is append-only. Every row transition (``pending`` first,
  then a terminal status) writes one JSON object per line.
* ``read_latest_rows`` collapses to "latest entry per row_id wins" so the
  orchestrator can ask "which rows are already terminal?" at start and
  on resume. A ``pending`` line without a successor + an existing
  manifest is the orchestrator's signal that a prior crash interrupted
  the write — translated to ``done_recovered`` in Unit 8, not
  ``skipped(already_imported)``.
* ``session.yaml`` is a derived cache rewritten once on success. If it
  goes stale during a crash, the next ``import-status`` re-derives
  counts from ``rows.jsonl`` directly.
* ``delete_template_dir`` is failure-safe: on rmtree error it writes a
  ``.cleanup_failed`` marker that the loader rejects on next resume
  attempt, giving the user a clear "manual cleanup needed" signal
  instead of partial-state confusion.

Durability: each append calls ``flush + fsync`` so SIGINT mid-write
doesn't drop the latest line. Cost: a few ms per row; acceptable for
import workloads where rows arrive at human speed.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from claritymed.stores.paths import (
    import_dir,
    import_rows_jsonl,
    import_session_yaml,
    import_template_dir,
    imports_root,
)

logger = logging.getLogger(__name__)

CLEANUP_FAILED_MARKER = ".cleanup_failed"
RowKind = Literal["case", "fact"]
FactKind = Literal["profile", "allergy", "condition", "medication"]
RowStatus = Literal[
    "pending",
    "done",
    "done_recovered",
    "skipped",
    "error",
]


class RowRecord(BaseModel):
    """One transition line in ``rows.jsonl``. PHI-free by construction.

    The orchestrator builds these from ``ApplyResult`` returns; this
    module only writes / reads / collapses them. ``slug`` is the
    manifest slug for case rows (PHI-free — sha-based suffix), null
    for fact rows. Field names are deliberate JSON keys, not pydantic
    aliases, so a ``rows.jsonl`` line is a literal dump of this model.
    """

    model_config = ConfigDict(extra="forbid")

    kind: RowKind
    user_id: str
    row_id: str
    ts: datetime
    status: RowStatus
    fact_kind: FactKind | None = None
    slug: str | None = None
    error: str | None = None
    reason: str | None = None


class ImportState:
    """Read/write facade over one import's state directory.

    Construction is cheap — just stores the import_id and computes the
    paths. Side effects happen at method-call time so a partially-set-up
    ``ImportState`` doesn't accidentally create directories.
    """

    def __init__(self, import_id: str) -> None:
        self.import_id = import_id
        self.dir_path = import_dir(import_id)
        self.template_dir = import_template_dir(import_id)
        self.rows_jsonl = import_rows_jsonl(import_id)
        self.session_yaml = import_session_yaml(import_id)

    # --- bootstrap -----------------------------------------------------

    def ensure_dir(self) -> None:
        """Create ``data/_imports/<id>/`` (mode 0700) if missing.

        Also ensures ``imports_root`` exists with mode 0700. Idempotent.
        """
        root = imports_root()
        root.mkdir(parents=True, exist_ok=True)
        _chmod_safe(root, 0o700)
        self.dir_path.mkdir(parents=True, exist_ok=True)
        _chmod_safe(self.dir_path, 0o700)

    def ensure_template_dir(self) -> None:
        """Create ``template/`` (mode 0700) if missing."""
        self.ensure_dir()
        self.template_dir.mkdir(parents=True, exist_ok=True)
        _chmod_safe(self.template_dir, 0o700)

    # --- rows.jsonl ----------------------------------------------------

    def append_row(self, row: RowRecord) -> None:
        """Atomically append one row transition.

        ``flush + fsync`` so SIGINT between two transitions can't drop
        the latest line. The orchestrator's SIGINT handler relies on
        this to know that "row appears in rows.jsonl" implies "row was
        durably observed."
        """
        self.ensure_dir()
        line = row.model_dump_json(exclude_none=True)
        with open(self.rows_jsonl, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        try:
            os.chmod(self.rows_jsonl, 0o600)
        except OSError:
            # filesystems that don't support chmod (FAT, some network FS)
            # — best effort, don't fail the append.
            pass

    def read_latest_rows(self) -> dict[str, RowRecord]:
        """Collapse to ``{row_id: latest_RowRecord}``.

        Missing file → empty dict (resume from never-attempted state).
        Malformed JSON line → ``ValueError`` (do NOT silently skip; the
        operator must see and resolve, otherwise a stale ``pending`` row
        could be misclassified on resume).
        """
        if not self.rows_jsonl.exists():
            return {}
        latest: dict[str, RowRecord] = {}
        with open(self.rows_jsonl, encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                    row = RowRecord.model_validate(data)
                except Exception as exc:
                    raise ValueError(
                        f"rows.jsonl line {line_no} unparseable: {exc}"
                    ) from exc
                latest[row.row_id] = row
        return latest

    def has_pending_row(self, row_id: str) -> bool:
        """Return True iff a literal ``pending`` line exists for ``row_id``.

        Used by the WAL orchestrator (Unit 8) to distinguish "crash-
        recovered done" (``pending`` line exists, then ``FileExistsError``
        from the store) from "earlier run already wrote this" (no
        ``pending`` line in this rows.jsonl).
        """
        if not self.rows_jsonl.exists():
            return False
        with open(self.rows_jsonl, encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if data.get("row_id") == row_id and data.get("status") == "pending":
                    return True
        return False

    # --- session.yaml --------------------------------------------------

    def write_session(self, payload: dict) -> None:
        """Atomically rewrite ``session.yaml`` (tmp + os.rename)."""
        self.ensure_dir()
        tmp = self.session_yaml.with_suffix(self.session_yaml.suffix + ".tmp")
        body = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        try:
            tmp.write_text(body, encoding="utf-8")
            os.rename(tmp, self.session_yaml)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
        _chmod_safe(self.session_yaml, 0o600)

    def read_session(self) -> dict | None:
        """Return the cached snapshot, or ``None`` if no session.yaml exists."""
        if not self.session_yaml.exists():
            return None
        return yaml.safe_load(self.session_yaml.read_text(encoding="utf-8")) or {}

    # --- template cleanup ---------------------------------------------

    def delete_template_dir(self) -> None:
        """Tear down ``template/`` after a successful import.

        Failure-safe — on rmtree error (permission, EBUSY) writes a
        ``.cleanup_failed`` marker that the loader rejects on next resume
        attempt, giving the operator a clear "manual cleanup needed"
        signal instead of silently leaking PHI in a partially-deleted
        state. Missing-dir is a no-op (idempotent).
        """
        if not self.template_dir.exists():
            return
        errors: list[str] = []

        def _on_error(_func, path, exc_info):
            errors.append(f"{path}: {exc_info[1]}")

        shutil.rmtree(self.template_dir, onerror=_on_error)
        if errors:
            try:
                self.template_dir.mkdir(parents=True, exist_ok=True)
                marker = self.template_dir / CLEANUP_FAILED_MARKER
                marker.write_text(
                    f"cleanup failed for import_id={self.import_id}:\n"
                    + "\n".join(errors),
                    encoding="utf-8",
                )
                _chmod_safe(marker, 0o600)
            except OSError:
                # If we can't even write the marker, log and move on —
                # the orchestrator's caller will see the original failure
                # via the surfaced summary state. Don't shadow the real
                # error by raising on a recovery-path failure.
                logger.warning(
                    "import %s: cleanup failed AND could not write marker",
                    self.import_id,
                    exc_info=True,
                )


# --- helpers ----------------------------------------------------------


def _chmod_safe(path: Path, mode: int) -> None:
    """``os.chmod`` swallowing OSError on filesystems that don't support it.

    Per-import dirs / files are PHI-bearing; we *want* 0700 / 0600 on
    POSIX. On exotic mounts (FAT, some network shares) the chmod silently
    no-ops and the OS falls back to the umask. That's acceptable for a
    single-user laptop deployment — fail-loud here would block import on
    a filesystem we have no other choice about.
    """
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def now_utc() -> datetime:
    """UTC ``datetime`` for row timestamps. Centralized so tests can
    monkeypatch a fixed clock."""
    return datetime.now(timezone.utc)
