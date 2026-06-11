"""Per-session attachment tray, persisted to ``session/<sid>/attachments.json``.

The TUI (Ctrl+V paste, ``/upload``, drag-drop) and the OCR worker share
this little file. Every entry maps one sha256 to its display metadata
plus OCR status, and ``AskService`` renders the file into the prompt
envelope each turn.

Single-writer locking via the project's composed file lock — the OCR
worker may finish on a background task at the same moment a foreground
paste adds a new attachment, so the write contention is real even for
a single-user TUI.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from claritymed.core.locks import file_lock
from claritymed.core.schemas.records import (
    Attachment,
    AttachmentSource,
    OcrStatus,
)
from claritymed.stores.paths import (
    user_session_attachments_path,
    user_session_dir,
    validate_user_id,
)

SourceKind = AttachmentSource


class SessionAttachment(BaseModel):
    """One attachment row in the session tray.

    Mirrors ``records.Attachment`` (sha256/filename/mime/size/ocr_*)
    plus a wall-clock timestamp so the TUI can show the order in which
    attachments arrived.
    """

    model_config = ConfigDict(extra="forbid")

    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    filename: str
    mime: str
    size: int = Field(ge=0)
    source: SourceKind | None = None
    ocr_status: OcrStatus = "pending"
    ocr_provider: str | None = None
    ocr_reason: str | None = None
    added_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionAttachments:
    """One per (user_id, session_id). Atomic JSON read/modify/write."""

    def __init__(self, user_id: str, session_id: str) -> None:
        self.user_id = validate_user_id(user_id)
        self.session_id = session_id
        self.path: Path = user_session_attachments_path(self.user_id, session_id)

    # --- write -------------------------------------------------------

    def add(
        self,
        *,
        sha256: str,
        filename: str,
        mime: str,
        size: int,
        source: SourceKind | None = None,
    ) -> SessionAttachment:
        """Append (or refresh) the row for ``sha256``. Idempotent.

        Same sha → same row; we refresh ``added_at`` and ``filename``
        because the user may have pasted the same blob under a
        different display name.
        """
        entry = SessionAttachment(
            sha256=sha256,
            filename=filename,
            mime=mime,
            size=size,
            source=source,
        )
        with file_lock(self._lock_path()):
            rows = self._load()
            rows = [r for r in rows if r.sha256 != sha256]
            rows.append(entry)
            self._persist(rows)
        return entry

    def mark_ocr_status(
        self,
        sha256: str,
        status: OcrStatus,
        *,
        provider: str | None = None,
        reason: str | None = None,
    ) -> SessionAttachment | None:
        with file_lock(self._lock_path()):
            rows = self._load()
            updated: SessionAttachment | None = None
            for i, row in enumerate(rows):
                if row.sha256 == sha256:
                    updated = row.model_copy(
                        update={
                            "ocr_status": status,
                            "ocr_provider": provider,
                            "ocr_reason": reason,
                        }
                    )
                    rows[i] = updated
                    break
            if updated is not None:
                self._persist(rows)
            return updated

    # --- read --------------------------------------------------------

    def list(self) -> list[SessionAttachment]:
        return self._load()

    def get(self, sha256: str) -> SessionAttachment | None:
        for row in self._load():
            if row.sha256 == sha256:
                return row
        return None

    # --- envelope helper ---------------------------------------------

    def to_manifest_attachments(self) -> list[Attachment]:
        """Promote rows to ``records.Attachment`` for use in a manifest."""
        out: list[Attachment] = []
        for row in self._load():
            out.append(
                Attachment(
                    sha256=row.sha256,
                    filename=row.filename,
                    mime=row.mime,
                    size=row.size,
                    ocr_status=row.ocr_status,
                    ocr_provider=row.ocr_provider,
                    source=row.source,
                )
            )
        return out

    # --- internals ---------------------------------------------------

    def _lock_path(self) -> Path:
        return self.path.with_suffix(".json.lock")

    def _load(self) -> list[SessionAttachment]:
        if not self.path.exists():
            return []
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return []
        if not isinstance(raw, list):
            return []
        out: list[SessionAttachment] = []
        for entry in raw:
            try:
                out.append(SessionAttachment.model_validate(entry))
            except Exception:  # noqa: BLE001
                continue
        return out

    def _persist(self, rows: list[SessionAttachment]) -> None:
        user_session_dir(self.user_id, self.session_id).mkdir(
            parents=True, exist_ok=True
        )
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                [r.model_dump(mode="json") for r in rows],
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        tmp.replace(self.path)
