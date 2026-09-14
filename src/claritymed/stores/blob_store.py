"""Content-addressable blob storage for the per-user data tree.

``data/users/<id>/blobs/<sha[:2]>/<sha>/content.<ext>`` holds the raw bytes
of every attachment the user has ever uploaded. The directory is the unit
of deduplication: same bytes always hash to the same sha, so re-uploading
the same PDF resolves to the same directory and the same OCR cache.

Sibling files inside the blob directory:

* ``content.<ext>`` — original bytes (written by this module).
* ``ocr.md`` — extracted markdown text (written by ``OcrWorker``).
  For text-kind blobs (csv/md/txt/...) this file is NOT written —
  the source ``content.<ext>`` already is the text and the sentinel
  records ``kind: "text"`` so the reader knows to read it directly.
* ``ocr.json`` — completion sentinel ({ status, kind, ext, provider,
  chain_tried, ... } written by ``OcrWorker`` or the paste fast-path;
  the *presence* of this file is the "extraction complete" signal, so
  it must be written last via atomic rename to avoid half-completed
  states being mistaken for done).

Writes are atomic-by-rename: bytes go to ``content.<ext>.tmp`` first, then
``os.rename`` swaps them in. Idempotent: if ``content.<ext>`` already
exists for this sha, the second ``store`` is a no-op and returns the same
sha without rewriting (the bytes match by construction).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

from claritymed.stores.paths import (
    _validate_sha256,
    user_blob_dir,
    user_blob_path,
    validate_user_id,
)

logger = logging.getLogger(__name__)


class BlobStore:
    """One per user. Constructed lazily; the directory tree is created on
    first write so a fresh install has no empty ``blobs/`` to ship.
    """

    def __init__(self, user_id: str) -> None:
        self.user_id = validate_user_id(user_id)

    # --- write --------------------------------------------------------

    def store(self, content: bytes, ext: str) -> str:
        """Persist ``content`` under its sha256 and return the hex digest.

        ``ext`` is the file's natural extension (``"pdf"``, ``"png"``,
        ``"txt"``); we do not infer it because the upstream caller already
        knows it (paste handler from clipboard MIME, upload from filename
        suffix, etc.) and silently picking an ext from magic bytes risks
        a wrong assumption on encrypted / proprietary formats.

        Atomic-by-rename: never leaves a half-written ``content.<ext>``
        even on disk-full.
        """
        if not content:
            # The blob pool's deduplication relies on sha256 of meaningful
            # bytes; a zero-byte blob would all hash to the same id and
            # create a degenerate collision case. Reject loudly.
            raise ValueError("refusing to store zero-byte blob")

        sha = hashlib.sha256(content).hexdigest()
        target_dir = user_blob_dir(self.user_id, sha)
        target = user_blob_path(self.user_id, sha, ext)

        if target.exists():
            # Same bytes were stored before. Idempotent path: skip rewrite
            # so the caller does not race a concurrent OCR worker that is
            # actively reading from this directory.
            return sha

        target_dir.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        try:
            tmp.write_bytes(content)
            os.rename(tmp, target)
        except OSError:
            tmp.unlink(missing_ok=True)
            raise
        return sha

    def write_ocr_result(
        self,
        sha256: str,
        *,
        status: str,
        kind: str,
        ext: str,
        provider: str | None,
        chain_tried: list[str],
        reason: str | None,
        text: str,
        original_filename: str | None = None,
        modality: str | None = None,
        modality_confidence: float | None = None,
        is_medical: bool | None = None,
        ocr_has_report: bool | None = None,
        vision_warnings: list[str] | None = None,
    ) -> None:
        """Write the OCR/text-extraction sentinel for this blob.

        ``kind="ocr"``: an ``ocr.md`` file is written alongside the
        sentinel — the canonical OCR output path. ``kind="text"``: the
        source ``content.<ext>`` already IS the text, so writing
        ``ocr.md`` would duplicate the bytes for zero benefit; only the
        sentinel lands. Readers use :meth:`read_extracted_text` to be
        oblivious of which case occurred.

        ``ext`` is the source file's extension (without the dot) so the
        text reader can locate ``content.<ext>`` without scanning the
        directory.

        ``original_filename`` is the user-facing name at first ingest
        (clipboard.png, 化验单.png, "Lab Report 2026.pdf", ...). It is
        sanitized and stored under first-write-wins semantics — if an
        earlier sentinel already carried a name, keep it. The blob CAS
        deduplicates by bytes, so the *same* blob can be re-uploaded
        under different display names; ``SessionAttachments`` is the
        per-upload authority for naming, and this field is just a
        disaster-recovery anchor pointing at "what the user first
        called this blob."

        Order matters: ``ocr.md`` (when written) lands first, then
        ``ocr.json`` is the last rename. ``ocr_done`` checks
        ``ocr.json`` existence only, so a crash between the two leaves
        an inconsistent-but-recoverable state (``ocr.md`` orphan re-runs
        cleanly on next extraction).
        """
        from claritymed.core.filetype.safe_filename import safe_filename

        _validate_sha256(sha256)
        target_dir = user_blob_dir(self.user_id, sha256)
        target_dir.mkdir(parents=True, exist_ok=True)

        if kind == "ocr":
            ocr_md = self.ocr_path(sha256)
            tmp_md = ocr_md.with_suffix(".md.tmp")
            tmp_md.write_text(text or "", encoding="utf-8")
            tmp_md.replace(ocr_md)

        existing_name = self._existing_original_filename(sha256)
        safe_name = existing_name or safe_filename(original_filename)

        ocr_json = self.ocr_meta_path(sha256)
        tmp_json = ocr_json.with_suffix(".json.tmp")
        payload: dict = {
            "status": status,
            "kind": kind,
            "ext": ext,
            "provider": provider,
            "chain_tried": chain_tried,
            "reason": reason,
            "chars": len(text or ""),
        }
        if safe_name is not None:
            payload["original_filename"] = safe_name
        # Vision-tag fields (Unit 3, plan KTD-V3 + KTD-V6). Omitted when
        # the worker doesn't supply them — preserves the legacy sentinel
        # shape so back-compat readers don't trip on unknown keys, and
        # keeps non-image blobs (PDF, plain text) free of fields that
        # have no meaning for them.
        if modality is not None:
            payload["modality"] = modality
        if modality_confidence is not None:
            payload["modality_confidence"] = modality_confidence
        if is_medical is not None:
            payload["is_medical"] = is_medical
        if ocr_has_report is not None:
            payload["ocr_has_report"] = ocr_has_report
        if vision_warnings:
            payload["vision_warnings"] = list(vision_warnings)
        tmp_json.write_text(
            # ensure_ascii=False keeps CJK / emoji / accented names readable
            # on disk — the audit and recovery story leans on people
            # grepping these files, not on machines parsing them.
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_json.replace(ocr_json)

    def _existing_original_filename(self, sha256: str) -> str | None:
        """Return the prior sentinel's ``original_filename``, if any.

        Powers first-write-wins. Silent on every failure mode (missing
        file, malformed JSON, missing key) — the rewrite path will
        either supply a fresh name or write ``None``; we never raise
        from this read.
        """
        path = self.ocr_meta_path(sha256)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        name = data.get("original_filename") if isinstance(data, dict) else None
        return name if isinstance(name, str) and name else None

    def read_extracted_text(self, sha256: str) -> str:
        """Return the extracted-text content for a completed blob.

        Dispatches off the sentinel's ``kind`` so callers don't have
        to know whether a real OCR pass ran or a text fast-path was
        used. ``kind="text"`` reads ``content.<ext>`` directly;
        ``kind="ocr"`` (or any other / missing value for back-compat)
        reads ``ocr.md``.

        Raises ``FileNotFoundError`` if the sentinel is missing
        (caller forgot to check :meth:`ocr_done`) or the referenced
        source file disappeared.
        """
        _validate_sha256(sha256)
        sentinel_path = self.ocr_meta_path(sha256)
        sentinel = json.loads(sentinel_path.read_text(encoding="utf-8"))
        if sentinel.get("kind") == "text":
            ext = sentinel.get("ext", "")
            source = (
                self.dir(sha256) / f"content.{ext}" if ext else self.ocr_path(sha256)
            )
            # ``errors="replace"`` mirrors the writer side — a binary
            # file that snuck into the text fast-path still yields
            # something readable rather than crashing the rendering
            # pipeline.
            return source.read_text(encoding="utf-8", errors="replace")
        return self.ocr_path(sha256).read_text(encoding="utf-8")

    # --- read paths (do not assert existence; callers check) ----------

    def path(self, sha256: str, ext: str) -> Path:
        """Compose ``content.<ext>`` for a sha — does NOT assert it exists."""
        return user_blob_path(self.user_id, sha256, ext)

    def dir(self, sha256: str) -> Path:
        """The full blob directory for one sha."""
        return user_blob_dir(self.user_id, sha256)

    def ocr_path(self, sha256: str) -> Path:
        """``<blob_dir>/ocr.md`` — extracted text. May not exist yet."""
        return user_blob_dir(self.user_id, sha256) / "ocr.md"

    def ocr_meta_path(self, sha256: str) -> Path:
        """``<blob_dir>/ocr.json`` — the completion sentinel.

        Whoever writes this file is the one signalling "OCR done."  The
        worker writes ``ocr.md.tmp`` + ``ocr.json.tmp`` first, renames
        ``ocr.md`` into place, then renames ``ocr.json`` last. Any reader
        consults this file's presence — never ``ocr.md`` alone.
        """
        return user_blob_dir(self.user_id, sha256) / "ocr.json"

    def read_ocr_metadata(self, sha256: str) -> dict | None:
        """Return the parsed ``ocr.json`` contents, or ``None`` if absent.

        Used by readers that need the vision-tag fields (``modality``,
        ``is_medical``, ``ocr_has_report``) without separately copying
        them into ``SessionAttachments``. Silent on corruption: returns
        ``None`` so callers fall through to the legacy "no extra attrs"
        path rather than crashing the prompt-assembly step on a malformed
        sentinel.
        """
        _validate_sha256(sha256)
        path = self.ocr_meta_path(sha256)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def ocr_done(self, sha256: str) -> bool:
        """True iff the sentinel ``ocr.json`` exists.

        Read-only; safe to call from any thread. Used by the OCR worker
        to skip re-extraction of an already-processed blob, and by the
        envelope renderer to decide whether to inline OCR text or a
        ``<note>OCR pending</note>`` placeholder.
        """
        _validate_sha256(sha256)
        return self.ocr_meta_path(sha256).exists()

    # --- introspection (used by audit + reconciliation) ---------------

    def exists(self, sha256: str) -> bool:
        """True iff *any* ``content.*`` file exists for this sha."""
        d = self.dir(sha256)
        if not d.exists():
            return False
        return any(
            p.name.startswith("content.") and not p.name.endswith(".tmp")
            for p in d.iterdir()
        )


def make_blob_store(user_id: str) -> BlobStore:
    """Factory matching ``stores/user_rag.py`` convention."""
    return BlobStore(user_id)
