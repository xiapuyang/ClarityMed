"""Content-addressable blob storage for the per-user data tree.

``data/users/<id>/blobs/<sha[:2]>/<sha>/content.<ext>`` holds the raw bytes
of every attachment the user has ever uploaded. The directory is the unit
of deduplication: same bytes always hash to the same sha, so re-uploading
the same PDF resolves to the same directory and the same OCR cache.

Sibling files inside the blob directory:

* ``content.<ext>`` — original bytes (written by this module).
* ``ocr.md`` — extracted markdown text (written by ``OcrWorker``).
* ``ocr.json`` — completion sentinel ({ status, provider, chain_tried, ... }
  written by ``OcrWorker``; the *presence* of this file is the
  "extraction complete" signal, so it must be written last via atomic
  rename to avoid half-completed states being mistaken for done).

Writes are atomic-by-rename: bytes go to ``content.<ext>.tmp`` first, then
``os.rename`` swaps them in. Idempotent: if ``content.<ext>`` already
exists for this sha, the second ``store`` is a no-op and returns the same
sha without rewriting (the bytes match by construction).
"""

from __future__ import annotations

import hashlib
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
