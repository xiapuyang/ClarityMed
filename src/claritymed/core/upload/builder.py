"""Build ``UploadBundle`` instances from raw input-bar text.

The builder is the only module that talks to ``SessionAttachments``
and ``BlobStore`` for the upload pipeline. Everything downstream
(``UploadBundle.validate``, the modal preview, ``RagService``) sees
only the resolved value objects.

Input shape — what the input bar contains when the user types
``/upload``:

    "These are my notes on the paper:
    [File sha:abc12345]

    And here's the figure I'm asking about:
    [Image sha:def67890]"

Output — one ``UploadPart`` per logical unit, in document order:

    [
        UploadPart(kind="text", content="These are my notes…", …),
        UploadPart(kind="file", content="<paper OCR text>", …),
        UploadPart(kind="text", content="And here's the figure…", …),
        UploadPart(kind="image", content="<figure OCR text>", …),
    ]

Per-bundle dedupe (Layer 2 from the design): if the same placeholder
appears twice in the input (or the same inline-text segment is
repeated verbatim modulo whitespace), only the first occurrence
contributes a part.

Unresolvable placeholders (no session, bad prefix, ambiguous prefix)
are silently treated as inline text — same fallback shape that
``AttachmentsFeature.expand_placeholders`` uses, so the LLM still sees
the literal placeholder. The user will see "total_too_short" or
similar from the validator if this leaves the bundle below the floor.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from claritymed import config as _cfg
from claritymed.core.upload.bundle import (
    PartKind,
    PartStatus,
    UploadBundle,
    UploadPart,
    count_meaningful_chars,
    hash_inline_text,
)
from claritymed.stores.blob_store import BlobStore
from claritymed.stores.session_attachments import (
    SessionAttachment,
    SessionAttachments,
)

logger = logging.getLogger(__name__)

# Mirrors ``AttachmentsFeature._PLACEHOLDER_RE``. The capture groups
# expose the kind ("Image"/"File") + the 8-char sha prefix.
_PLACEHOLDER_RE = re.compile(r"\[(Image|File) sha:([0-9a-f]+)\]")


def build_upload_bundle(
    text: str,
    *,
    user_id: str,
    session_id: str | None,
) -> UploadBundle:
    """Parse ``text`` into an ordered, deduplicated ``UploadBundle``.

    ``session_id=None`` disables placeholder resolution — useful for
    the path-mode entry point (``/upload /path/to/file.txt``) where
    the caller already loaded the file content and there are no
    placeholders to expand. In that mode the entire ``text`` becomes a
    single inline-text part.

    Returns an empty bundle when ``text`` is empty/whitespace-only.
    Callers should run ``bundle.validate()`` before dispatching.
    """
    if not text or not text.strip():
        return UploadBundle(parts=())

    rows_by_prefix: dict[str, list[SessionAttachment]] = {}
    if session_id is not None:
        try:
            rows = SessionAttachments(user_id, session_id).list()
        except Exception:  # noqa: BLE001 — store I/O isn't fatal here
            logger.exception("upload bundle: session attachments read failed")
            rows = []
        for row in rows:
            rows_by_prefix.setdefault(row.sha256[:8], []).append(row)

    blob_store = BlobStore(user_id) if rows_by_prefix else None
    parts: list[UploadPart] = []
    seen_hashes: set[str] = set()

    cursor = 0
    for match in _PLACEHOLDER_RE.finditer(text):
        # Inline-text segment before this placeholder (if any).
        if match.start() > cursor:
            segment = text[cursor : match.start()]
            _append_text_part(segment, parts, seen_hashes)

        # Placeholder itself — resolve or fall through as text.
        kind_raw, prefix = match.group(1), match.group(2)
        hits = rows_by_prefix.get(prefix, [])
        if len(hits) == 1 and blob_store is not None:
            part = _resolve_attachment_part(blob_store, kind_raw, hits[0])
            if part.source_hash not in seen_hashes:
                seen_hashes.add(part.source_hash)
                parts.append(part)
        else:
            # No unique match: treat the literal placeholder as
            # inline text so the user can see in the assembled
            # preview that something didn't resolve.
            _append_text_part(match.group(0), parts, seen_hashes)
        cursor = match.end()

    # Trailing inline text.
    if cursor < len(text):
        _append_text_part(text[cursor:], parts, seen_hashes)

    return UploadBundle(parts=tuple(parts))


def build_path_mode_bundle(
    path: Path,
    *,
    max_bytes: int,
    min_part_chars: int,
) -> tuple[UploadBundle | None, str | None]:
    """Read ``path`` off disk and wrap it as a one-part bundle.

    Exactly one of the returned values is non-None:

    * ``(bundle, None)`` on success — caller dispatches it.
    * ``(None, error_message)`` on a size or decode failure — caller
      surfaces ``error_message`` (a short English fragment) as a toast.

    Keeping this function pure (no ``self._toast`` side effects) means
    the App layer can test the gate logic without spinning up a
    Textual app. The size + decode gates mirror the old in-modal
    behaviour so existing oversize-file UX is preserved.
    """
    try:
        size = path.stat().st_size
    except OSError as exc:
        return None, f"Stat failed: {exc}"
    if size > max_bytes:
        return None, f"File too large: {size} bytes (limit {max_bytes} bytes)."
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return None, f"Read failed: {exc}"

    chars = count_meaningful_chars(text)
    status: PartStatus = "ok" if chars >= min_part_chars else "low_content"
    bundle = UploadBundle(
        parts=(
            UploadPart(
                kind="file",
                source=path.name,
                content=text,
                source_hash=hash_inline_text(text),
                status=status,
                chars=chars,
            ),
        )
    )
    return bundle, None


def _append_text_part(
    segment: str,
    parts: list[UploadPart],
    seen_hashes: set[str],
) -> None:
    """Append an inline-text part if ``segment`` carries enough content.

    Three cases produce nothing:

    1. Whitespace-only segments — natural gaps between placeholders.
    2. Sub-threshold text segments (chars < ``upload.min_part_chars``)
       — typically connector phrases like "and " or "Here's the file:"
       that the user wrote between placeholders. Silently skipped, not
       surfaced as a ``low_content`` gate reason, because the user did
       not deliberately ask to upload them. Attachments (image/file)
       below the same floor stay as ``low_content`` since the user
       explicitly attached them.
    3. Duplicates of an earlier ``source_hash`` — collapses repeated
       paragraphs.
    """
    if not segment.strip():
        return
    part = UploadPart.from_text(segment, source="inline")
    if part.status == "low_content":
        return
    if part.chars < _cfg.upload_min_part_chars():
        # Defensive: ``from_text`` already gates on the same floor, but
        # this keeps the policy explicit at the call site and shields
        # against a future change in ``from_text``.
        return
    if part.source_hash in seen_hashes:
        return
    seen_hashes.add(part.source_hash)
    parts.append(part)


def _resolve_attachment_part(
    blob_store: BlobStore,
    kind_raw: str,
    row: SessionAttachment,
) -> UploadPart:
    """Build a Part for a single attachment row.

    Maps ``SessionAttachment.ocr_status`` to ``PartStatus``:

    * ``done`` → read ``ocr.md`` from disk; status is ``ok`` when the
      char count clears the per-part floor, else ``low_content``.
    * ``pending`` → empty content, status ``ocr_pending``.
    * ``failed`` → empty content, status ``ocr_failed``. The caller's
      preview surfaces the original ``ocr_reason`` when available.
    * ``empty`` (sentinel says "nothing extractable") → ``low_content``
      so the gate blocks instead of pretending OCR succeeded with no
      text.

    The blob sha256 is the dedupe key. ``content`` for failed / pending
    parts is empty string, not the raw placeholder — that's the
    builder's job, not this helper's.
    """
    kind = "image" if kind_raw == "Image" else "file"
    sha = row.sha256

    if row.ocr_status == "done":
        try:
            content = blob_store.read_extracted_text(sha)
        except (OSError, ValueError) as exc:
            logger.warning(
                "upload bundle: OCR read failed for sha=%s: %s", sha[:8], exc
            )
            return _attachment_part(
                kind=kind,
                source=row.filename,
                source_hash=sha,
                content="",
                status="ocr_failed",
            )
        chars = count_meaningful_chars(content)
        status: PartStatus = (
            "ok" if chars >= _cfg.upload_min_part_chars() else "low_content"
        )
        return UploadPart(
            kind=kind,
            source=row.filename,
            content=content,
            source_hash=sha,
            status=status,
            chars=chars,
        )

    status_map: dict[str, PartStatus] = {
        "pending": "ocr_pending",
        "failed": "ocr_failed",
        "empty": "low_content",
    }
    return _attachment_part(
        kind=kind,
        source=row.filename,
        source_hash=sha,
        content="",
        status=status_map.get(row.ocr_status, "ocr_pending"),
    )


def _attachment_part(
    *,
    kind: PartKind,
    source: str,
    source_hash: str,
    content: str,
    status: PartStatus,
) -> UploadPart:
    """Construct an attachment Part with derived char count."""
    return UploadPart(
        kind=kind,
        source=source,
        content=content,
        source_hash=source_hash,
        status=status,
        chars=count_meaningful_chars(content),
    )
