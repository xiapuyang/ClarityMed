"""Upload bundle: data + validation for mixed-content /upload payloads.

A bundle is an ordered list of ``UploadPart``s, each one a single
self-contained unit of text that should land in the user's library.
Parts come from three sources:

* ``"text"`` — an inline text segment the user typed in the input bar
  (between or around placeholders).
* ``"image"`` — an attachment that began life as image bytes; the
  ``content`` field holds the OCR output.
* ``"file"`` — an attachment ingested as a file (PDF, doc, txt, …);
  the ``content`` field holds the extracted text.

The dataclasses here are pure value objects — they do no I/O. The
``build_upload_bundle`` resolver (separate module) is the one that
walks ``SessionAttachments`` + ``BlobStore`` and produces a populated
bundle.

Dedupe identity (``source_hash``) is the key the whole upload pipeline
keys off:

* For attachments it is the blob's SHA-256 (already content-addressed
  by ``BlobStore``).
* For inline text it is ``sha256(normalize_text(content))`` where
  ``normalize_text`` does NFKC + casefold + whitespace collapse so that
  trivial reformatting does not create a "new" duplicate.

Per-bundle dedupe (Layer 2) drops repeated ``source_hash`` values
during build. Cross-upload dedupe (Layer 3) lives in ``RagService`` and
uses the same key.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Literal

from claritymed import config as _cfg

PartKind = Literal["text", "image", "file"]
"""Source category of a part. Affects modal icons and status reporting,
not the ingest pipeline — by the time a part is built, everything is
text."""

PartStatus = Literal["ok", "low_content", "ocr_failed", "ocr_pending"]
"""Per-part status used by ``UploadBundle.validate``.

* ``ok`` — content is non-empty and clears ``min_part_chars``; eligible
  for ingest.
* ``low_content`` — has some text but below the per-part floor.
* ``ocr_failed`` — the OCR worker recorded a failure for this blob.
* ``ocr_pending`` — the OCR worker has not yet produced output. The
  bundle holds whatever ``BlobStore.read_extracted_text`` returned (may
  be empty); validation blocks until OCR finishes.
"""

# Inline-text dedupe: NFKC fold + casefold + collapse all whitespace runs
# to a single space. Punctuation is preserved — "Hello!" and "hello"
# are still considered different paragraphs. The point is to absorb
# trivial reformatting (extra newlines, casing of the first letter
# after a paste), not to do semantic dedupe (that's task 4's KNN job).
_WS_RUN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Return a stable form of ``text`` for hash comparison.

    NFKC normalises full-width / half-width forms (Chinese punctuation
    interop), ``casefold`` is the Unicode-aware lowercase, and the
    whitespace collapse soaks up newline / tab differences. The output
    is not meant to be human-readable — it only feeds the hash.
    """
    return _WS_RUN.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


def hash_inline_text(text: str) -> str:
    """Stable SHA-256 of an inline text segment after normalisation."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def count_meaningful_chars(text: str) -> int:
    """Count chars after stripping whitespace.

    Used by the validator as a content floor. We collapse whitespace
    first so a 200-char file of newlines doesn't pretend to be content.
    """
    return len(_WS_RUN.sub("", text))


@dataclass(frozen=True, slots=True)
class UploadPart:
    """One ingestable unit inside a bundle. Immutable by design.

    ``source_hash`` is the dedupe identity (see module docstring).
    ``content`` is the text that will be passed to the chunker — for
    images/files this is the OCR output, for inline text it's the
    segment verbatim.

    ``status`` is computed at build time, *not* at validate time. The
    builder is the one that talks to ``SessionAttachments`` /
    ``BlobStore`` and knows whether OCR completed; the validator just
    enforces gate rules based on the recorded status + char counts.
    This split keeps validation a pure function (testable in isolation
    without any store).
    """

    kind: PartKind
    source: str  # display label: filename for attachments, "inline" for text
    content: str
    source_hash: str
    status: PartStatus
    chars: int

    @classmethod
    def from_text(cls, content: str, *, source: str = "inline") -> "UploadPart":
        """Build an inline-text part. Status is derived from char count."""
        chars = count_meaningful_chars(content)
        status: PartStatus = (
            "ok" if chars >= _cfg.upload_min_part_chars() else "low_content"
        )
        return cls(
            kind="text",
            source=source,
            content=content,
            source_hash=hash_inline_text(content),
            status=status,
            chars=chars,
        )


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Outcome of ``UploadBundle.validate``.

    ``reasons`` carries human-readable lines the modal can render
    verbatim (already localised by the caller, since validate is pure
    and language-agnostic). When ``ok`` is true, ``reasons`` is empty
    and the bundle is safe to dispatch.
    """

    ok: bool
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class UploadBundle:
    """An ordered, deduped list of parts plus convenience accessors.

    Construction is via ``build_upload_bundle`` — this dataclass does
    not police its own invariants beyond what the type system enforces
    (it trusts the builder to have deduped by ``source_hash``).
    """

    parts: tuple[UploadPart, ...]

    @property
    def assembled_text(self) -> str:
        """Concatenate all part contents with a blank-line separator.

        The separator is two newlines so the chunker sees clear
        paragraph boundaries between parts of different provenance.
        Empty-content parts (pending OCR with no text yet) contribute
        nothing — they're surfaced via ``status`` but don't pollute the
        payload.
        """
        return "\n\n".join(p.content for p in self.parts if p.content.strip())

    @property
    def total_chars(self) -> int:
        """Sum of ``chars`` over every part — what the global floor compares against."""
        return sum(p.chars for p in self.parts)

    @property
    def has_failed_parts(self) -> bool:
        return any(p.status == "ocr_failed" for p in self.parts)

    @property
    def has_pending_parts(self) -> bool:
        return any(p.status == "ocr_pending" for p in self.parts)

    @property
    def has_low_content_parts(self) -> bool:
        return any(p.status == "low_content" for p in self.parts)

    def validate(self) -> ValidationResult:
        """Run the gate. Returns ``(ok, reasons)``.

        Reasons are short English fragments referencing the offending
        part by ``source``. The modal layer translates these via i18n
        keys keyed off the reason shape — see ``rag.upload.gate.*`` in
        ``configs/i18n/<lang>.yaml``.

        Empty bundles fail with a single ``empty`` reason rather than
        passing silently — "nothing to upload" should look the same as
        "everything was rejected" to the user.
        """
        if not self.parts:
            return ValidationResult(ok=False, reasons=["empty"])
        reasons: list[str] = []
        for part in self.parts:
            if part.status == "ocr_failed":
                reasons.append(f"ocr_failed:{part.source}")
            elif part.status == "ocr_pending":
                reasons.append(f"ocr_pending:{part.source}")
            elif part.status == "low_content":
                reasons.append(f"low_content:{part.source}")
        total = self.total_chars
        floor = _cfg.upload_min_total_chars()
        if total < floor:
            reasons.append(f"total_too_short:{total}/{floor}")
        return ValidationResult(ok=not reasons, reasons=reasons)
