"""Pydantic contracts for ``manifest.yaml`` (records + library scope).

A record is one event: a checkup, a lab panel, an imaging session. The
manifest is the metadata wrapper around its CAS-stored attachments.  Library
entries reuse the same shape with a slightly different content profile
(``papers`` / ``articles`` instead of ``exam-reports`` etc.); a single
``Manifest`` model spans both so ManifestStore can write either with a
``scope`` parameter rather than two parallel models.

Why frozen=True: a manifest is the source of truth for a record on disk. The
in-memory model is read once, mutated in a ``mutator(dict)`` callback by
ManifestStore.update, then re-validated — never edited in place — so the
optimistic-lock revision counter has a single write boundary.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Embed-pipeline status. Set on ``save_record`` after the manifest is written:
# ``ok`` when the Qdrant chunk landed; ``pending_retry`` when the embed call
# failed (startup reconcile re-embeds); ``pending_delete`` when the user
# requested deletion but the Qdrant cascade hasn't completed yet (used by the
# reconcile scan to distinguish ghost-from-failed-delete vs.
# pending-retry-embed). Default ``ok`` keeps existing fixtures simple.
EmbedStatus = Literal["ok", "pending_retry", "pending_delete"]

# OCR completion state for one attachment. The lifecycle is single-arrow:
# pending → (done | empty | failed | timeout). The completion sentinel on
# disk is the existence of ``ocr.json`` next to ``content.<ext>``.
OcrStatus = Literal["pending", "done", "empty", "failed", "timeout"]

# Source flag — where the attachment came from. Used by the TUI to render the
# correct chip glyph and by audit rows. Free-form choice rather than enum so
# new channels can be added without a migration.
AttachmentSource = Literal["paste", "upload", "slash_command", "cli"]


class Attachment(BaseModel):
    """One file referenced from a manifest. Lives in the CAS blob pool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    filename: str = Field(min_length=1, max_length=255)
    mime: str = Field(min_length=1, max_length=128)
    size: int = Field(ge=0)
    ocr_status: OcrStatus = "pending"
    ocr_provider: str | None = None
    source: AttachmentSource | None = None


class ExtractedLab(BaseModel):
    """One lab value pulled out of a record by the LLM.

    Mirrors ``LabValue`` but stays in the manifest rather than being upserted
    to ``profile.db`` — per the v1 brainstorm decision S? (Q2 resolution):
    extracted labs stay file-shaped; SQLite promotion is deferred.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=128)
    value: float | str
    unit: str | None = Field(default=None, max_length=32)
    loinc: str | None = Field(default=None, max_length=32)
    ref_low: float | None = None
    ref_high: float | None = None
    flag: str | None = None


class Manifest(BaseModel):
    """One event's metadata. Persisted as ``manifest.yaml`` on disk.

    The ``kind`` field is the LLM-supplied event type (``exam-report``,
    ``lab-report``, ``prescription``, ``paper``, ``article``…); ``category``
    is the on-disk folder one level up (e.g. ``exam-reports``). They diverge
    so renaming a folder doesn't require touching every event's ``kind``.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    revision: int = Field(ge=1, description="Monotonic optimistic-lock counter.")
    schema_version: int = Field(default=1, ge=1)
    kind: str = Field(min_length=1, max_length=64)
    category: str = Field(min_length=1, max_length=64)
    slug: str = Field(min_length=1, max_length=128)
    # On-disk YAML uses ``date:`` but the Python attribute is ``event_date`` so
    # the type ``datetime.date`` isn't shadowed by the field name. Alias keeps
    # the YAML readable for humans without renaming the field at the file
    # layer.
    event_date: date | None = Field(default=None, alias="date")
    title: str = Field(min_length=1, max_length=256)
    provider: str | None = Field(default=None, max_length=128)
    attachments: list[Attachment] = Field(default_factory=list)
    extracted_labs: list[ExtractedLab] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None
    # Library-only metadata. Records leave these unset.
    authors: list[str] = Field(default_factory=list)
    year: int | None = Field(default=None, ge=1800, le=2200)
    public: bool = False
    # Embed pipeline metadata. Underscore prefix matches the on-disk
    # convention for internal fields the LLM should not surface.
    embed_status: EmbedStatus = Field(default="ok", alias="_embed_status")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
