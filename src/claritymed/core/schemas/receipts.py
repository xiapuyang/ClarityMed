"""Receipts returned by ingest and rag modes.

Distinct from ``GroundedAnswer`` (the ask-mode output): receipts confirm a
deterministic write to a store and never carry LLM-generated free text.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

IngestKind = Literal["profile", "lab_record", "vision_record", "timeline"]


class IngestRecord(BaseModel):
    """One persisted record within an ingest receipt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: IngestKind
    record_id: str = Field(min_length=1)


class IngestReceipt(BaseModel):
    """Output of ``ingest`` mode: what got written, and a deterministic summary.

    The summary is a short factual statement built from the record kinds (e.g.
    "2 lab records and 1 profile field saved") — never an LLM-generated free
    response. Numerical interpretation belongs in ``ask`` mode.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    records: list[IngestRecord] = Field(default_factory=list)
    summary: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class IngestionReceipt(BaseModel):
    """Output of ``rag`` mode: a document landed in the user RAG store.

    ``chunk_count`` is the number of *new* chunks written. Per-chunk
    cosine-similarity dedupe (``upload.dedupe_cosine_threshold``) can
    drop chunks that closely match existing content; those land in
    ``skipped_chunk_count`` instead. A document whose every chunk is a
    near-duplicate of existing material returns
    ``chunk_count=0, skipped_chunk_count=N > 0`` — callers treat that
    as "already in library" semantically equivalent to
    ``DuplicateDocumentError``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_id: str = Field(min_length=1)
    chunk_count: int = Field(ge=0)
    skipped_chunk_count: int = Field(default=0, ge=0)
    embedding_status: Literal["ok", "stub", "failed"] = "ok"
    public: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
