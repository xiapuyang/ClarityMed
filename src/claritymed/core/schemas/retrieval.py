"""Retrieval-layer schemas.

``RetrievedChunk`` is the unit returned by any RAG retriever (system or
per-user). It carries the chunk text plus the metadata that the PHI guard
and the prompt assembler need to make decisions: was this chunk PHI? is it
safe to send to a cloud LLM? where did it come from?
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Source = Literal["system_rag", "user_rag"]


class RetrievedChunk(BaseModel):
    """One chunk returned by a retriever, with the metadata downstream layers
    need (PHI filtering, citation, dedup).
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    source: Source = Field(
        ...,
        description=(
            "Which retriever produced this chunk. user_rag chunks default to "
            "is_phi=True; system_rag chunks default to is_phi=False."
        ),
    )
    score: float = Field(..., description="Retriever similarity score.")
    doc_id: str = Field(..., description="Stable id of the source document.")
    chunk_index: int = Field(0, ge=0)
    is_phi: bool = Field(
        True,
        description=(
            "True when the chunk may contain PHI (default for any free-text "
            "user upload). Cloud LLM path will filter these unless can_cloud."
        ),
    )
    can_cloud: bool = Field(
        False,
        description=(
            "True when the user has explicitly marked the source doc as "
            "public reference material (e.g. a published paper)."
        ),
    )
    user_id: int | None = Field(
        None,
        description="Owning user id for user_rag chunks; None for system_rag.",
    )
    source_uri: str | None = Field(
        None, description="Optional pointer to the original source for citation."
    )
    ingested_at: datetime | None = Field(None)
