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
    doc_title: str | None = Field(
        None, description="Human-readable document title for citation display."
    )
    ingested_at: datetime | None = Field(None)

    # --- hybrid-retrieval fields (Unit 1 of RAG plan) -------------------
    # All optional so existing v1 callers (UserRagStore stub, mode plan
    # tests) keep validating; populated by HybridRetriever in Unit 6.

    collection_name: str | None = Field(
        None,
        description=(
            "Source collection (e.g. ``statpearls_en``, ``user_rag_alice``). "
            "Used by ParentStore lookup and citation grouping."
        ),
    )
    parent_id: str | None = Field(
        None,
        description=(
            "Parent chunk id (parent-child layout). ``None`` for legacy single-"
            "level chunks."
        ),
    )
    parent_text: str | None = Field(
        None,
        description=(
            "Hydrated parent chunk text. Filled by HybridRetriever after a "
            "ParentStore lookup; prompt assembly uses parent_text when "
            "present, falling back to text otherwise."
        ),
    )
    dense_score: float | None = Field(
        None, description="Dense (bge-m3 dense) similarity score, when known."
    )
    sparse_score: float | None = Field(
        None, description="Sparse (bge-m3 lexical) similarity score, when known."
    )
    rerank_score: float | None = Field(
        None,
        description=(
            "Cross-encoder rerank score (e.g. bge-reranker-v2-m3). "
            "Authoritative ranking signal after rerank stage."
        ),
    )
