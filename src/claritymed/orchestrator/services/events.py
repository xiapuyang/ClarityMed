"""Event types streamed by services to UI callers.

Frozen Pydantic models so the TUI can do exhaustive ``match`` over the
``Event`` union without missing a case. Schema is intentionally narrow:
audit-safe (no PHI in payloads), small enough to flow through a Textual
``Worker`` without serialization overhead.
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict

ErrorType = Literal[
    "retrieval_failed",
    "llm_error",
    "phi_violation",
    "permission_denied",
    "config_error",
    "user_cancelled",
]

ModeName = Literal["ingest", "ask", "rag"]


class _EventBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ToolStarted(_EventBase):
    type: Literal["tool_started"] = "tool_started"
    tool_name: str
    args_preview: str = ""


class ToolCompleted(_EventBase):
    type: Literal["tool_completed"] = "tool_completed"
    tool_name: str
    duration_ms: int = 0
    summary: str = ""


class TokenChunk(_EventBase):
    """Streamed token from ask mode's LLM call."""

    type: Literal["token_chunk"] = "token_chunk"
    text: str


class RetrievalFiltered(_EventBase):
    """Counts only — no chunk content. Drives the 'N sources filtered' UI hint."""

    type: Literal["retrieval_filtered"] = "retrieval_filtered"
    total: int
    kept: int
    filtered_phi: int
    reason: str = ""


class RetrievalStarted(_EventBase):
    """RAG retrieval begun. Drives the 'searching N collections' UI hint."""

    type: Literal["retrieval_started"] = "retrieval_started"
    active_collections: list[str] = []
    strategy: str = "naive_hybrid"


class RetrievalCompleted(_EventBase):
    """RAG retrieval finished. ``trace_summary`` is a small dict the TUI
    can render directly — no chunk content (audit-safe)."""

    type: Literal["retrieval_completed"] = "retrieval_completed"
    num_chunks: int
    fallback_triggered: bool = False
    rerank_fallback: bool = False
    embed_ms: int = 0
    search_ms: int = 0
    rerank_ms: int = 0
    parent_expand_ms: int = 0


class ModeRouted(_EventBase):
    """Router decided the mode for this turn. Drives the status-bar flash."""

    type: Literal["mode_routed"] = "mode_routed"
    detected_mode: ModeName
    confidence: float
    source: Literal["rule", "llm", "explicit"]
    reason: str = ""


class Cancelled(_EventBase):
    type: Literal["cancelled"] = "cancelled"
    reason: str = ""


class Done(_EventBase):
    """Final payload. ``final`` is the mode-specific receipt or text."""

    type: Literal["done"] = "done"
    final: Any = None


class Error(_EventBase):
    type: Literal["error"] = "error"
    error_type: ErrorType
    message: str
    retryable: bool = False


Event = Union[
    ToolStarted,
    ToolCompleted,
    TokenChunk,
    RetrievalStarted,
    RetrievalCompleted,
    RetrievalFiltered,
    ModeRouted,
    Cancelled,
    Done,
    Error,
]
