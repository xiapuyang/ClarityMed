"""Event types streamed by services to UI callers.

Frozen Pydantic models so the TUI can do exhaustive ``match`` over the
``Event`` union without missing a case. Schema is intentionally narrow:
audit-safe (no PHI in payloads), small enough to flow through a Textual
``Worker`` without serialization overhead.
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from claritymed.core.schemas.answer import Language
from claritymed.core.symptoms.schemas import SeverityTier

# Confidence bucket for the multi-card renderer. Kept as a Literal so
# the schema stays typed; the display label + threshold floats live in
# ``configs/i18n/<lang>/symptoms.yaml`` under
# ``symptoms.confidence.labels.*`` / ``symptoms.confidence.thresholds.*``.
ConfidenceBucket = Literal["low", "moderate", "high", "very_high"]

ErrorType = Literal[
    "retrieval_failed",
    "llm_error",
    "phi_violation",
    "permission_denied",
    "config_error",
    "user_cancelled",
    "scrub_unavailable",
]

ModeName = Literal["ask"]


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


class RetrievalPending(_EventBase):
    """RAG retrieval kicked off — the embed + search + rerank pipeline has
    started but bundle.trace metadata isn't available yet. Drives an
    immediate 'Retrieving…' hint in the UI so the user has feedback
    during the slow path (typically 1-5s on local Qdrant + reranker).
    Distinct from ``RetrievalStarted`` which fires *after* retrieve()
    returns and carries the actual collection list.
    """

    type: Literal["retrieval_pending"] = "retrieval_pending"


class RetrievalStarted(_EventBase):
    """RAG retrieval completed at the strategy layer; reports which
    collections were searched. Despite the name, this fires *after*
    ``await strategy.retrieve(ctx)`` returns — ``active_collections`` is
    populated from the router decision inside the bundle trace.
    Consumers wanting a pre-retrieve signal should watch
    ``RetrievalPending`` instead.
    """

    type: Literal["retrieval_started"] = "retrieval_started"
    active_collections: list[str] = []
    strategy: str = "naive_hybrid"


class LlmCallStarted(_EventBase):
    """LLM streaming call begun. Drives a 'generating response…' UI hint
    so the user knows the slow path is the model, not a stuck pipeline.
    Local 30B+ models on Apple Silicon can take 30-180s to first token —
    without this event the assistant bubble sits empty for that whole
    stretch and looks frozen.
    """

    type: Literal["llm_call_started"] = "llm_call_started"
    model_name: str = ""
    provider_id: str = ""


class LlmFirstToken(_EventBase):
    """First token arrived. Carries TTFT in ms so the UI can mark the
    LlmCallStarted step complete and report wall-clock time-to-first-token.
    """

    type: Literal["llm_first_token"] = "llm_first_token"
    ttft_ms: int = 0


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


class TokensUsed(_EventBase):
    """Per-turn LLM token usage.

    Emitted by ask mode just before ``Done`` so a SPA can render a
    ``12k / 128k`` context-window ratio next to the model picker. All
    fields default to zero — emitters that don't know one of these
    values (local providers that omit usage) leave the field at 0 and
    the consumer treats it as "unknown".
    """

    type: Literal["tokens_used"] = "tokens_used"
    model_name: str = ""
    provider_id: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    context_window: int = 0


class DifferentialSessionMeta(BaseModel):
    """Session flags relevant to the multi-card renderer above-cards banner.

    Server-computed ``banner_key`` is an i18n lookup key the frontend
    consumes with one dict lookup — no branching logic. ``None`` means
    "no banner" (normal completion). See
    ``symptoms_plugin/_card_builder.py:build_session_meta`` for the
    flag → key mapping.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    cancelled: bool = False
    hit_cap: bool = False
    meets_confidence_threshold: bool = True
    severity_override: bool = False
    max_low_severity_seen: int | None = None
    banner_key: str | None = None


class DifferentialCard(BaseModel):
    """One card in the differential-diagnosis card list.

    Everything a card needs to render is on this payload — the frontend
    does no lookups. Confidence bucketing + directional headline +
    curated report are all server-hydrated so the client stays
    presentation-only. Per-condition ``suggestion`` / ``citations`` are
    intentionally not on the card — the LLM composes the aggregate
    summary paragraph (via ``symptoms_final_reply``) that renders
    around the card list and carries the actionable guidance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # From tool payload (DifferentialRow) — canonical/authoritative:
    condition_id: str = Field(min_length=1)
    condition_name: str = Field(min_length=1)
    probability: float = Field(ge=0.0, le=1.0)
    severity_tier: SeverityTier

    # Derived / curated:
    confidence_bucket: ConfidenceBucket
    confidence_label: str = Field(min_length=1)
    # ``None`` for standard rank 2+ cards — the card title already carries
    # ``{condition_name} — {confidence_label} · {probability}%`` so the
    # previous ``Also consider {condition}`` template was a bold duplicate.
    # Rank 1 always has a directional cue (Likely/Probably/Consider); rank
    # 2+ keeps it only for the Critical/Urgent "unlikely but rule out"
    # safety variant. Renderers must skip a ``None`` headline entirely.
    headline: str | None = Field(default=None, min_length=1)
    report: str = Field(min_length=1)


class DifferentialReady(_EventBase):
    """Sidecar event emitted once by symptoms_plugin after ``_run_sub_session``
    returns a usable differential.

    Fully hydrated payload — frontend renders directly. Not emitted on
    ``user_declined`` / ``eligible=False`` / ``server_error`` /
    ``session_expired`` branches (those fall through to a normal
    free-text reply and never build cards).
    """

    type: Literal["differential_ready"] = "differential_ready"
    cards: list[DifferentialCard]
    session: DifferentialSessionMeta
    language: Language


class InteractionRequested(_EventBase):
    """A tool wants the user to answer something or approve a write.

    ``kind`` selects which channel raised the request:

    * ``ask_user_question`` — payload is the JSON dump of
      :class:`~claritymed.core.interaction.schemas.AskUserQuestionInput`.
    * ``tool_approval`` — payload carries ``tool_name``, ``args``, and
      a short breadcrumb (``"Tool 2/5"``) the UI can show.

    The turn stream pauses on this event; the host UI POSTs to
    ``/sessions/{id}/interactions/{interaction_id}`` to resume.
    Channels in headless/CLI contexts never emit this event — the
    rendezvous lives entirely inside the web layer.
    """

    type: Literal["interaction_requested"] = "interaction_requested"
    interaction_id: str
    kind: Literal["ask_user_question", "tool_approval"]
    payload: dict[str, Any] = {}


Event = Union[
    ToolStarted,
    ToolCompleted,
    TokenChunk,
    RetrievalPending,
    RetrievalStarted,
    RetrievalCompleted,
    RetrievalFiltered,
    LlmCallStarted,
    LlmFirstToken,
    Cancelled,
    Done,
    Error,
    TokensUsed,
    InteractionRequested,
    DifferentialReady,
]
