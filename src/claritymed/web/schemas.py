"""Pydantic request / response models for the web v1 API.

Every endpoint declares its request and response model here (or in a
per-router schema module). FastAPI's auto-generated ``/openapi.json``
becomes the published contract; the frontend hand-mirrors these shapes
in TypeScript until codegen lands with the first admin-endpoint PR.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from claritymed.core.schemas.account import Language, Role

# Mirrors ``stores.paths.USER_ID_RE``. Declared in regex form for
# Pydantic so a malformed user_id arrives back as 422 with a clear
# constraint message rather than tripping the store's validator with a
# 500.
USER_ID_PATTERN = r"^[a-zA-Z0-9_-]{1,32}$"


class ErrorEnvelope(BaseModel):
    """Common error body. FastAPI's default 4xx/5xx body uses ``{"detail"}``
    too — declaring it as a schema lets OpenAPI consumers pick it up.
    """

    model_config = ConfigDict(extra="forbid")

    detail: str


# --- /auth/login -------------------------------------------------------


class LoginRequest(BaseModel):
    """Credentials body for ``POST /auth/login``.

    ``user_id`` is regex-validated so the store layer never sees a
    path-traversal attempt. ``password`` is a plain string with no
    upper bound; bcrypt itself caps at 72 bytes and silently truncates
    anything longer (passlib emits a deprecation warning).
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(pattern=USER_ID_PATTERN)
    password: str = Field(min_length=1)


class LoginResponse(BaseModel):
    """Successful-login response body.

    The JWT itself is NEVER in the body — it lives only in the
    ``Set-Cookie: access_token`` header with ``HttpOnly``. The body
    carries the Account fields the SPA needs to render the header /
    nav bar without an extra ``GET /api/v1/me`` round-trip.
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str
    display_name: str
    role: Role
    language: Language


# --- /auth/logout: no body, no response model -------------------------
# 204 No Content; cookie cleared via Set-Cookie.

LogoutMethod = Literal["POST"]


# --- /api/v1/me --------------------------------------------------------


class AccountResponse(BaseModel):
    """Shape returned by ``GET /api/v1/me`` and ``PATCH /api/v1/me``.

    Mirrors :class:`claritymed.core.schemas.account.Account` minus the
    timestamps and the future-PHI fields. ``provider_id`` is exposed
    because the SPA chrome surfaces it ("Connected to: Ollama"); the
    role is read-only via this endpoint (admin promotion is out-of-band
    via direct YAML edit per the MVP scope).
    """

    model_config = ConfigDict(extra="forbid")

    user_id: str
    display_name: str
    role: Role
    language: Language
    provider_id: str | None = None


# --- /api/v1/sessions --------------------------------------------------


# Mirrors ``ChatSession._path_for``'s validation: no slashes, no leading
# dot, length <= 128. The Pydantic regex is the API-boundary version of
# the same constraint; both layers stay in agreement.
SESSION_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$"


class SessionMetaResponse(BaseModel):
    """One entry in ``GET /api/v1/sessions``.

    Mirrors :class:`claritymed.orchestrator.services.chat_session.SessionMeta`
    minus the on-disk ``path`` (an implementation detail) and ``size_bytes``
    (frontend doesn't need it for MVP).
    """

    model_config = ConfigDict(extra="forbid")

    session_id: str
    preview: str
    # ``datetime`` so OpenAPI emits ``format: date-time`` and the SPA
    # codegen receives a typed Date. Pydantic v2 serializes to ISO-8601
    # in JSON, matching the previous wire shape exactly.
    modified_at: datetime


class NewSessionResponse(BaseModel):
    """Returned by ``POST /api/v1/sessions``."""

    model_config = ConfigDict(extra="forbid")

    session_id: str


class TurnResponse(BaseModel):
    """One turn in ``GET /api/v1/sessions/{id}/turns``.

    Projection of ``ChatTurn`` plus a ``cancelled`` flag so the SPA can
    render half-finished assistant bubbles with the right state.

    ``differential`` is the multi-card renderer's sidecar payload
    persisted on symptoms turns; ``None`` for every other turn.
    Serialized as a plain dict so the wire schema doesn't pull
    ``DifferentialReady`` into every downstream typing dependency —
    the SPA reconstitutes it against its own hand-mirrored TS types.
    """

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant", "system"]
    text: str
    cancelled: bool = False
    differential: dict[str, Any] | None = None


class StreamRequest(BaseModel):
    """Body of ``POST /api/v1/sessions/{id}/stream``.

    ``q`` is the user's question — PHI lives in the request body (not
    the URL) so uvicorn's access log never captures it. The 8000-char
    cap matches the plan's chosen bound; longer questions should be
    broken into multiple turns.

    ``attachment_ids`` references blobs already uploaded for this
    session via ``POST /sessions/{id}/attachments``. The backend
    converts each id into the ``[Image sha:…]`` placeholder text the
    LLM expansion path expects so the frontend never needs to know
    about placeholders.

    ``provider_id`` is a per-turn override of the persisted
    ``Account.provider_id``. Unset → resolve via account preference →
    catalog default. Persisting a switch happens via PATCH /api/v1/me;
    this field is for one-off "try another model" without changing the
    default.
    """

    model_config = ConfigDict(extra="forbid")

    q: str = Field(min_length=1, max_length=8000)
    attachment_ids: list[str] = Field(
        default_factory=list,
        max_length=8,
        description=(
            "Attachment ids returned by POST /sessions/{id}/attachments. "
            "Each one becomes an inline image/file placeholder visible to "
            "the LLM."
        ),
    )
    provider_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=64,
        description=(
            "Per-turn provider override; persisted default still lives on "
            "Account.provider_id."
        ),
    )
    language: Language | None = Field(
        default=None,
        description=(
            "Per-turn language override; if absent the request falls back "
            "to Account.language. Lets the SPA flip the picker and have "
            "the very next turn use the new language without waiting on a "
            "PATCH /api/v1/me round-trip."
        ),
    )


class MePatch(BaseModel):
    """Mutable subset of ``Account`` for ``PATCH /api/v1/me``.

    ``extra="forbid"`` keeps the frontend from accidentally promoting a
    user to admin via this endpoint — ``role`` is not mutable here.
    ``active_system_rag_collections`` remains read-only via this endpoint;
    admin pages will expose it later.

    ``provider_id`` is mutable so the web model picker can persist the
    user's chosen model across sessions. The router validates the id
    against the live catalog and rejects unknown values with 422.

    All fields are optional; the patch is a partial update. An empty
    body is allowed and is a no-op (router returns 200 with the
    unchanged Account).
    """

    model_config = ConfigDict(extra="forbid")

    display_name: str | None = Field(default=None, min_length=1, max_length=64)
    language: Language | None = None
    provider_id: str | None = Field(default=None, min_length=1, max_length=64)


# --- /api/v1/providers ------------------------------------------------


class ProviderResponse(BaseModel):
    """One entry in ``GET /api/v1/providers``.

    The shape is intentionally wide so future UI additions (capability
    chips, thinking-toggle indicator, family icon, cost tier) can be
    rendered without an API change. Fields the catalog does not supply
    are filled with safe defaults — the frontend treats absent /
    zero / None as "feature unknown" and degrades gracefully.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: Literal["local", "cloud"]
    model: str
    family: str
    display_name: str
    context_window: int = 0
    available: bool = True
    thinking: str | bool | None = None


class ProviderListResponse(BaseModel):
    """Body returned by ``GET /api/v1/providers``."""

    model_config = ConfigDict(extra="forbid")

    providers: list[ProviderResponse]
    default_provider_id: str
    current_provider_id: str


# --- /api/v1/sessions/{id}/attachments -------------------------------


class AttachmentResponse(BaseModel):
    """One uploaded attachment.

    The frontend renders a preview chip from ``filename`` + ``kind`` +
    ``ocr_status``. ``id`` is the blob's sha256 and is what the
    StreamRequest references on the next turn.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    filename: str
    mime_type: str
    size_bytes: int
    kind: Literal["image", "text", "other"]
    ocr_status: Literal["pending", "done", "empty", "failed"] = "pending"


class AttachmentListResponse(BaseModel):
    """Body returned by ``POST /api/v1/sessions/{id}/attachments``."""

    model_config = ConfigDict(extra="forbid")

    attachments: list[AttachmentResponse]


class AttachmentMetaItem(BaseModel):
    """One entry in the batch meta-lookup response.

    ``requested`` echoes the input verbatim so the caller can correlate
    results without depending on response order. Exactly one of
    ``resolved`` / ``error`` is populated: success → ``resolved`` carries
    the full attachment shape; failure → ``error`` names the failure mode
    so the frontend can render an appropriate placeholder chip.
    """

    model_config = ConfigDict(extra="forbid")

    requested: str
    resolved: AttachmentResponse | None = None
    error: Literal["not_found", "ambiguous", "invalid"] | None = None


class AttachmentMetaResponse(BaseModel):
    """Body returned by ``GET /api/v1/sessions/{id}/attachments/meta``."""

    model_config = ConfigDict(extra="forbid")

    items: list[AttachmentMetaItem]


# --- /api/v1/sessions/{id}/interactions/{id} -------------------------


class InteractionResponse(BaseModel):
    """Body submitted to ``POST /sessions/{id}/interactions/{interaction_id}``.

    One generic shape carries every interaction kind. The web layer
    rendezvous looks up the pending interaction by id and dispatches
    based on its declared kind, then validates ``payload`` against the
    matching channel schema (ask_user_question → AskUserQuestionResult,
    tool_approval → ApprovalDecision). Keeping the wire shape generic
    means the SPA can support new interaction kinds without minting a
    new endpoint per kind.
    """

    model_config = ConfigDict(extra="forbid")

    kind: Literal["ask_user_question", "tool_approval"]
    # ``dict[str, Any]`` so OpenAPI exposes a non-empty schema; the
    # per-kind payload validation still happens in the rendezvous
    # handler against the matching channel schema.
    payload: dict[str, Any] = Field(default_factory=dict)


# --- /api/v1/library --------------------------------------------------


class LibrarySystemCollection(BaseModel):
    """One system-managed RAG collection entry in ``GET /api/v1/library``.

    Mirrors :class:`claritymed.core.rag.schemas.CollectionMetadata` minus
    the routing-only fields (``source_uri_prefix``, ``disease_codes``,
    ``cross_lingual``) the SPA does not yet render. ``chunk_count`` is
    ``None`` when the shared Qdrant server does not yet know about the
    collection — the row displays as ``?`` chunks rather than failing
    the whole list.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    language: Literal["en", "zh"]
    authority_tier: int
    topics: list[str] = Field(default_factory=list)
    license: str | None = None
    chunk_count: int | None = None


class LibraryUserCollection(BaseModel):
    """The current user's per-account ``user_rag_<uid>`` collection.

    ``chunk_count`` is ``None`` when the user has never ingested anything
    (collection not yet created). Zero is a valid value separate from
    None — it means every previously-ingested document was later removed.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    chunk_count: int | None = None


class LibraryListResponse(BaseModel):
    """Body returned by ``GET /api/v1/library``.

    ``rag_enabled`` carries the same flag the chat router consults: when
    ``False`` the search endpoint will refuse with 503, and the SPA
    should render a banner rather than calling it.
    """

    model_config = ConfigDict(extra="forbid")

    system_collections: list[LibrarySystemCollection]
    user_collection: LibraryUserCollection
    rag_enabled: bool


class LibrarySearchRequest(BaseModel):
    """Body of ``POST /api/v1/library/search``.

    Mirrors the TUI library modal's input. The 8000-char cap matches
    ``StreamRequest.q`` so an oversized query is rejected the same way
    in both surfaces.
    """

    model_config = ConfigDict(extra="forbid")

    q: str = Field(min_length=1, max_length=8000)


class LibrarySearchChunk(BaseModel):
    """One retrieved chunk in the search response.

    ``tag`` is the coarse origin label the SPA uses to badge results:
    ``USER`` for ``user_rag_<uid>`` chunks, ``SYS`` for everything else.
    ``score`` falls back to the pre-rerank score when the reranker did
    not produce one (TUI mirrors this same fallback).
    """

    model_config = ConfigDict(extra="forbid")

    tag: Literal["USER", "SYS"]
    collection_name: str
    doc_id: str
    score: float | None = None
    snippet: str


class LibrarySearchTrace(BaseModel):
    """Per-stage timings + active collections, mirrors ``RetrievalTrace``."""

    model_config = ConfigDict(extra="forbid")

    active_collections: list[str] = Field(default_factory=list)
    expanded_query: str | None = None
    embed_ms: int = 0
    search_ms: int = 0
    rerank_ms: int = 0
    parent_expand_ms: int = 0


class LibrarySearchResponse(BaseModel):
    """Body returned by ``POST /api/v1/library/search``."""

    model_config = ConfigDict(extra="forbid")

    chunks: list[LibrarySearchChunk]
    trace: LibrarySearchTrace


class LibraryIngestRequest(BaseModel):
    """Body of ``POST /api/v1/library/ingest``.

    ``text`` is the raw input bar contents — it may carry
    ``[Image sha:…]`` / ``[File sha:…]`` placeholders that resolve
    against ``session_id``'s SessionAttachments. ``session_id`` is
    optional only when ``text`` carries no placeholders (pure inline
    text). The server runs ``UploadBundle.validate`` and returns 422
    with the reasons list when the bundle fails the floor / pending /
    failed gates — identical to the TUI's modal gate.

    ``public`` is the user's explicit consent to ingest *without* the
    PHI scrub layer and to mark resulting chunks ``can_cloud=True``.
    Default ``False`` — text is PHI-scrubbed at ingest and chunks land
    ``can_cloud=False`` (cloud-bound retrieval skips them). The SPA
    must surface this as a per-upload toggle when offering to make
    library content cloud-queryable; passing ``true`` silently would
    re-introduce the flag-error that retrieval defense-in-depth was
    designed to compensate for.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=200_000)
    session_id: str | None = Field(default=None, pattern=SESSION_ID_PATTERN)
    public: bool = False


class LibraryIngestPart(BaseModel):
    """Per-part outcome included in the ingest response.

    ``status`` is the post-ingest verdict, distinct from the pre-ingest
    ``UploadPart.status`` gate result: it captures what happened when
    the part was actually sent through ``RagService`` — ``added`` (new
    chunks landed), ``skipped`` (cosine-dedup or duplicate source_uri),
    or ``failed`` (the LLM / embedder raised).
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    kind: Literal["text", "image", "file"]
    status: Literal["added", "skipped", "failed"]
    chunks: int = 0
    error: str | None = None


class LibraryIngestResponse(BaseModel):
    """Body returned by ``POST /api/v1/library/ingest``."""

    model_config = ConfigDict(extra="forbid")

    added_parts: int
    skipped_parts: int
    failed_parts: int
    added_chunks: int
    parts: list[LibraryIngestPart]


class LibraryIngestValidationError(BaseModel):
    """422 body when ``UploadBundle.validate`` fails.

    The same shape FastAPI uses for built-in validation errors would
    work, but the per-part ``reasons`` list is more useful for the SPA
    than a generic ``detail`` string — the modal can map each reason to
    a per-attachment chip warning.
    """

    model_config = ConfigDict(extra="forbid")

    detail: str
    reasons: list[str] = Field(default_factory=list)
