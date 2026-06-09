"""RAG-layer pydantic contracts and the retrieval.yaml loader.

The catalog + active-id pattern mirrors ``stores/models.py``:

* Each section that has a ``catalog`` also has an ``active`` id. A typo in
  ``active`` raises ``Unknown<Section>Error`` at config load time — never
  silently falls back to a default, because the default may not be what
  the operator intended (and certainly not what a paper experiment
  expects).
* ``CollectionMetadata`` is the per-collection routing input the
  ``CollectionRouter`` consumes (see Unit 5 of the RAG plan).
* ``RetrievalTrace`` + ``EvidenceBundle`` are the strategy → service
  return contract. ``RetrievalTrace.fallback_triggered=True`` requires
  ``grader`` to be populated, since CRAG-lite is the only thing that can
  raise the flag in v1.

Optional ``CollectionMetadata.size_chunks`` defaults to 0 so a YAML entry
can be written before ingest has actually populated the collection;
ingest code is expected to update the YAML after a successful run, or
the router treats 0 as "not yet ingested" and may skip it.
"""

from __future__ import annotations

import os
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from claritymed import config as _cfg
from claritymed.core.schemas.retrieval import RetrievedChunk
from claritymed.errors import (
    UnknownChunkerError,
    UnknownEmbedderError,
    UnknownRerankerError,
    UnknownRouterError,
    UnknownStrategyError,
    UnknownTermServiceError,
)

_RAG_ENABLED_ENV = "CLARITYMED_RAG_ENABLED"
_QDRANT_URL_ENV = "CLARITYMED_QDRANT_URL"
_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}

# --- collection metadata (router input) ---------------------------------

CollectionLanguage = Literal["en", "zh"]
COLLECTION_NAME_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"


class CollectionMetadata(BaseModel):
    """Static per-collection metadata the CollectionRouter routes on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=COLLECTION_NAME_PATTERN)
    language: CollectionLanguage
    cross_lingual: bool = False
    authority_tier: int = Field(ge=1, le=3)
    size_chunks: int = Field(default=0, ge=0)
    topics: list[str] = Field(default_factory=list)
    disease_codes: list[str] = Field(default_factory=list)
    source_uri_prefix: str | None = None
    license: str | None = None


# --- retrieval trace / grader / evidence bundle -------------------------


class GraderReport(BaseModel):
    """CRAG-lite grader output for a single retrieval pass."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    threshold: float = Field(ge=0.0, le=1.0)
    mean_rerank_score: float
    decision: Literal["pass", "rewrite"]


class RetrievalTrace(BaseModel):
    """Observability payload returned alongside chunks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: str
    active_collections: list[str] = Field(default_factory=list)
    expanded_query: str | None = None
    embed_ms: int = Field(default=0, ge=0)
    search_ms: int = Field(default=0, ge=0)
    rerank_ms: int = Field(default=0, ge=0)
    parent_expand_ms: int = Field(default=0, ge=0)
    grader: GraderReport | None = None
    fallback_triggered: bool = False
    rerank_fallback: bool = Field(
        default=False,
        description=(
            "True when the reranker raised RerankerUnreachableError and the "
            "retriever fell back to RRF order. Surfaced here so the service "
            "layer can audit / degrade; retriever stays free of context-bound "
            "side effects."
        ),
    )
    hyde_fallback: bool = Field(
        default=False,
        description=(
            "True when the HyDE LLM call failed and the strategy fell back to "
            "embedding the original query unchanged. Surfaced for audit."
        ),
    )

    @model_validator(mode="after")
    def _check_fallback_requires_grader(self) -> "RetrievalTrace":
        # CRAG-lite is still the only thing that sets fallback_triggered;
        # HyDE uses its own ``hyde_fallback`` flag so the audit can tell
        # the two apart.
        if self.fallback_triggered and self.grader is None:
            raise ValueError(
                "fallback_triggered=True requires a GraderReport (CRAG-lite "
                "is the only path that raises this flag; HyDE failures set "
                "hyde_fallback instead)"
            )
        return self


class EvidenceBundle(BaseModel):
    """Return type of ``RagStrategy.retrieve``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunks: list[RetrievedChunk]
    trace: RetrievalTrace


# --- retrieval.yaml top-level config ------------------------------------


class GraderConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    enabled: bool = False
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    top_n: int = Field(default=5, ge=1)
    rewrite_mode: Literal["deterministic", "llm"] = "deterministic"


class NaiveHybridStrategyConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: Literal["naive_hybrid"]
    grader: GraderConfig = Field(default_factory=GraderConfig)


class HydeStrategyConfig(BaseModel):
    """HyDE: LLM drafts a hypothetical answer, embed *that*, retrieve."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    id: Literal["hyde"]
    # Concatenate the hypothetical doc with the original query when
    # forming the embedding text. Default True follows the LlamaIndex /
    # original-paper recommendation — including the original query
    # degrades gracefully when the LLM's draft is off-topic.
    include_original: bool = True


class AgenticStrategyConfig(BaseModel):
    """Agentic mode: LLM drives retrieval via the tool loop.

    No retrieval-side parameters here — Agentic mode is selected via the
    ``id`` and consumes the same underlying NaiveHybridStrategy; the
    behavioural switch lives in ``AskService`` (skip pre-retrieval, let
    the agent call ``retrieve_medical_literature`` 1..N times).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    id: Literal["agentic"]
    grader: GraderConfig = Field(default_factory=GraderConfig)


# Discriminated union on ``id`` so YAML validation routes to the right
# variant. Add new entries here when introducing a new strategy.
StrategyConfig = NaiveHybridStrategyConfig | HydeStrategyConfig | AgenticStrategyConfig


class StrategiesConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    active: str
    # Discriminated on ``id`` so a typo in the YAML routes a malformed
    # entry to the right error message rather than complaining about an
    # unrelated variant's required fields.
    catalog: list[Annotated[StrategyConfig, Field(discriminator="id")]]

    @model_validator(mode="after")
    def _resolve_active(self) -> "StrategiesConfig":
        ids = {entry.id for entry in self.catalog}
        if self.active not in ids:
            raise UnknownStrategyError(
                f"strategies.active={self.active!r} not in catalog {sorted(ids)!r}"
            )
        return self

    def resolved(self) -> StrategyConfig:
        for entry in self.catalog:
            if entry.id == self.active:
                return entry
        # Validator guarantees this is unreachable.
        raise UnknownStrategyError(self.active)


class ParentChildChunkerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: Literal["parent_child"]
    child_tok: int = Field(ge=10)
    parent_tok: int = Field(ge=50)
    overlap_tok: int = Field(default=0, ge=0)


ChunkerEntry = ParentChildChunkerConfig


class ChunkerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    active: str
    catalog: list[ChunkerEntry]

    @model_validator(mode="after")
    def _resolve_active(self) -> "ChunkerConfig":
        ids = {entry.id for entry in self.catalog}
        if self.active not in ids:
            raise UnknownChunkerError(
                f"chunker.active={self.active!r} not in catalog {sorted(ids)!r}"
            )
        return self

    def resolved(self) -> ChunkerEntry:
        for entry in self.catalog:
            if entry.id == self.active:
                return entry
        raise UnknownChunkerError(self.active)


class EmbedderEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str = Field(min_length=1)
    kind: Literal["http"]
    base_url: str = Field(min_length=1)
    dense_dim: int = Field(ge=1)
    batch_size: int = Field(default=32, ge=1)
    timeout_s: int = Field(default=30, ge=1)
    api_key_env: str | None = None


class EmbedderConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    active: str
    catalog: list[EmbedderEntry]

    @model_validator(mode="after")
    def _resolve_active(self) -> "EmbedderConfig":
        ids = {entry.id for entry in self.catalog}
        if self.active not in ids:
            raise UnknownEmbedderError(
                f"embedders.active={self.active!r} not in catalog {sorted(ids)!r}"
            )
        return self

    def resolved(self) -> EmbedderEntry:
        for entry in self.catalog:
            if entry.id == self.active:
                return entry
        raise UnknownEmbedderError(self.active)


class RerankerEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str = Field(min_length=1)
    kind: Literal["http"]
    base_url: str = Field(min_length=1)
    batch_size: int = Field(default=32, ge=1)
    timeout_s: int = Field(default=30, ge=1)
    api_key_env: str | None = None


class RerankerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    active: str
    catalog: list[RerankerEntry]

    @model_validator(mode="after")
    def _resolve_active(self) -> "RerankerConfig":
        ids = {entry.id for entry in self.catalog}
        if self.active not in ids:
            raise UnknownRerankerError(
                f"rerankers.active={self.active!r} not in catalog {sorted(ids)!r}"
            )
        return self

    def resolved(self) -> RerankerEntry:
        for entry in self.catalog:
            if entry.id == self.active:
                return entry
        raise UnknownRerankerError(self.active)


class TermServiceEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str = Field(min_length=1)
    kind: Literal["local", "noop"]
    data_dir: str | None = None


class TermServiceConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    active: str
    catalog: list[TermServiceEntry]

    @model_validator(mode="after")
    def _resolve_active(self) -> "TermServiceConfig":
        ids = {entry.id for entry in self.catalog}
        if self.active not in ids:
            raise UnknownTermServiceError(
                f"term_service.active={self.active!r} not in catalog {sorted(ids)!r}"
            )
        return self

    def resolved(self) -> TermServiceEntry:
        for entry in self.catalog:
            if entry.id == self.active:
                return entry
        raise UnknownTermServiceError(self.active)


class RouterEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    # ``id`` is an open string so adding a router (e.g. ``centroid_classifier``)
    # only requires a catalog entry + a factory branch — no schema change.
    # Unknown ids fail loud at ``build_router`` (UnknownRouterError), not here.
    id: str = Field(min_length=1)
    max_active: int = Field(default=3, ge=1)
    # Keys are stringified tier numbers in YAML for stable YAML int-key
    # behavior; convert to int-keyed dict here. Only consulted by the
    # ``rule_based`` router; ignored by ``centroid_classifier``.
    authority_bias: dict[int, float] = Field(default_factory=dict)
    # Cosine-similarity threshold for the embedding-based router. Ignored
    # by ``rule_based``; tunable per environment for ``centroid_classifier``.
    min_similarity: float = Field(default=0.1, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def _coerce_authority_bias_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            bias = data.get("authority_bias")
            if isinstance(bias, dict):
                data = {
                    **data,
                    "authority_bias": {int(k): float(v) for k, v in bias.items()},
                }
        return data


class RouterConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    active: str
    catalog: list[RouterEntry]

    @model_validator(mode="after")
    def _resolve_active(self) -> "RouterConfig":
        ids = {entry.id for entry in self.catalog}
        if self.active not in ids:
            raise UnknownRouterError(
                f"router.active={self.active!r} not in catalog {sorted(ids)!r}"
            )
        return self

    def resolved(self) -> RouterEntry:
        for entry in self.catalog:
            if entry.id == self.active:
                return entry
        raise UnknownRouterError(self.active)


class SystemRagConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    default_active: list[str] = Field(default_factory=list)
    collections: list[CollectionMetadata] = Field(default_factory=list)
    score_threshold: float = Field(default=0.4, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _default_active_must_be_known(self) -> "SystemRagConfig":
        known = {c.name for c in self.collections}
        unknown = [n for n in self.default_active if n not in known]
        if unknown:
            raise ValueError(
                f"system_rag.default_active references unknown collections: {unknown!r}"
            )
        return self


class UserRagConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    top_k: int = Field(default=5, ge=1)
    rerank_k: int = Field(default=3, ge=1)
    score_threshold: float = Field(default=0.4, ge=0.0, le=1.0)


class TranslationConfig(BaseModel):
    """Configuration for the translation provider.

    ``provider`` selects the backend used for query and answer translation:

    * ``"llm"`` — pydantic-ai Agent; works with any configured model, zero
      extra infra.  Adds one LLM call per translated query/answer.
    * (planned) ``"bge_m3"`` — BGE-M3 multilingual instruction embedding;
      no LLM call, uses the already-running embedder server.
    * (planned) ``"deepl"`` / ``"google"`` — cloud translation APIs; requires
      ``api_key_env`` to be set.

    Default is ``"llm"`` so the section can be omitted from ``retrieval.yaml``
    without breaking startup.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    provider: Literal["llm"] = "llm"


class RagBootstrapConfig(BaseModel):
    """Top-level RAG bootstrap switch.

    Off by default. When ``enabled=False``, ``AskService`` runs without a
    strategy (no retrieval; no calls to the embedder / reranker / Qdrant).
    When ``enabled=True``, the CLI / TUI build a HybridRetriever + strategy
    at startup; any missing dependency (embedder server down, qdrant path
    unreachable) raises fail-loud — silent fallback to LLM-only would mask
    a misconfiguration the operator is opting in to.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
    enabled: bool = False
    max_evidence: int = Field(default=5, ge=1)


class QdrantConfig(BaseModel):
    """Qdrant server connection settings.

    Server-only — local file-locked mode was removed because (a) the
    SQLite storage format is incompatible with server segment format
    (no in-place migration; switching backends costs a full re-embed),
    (b) the file lock forces a single-process model that breaks under
    ingest + TUI concurrency, and (c) maintaining two storage layouts
    doubled the test matrix without adding any production value. Run
    Qdrant via Docker / native binary / Qdrant Cloud — see
    ``docs/rag-setup.md`` §1.

    ``api_key_env`` is the name of an env var holding the Qdrant Cloud
    API key. Declaring it without setting the env var fail-louds at
    startup rather than silently sending unauthenticated requests.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    url: str
    api_key_env: str | None = None


class RetrievalConfig(BaseModel):
    """Root of ``configs/retrieval.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rag: RagBootstrapConfig = Field(default_factory=RagBootstrapConfig)
    qdrant: QdrantConfig
    strategies: StrategiesConfig
    chunker: ChunkerConfig
    embedders: EmbedderConfig
    rerankers: RerankerConfig
    term_service: TermServiceConfig
    router: RouterConfig
    system_rag: SystemRagConfig
    user_rag: UserRagConfig
    translation: TranslationConfig = Field(default_factory=TranslationConfig)


def load_retrieval_config() -> RetrievalConfig:
    """Parse ``configs/retrieval.yaml`` through the mtime-cached loader.

    Two env vars can override yaml fields so local dev can flip without
    editing (and accidentally committing) the shipped defaults:

    * ``CLARITYMED_RAG_ENABLED`` — overrides ``rag.enabled``. Accepts
      ``1/true/yes/on`` (case-insensitive) for on, ``0/false/no/off``
      for off; anything else is ignored so a typo cannot silently flip.
    * ``CLARITYMED_QDRANT_URL`` — overrides ``qdrant.url``. Any
      non-empty value wins; unset / empty leaves yaml intact.

    Tests can call ``claritymed.config.reload_configs()`` to force a
    re-read after redirecting CONFIG_DIR.
    """
    raw = _cfg.load_yaml("retrieval.yaml")
    rag_override = os.environ.get(_RAG_ENABLED_ENV, "").strip().lower()
    if rag_override in _TRUTHY or rag_override in _FALSY:
        raw = {
            **raw,
            "rag": {**raw.get("rag", {}), "enabled": rag_override in _TRUTHY},
        }
    qdrant_override = os.environ.get(_QDRANT_URL_ENV, "").strip()
    if qdrant_override:
        raw = {
            **raw,
            "qdrant": {**raw.get("qdrant", {}), "url": qdrant_override},
        }
    return RetrievalConfig.model_validate(raw)
