"""Eligibility Protocol + result type.

The eligibility check runs before the sub-session starts: it inspects
the user's free-text complaint and decides whether the dataset's model
is likely to produce a useful differential. A miss returns a structured
result instead of raising — the orchestrator falls back to the LLM's
normal RAG / free-text answer (KTD-11: out-of-scope is silent).

Strategies are instantiated once at plugin startup with whatever
dependencies they need (vocab maps, term services, LLM providers) and
called per request. The Protocol intentionally takes the rich
:class:`~claritymed.core.schemas.patient.Profile` rather than a slim
``{age_years, sex}`` shape so future strategies can branch on residence
/ birthplace / age bucketing without a Protocol revision.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from claritymed.core.rag.terms.base import ConceptLanguage
from claritymed.core.schemas.patient import Profile
from claritymed.core.symptoms.schemas import DatasetSpec

EligibilityReason = Literal[
    "in_scope",
    "out_of_scope",
    "demographic_mismatch",
    "strategy_unavailable",
]


class EligibilityResult(BaseModel):
    """Structured outcome of one eligibility check.

    ``evidence_hits`` carries the matched evidence identifiers so the
    audit log can record *what* the strategy keyed on without exposing
    the raw complaint. ``confidence`` is the strategy's self-reported
    match strength; the registry uses it as a tiebreaker when multiple
    datasets are eligible (KTD-9).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    eligible: bool
    reason: EligibilityReason
    evidence_hits: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


@runtime_checkable
class EligibilityStrategy(Protocol):
    """Pluggable "is this complaint in scope for this dataset?" check.

    Implementations are stateless across calls — any cached vocab /
    sidecar is loaded at construct time. ``check`` must never raise on a
    well-formed complaint; transient backing-service failures should
    return ``EligibilityResult(eligible=False, reason="strategy_unavailable")``.
    Hard config errors (cloud provider on a local-only strategy) raise
    at construct time, not at check time.

    ``check`` is async because the ``translation`` strategy (Unit 9) calls
    a pydantic-ai ``Agent``. Sync-only implementations (``direct``,
    ``term_service``) still declare ``async def`` so the Protocol stays
    homogeneous and the plugin can ``await`` without branching.
    """

    async def check(
        self,
        complaint: str,
        language: ConceptLanguage,
        profile: Profile,
        dataset: DatasetSpec,
    ) -> EligibilityResult: ...
