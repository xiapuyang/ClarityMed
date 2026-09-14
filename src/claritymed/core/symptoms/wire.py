"""Pydantic models for the symptoms-server HTTP wire format.

Per KTD-7 (disease-prediction plan), the server is data-only — no
``safety_sentence``, ``severity_tier``, or ``forced_caveat_sentence``
fields. Per-disease ``severity: int`` (1-5) is included in every
differential row so the LLM downstream can compose tier-appropriate
prose and the plugin's post_process can audit keyword compliance.

Question payloads are :class:`Question` instances from
:mod:`claritymed.core.interaction.schemas` so the server's first-question
field round-trips straight into the plugin's PromptChannel without
re-shaping.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from claritymed.core.interaction.schemas import Question


class ProfilePayload(BaseModel):
    """User-supplied profile fields the model conditions on.

    ``age_years`` is the raw integer per KTD-13 — bucketing happens
    server-side via :func:`claritymed.ingest.symptoms.typed_basd.age_bucket`.
    Datasets with different bucketing override the helper in their
    adapter, never the wire format.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    age_years: int = Field(ge=0, le=120)
    sex: Literal["M", "F"]


class StartSessionRequest(BaseModel):
    """Body of ``POST /v1/datasets/{id}/sessions``.

    ``language`` selects the locale every Question + option in the
    sub-session is rendered in. Set by the plugin from the active
    ``language_ctx``; the server doesn't read its own ContextVar
    because uvicorn workers don't carry the orchestrator's context.

    ``symptom_summary`` is the LLM's distilled chief-complaint
    summary — a 1-2 sentence English description of the user's
    presenting symptoms, synthesized from the full conversation by
    the calling LLM. Used exclusively as input to the init-symptom
    matcher (complaint → candidate evidence cosine match); the
    differential model itself never sees it. Falls back to
    ``complaint`` server-side when absent. Kept separate from
    ``complaint`` because eligibility / audit / PHI guard all want
    the raw user text, while the embedder wants a tight clinical
    phrase — the two uses have opposite preferences.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    complaint: str = Field(min_length=1, max_length=4_000)
    profile: ProfilePayload
    language: Literal["en", "zh"] = "en"
    symptom_summary: str | None = Field(
        default=None,
        max_length=4_000,
        description=(
            "LLM-distilled clinical chief complaint (1-2 sentences, EN). "
            "Used as input to the init-symptom matcher; falls back to "
            "`complaint` when absent."
        ),
    )


class StartSessionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str
    first_question: Question


class TurnRequest(BaseModel):
    """Body of ``POST /v1/datasets/{id}/sessions/{sid}/turn``.

    ``answer`` is loose-typed because the previous question may have
    been any of: binary (``"Yes"`` / ``"No"``), categorical label, list
    of labels (multi-select), or a numeric value. Per-evidence
    validation happens in :func:`questions.synth_patient`.

    ``answer_value`` is the raw value identifier (e.g. ``"V_14"`` for
    DDXPlus) when the plugin knows it — set by the client after
    matching the user's clicked option label back to its raw value.
    When set, the server uses it directly rather than re-matching
    ``answer`` against the localized labels (avoids translation
    round-trip failures). ``language`` is re-supplied so a follow-up
    turn renders the next question in the same locale the session
    started with — also defensible against the plugin issuing turns
    from a context with a different active language.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: str | int | float | list[str]
    answer_value: str | list[str] | None = None
    language: Literal["en", "zh"] = "en"


class EvidenceCollectedRow(BaseModel):
    """One Q&A round-trip the server applied to internal state.

    ``source`` distinguishes user-answered evidence from
    init-matcher-inferred evidence. Under the pre-question flow, an
    ``init_matcher`` entry means "SapBERT proposed this evidence based
    on the symptom summary and the user confirmed with Yes/No" — the
    answer is authoritative, only the *choice of question* was inferred.
    Defaults to ``"modal_answer"`` so existing call sites stay
    schema-compatible.

    ``match_score`` is the SapBERT cosine score (0.0-1.0) that led the
    matcher to pick this evidence, populated only when ``source ==
    "init_matcher"``. Kept in the payload so audit + eval can inspect
    which matches were weak vs strong; downstream reply generation is
    free to ignore it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    evidence_name: str
    evidence_type: Literal["B", "C", "M"]
    answer: str | int | float | list[str]
    source: Literal["modal_answer", "init_matcher"] = "modal_answer"
    match_score: float | None = None


class DifferentialRow(BaseModel):
    """One pathology in the ranked differential.

    ``condition_id`` is the canonical slug (stable across re-exports,
    suitable as an i18n key or filesystem segment); ``condition_idx`` is
    the algorithm-internal integer index — useful for audit logs that
    want a compact identifier without dereferencing the slug table.
    ``severity`` is the raw 1-5 from the dataset's conditions schema;
    the LLM and the audit layer compute tier semantics from it.
    ``icd10`` is best-effort metadata sourced from the corpus and may
    be ``None`` for datasets that don't carry it.

    ``condition_idx=None`` is reserved for the synthetic ``Other`` row
    v3 subset-parametric datasets emit — Other is not a canonical
    catalog entry, so it has no pidx integer to reference. Callers that
    key audit logs on the integer index must handle the None case (or
    fall back to the slug, which is always present).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    condition_id: str = Field(min_length=1, max_length=128)
    condition_idx: int | None = Field(default=None, ge=0)
    condition_name: str
    probability: float = Field(ge=0.0, le=1.0)
    severity: int = Field(ge=1, le=5)
    icd10: str | None = None


class TurnResponse(BaseModel):
    """Union response from ``/turn``.

    Exactly one of ``next_question`` / ``done`` / ``hit_cap`` is set.
    Distinguishing fields rather than a discriminator keeps the wire
    payload compact and the plugin's pattern-match branches obvious.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    next_question: Question | None = None
    done: bool = False
    hit_cap: bool = False
    differential: list[DifferentialRow] = Field(default_factory=list)
    partial_differential: list[DifferentialRow] = Field(default_factory=list)
    evidence_collected: list[EvidenceCollectedRow] = Field(default_factory=list)
    turn_count: int = 0
    partial_confidence: float = 0.0


class CancelResponse(BaseModel):
    """Response body of ``DELETE /v1/datasets/{id}/sessions/{sid}``.

    ``severity_override`` flags the "low-confidence but a severity ≤2
    disease has prob > 0.1" case so the LLM knows to lead with
    urgent-care language even without a full differential.
    ``max_low_severity_seen`` accompanies ``severity_override=True`` so
    the post_process audit knows which tier to check keywords against.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    cancelled: bool = True
    partial_differential: list[DifferentialRow] = Field(default_factory=list)
    evidence_collected: list[EvidenceCollectedRow] = Field(default_factory=list)
    turn_count: int = 0
    partial_confidence: float = 0.0
    meets_confidence_threshold: bool = False
    severity_override: bool = False
    max_low_severity_seen: int | None = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["ok", "loading"]
    datasets_loaded: list[str] = Field(default_factory=list)
    models_loaded: list[str] = Field(default_factory=list)
