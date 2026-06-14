"""Pydantic contracts for ``configs/symptoms.yaml``.

Mirrors the catalog + active-id pattern used by ``core/rag/schemas.py`` —
``eligibility.active`` must resolve to a catalog entry id, a typo raises
:class:`~claritymed.errors.UnknownEligibilityStrategyError` at load time
rather than silently degrading to a default.

Two-level integrity chain on model weights (see KTD-6 in the
disease-prediction plan): :class:`ModelSpec.manifest_sha256` pins the
manifest's own digest from this file, so a tampered manifest pointing at
different weights cannot pass the file-against-manifest check the server
runs on startup. The chain is rooted in the committed config.

Audit-only safety keywords are NOT part of this schema — they live in
``configs/i18n/<lang>/symptoms.yaml`` under
``symptoms.safety_keywords.<tier>`` and are read by the plugin's
``post_process`` hook via :func:`claritymed.core.i18n.loader.t_list`.
Keeping them in the i18n bundle lets translators edit them alongside
the rest of the localized copy without touching Pydantic.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from claritymed.errors import UnknownEligibilityStrategyError

# Per-disease severity is sourced from DDXPlus ``release_conditions.json``
# on a 1-5 scale (1 = most severe). Tier names are stable across the
# audit, prompt registry, and config so a Critical-tier mock in tests
# matches the field used by the post_process check.
SeverityTier = Literal["Critical", "Urgent", "Moderate", "Mild"]

# Eligibility strategy ids accepted by the catalog discriminator. Adding
# a new strategy requires (a) a new entry class below, (b) a factory
# branch in ``core/symptoms/eligibility/factory.py`` (Unit 7), and (c)
# a YAML catalog entry. Anything else (typo, missing branch) fails loud.
EligibilityStrategyKind = Literal["direct", "term_service", "translation"]

# Evidence dtypes the init-symptom matcher will consider as candidates.
# Mila's BASD only injects binary evidences as the turn-0 free
# observation (env.py:267 — ``if data_type == "B" and is_present and
# (not is_antecedent)``). We mirror that by default; relaxing to
# ``["B", "M"]`` requires the matcher to also pick a value, which v1
# does not implement.
EvidenceDtype = Literal["B", "C", "M"]


# --- init-symptom matcher --------------------------------------------------


class InitSymptomFilter(BaseModel):
    """Per-dataset candidate-pool filter for the init-symptom matcher.

    Default mirrors Mila BASD: exclude antecedent evidences and accept
    only binary dtype. Datasets where the binary pool is too small (or
    where the matcher should also consider M-type localizers) can
    override either field — but enabling M requires runtime support
    for value resolution that v1 doesn't ship.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    exclude_antecedent: bool = True
    allowed_dtypes: list[EvidenceDtype] = Field(
        default_factory=lambda: ["B"],
        min_length=1,
        max_length=3,
    )


class InitMatcherConfig(BaseModel):
    """Top-level init-symptom matcher configuration.

    The embedder is a process-wide singleton loaded once at server
    lifespan startup. Per-dataset candidate vectors are encoded
    against this singleton at dataset load time. Disabling the
    matcher globally (``enabled=False``) makes ``init_catalog`` on
    every :class:`LoadedDataset` ``None`` and the runtime falls back
    to zero-init state — matching pre-matcher behaviour.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    model_id: str = Field(
        default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        min_length=1,
        max_length=128,
        description=(
            "sentence-transformers / HuggingFace model id. SapBERT is "
            "the default — trained on UMLS synonyms, well-suited to "
            "short medical-phrase similarity."
        ),
    )
    device: Literal["cpu", "cuda", "mps", "auto"] = Field(
        default="cpu",
        description=(
            "Where to load the matcher model. CPU keeps it from "
            "competing with BASD inference for GPU memory; matching "
            "is a turn-0 one-shot so latency is not critical."
        ),
    )
    threshold: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum cosine score for a match to be injected. Scores "
            "below this fall through to the zero-init branch (no "
            "evidence pre-revealed). 0.55 is a SapBERT-on-DDXPlus "
            "starting point; sweep on a dev set to tune."
        ),
    )


# --- datasets + models -----------------------------------------------------


class DatasetSpec(BaseModel):
    """One dataset registered with the symptoms server.

    ``model_ids`` references entries in the top-level ``models`` list —
    cross-checked by :meth:`SymptomsConfig._model_refs_resolve`. Multiple
    models per dataset are supported (shadow inference, future A/B);
    :attr:`model_selection` decides which the server uses per-request.
    The first id in the list is the canonical primary used by the
    "first" selection strategy. ``maxstep`` is the per-session question
    budget chosen by the Phase 0 ablation (Unit 3 in the plan);
    ``partial_min_confidence`` is the top-3 mass cutoff below which a
    cancelled session is treated as "no usable differential" rather
    than "show partial results".

    i18n: every text-bearing field in the question payload is rendered
    via ``t(key, lang=...)``. Keys follow the convention
    ``{i18n_key_prefix}.<evidence_id>.question`` and
    ``{i18n_key_prefix}.<evidence_id>.values.<raw_value>``;
    condition names use ``{i18n_key_prefix}.conditions.<slug>.name``.
    Per-language YAML files live under ``configs/i18n/<lang>/`` and are
    merged into the language dict by the loader. ``binary_yes_key`` /
    ``binary_no_key`` default to a shared global so multiple datasets
    don't each ship their own Yes/No translation.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]{0,63}$")
    enabled: bool = True
    model_ids: list[str] = Field(
        min_length=1,
        max_length=8,
        description=(
            "One or more model ids this dataset can serve. The first "
            "entry is the canonical primary; additional entries enable "
            "multi-model selection via model_selection."
        ),
    )
    model_selection: Literal["first", "round_robin"] = Field(
        default="first",
        description=(
            "Per-request model picker. 'first' always uses model_ids[0] "
            "(default); 'round_robin' cycles in spec order — useful for "
            "A/B and shadow inference."
        ),
    )
    partial_min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    severity_high_specificity_evidence_ids: list[str] = Field(default_factory=list)
    # Native language of the dataset's evidence vocab — the language
    # the ``direct`` matcher expects to see. Drives the translation
    # eligibility strategy: complaints in any other language get
    # translated **to this** before the direct match runs. DDXPlus
    # ships English questions/values, so ``"en"`` is the right default;
    # a future Chinese dataset would set ``"zh"`` here without code
    # changes elsewhere.
    native_language: Literal["en", "zh"] = "en"
    # KTD-12: in-memory session TTL. Default 30 minutes; raise to 7200 if
    # dogfood shows TTL eviction dominates the cancel reason. Config-driven
    # so the change is one line, not an architectural shift.
    session_ttl_seconds: int = Field(default=1800, ge=60, le=86400)
    # Maximum categorical options shown per question. DDXPlus travel-region
    # evidence (E_204) has 12 values; 12 fits the TUI picker without a long
    # scroll. Raise for datasets with denser categorical vocabularies.
    max_options: int = Field(default=12, ge=2, le=20)

    # Init-symptom matching (parity with Mila BASD's INITIAL_EVIDENCE).
    # When ``True`` and a ``SymptomsConfig.init_matcher`` is enabled +
    # reachable, the server attempts to map the user's complaint to one
    # candidate evidence and pre-reveal it on turn 0 — same shape as
    # ``Patient.init`` in training (typed_basd.py:247). Defaults to
    # ``True`` to close the train/serve skew; setting ``False`` reverts
    # to zero-init state.
    use_initial_symptom_flag: bool = True
    init_symptom_filter: "InitSymptomFilter" = Field(
        default_factory=lambda: InitSymptomFilter()
    )

    @model_validator(mode="after")
    def _model_ids_unique(self) -> "DatasetSpec":
        if len(set(self.model_ids)) != len(self.model_ids):
            raise ValueError(f"datasets[id={self.id!r}].model_ids must be unique")
        return self

    def primary_model_id(self) -> str:
        """Return the first model id — the canonical 'main' checkpoint."""
        return self.model_ids[0]

    # i18n key wiring (see class docstring). ``None`` for prefix means
    # derive as ``symptoms.<id>``; explicit override accepted for the
    # rare dataset that needs to share keys with a sibling.
    i18n_key_prefix: str | None = Field(
        default=None,
        max_length=128,
        description=(
            "Root prefix for evidence i18n keys. None → 'symptoms.<id>'. "
            "Resolved keys: '<prefix>.<evidence_id>.question' and "
            "'<prefix>.<evidence_id>.values.<raw_value>'."
        ),
    )
    binary_yes_key: str = Field(
        default="symptoms.binary.yes",
        min_length=1,
        max_length=128,
    )
    binary_no_key: str = Field(
        default="symptoms.binary.no",
        min_length=1,
        max_length=128,
    )

    def resolved_i18n_prefix(self) -> str:
        """Return the active prefix (explicit override or convention)."""
        return self.i18n_key_prefix or f"symptoms.{self.id}"

    def question_key(self, evidence_id: str) -> str:
        return f"{self.resolved_i18n_prefix()}.{evidence_id}.question"

    def value_key(self, evidence_id: str, raw_value: str) -> str:
        return f"{self.resolved_i18n_prefix()}.{evidence_id}.values.{raw_value}"

    def condition_name_key(self, condition_slug: str) -> str:
        """Localized display name key for a canonical condition slug."""
        return f"{self.resolved_i18n_prefix()}.conditions.{condition_slug}.name"


class ModelSpec(BaseModel):
    """One model checkpoint registered for a dataset.

    ``weights_subpath`` is resolved relative to
    ``CLARITYMED_HOME/models/symptoms/`` at server load time. Absolute
    paths and traversal segments are rejected to keep the load surface
    scoped to the per-user runtime root.

    ``manifest_sha256`` pins the digest of the ``manifest.json`` that
    sits next to the weights; the server hashes the manifest at startup
    and refuses to start on mismatch (KTD-6).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]{0,63}$")
    algorithm_module: str = Field(min_length=1, max_length=64)
    weights_subpath: str = Field(min_length=1, max_length=256)
    manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    # Per-session question budget. Populated from claritymed-symptoms-tune-ddxplus.
    maxstep: int = Field(ge=1, le=50)
    # Softmax temperature for the pathology classifier (overrides checkpoint).
    # null → fall back to the value baked into the checkpoint.
    patho_temp: float | None = Field(default=None, gt=0.0, le=10.0)
    # Stop-gate threshold (overrides checkpoint). Heuristic mode: max symptom
    # prob must drop below this to keep asking. null → use checkpoint value.
    stop_thres: float | None = Field(default=None, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _weights_subpath_relative(self) -> "ModelSpec":
        path = self.weights_subpath
        if path.startswith("/") or path.startswith("~"):
            raise ValueError(
                f"weights_subpath must be relative to "
                f"CLARITYMED_HOME/models/symptoms/, got absolute path: {path!r}"
            )
        if ".." in path.replace("\\", "/").split("/"):
            raise ValueError(
                f"weights_subpath must not contain '..' segments: {path!r}"
            )
        return self


# --- eligibility catalog ---------------------------------------------------


class _EligibilityEntryBase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]{0,63}$")


class DirectEligibilityEntry(_EligibilityEntryBase):
    """EN-only token-match strategy. Cheapest, zero external deps."""

    kind: Literal["direct"]


class TermServiceEligibilityEntry(_EligibilityEntryBase):
    """Concept-grounded strategy. Cross-lingual via the active TermService.

    Requires ``term_service.active != "none"`` in ``retrieval.yaml``; the
    factory raises :class:`~claritymed.errors.EligibilityStrategyUnavailableError`
    at construct time when a NoOp service is configured.
    """

    kind: Literal["term_service"]


class TranslationEligibilityEntry(_EligibilityEntryBase):
    """LLM-translation strategy. ``provider_id`` must be ``kind: local``.

    The factory asserts ``ProviderConfig.kind == "local"`` and raises
    :class:`~claritymed.errors.EligibilityStrategyConfigError` for a cloud
    provider — PHI must not cross the network for translation.
    """

    kind: Literal["translation"]
    provider_id: str = Field(min_length=1, max_length=64)
    prompt_name: str = Field(min_length=1, max_length=64)
    max_tokens: int = Field(default=256, ge=16, le=4096)


EligibilityCatalogEntry = Annotated[
    DirectEligibilityEntry | TermServiceEligibilityEntry | TranslationEligibilityEntry,
    Field(discriminator="kind"),
]


EligibilityInputSource = Literal["complaint", "symptom_summary"]


class EligibilityCatalogConfig(BaseModel):
    """The catalog + active-id resolution for symptom eligibility.

    ``input_source`` selects which LLM-supplied string the plugin
    feeds into the active strategy's ``check``:

    * ``"complaint"`` (default) — the user's raw text. Preserves
      original-language tokens; lets ``term_service`` exploit the
      cross-lingual surface forms and lets ``translation`` start from
      the user's tongue. Audit-friendliest: the eligibility verdict
      references what the user actually said, not a paraphrase.
    * ``"symptom_summary"`` — the LLM-distilled 1-2 sentence English
      clinical phrase. Useful when the active strategy is ``direct``
      and the user often types in EN but with noisy / lay phrasing
      that the summary's tight tokens clean up. Adds a hidden
      "the model thought you meant…" step in the audit trail and
      makes eligibility depend on the calling LLM being up.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    active: str = Field(min_length=1)
    catalog: list[EligibilityCatalogEntry] = Field(min_length=1)
    input_source: EligibilityInputSource = Field(
        default="complaint",
        description=(
            "Which input the plugin forwards to the active eligibility "
            "strategy. See class docstring for tradeoffs."
        ),
    )

    @model_validator(mode="after")
    def _resolve_active(self) -> "EligibilityCatalogConfig":
        ids = {entry.id for entry in self.catalog}
        if self.active not in ids:
            raise UnknownEligibilityStrategyError(
                f"eligibility.active={self.active!r} not in catalog {sorted(ids)!r}"
            )
        return self

    def resolved(self) -> EligibilityCatalogEntry:
        for entry in self.catalog:
            if entry.id == self.active:
                return entry
        # ``_resolve_active`` guarantees this is unreachable.
        raise UnknownEligibilityStrategyError(self.active)


# --- top-level symptoms config --------------------------------------------


class SymptomsConfig(BaseModel):
    """Root of ``configs/symptoms.yaml``.

    Cross-references are validated up-front: every dataset's ``model_id``
    must resolve to a registered model. A typo'd reference raises at
    load time so the server never starts against a half-broken catalog.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    datasets: list[DatasetSpec] = Field(min_length=1)
    models: list[ModelSpec] = Field(min_length=1)
    eligibility: EligibilityCatalogConfig
    init_matcher: InitMatcherConfig = Field(default_factory=lambda: InitMatcherConfig())

    @model_validator(mode="after")
    def _model_refs_resolve(self) -> "SymptomsConfig":
        known = {m.id for m in self.models}
        for ds in self.datasets:
            unknown = [mid for mid in ds.model_ids if mid not in known]
            if unknown:
                raise ValueError(
                    f"datasets[id={ds.id!r}].model_ids reference unknown "
                    f"models {unknown!r}; known: {sorted(known)!r}"
                )
        return self

    @model_validator(mode="after")
    def _unique_dataset_ids(self) -> "SymptomsConfig":
        ids = [d.id for d in self.datasets]
        if len(set(ids)) != len(ids):
            raise ValueError(f"datasets[].id must be unique, got {ids!r}")
        return self

    @model_validator(mode="after")
    def _unique_model_ids(self) -> "SymptomsConfig":
        ids = [m.id for m in self.models]
        if len(set(ids)) != len(ids):
            raise ValueError(f"models[].id must be unique, got {ids!r}")
        return self
