"""Symptoms feature — disease-prediction sub-session plugin.

Tool-mode plugin per KTD-1: the tool body holds the
:class:`~claritymed.core.interaction.prompt_channel.PromptChannel` open
and awaits per-turn modals while the symptoms-server drives the model
loop. The LLM is not invoked between turns; it sees one structured
result at the end and composes the final reply guided by
``symptoms_final_reply.yaml``.

Why not ``agentic`` mode: pydantic-ai tool bodies are plain async
functions and can ``await`` repeatedly. The state-graph machinery
would be overkill for "ask N questions then return one payload". See
the disease-prediction plan's KTD-1 for the full rationale.

``post_process`` implements :class:`~claritymed.core.features.base.PostProcessHook`
as an audit-only safety-keyword check (KTD-2). The reply is returned
unchanged; the audit signal feeds the
``symptoms.safety_keywords.missing`` metric. No mutation, no flicker,
no prompt-driven retry — the v1 safety net is observability.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

import httpx

from claritymed.context import get_context_or_raise
from claritymed.core.features.base import FeatureMode, TurnContext
from claritymed.core.i18n.loader import t, t_list
from claritymed.core.interaction.prompt_channel import (
    InteractiveChannelUnavailable,
    UserDeclinedAnswer,
)
from claritymed.core.interaction.schemas import (
    AskUserQuestionInput,
    AskUserQuestionResult,
    NumericSpec,
    Question,
    QuestionOption,
)
from claritymed.core.observability.audit import audit_event
from claritymed.core.observability.audit_payloads import write_payload
from claritymed.core.prompts.registry import PromptRegistry
from claritymed.core.schemas.patient import Profile
from claritymed.core.symptoms.client import SymptomsServerClient
from claritymed.core.symptoms.conditions_catalog import (
    SymptomsConditionsCatalog,
    enumerate_dataset_condition_ids,
    validate_conditions_catalog,
)
from claritymed.core.symptoms.eligibility.base import (
    EligibilityResult,
    EligibilityStrategy,
)
from claritymed.core.symptoms.registry import DatasetRegistry
from claritymed.core.symptoms.schemas import DatasetSpec, SeverityTier, SymptomsConfig
from claritymed.core.symptoms.severity import tier_for_severity
from claritymed.errors import SymptomsServerUnreachableError
from claritymed.core.symptoms.wire import (
    DifferentialRow,
    StartSessionResponse,
    TurnResponse,
)

# Runtime imports for forward refs in tool-method annotations: pydantic-ai
# resolves the annotations via ``get_type_hints`` at toolset build time, so
# any name that appears in the string form (``RunContext[TurnState]``) must
# exist in this module's globals. Keeping them under TYPE_CHECKING produced
# a NameError once pydantic-ai actually inspected ``SymptomsFeature._predict``.
from pydantic_ai import RunContext

from claritymed.core.turn_state import TurnState

if TYPE_CHECKING:
    from pydantic_ai.toolsets import AbstractToolset

logger = logging.getLogger(__name__)

TOOL_NAME = "predict_disease_from_symptoms"
PROMPT_NAMES = (
    "predict_disease_from_symptoms_tool",
    "translate_complaint",
    "symptoms_final_reply",
)

# Tiers we audit for keyword compliance. Moderate / Mild reply structure
# doesn't carry urgent-care obligations, so missing a "see your doctor"
# keyword there is not a safety signal worth firing.
_AUDIT_TIERS = {"Critical", "Urgent"}

# Window of the LLM's reply scanned for safety keywords. Reply structure
# mandates safety language opens the message; checking the entire body
# would let "rest if it feels better" later in the message satisfy a
# Critical-tier audit.
_AUDIT_LEADING_CHARS = 600


_SAFETY_KEYWORD_TIERS: tuple[SeverityTier, ...] = (
    "Critical",
    "Urgent",
    "Moderate",
    "Mild",
)


def _validate_safety_keywords() -> None:
    """Fail-loud check: every safety-keyword tier resolves in en + zh.

    Used to live in Pydantic on ``SafetyKeywordsByTier``; moved here
    when the lists migrated into the i18n bundle. Same contract: a
    bilingual omission cannot silently disable the
    ``symptoms.safety_keywords.missing`` audit signal.
    """
    missing: list[str] = []
    for tier in _SAFETY_KEYWORD_TIERS:
        key = f"symptoms.safety_keywords.{tier}"
        for lang in ("en", "zh"):
            entries = t_list(key, lang=lang)
            if not entries:
                missing.append(f"{key}.{lang}")
                continue
            if any(not item.strip() for item in entries):
                raise RuntimeError(
                    f"i18n key {key!r} ({lang}) contains a blank entry; "
                    f"fill or delete it in configs/i18n/{lang}/symptoms.yaml"
                )
    if missing:
        raise RuntimeError(
            "Missing safety_keywords i18n entries: "
            + ", ".join(missing)
            + ". Define them in configs/i18n/<lang>/symptoms.yaml under "
            "symptoms.safety_keywords.<tier>."
        )


def _validate_symptoms_conditions_catalog(
    catalog: SymptomsConditionsCatalog, registry: DatasetRegistry
) -> None:
    """Fail-loud check: every enabled dataset's conditions are catalogued.

    Runs at plugin construct after prompts and safety-keyword checks.
    Enumerates ``condition_id`` slugs from each enabled dataset's own
    i18n YAML (the operator-owned source of truth for name → slug
    mapping) and calls :func:`validate_conditions_catalog` to assert
    both languages carry curated content. Empty registry (no datasets
    enabled) short-circuits to no-op — the plugin is effectively off
    in that case and the card renderer is unreachable.
    """
    condition_ids: set[str] = set()
    for dataset in registry.list_enabled():
        condition_ids.update(enumerate_dataset_condition_ids(dataset))
    validate_conditions_catalog(catalog, condition_ids)


def _validate_symptoms_prompts(registry: PromptRegistry) -> None:
    """Fail-loud check: every Unit 13 prompt loads bilingually.

    Mirrors ``_validate_ingest_prompts`` — a missing YAML at first call
    would silently degrade the LLM to no description. The toolset build
    is the right place to catch it; the registry is already in memory.
    """
    missing: list[str] = []
    for prompt_name in PROMPT_NAMES:
        for lang in ("en", "zh"):
            try:
                registry.get(prompt_name, language=lang)  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001
                missing.append(f"{prompt_name}.{lang}")
    if missing:
        raise RuntimeError(
            "Missing symptoms prompts: "
            + ", ".join(missing)
            + ". Add the YAML(s) under core/prompts/store/ before "
            "wiring the symptoms feature."
        )


# --- modal builders --------------------------------------------------------


def _confirm_question(language: str) -> Question:
    """Build the confirm modal from i18n keys.

    Copy lives in ``configs/i18n/<lang>/symptoms.yaml`` under
    ``symptoms.confirm.*`` so translators can edit it without touching
    Python. The modal renderer surfaces the rendered question and the
    yes/no options to the user.
    """
    return Question(
        question=t("symptoms.confirm.text", lang=language),
        header=t("symptoms.confirm.header", lang=language),
        options=[
            QuestionOption(
                label=t("symptoms.confirm.yes_label", lang=language),
                description=t("symptoms.confirm.yes_description", lang=language),
            ),
            QuestionOption(
                label=t("symptoms.confirm.no_label", lang=language),
                description=t("symptoms.confirm.no_description", lang=language),
            ),
        ],
    )


def _initial_batch(profile: Profile, language: str) -> AskUserQuestionInput:
    """Build the age + sex modal, omitting fields already on profile.

    KTD-13: age is collected as :class:`NumericSpec`; the server applies
    its own bucketing.
    """
    questions: list[Question] = []
    if profile.age is None:
        questions.append(_age_question(language))
    if profile.sex is None:
        questions.append(_sex_question(language))
    if not questions:
        # Should never reach here — caller checks first. Defensive: pad
        # with a no-op trivial question to satisfy schema's min_length.
        questions.append(_age_question(language))
    return AskUserQuestionInput(questions=questions)


def _age_question(language: str) -> Question:
    return Question(
        question=t("symptoms.age.question", lang=language),
        header=t("symptoms.age.header", lang=language),
        numeric=NumericSpec(
            min=0,
            max=120,
            step=1,
            unit=t("symptoms.age.unit", lang=language),
        ),
    )


def _sex_question(language: str) -> Question:
    return Question(
        question=t("symptoms.sex.question", lang=language),
        header=t("symptoms.sex.header", lang=language),
        options=[
            QuestionOption(
                label=t("symptoms.sex.female_label", lang=language),
                description=t("symptoms.sex.female_description", lang=language),
            ),
            QuestionOption(
                label=t("symptoms.sex.male_label", lang=language),
                description=t("symptoms.sex.male_description", lang=language),
            ),
        ],
    )


def _resolve_initial_batch(
    result: AskUserQuestionResult,
    *,
    profile: Profile,
    language: str,
) -> dict[str, Any]:
    """Reduce the modal result into the wire-format profile dict.

    Pulls ``age_years`` from ``numeric_values`` per the canonical
    Unit 11 shape, and translates the sex label to the wire literal
    (``"M"`` / ``"F"``). Falls back to the existing profile when a
    field was not asked. The wire format uses single-letter codes
    while the profile literal uses spelled-out values
    (``"female"`` / ``"male"`` / ``"intersex"`` / ``"unknown"``);
    :func:`_map_sex` handles both spellings.
    """
    # Answers come back keyed by the rendered question string — look up
    # the active language's rendering first, fall back to EN so a modal
    # rendered in one locale and resolved in another (mid-session
    # language flip) still picks up the answer.
    age_keys = (
        t("symptoms.age.question", lang=language),
        t("symptoms.age.question", lang="en"),
    )
    age = next(
        (result.numeric_values[k] for k in age_keys if k in result.numeric_values),
        None,
    )
    if age is None:
        age = profile.age if profile.age is not None else 30
    sex_keys = (
        t("symptoms.sex.question", lang=language),
        t("symptoms.sex.question", lang="en"),
    )
    sex_raw = next(
        (result.answers[k] for k in sex_keys if k in result.answers),
        None,
    )
    sex = _map_sex(sex_raw) or _map_sex(profile.sex) or "M"
    return {"age_years": int(age), "sex": sex}


def _map_sex(value: Any) -> str | None:
    """Coerce a sex label (modal pick or stored Profile literal) to
    the wire-format ``"M"`` / ``"F"`` code.

    DDXPlus only carries the binary axis; ``intersex`` / ``unknown``
    fall through and the caller defaults to ``"M"``. That's a known
    limitation of the upstream corpus, not a stance — flagged in the
    plan's Scope Boundaries.
    """
    if isinstance(value, list):
        value = value[0] if value else None
    if not value:
        return None
    s = str(value).strip().lower()
    if s in ("female", "f", "女"):
        return "F"
    if s in ("male", "m", "男"):
        return "M"
    return None


# --- tracing baggage (KTD-8) ------------------------------------------------


def _attach_symptoms_baggage(
    dataset_id: str, model_id: str, session_id: str
) -> object | None:
    """Attach the three KTD-8 baggage keys onto the active OTel context.

    Returns the attach token; ``None`` if OpenTelemetry is unavailable
    (mirrors :func:`claritymed.context.attach_session_baggage`).
    """
    try:
        from opentelemetry import baggage
        from opentelemetry import context as otel_context
        from opentelemetry.trace import get_current_span

        ctx = otel_context.get_current()
        ctx = baggage.set_baggage(
            "claritymed.symptoms.dataset_id", dataset_id, context=ctx
        )
        ctx = baggage.set_baggage("claritymed.symptoms.model_id", model_id, context=ctx)
        ctx = baggage.set_baggage(
            "claritymed.symptoms.session_id", session_id, context=ctx
        )
        span = get_current_span()
        if span.get_span_context() is not None and span.get_span_context().is_valid:
            span.set_attribute("claritymed.symptoms.dataset_id", dataset_id)
            span.set_attribute("claritymed.symptoms.model_id", model_id)
            span.set_attribute("claritymed.symptoms.session_id", session_id)
        return otel_context.attach(ctx)
    except ImportError:
        return None
    except Exception:  # noqa: BLE001
        logger.warning("symptoms baggage attach failed", exc_info=True)
        return None


def _detach_baggage(token: object | None) -> None:
    if token is None:
        return
    try:
        from opentelemetry import context as otel_context

        otel_context.detach(token)
    except Exception:  # noqa: BLE001
        pass


# --- plugin -----------------------------------------------------------------


class SymptomsFeature:
    """Tool-mode plugin exposing ``predict_disease_from_symptoms``.

    The constructor stores collaborators only; nothing is materialized
    until the LLM actually calls the tool. ``_validate_symptoms_prompts``
    runs at construct so a missing YAML fails the plugin build rather
    than the first request.
    """

    name = "symptoms"
    mode: FeatureMode = "tool"

    def __init__(
        self,
        *,
        config: SymptomsConfig,
        registry: DatasetRegistry,
        client: SymptomsServerClient,
        eligibility: EligibilityStrategy,
        prompt_registry: PromptRegistry | None = None,
        profile_loader: Callable[[str], Profile] | None = None,
        conditions_catalog: SymptomsConditionsCatalog | None = None,
    ) -> None:
        self._config = config
        self._registry = registry
        self._client = client
        self._eligibility = eligibility
        self._prompt_registry = prompt_registry or PromptRegistry()
        self._profile_loader = profile_loader or _default_profile_loader
        self._conditions_catalog = conditions_catalog or SymptomsConditionsCatalog()
        _validate_symptoms_prompts(self._prompt_registry)
        _validate_safety_keywords()
        _validate_symptoms_conditions_catalog(self._conditions_catalog, self._registry)
        # Per-request post_process state — keyed by request_id. Cleared
        # after post_process consumes it.
        self._stash: dict[str, dict[str, Any]] = {}

    async def pre_invoke(self, ctx: TurnContext) -> str:
        """No-op pre-invoke hook; symptoms feature needs no preamble injection."""
        return ""

    def system_prompt_fn(self) -> "Callable":
        """Return a dynamic system-prompt function for pydantic-ai.

        pydantic-ai calls this before each LLM request in the agent run.
        Returns the ``symptoms_final_reply`` prompt only after the tool
        has set ``deps.symptoms_reply_guide`` (i.e. on the reply-composition
        call, not the tool-selection call). Empty string when the tool was
        not called or returned user_declined / eligible:false / server_error.
        """

        def _fn(ctx: "RunContext[Any]") -> str:
            return getattr(ctx.deps, "symptoms_reply_guide", None) or ""

        return _fn

    def as_toolset(self) -> "AbstractToolset[Any] | None":
        """Return None; this feature exposes a single tool, not a toolset."""
        return None

    def as_tool(self) -> Callable | None:
        """Return a pydantic-ai Tool wrapping the predict_disease_from_symptoms body."""
        from pydantic_ai import Tool

        return Tool(
            self._predict,
            name=TOOL_NAME,
            description=self._build_tool_description(),
        )

    def _build_tool_description(self) -> str | None:
        """Build the tool description, injecting per-dataset domain text.

        The prompt YAML (v3+) contains a ``{covered_conditions}``
        placeholder where the dataset domain descriptions go.  This
        method reads each enabled dataset's ``domain_description["en"]``
        and substitutes the block so adding a new dataset only requires
        a config entry, not a prompt edit.  Falls back gracefully when
        the prompt is missing or the placeholder is absent (v1/v2).
        """
        try:
            template = self._prompt_registry.get(
                "predict_disease_from_symptoms_tool", language="en"
            )
        except Exception:  # noqa: BLE001
            return None
        if "{covered_conditions}" not in template:
            return template
        parts: list[str] = []
        for ds in self._registry.list_enabled():
            desc = ds.domain_description.get("en", "").strip()
            if desc:
                parts.append(desc)
        covered = (
            "\n\n".join(parts) if parts else "(no dataset descriptions configured)"
        )
        return template.replace("{covered_conditions}", covered)

    # --- tool body ----------------------------------------------------------

    async def _predict(
        self,
        ctx: "RunContext[TurnState]",
        complaint: str,
        symptom_summary: str | None = None,
        dataset_hint: str | None = None,
    ) -> dict[str, Any]:
        """Run the eligibility → confirm → loop pipeline.

        Always returns a dict (never raises into the LLM). Error
        translation happens here so the LLM sees a structured result
        for every branch.

        ``symptom_summary`` is the LLM's distilled clinical chief
        complaint — passed straight to the server's init-symptom
        matcher. Eligibility / audit / PHI guard all stay on
        ``complaint`` because they want the raw user text. See
        ``predict_disease_from_symptoms_tool.yaml`` for the LLM-facing
        contract.
        """
        deps = ctx.deps
        language = getattr(deps, "language", "en") or "en"
        # Guard against oversized inputs before they hit the wire schema validator.
        complaint = complaint[:4000] if complaint else complaint
        if symptom_summary:
            symptom_summary = symptom_summary[:4000]
        request_id, user_id, _ = get_context_or_raise()
        audit_event(
            "tool.predict_disease_from_symptoms",
            {
                "tool_name": TOOL_NAME,
                "dataset_hint": dataset_hint,
                "has_symptom_summary": symptom_summary is not None
                and bool(symptom_summary.strip()),
            },
        )

        eligibility_input = self._eligibility_input(complaint, symptom_summary)

        # Dataset routing: hint or single dataset → fast path.
        # No hint + multiple enabled datasets → score all via eligibility.
        enabled = self._registry.list_enabled()
        if not enabled:
            return {"eligible": False, "reason": "out_of_scope"}

        if dataset_hint or len(enabled) == 1:
            dataset = self._resolve_dataset(dataset_hint, complaint)
            if dataset is None:
                return {"eligible": False, "reason": "out_of_scope"}
            elig = await self._run_eligibility(
                deps, eligibility_input, dataset, language
            )
        else:
            dataset, elig = await self._route_by_eligibility(
                deps, eligibility_input, language
            )
            if dataset is None:
                return {"eligible": False, "reason": "out_of_scope"}

        if not elig.eligible:
            return self._reject(dataset.id, elig)

        channel = getattr(deps, "prompt_channel", None)
        if channel is None:
            audit_event(
                "symptoms.session.ineligible",
                {
                    "dataset_id": dataset.id,
                    "strategy_id": self._strategy_id(),
                    "reason": "no_interactive_channel",
                },
            )
            return {"eligible": False, "reason": "no_interactive_channel"}

        try:
            if not await self._confirm(channel, language):
                audit_event(
                    "symptoms.session.cancelled",
                    {"phase": "confirm", "turn_index": 0},
                )
                return {"eligible": True, "user_declined": True}
        except InteractiveChannelUnavailable:
            return {"eligible": False, "reason": "no_interactive_channel"}

        profile = await self._load_profile(user_id)
        try:
            wire_profile = await self._collect_initial_batch(channel, profile, language)
        except UserDeclinedAnswer:
            audit_event(
                "symptoms.session.cancelled",
                {"phase": "initial_batch", "turn_index": 0},
            )
            return {"eligible": True, "user_declined": True}
        except InteractiveChannelUnavailable:
            return {"eligible": False, "reason": "no_interactive_channel"}

        result = await self._run_sub_session(
            deps=deps,
            channel=channel,
            dataset=dataset,
            complaint=complaint,
            symptom_summary=symptom_summary,
            wire_profile=wire_profile,
            language=language,
            request_id=request_id,
            user_id=user_id,
        )
        # Emit the sidecar DifferentialReady event before the LLM starts
        # composing the summary above the cards. Cards render as soon as
        # the event hits the SSE stream; subsequent token_chunk events
        # populate the summary paragraph above them. See PR-C of the
        # feat-symptoms-multi-card-render spec.
        await self._maybe_emit_differential_ready(deps, result, language)
        # Inject the composing-guide into deps so the dynamic system_prompt
        # fn (registered by make_ask_agent) can surface it on the second
        # LLM call (reply composition). Only set when there's a usable
        # differential — user_declined / eligible:false / server_error leave
        # deps.symptoms_reply_guide as None so no extra prompt is added.
        if result.get("differential") or result.get("partial_differential"):
            try:
                guide = self._prompt_registry.get(
                    "symptoms_final_reply", language=language
                )
                ctx.deps.symptoms_reply_guide = guide  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass
        return result

    async def _maybe_emit_differential_ready(
        self, deps: "TurnState", result: dict[str, Any], language: str
    ) -> None:
        """Hydrate and emit the ``DifferentialReady`` sidecar event.

        No-op on branches the card renderer intentionally skips
        (``user_declined`` / ``eligible=False`` / ``server_error``, plus
        any tool-side error where the raw row stash was not populated).
        Failure to emit is logged and swallowed — a broken event must
        not break the user's turn.
        """
        try:
            from claritymed.orchestrator.features.symptoms_card_builder import (
                build_differential_ready,
            )

            event = build_differential_ready(
                result,
                catalog=self._conditions_catalog,
                language=language,  # type: ignore[arg-type]
                top_n_cap=self._config.card_renderer.top_n_cap,
            )
            if event is None:
                return
            eq = getattr(deps, "event_queue", None)
            if eq is None:
                return
            await eq.put(event)
            # Stash on deps too so ``AskService._finalize_turn`` can
            # persist the payload on the assistant turn's JSONL event.
            # Without this the cards live only in the SSE stream and
            # vanish on page refresh — see ChatTurn.differential + the
            # session_resume regression test in tests/orchestrator/
            # services/test_chat_session.py.
            try:
                deps.differential_ready = event  # type: ignore[union-attr]
            except AttributeError:
                # ``deps`` is a Protocol; concrete stubs in tests may
                # not carry the slot. Safe to skip — persistence is
                # a nice-to-have, not the emit contract.
                pass
        except Exception:  # noqa: BLE001
            logger.warning("symptoms: failed to emit DifferentialReady", exc_info=True)

    # --- helpers ------------------------------------------------------------

    def _eligibility_input(self, complaint: str, symptom_summary: str | None) -> str:
        """Pick the eligibility-check input per the catalog config.

        ``input_source="symptom_summary"`` opts into the LLM-distilled
        clinical phrase when it exists; falls back to ``complaint`` if
        the LLM omitted the optional argument or sent whitespace. The
        default (``"complaint"``) always returns the raw user text —
        see :class:`EligibilityCatalogConfig` for the tradeoff.
        """
        source = self._config.eligibility.input_source
        if source == "symptom_summary" and symptom_summary and symptom_summary.strip():
            return symptom_summary.strip()
        return complaint

    def _resolve_dataset(self, hint: str | None, complaint: str) -> DatasetSpec | None:
        dataset = self._registry.resolve(hint)
        if dataset is None and hint:
            audit_event(
                "symptoms.session.ineligible",
                {
                    "dataset_id": "<none>",
                    "strategy_id": self._strategy_id(),
                    "reason": "unknown_hint",
                },
            )
        return dataset

    async def _run_eligibility(
        self,
        deps: "TurnState",
        complaint: str,
        dataset: DatasetSpec,
        language: str,
    ) -> EligibilityResult:
        profile = await self._load_profile(deps.user_id)
        strategy_id = self._strategy_id()
        logger.info(
            "eligibility check: dataset=%s strategy=%s",
            dataset.id,
            strategy_id,
        )
        result = await self._eligibility.check(complaint, language, profile, dataset)
        logger.info(
            "eligibility result: dataset=%s strategy=%s eligible=%s confidence=%.3f reason=%r",
            dataset.id,
            strategy_id,
            result.eligible,
            result.confidence,
            result.reason,
        )
        audit_event(
            "symptoms.eligibility.checked",
            {
                "dataset_id": dataset.id,
                "strategy_id": strategy_id,
                "confidence": result.confidence,
                "eligible": result.eligible,
            },
        )
        return result

    async def _route_by_eligibility(
        self,
        deps: "TurnState",
        eligibility_input: str,
        language: str,
    ) -> "tuple[DatasetSpec | None, EligibilityResult]":
        """Score all enabled datasets and return the best match.

        Called only when no dataset_hint is provided and more than one
        dataset is enabled.  Each dataset gets its own eligibility check
        and audit event so the routing decision is fully traceable.
        Single-dataset deployments never reach this path.
        """
        from claritymed.core.symptoms.eligibility.base import EligibilityResult as _ER

        profile = await self._load_profile(deps.user_id)
        strategy_id = self._strategy_id()
        scores: dict[str, float] = {}
        results: dict[str, EligibilityResult] = {}
        for ds in self._registry.list_enabled():
            r = await self._eligibility.check(eligibility_input, language, profile, ds)
            logger.info(
                "eligibility routing: dataset=%s strategy=%s eligible=%s confidence=%.3f",
                ds.id,
                strategy_id,
                r.eligible,
                r.confidence,
            )
            scores[ds.id] = r.confidence
            results[ds.id] = r
            audit_event(
                "symptoms.eligibility.checked",
                {
                    "dataset_id": ds.id,
                    "strategy_id": strategy_id,
                    "confidence": r.confidence,
                    "eligible": r.eligible,
                },
            )
        dataset = self._registry.resolve(None, eligibility_scores=scores)
        if dataset is None:
            return None, _ER(eligible=False, reason="out_of_scope", confidence=0.0)
        return dataset, results[dataset.id]

    def _reject(self, dataset_id: str, elig: EligibilityResult) -> dict[str, Any]:
        audit_event(
            "symptoms.session.ineligible",
            {
                "dataset_id": dataset_id,
                "strategy_id": self._strategy_id(),
                "reason": elig.reason,
            },
        )
        return {"eligible": False, "reason": elig.reason}

    def _strategy_id(self) -> str:
        return self._config.eligibility.active

    async def _load_profile(self, user_id: str) -> Profile:
        try:
            return self._profile_loader(user_id)
        except Exception:  # noqa: BLE001
            logger.warning("profile load failed for %s", user_id, exc_info=True)
            return Profile()

    async def _confirm(self, channel: Any, language: str) -> bool:
        payload = AskUserQuestionInput(questions=[_confirm_question(language)])
        try:
            result = await channel.ask(payload)
        except UserDeclinedAnswer:
            return False
        # The user picks Yes / No. Anything truthy non-No is treated as Yes.
        text = next(iter(result.answers.values()), "")
        if isinstance(text, list):
            text = text[0] if text else ""
        return str(text).strip().lower() in ("yes", "是", "y")

    async def _collect_initial_batch(
        self, channel: Any, profile: Profile, language: str
    ) -> dict[str, Any]:
        profile_sex_wire = _map_sex(profile.sex)
        if profile.age is not None and profile_sex_wire is not None:
            return {
                "age_years": int(profile.age),
                "sex": profile_sex_wire,
            }
        payload = _initial_batch(profile, language)
        result = await channel.ask(payload)
        return _resolve_initial_batch(result, profile=profile, language=language)

    async def _run_sub_session(
        self,
        *,
        deps: "TurnState",
        channel: Any,
        dataset: DatasetSpec,
        complaint: str,
        symptom_summary: str | None,
        wire_profile: dict[str, Any],
        language: str,
        request_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        try:
            start = await self._client.start_session(
                dataset.id,
                complaint,
                wire_profile,
                language=language,
                symptom_summary=symptom_summary,
                request_id=request_id,
            )
        except SymptomsServerUnreachableError:
            audit_event(
                "symptoms.session.cancelled",
                {"phase": "start", "reason": "server_unreachable"},
            )
            return {"eligible": True, "server_error": True}
        except httpx.HTTPStatusError:
            audit_event(
                "symptoms.session.cancelled",
                {"phase": "start", "reason": "server_error"},
            )
            return {"eligible": True, "server_error": True}

        baggage_token = _attach_symptoms_baggage(
            dataset.id, dataset.primary_model_id(), start.session_id
        )
        audit_event(
            "symptoms.session.started",
            {
                "dataset_id": dataset.id,
                "model_id": dataset.primary_model_id(),
                "session_id": start.session_id,
            },
        )
        try:
            return await self._loop_until_done(
                deps=deps,
                channel=channel,
                dataset=dataset,
                start=start,
                language=language,
                request_id=request_id,
                user_id=user_id,
            )
        finally:
            _detach_baggage(baggage_token)

    async def _loop_until_done(
        self,
        *,
        deps: "TurnState",
        channel: Any,
        dataset: DatasetSpec,
        start: StartSessionResponse,
        language: str,
        request_id: str,
        user_id: str,
    ) -> dict[str, Any]:
        """Drive the question-answer loop until done, cap-hit, cancel, or error."""
        question = start.first_question
        turn_index = 0
        transcript: list[dict[str, Any]] = []
        while True:
            payload = AskUserQuestionInput(questions=[question])
            try:
                answer = await channel.ask(payload)
            except UserDeclinedAnswer:
                return await self._handle_cancel(
                    dataset=dataset,
                    session_id=start.session_id,
                    request_id=request_id,
                    user_id=user_id,
                    transcript=transcript,
                    turn_index=turn_index,
                )
            except InteractiveChannelUnavailable:
                await self._handle_cancel(
                    dataset=dataset,
                    session_id=start.session_id,
                    request_id=request_id,
                    user_id=user_id,
                    transcript=transcript,
                    turn_index=turn_index,
                )
                return {"eligible": False, "reason": "no_interactive_channel"}
            answer_text = _first_answer(answer)
            answer_value = _pick_answer_value(answer, question)
            transcript.append({"question": question.question, "answer": answer_text})
            try:
                turn_resp = await self._client.turn(
                    dataset.id,
                    start.session_id,
                    answer_text,
                    answer_value=answer_value,
                    language=language,
                    request_id=request_id,
                )
            except SymptomsServerUnreachableError:
                audit_event(
                    "symptoms.session.cancelled",
                    {"phase": "turn", "reason": "server_unreachable"},
                )
                return {"eligible": True, "server_error": True}
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    return {"eligible": True, "session_expired": True}
                return {"eligible": True, "server_error": True}

            turn_index += 1
            audit_event(
                "symptoms.session.turn",
                {
                    "turn_index": turn_index,
                    "question_id": question.header,
                    "answer_type": "numeric" if question.numeric else "categorical",
                },
            )
            if turn_resp.done:
                return self._handle_done(
                    dataset=dataset,
                    turn_resp=turn_resp,
                    request_id=request_id,
                    user_id=user_id,
                    transcript=transcript,
                )
            if turn_resp.hit_cap:
                return self._handle_cap(
                    dataset=dataset,
                    turn_resp=turn_resp,
                    request_id=request_id,
                    user_id=user_id,
                    transcript=transcript,
                )
            if turn_resp.next_question is None:
                # Defensive: malformed server response — surface as error
                # rather than infinite-loop.
                return {"eligible": True, "server_error": True}
            question = turn_resp.next_question

    def _finalize_session(
        self,
        *,
        user_id: str,
        request_id: str,
        diff: list[DifferentialRow],
        transcript: list[dict[str, Any]],
        payload_kind: str,
        diff_key: str,
    ) -> list[dict[str, Any]]:
        """Write PHI payload and populate the post_process stash.

        Shared by done and cap outcomes — both write the differential
        (under different keys) and stash max_severity for keyword audit.
        Returns the serialized diff_raw list so callers can reference it
        in their audit events without serializing twice.
        """
        diff_raw = [d.model_dump() for d in diff]
        write_payload(
            user_id,
            request_id,
            {"kind": payload_kind, diff_key: diff_raw, "transcript": transcript},
        )
        self._stash[request_id] = {"max_severity": _max_severity(diff)}
        return diff_raw

    def _handle_done(
        self,
        *,
        dataset: DatasetSpec,
        turn_resp: TurnResponse,
        request_id: str,
        user_id: str,
        transcript: list[dict[str, Any]],
    ) -> dict[str, Any]:
        diff_raw = self._finalize_session(
            user_id=user_id,
            request_id=request_id,
            diff=turn_resp.differential,
            transcript=transcript,
            payload_kind="symptoms.session.completed",
            diff_key="differential",
        )
        audit_event(
            "symptoms.session.completed",
            {
                "dataset_id": dataset.id,
                "model_id": dataset.primary_model_id(),
                "session_id": "<elided>",
                "turns_used": turn_resp.turn_count,
                "severity_tier": _tier_for(turn_resp.differential),
                "top_condition_id": diff_raw[0]["condition_id"] if diff_raw else None,
            },
        )
        return {
            "eligible": True,
            "differential": _format_differential(turn_resp.differential),
            "turns_used": turn_resp.turn_count,
            # Raw rows retained for the card builder — carries the
            # canonical condition_id slug the plugin's LLM-facing
            # _format_differential drops. Not sent to the LLM (the
            # underscore prefix keeps pydantic-ai's tool-result summary
            # readable) but used by _maybe_emit_differential_ready.
            "_raw_differential": list(turn_resp.differential),
        }

    def _handle_cap(
        self,
        *,
        dataset: DatasetSpec,
        turn_resp: TurnResponse,
        request_id: str,
        user_id: str,
        transcript: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self._finalize_session(
            user_id=user_id,
            request_id=request_id,
            diff=turn_resp.partial_differential,
            transcript=transcript,
            payload_kind="symptoms.session.cap_hit",
            diff_key="partial_differential",
        )
        audit_event(
            "symptoms.session.cap_hit",
            {
                "turns_used": turn_resp.turn_count,
                "partial_confidence": turn_resp.partial_confidence,
                "severity_tier": _tier_for(turn_resp.partial_differential),
            },
        )
        return {
            "eligible": True,
            "hit_cap": True,
            "partial_differential": _format_differential(
                turn_resp.partial_differential
            ),
            "turns_used": turn_resp.turn_count,
            "partial_confidence": turn_resp.partial_confidence,
            "_raw_partial_differential": list(turn_resp.partial_differential),
        }

    async def _handle_cancel(
        self,
        *,
        dataset: DatasetSpec,
        session_id: str,
        request_id: str,
        user_id: str,
        transcript: list[dict[str, Any]],
        turn_index: int,
    ) -> dict[str, Any]:
        """Cancel the active session and return a structured partial-result payload."""
        try:
            cancel = await self._client.cancel(
                dataset.id, session_id, request_id=request_id
            )
        except SymptomsServerUnreachableError:
            audit_event(
                "symptoms.session.cancelled",
                {"phase": "cancel", "reason": "server_unreachable"},
            )
            return {"eligible": True, "server_error": True}
        diff = [d.model_dump() for d in cancel.partial_differential]
        audit_event(
            "symptoms.session.cancelled",
            {
                "phase": "loop",
                "turn_index": turn_index,
                "partial_confidence": cancel.partial_confidence,
                "meets_confidence_threshold": cancel.meets_confidence_threshold,
                "severity_override_fired": cancel.severity_override,
            },
        )
        write_payload(
            user_id,
            request_id,
            {
                "kind": "symptoms.session.cancelled",
                "partial_differential": diff,
                "transcript": transcript,
            },
        )
        # Stash for post_process — pick tier source per the cancel
        # decision table.
        if cancel.severity_override and cancel.max_low_severity_seen is not None:
            self._stash[request_id] = {"max_severity": cancel.max_low_severity_seen}
        elif cancel.meets_confidence_threshold:
            self._stash[request_id] = {
                "max_severity": _max_severity(cancel.partial_differential),
            }
        # else: no stash entry → post_process treats as no audit needed.
        partial = _format_differential(cancel.partial_differential)
        # Surface partial_differential when either confidence threshold is
        # met (case C) OR severity_override fires (case E). The server
        # already populates partial_differential for both conditions via
        # ``show_partial = meets_threshold or override_fired``; here we
        # mirror that logic so the LLM has the actual conditions for case E.
        show_partial = cancel.meets_confidence_threshold or cancel.severity_override
        return {
            "eligible": True,
            "cancelled": True,
            "partial_differential": partial if show_partial else None,
            "turns_used": cancel.turn_count,
            "partial_confidence": cancel.partial_confidence,
            "meets_confidence_threshold": cancel.meets_confidence_threshold,
            "severity_override": cancel.severity_override,
            "max_low_severity_seen": cancel.max_low_severity_seen,
            "_raw_partial_differential": (
                list(cancel.partial_differential) if show_partial else []
            ),
        }

    # --- post_process (PostProcessHook) -------------------------------------

    async def post_process(self, text: str, tool_result: dict[str, Any]) -> str:
        """Audit-only safety-keyword check (KTD-2).

        Always returns ``text`` unchanged. Emits
        ``symptoms.safety_keywords.missing`` when a Critical / Urgent
        tier reply does not contain a tier-appropriate keyword. The
        keyword lists live in ``configs/i18n/<lang>/symptoms.yaml``
        under ``symptoms.safety_keywords.<tier>`` and are read via
        :func:`t_list` so translators own them.
        """
        try:
            request_id, _, language = get_context_or_raise()
        except Exception:  # noqa: BLE001
            return text
        stash = self._stash.pop(request_id, None)
        if stash is None:
            return text
        max_sev = stash.get("max_severity")
        if max_sev is None:
            return text
        tier = tier_for_severity(max_sev)
        if tier not in _AUDIT_TIERS:
            return text
        keywords = t_list(f"symptoms.safety_keywords.{tier}", lang=language)
        leading = text[:_AUDIT_LEADING_CHARS].lower()
        if any(kw.lower() in leading for kw in keywords):
            return text
        audit_event(
            "symptoms.safety_keywords.missing",
            {
                "max_severity": int(max_sev),
                "tier": tier,
                "observed_leading_chars": min(_AUDIT_LEADING_CHARS, len(text)),
            },
        )
        return text


# --- module helpers --------------------------------------------------------


def _format_differential(rows: list[DifferentialRow]) -> list[dict[str, Any]]:
    """Build the LLM-facing differential — only the fields the reply prompt uses.

    Drops condition_id (internal slug), condition_idx (algorithm index),
    and icd10 (the prompt explicitly tells the LLM not to show ICD codes).
    Severity is a 1-5 integer: 1=Critical, 2=Urgent, 3=Moderate, 4-5=Mild.
    """
    return [
        {
            "condition_name": d.condition_name,
            "probability": round(d.probability, 3),
            "severity": d.severity,
        }
        for d in rows
    ]


def _first_answer(result: AskUserQuestionResult) -> Any:
    if result.numeric_values:
        return next(iter(result.numeric_values.values()))
    if result.answers:
        v = next(iter(result.answers.values()))
        if isinstance(v, list):
            return v[0] if v else ""
        return v
    return ""


def _pick_answer_value(
    result: AskUserQuestionResult,
    question: "Question",
) -> "str | list[str] | None":
    """Extract the ``QuestionOption.value`` for the user's pick.

    Returns the raw value identifier (e.g. ``"V_0"``, ``"yes"``) that the
    server can use as ``answer_value`` to skip label re-matching. Returns
    ``None`` for numeric answers (no raw code) and when options carry no
    ``value`` field.
    """
    if result.numeric_values or not result.answers or not question.options:
        return None
    raw_answer = next(iter(result.answers.values()))
    by_label: dict[str, str | None] = {opt.label: opt.value for opt in question.options}
    if isinstance(raw_answer, list):
        values = [by_label.get(lbl) for lbl in raw_answer]
        filtered = [v for v in values if v is not None]
        return filtered if filtered else None
    return by_label.get(raw_answer)


def _max_severity(diff: list[DifferentialRow]) -> int | None:
    """Lowest severity number = most severe; ``None`` for empty diffs."""
    if not diff:
        return None
    return min(d.severity for d in diff)


def _tier_for(diff: list[DifferentialRow]) -> str | None:
    sev = _max_severity(diff)
    if sev is None:
        return None
    return tier_for_severity(sev)


def _default_profile_loader(user_id: str) -> Profile:
    """Read the user's stored profile via :class:`ProfileStore`.

    Wrapped in the indirection layer so tests can inject a stub without
    setting up a temporary SQLite database.
    """
    from claritymed.stores.profile import ProfileStore

    store = ProfileStore(user_id)
    profile = store.get_profile()
    return profile or Profile()


# --- production factory --------------------------------------------------------


def _load_ddxplus_vocab(data_dir: str) -> "dict[str, frozenset[str]]":
    """Build evidence_id → EN phrase set from release_evidences.json.

    Used by :func:`make_symptoms_factory` to populate the vocab map for
    :class:`~claritymed.core.symptoms.eligibility.direct.DirectEligibility`.
    Returns an empty dict when the file is absent (data not yet downloaded).
    """
    import json
    from pathlib import Path as _Path

    evidences_path = _Path(data_dir) / "release_evidences.json"
    if not evidences_path.exists():
        return {}
    with evidences_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    per_evidence: dict[str, frozenset[str]] = {}
    for ev_code, ev_data in raw.items():
        phrases: set[str] = set()
        if q_en := ev_data.get("question_en"):
            phrases.add(q_en)
        for val_meanings in (ev_data.get("value_meaning") or {}).values():
            if isinstance(val_meanings, dict) and (en_label := val_meanings.get("en")):
                phrases.add(en_label)
        if phrases:
            per_evidence[ev_code] = frozenset(phrases)
    return per_evidence


def _load_ddxplus_sidecar(data_dir: str) -> "dict[str, str]":
    """Load evidence_id → concept_id map produced by ``prepare.py --build-sidecar``.

    Consumed by :class:`~claritymed.core.symptoms.eligibility.term_service.TermServiceEligibility`.
    Returns an empty dict when the file is absent (sidecar not yet built).
    """
    import json
    from pathlib import Path as _Path

    path = _Path(data_dir) / "evidence_concepts.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def make_symptoms_factory(
    base_url: str = "http://127.0.0.1:8084",
) -> "Callable[[], SymptomsFeature] | None":
    """Build the :class:`SymptomsFeature` factory for :class:`~claritymed.orchestrator.services.ask_service.AskService`.

    Returns ``None`` when:
    * ``configs/symptoms.yaml`` is absent or malformed.
    * All datasets have ``enabled: false``.
    * The active eligibility strategy cannot be constructed (provider
      unreachable, config error).

    ``None`` silently skips the symptoms tool so the LLM still answers
    with free text — the tool is treated as "not installed" rather than
    "broken".
    """
    from pathlib import Path as _Path

    from claritymed.config import DATA_DIR
    from claritymed.core.rag.terms.factory import build_term_service
    from claritymed.core.symptoms.eligibility.direct import EvidenceVocabMap
    from claritymed.core.symptoms.eligibility.factory import build_eligibility_strategy
    from claritymed.core.symptoms.eligibility.term_service import SidecarMap
    from claritymed.core.symptoms.registry import DatasetRegistry
    from claritymed.errors import (
        EligibilityStrategyConfigError,
        EligibilityStrategyUnavailableError,
    )

    try:
        from claritymed.config import load_symptoms_config

        config = load_symptoms_config()
    except Exception:
        logger.warning("symptoms: config load failed; feature disabled", exc_info=True)
        return None

    enabled = [d for d in config.datasets if d.enabled]
    if not enabled:
        logger.info("symptoms: no enabled datasets; feature disabled")
        return None

    vocabs: EvidenceVocabMap = {}
    sidecars: SidecarMap = {}
    for ds in enabled:
        # Subset variants (e.g. ddxplus_pneumonia_flu) share the parent
        # dataset's evidence pool + i18n; key vocabs off adapter_id, then
        # register under this dataset's own id so DirectEligibility's
        # per-dataset lookup finds them.
        if ds.resolved_adapter_id() == "ddxplus":
            data_dir = _Path(str(DATA_DIR)) / "symptoms" / "ddxplus"
            per_evidence = _load_ddxplus_vocab(data_dir)
            if per_evidence:
                vocabs[ds.id] = per_evidence
            else:
                logger.warning(
                    "symptoms: ddxplus vocab not found at %s; "
                    "eligibility direct-match will return strategy_unavailable",
                    data_dir,
                )
            ev_concepts = _load_ddxplus_sidecar(data_dir)
            if ev_concepts:
                sidecars[ds.id] = ev_concepts
            else:
                logger.info(
                    "symptoms: ddxplus evidence_concepts.json not found at %s; "
                    "term_service eligibility will report strategy_unavailable "
                    "until `prepare.py --build-sidecar` is run",
                    data_dir,
                )

    try:
        eligibility = build_eligibility_strategy(
            config.eligibility,
            vocabs=vocabs,
            sidecars=sidecars,
            term_service_factory=build_term_service,
        )
    except (EligibilityStrategyConfigError, EligibilityStrategyUnavailableError) as exc:
        logger.warning(
            "symptoms: eligibility strategy unavailable (%s); feature disabled", exc
        )
        return None
    except Exception:
        logger.warning(
            "symptoms: eligibility strategy build failed; feature disabled",
            exc_info=True,
        )
        return None

    registry = DatasetRegistry(config.datasets)
    client = SymptomsServerClient(base_url)

    def _factory() -> SymptomsFeature:
        return SymptomsFeature(
            config=config,
            registry=registry,
            client=client,
            eligibility=eligibility,
        )

    return _factory
