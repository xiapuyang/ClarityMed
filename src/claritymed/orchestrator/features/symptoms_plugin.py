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

from claritymed.context import get_context_or_raise
from claritymed.core.features.base import FeatureMode, TurnContext
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
from claritymed.core.symptoms.eligibility.base import (
    EligibilityResult,
    EligibilityStrategy,
)
from claritymed.core.symptoms.registry import DatasetRegistry
from claritymed.core.symptoms.schemas import DatasetSpec, SymptomsConfig
from claritymed.core.symptoms.severity import tier_for_severity
from claritymed.errors import SymptomsServerUnreachableError
from claritymed.servers.symptoms.wire import (
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
    "translate_complaint_to_en",
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
    """Bilingual confirm modal copy.

    Lives inline (not in the prompt registry) per Unit 13's scope note:
    the registry's YAMLs are LLM-facing; this string only reaches the
    user via the modal renderer.
    """
    en_text = (
        "I can run a short symptom-driven differential (~5-12 follow-up "
        "questions) to narrow down what this might be. Try it?"
    )
    zh_text = "我可以做一轮基于症状的鉴别诊断（约 5-12 个追问）来缩小范围。要试试吗？"
    en_opts = [
        QuestionOption(label="Yes", description="Run the symptom Q&A loop."),
        QuestionOption(label="No", description="Skip and answer with free text."),
    ]
    zh_opts = [
        QuestionOption(label="是", description="开始症状问答流程。"),
        QuestionOption(label="否", description="跳过，按自由文本回答。"),
    ]
    return Question(
        question=zh_text if language == "zh" else en_text,
        header="Try follow-up?" if language != "zh" else "试一下吗?",
        options=zh_opts if language == "zh" else en_opts,
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
    en_q = "How old are you, in years?"
    zh_q = "请问您今年多大（岁）？"
    return Question(
        question=zh_q if language == "zh" else en_q,
        header="Age" if language != "zh" else "年龄",
        numeric=NumericSpec(min=0, max=120, step=1, unit="years"),
    )


def _sex_question(language: str) -> Question:
    en_q = "Biological sex (for the differential model)?"
    zh_q = "生理性别（用于鉴别诊断模型）？"
    if language == "zh":
        opts = [
            QuestionOption(label="女", description="生理性别为女。"),
            QuestionOption(label="男", description="生理性别为男。"),
        ]
    else:
        opts = [
            QuestionOption(label="Female", description="Biological sex female."),
            QuestionOption(label="Male", description="Biological sex male."),
        ]
    return Question(
        question=zh_q if language == "zh" else en_q,
        header="Sex" if language != "zh" else "性别",
        options=opts,
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
    age = result.numeric_values.get("How old are you, in years?")
    age = age or result.numeric_values.get("请问您今年多大（岁）？")
    if age is None:
        age = profile.age if profile.age is not None else 30
    sex_raw = result.answers.get("Biological sex (for the differential model)?")
    sex_raw = sex_raw or result.answers.get("生理性别（用于鉴别诊断模型）？")
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

        ctx = otel_context.get_current()
        ctx = baggage.set_baggage(
            "claritymed.symptoms.dataset_id", dataset_id, context=ctx
        )
        ctx = baggage.set_baggage("claritymed.symptoms.model_id", model_id, context=ctx)
        ctx = baggage.set_baggage(
            "claritymed.symptoms.session_id", session_id, context=ctx
        )
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
    ) -> None:
        self._config = config
        self._registry = registry
        self._client = client
        self._eligibility = eligibility
        self._prompt_registry = prompt_registry or PromptRegistry()
        self._profile_loader = profile_loader or _default_profile_loader
        _validate_symptoms_prompts(self._prompt_registry)
        # Per-request post_process state — keyed by request_id. Cleared
        # after post_process consumes it.
        self._stash: dict[str, dict[str, Any]] = {}

    async def pre_invoke(self, ctx: TurnContext) -> str:
        return ""

    def as_toolset(self) -> "AbstractToolset[Any] | None":
        return None

    def as_tool(self) -> Callable | None:
        from pydantic_ai import Tool

        try:
            description = self._prompt_registry.get(
                "predict_disease_from_symptoms_tool", language="en"
            )
        except Exception:  # noqa: BLE001
            description = None
        return Tool(
            self._predict,
            name=TOOL_NAME,
            description=description,
        )

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

        dataset = self._resolve_dataset(dataset_hint, complaint)
        if dataset is None:
            return {"eligible": False, "reason": "out_of_scope"}

        elig = await self._run_eligibility(deps, complaint, dataset, language)
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

        return await self._run_sub_session(
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

    # --- helpers ------------------------------------------------------------

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
        result = await self._eligibility.check(complaint, language, profile, dataset)
        audit_event(
            "symptoms.eligibility.checked",
            {
                "dataset_id": dataset.id,
                "strategy_id": self._strategy_id(),
                "confidence": result.confidence,
                "eligible": result.eligible,
            },
        )
        return result

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
            )
        except SymptomsServerUnreachableError:
            audit_event(
                "symptoms.session.cancelled",
                {"phase": "start", "reason": "server_unreachable"},
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
            answer_text = _first_answer(answer)
            transcript.append({"question": question.question, "answer": answer_text})
            try:
                turn_resp = await self._client.turn(
                    dataset.id, start.session_id, answer_text, language=language
                )
            except SymptomsServerUnreachableError:
                audit_event(
                    "symptoms.session.cancelled",
                    {"phase": "turn", "reason": "server_unreachable"},
                )
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

    def _handle_done(
        self,
        *,
        dataset: DatasetSpec,
        turn_resp: TurnResponse,
        request_id: str,
        user_id: str,
        transcript: list[dict[str, Any]],
    ) -> dict[str, Any]:
        diff = [d.model_dump() for d in turn_resp.differential]
        audit_event(
            "symptoms.session.completed",
            {
                "dataset_id": dataset.id,
                "model_id": dataset.primary_model_id(),
                "session_id": "<elided>",
                "turns_used": turn_resp.turn_count,
                "severity_tier": _tier_for(turn_resp.differential),
                "top_condition_id": diff[0]["condition_id"] if diff else None,
            },
        )
        write_payload(
            user_id,
            request_id,
            {
                "kind": "symptoms.session.completed",
                "differential": diff,
                "transcript": transcript,
            },
        )
        result = {
            "eligible": True,
            "differential": diff,
            "evidence_collected": [
                e.model_dump() for e in turn_resp.evidence_collected
            ],
            "turn_count": turn_resp.turn_count,
        }
        self._stash[request_id] = {
            "max_severity": _max_severity(turn_resp.differential),
        }
        return result

    def _handle_cap(
        self,
        *,
        dataset: DatasetSpec,
        turn_resp: TurnResponse,
        request_id: str,
        user_id: str,
        transcript: list[dict[str, Any]],
    ) -> dict[str, Any]:
        diff = [d.model_dump() for d in turn_resp.partial_differential]
        tier = _tier_for(turn_resp.partial_differential)
        audit_event(
            "symptoms.session.cap_hit",
            {
                "turns_used": turn_resp.turn_count,
                "partial_confidence": turn_resp.partial_confidence,
                "severity_tier": tier,
            },
        )
        write_payload(
            user_id,
            request_id,
            {
                "kind": "symptoms.session.cap_hit",
                "partial_differential": diff,
                "transcript": transcript,
            },
        )
        self._stash[request_id] = {
            "max_severity": _max_severity(turn_resp.partial_differential),
        }
        return {
            "eligible": True,
            "hit_cap": True,
            "partial_differential": diff,
            "evidence_collected": [
                e.model_dump() for e in turn_resp.evidence_collected
            ],
            "turn_count": turn_resp.turn_count,
            "partial_confidence": turn_resp.partial_confidence,
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
        try:
            cancel = await self._client.cancel(dataset.id, session_id)
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
        return {
            "eligible": True,
            "cancelled": True,
            "partial_differential": diff if cancel.meets_confidence_threshold else None,
            "evidence_collected": [e.model_dump() for e in cancel.evidence_collected],
            "turn_count": cancel.turn_count,
            "partial_confidence": cancel.partial_confidence,
            "meets_confidence_threshold": cancel.meets_confidence_threshold,
            "severity_override": cancel.severity_override,
            "max_low_severity_seen": cancel.max_low_severity_seen,
        }

    # --- post_process (PostProcessHook) -------------------------------------

    async def post_process(self, text: str, tool_result: dict[str, Any]) -> str:
        """Audit-only safety-keyword check (KTD-2).

        Always returns ``text`` unchanged. Emits
        ``symptoms.safety_keywords.missing`` when a Critical / Urgent
        tier reply does not contain a tier-appropriate keyword from
        ``configs/symptoms.yaml.safety_keywords_by_tier`` in the first
        :data:`_AUDIT_LEADING_CHARS` characters.
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
        keywords_block = self._config.safety_keywords_by_tier.for_tier(tier)
        keywords = keywords_block.en if language != "zh" else keywords_block.zh
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


def _first_answer(result: AskUserQuestionResult) -> Any:
    if result.numeric_values:
        return next(iter(result.numeric_values.values()))
    if result.answers:
        v = next(iter(result.answers.values()))
        if isinstance(v, list):
            return v[0] if v else ""
        return v
    return ""


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
