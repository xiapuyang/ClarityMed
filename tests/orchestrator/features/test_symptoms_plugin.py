"""SymptomsFeature plugin — happy path + cancel + cap + post_process.

All collaborators (server client, eligibility, profile loader, prompt
channel) are mocked in-process. The plugin's tool body is exercised
directly without constructing a pydantic-ai ``RunContext`` — we pass a
``SimpleNamespace`` deps that satisfies the ``TurnState`` protocol
shape and call ``_predict`` as a plain awaitable. Coverage of the
pydantic-ai shim itself happens in Unit 18 e2e.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from claritymed.context import apply_context
from claritymed.core.interaction.prompt_channel import UserDeclinedAnswer
from claritymed.core.interaction.schemas import (
    AskUserQuestionInput,
    AskUserQuestionResult,
)
from claritymed.core.schemas.patient import Profile
from claritymed.core.symptoms.eligibility import EligibilityResult
from claritymed.core.symptoms.registry import DatasetRegistry
from claritymed.core.symptoms.schemas import (
    DatasetSpec,
    DirectEligibilityEntry,
    EligibilityCatalogConfig,
    ModelSpec,
    SafetyKeywordsByTier,
    SafetyKeywordsLang,
    SymptomsConfig,
)
from claritymed.errors import SymptomsServerUnreachableError
from claritymed.orchestrator.features.symptoms_plugin import (
    SymptomsFeature,
    _resolve_initial_batch,
)
from claritymed.servers.symptoms.wire import (
    CancelResponse,
    DifferentialRow,
    StartSessionResponse,
    TurnResponse,
)


# --- fixtures ---------------------------------------------------------------


_USER_ID = "test"


def _dataset_spec(*, id_: str = "ddxplus") -> DatasetSpec:
    return DatasetSpec(
        id=id_,
        enabled=True,
        model_ids=["typed_basd_v1"],
        maxstep=8,
    )


def _symptoms_config() -> SymptomsConfig:
    return SymptomsConfig(
        datasets=[_dataset_spec()],
        models=[
            ModelSpec(
                id="typed_basd_v1",
                algorithm_module="typed_basd",
                weights_subpath="ddxplus/typed_basd_v1",
                manifest_sha256="0" * 64,
            )
        ],
        eligibility=EligibilityCatalogConfig(
            active="direct",
            catalog=[DirectEligibilityEntry(id="direct", kind="direct")],
        ),
        safety_keywords_by_tier=SafetyKeywordsByTier(
            Critical=SafetyKeywordsLang(
                en=["call 911", "emergency"], zh=["120", "急救"]
            ),
            Urgent=SafetyKeywordsLang(en=["urgent care today"], zh=["今日就诊"]),
            Moderate=SafetyKeywordsLang(en=["see your doctor"], zh=["门诊"]),
            Mild=SafetyKeywordsLang(en=["rest"], zh=["休息"]),
        ),
    )


class _StubEligibility:
    def __init__(self, result: EligibilityResult) -> None:
        self.result = result
        self.calls = 0

    async def check(self, complaint, language, profile, dataset) -> EligibilityResult:
        self.calls += 1
        return self.result


class _StubClient:
    """Records calls and returns canned responses queue-style."""

    def __init__(
        self,
        *,
        start: StartSessionResponse | Exception,
        turns: list[TurnResponse | Exception] | None = None,
        cancel: CancelResponse | None = None,
    ) -> None:
        self._start = start
        self._turns = list(turns or [])
        self._cancel = cancel
        self.calls: list[tuple[str, dict]] = []

    async def start_session(
        self,
        dataset_id,
        complaint,
        profile,
        *,
        language="en",
        symptom_summary=None,
    ):
        self.calls.append(
            (
                "start",
                {
                    "dataset_id": dataset_id,
                    "language": language,
                    "symptom_summary": symptom_summary,
                },
            )
        )
        if isinstance(self._start, Exception):
            raise self._start
        return self._start

    async def turn(
        self, dataset_id, session_id, answer, *, answer_value=None, language="en"
    ):
        self.calls.append(("turn", {"answer": answer}))
        if not self._turns:
            raise AssertionError("StubClient.turn called with no canned responses left")
        nxt = self._turns.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    async def cancel(self, dataset_id, session_id):
        self.calls.append(("cancel", {"dataset_id": dataset_id}))
        if self._cancel is None:
            raise AssertionError("StubClient.cancel called without canned response")
        return self._cancel


class _StubChannel:
    """Drives the modal flow via a scripted answer queue."""

    def __init__(self, answers: list[AskUserQuestionResult | Exception]) -> None:
        self._answers = list(answers)
        self.calls: list[AskUserQuestionInput] = []

    async def ask(self, payload: AskUserQuestionInput) -> AskUserQuestionResult:
        self.calls.append(payload)
        if not self._answers:
            raise AssertionError("StubChannel.ask called with no scripted answers left")
        nxt = self._answers.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def _make_plugin(
    *,
    eligibility: _StubEligibility | None = None,
    client: _StubClient | None = None,
    profile: Profile | None = None,
) -> SymptomsFeature:
    config = _symptoms_config()
    registry = DatasetRegistry(config.datasets)
    return SymptomsFeature(
        config=config,
        registry=registry,
        client=client,  # type: ignore[arg-type]
        eligibility=eligibility,  # type: ignore[arg-type]
        profile_loader=lambda uid: profile or Profile(),
    )


def _yes() -> AskUserQuestionResult:
    return AskUserQuestionResult(answers={"Try follow-up?": "Yes"})


def _no() -> AskUserQuestionResult:
    return AskUserQuestionResult(answers={"Try follow-up?": "No"})


def _initial_batch_answer() -> AskUserQuestionResult:
    return AskUserQuestionResult(
        answers={"Biological sex (for the differential model)?": "Male"},
        numeric_values={"How old are you, in years?": 35},
    )


def _turn_answer(value: str = "Yes") -> AskUserQuestionResult:
    return AskUserQuestionResult(answers={"Any pain?": value})


def _start_resp(*, session_id: str = "sess-1") -> StartSessionResponse:
    from claritymed.core.interaction.schemas import Question, QuestionOption

    q = Question(
        question="Any pain?",
        header="Pain",
        options=[
            QuestionOption(label="Yes", description="Yes."),
            QuestionOption(label="No", description="No."),
        ],
    )
    return StartSessionResponse(session_id=session_id, first_question=q)


def _done_turn(*, severity: int = 3) -> TurnResponse:
    diff = [
        DifferentialRow(
            condition_id="acute_appendicitis",
            condition_idx=0,
            condition_name="acute appendicitis",
            probability=0.6,
            severity=severity,
        )
    ]
    return TurnResponse(
        done=True, differential=diff, evidence_collected=[], turn_count=1
    )


def _next_question_turn() -> TurnResponse:
    from claritymed.core.interaction.schemas import Question, QuestionOption

    q = Question(
        question="Any pain?",
        header="Pain",
        options=[
            QuestionOption(label="Yes", description="Yes."),
            QuestionOption(label="No", description="No."),
        ],
    )
    return TurnResponse(next_question=q, turn_count=1)


def _cap_turn(*, severity: int = 2) -> TurnResponse:
    diff = [
        DifferentialRow(
            condition_id="meningitis",
            condition_idx=0,
            condition_name="meningitis",
            probability=0.3,
            severity=severity,
        )
    ]
    return TurnResponse(
        hit_cap=True,
        partial_differential=diff,
        evidence_collected=[],
        turn_count=8,
        partial_confidence=0.4,
    )


def _cancel_resp(
    *, meets: bool = True, severity_override: bool = False, max_low: int | None = None
) -> CancelResponse:
    diff = (
        [
            DifferentialRow(
                condition_id="cluster_headache",
                condition_idx=0,
                condition_name="cluster headache",
                probability=0.4,
                severity=3,
            )
        ]
        if meets
        else []
    )
    return CancelResponse(
        partial_differential=diff,
        evidence_collected=[],
        turn_count=3,
        partial_confidence=0.6 if meets else 0.2,
        meets_confidence_threshold=meets,
        severity_override=severity_override,
        max_low_severity_seen=max_low,
    )


def _deps() -> SimpleNamespace:
    return SimpleNamespace(user_id=_USER_ID, language="en", prompt_channel=None)


def _with_context(coro):
    """Apply request/user/language context for the duration of one call.

    The context manager is intentionally not used here because some tests
    need the context to survive after the coroutine returns (post_process
    reads it). The ``apply_context`` call's side effect is what matters;
    the returned token is discarded.
    """
    apply_context(
        request_id="20260613000000ABCDEFAB",
        user_id=_USER_ID,
        language="en",
    )
    return coro


# --- tests ------------------------------------------------------------------


async def test_ineligible_returns_silent_reason(tmp_path) -> None:
    elig = _StubEligibility(EligibilityResult(eligible=False, reason="out_of_scope"))
    plugin = _make_plugin(eligibility=elig)
    deps = _deps()
    token = apply_context(
        request_id="20260613000000ABCDEFAB", user_id=_USER_ID, language="en"
    )
    try:
        result = await plugin._predict(
            SimpleNamespace(deps=deps), complaint="reset password"
        )
    finally:
        from claritymed.context import reset_context

        reset_context(token)
    assert result == {"eligible": False, "reason": "out_of_scope"}
    assert elig.calls == 1


async def test_user_declines_confirm_modal() -> None:
    elig = _StubEligibility(
        EligibilityResult(eligible=True, reason="in_scope", confidence=0.7)
    )
    channel = _StubChannel([_no()])
    client = _StubClient(start=_start_resp())
    plugin = _make_plugin(eligibility=elig, client=client)
    deps = _deps()
    deps.prompt_channel = channel
    token = apply_context(
        request_id="20260613000000ABCDEFAB", user_id=_USER_ID, language="en"
    )
    try:
        result = await plugin._predict(
            SimpleNamespace(deps=deps), complaint="my chest hurts"
        )
    finally:
        from claritymed.context import reset_context

        reset_context(token)
    assert result == {"eligible": True, "user_declined": True}
    assert client.calls == []  # server never reached


async def test_happy_path_returns_differential() -> None:
    elig = _StubEligibility(
        EligibilityResult(eligible=True, reason="in_scope", confidence=0.7)
    )
    channel = _StubChannel(
        [
            _yes(),  # confirm modal
            _initial_batch_answer(),  # initial batch
            _turn_answer("Yes"),  # answer to first question (loop ends here)
        ]
    )
    client = _StubClient(start=_start_resp(), turns=[_done_turn()])
    plugin = _make_plugin(eligibility=elig, client=client)
    deps = _deps()
    deps.prompt_channel = channel
    token = apply_context(
        request_id="20260613000000ABCDEFAB", user_id=_USER_ID, language="en"
    )
    try:
        result = await plugin._predict(
            SimpleNamespace(deps=deps), complaint="my chest hurts"
        )
    finally:
        from claritymed.context import reset_context

        reset_context(token)
    assert result["eligible"] is True
    assert result["turn_count"] == 1
    assert len(result["differential"]) == 1
    assert result["differential"][0]["condition_id"] == "acute_appendicitis"
    assert client.calls[0][0] == "start"
    assert client.calls[1][0] == "turn"


async def test_initial_batch_skipped_when_profile_complete() -> None:
    """Profile with age + sex preloaded → no initial-batch modal."""
    from datetime import date

    elig = _StubEligibility(
        EligibilityResult(eligible=True, reason="in_scope", confidence=0.7)
    )
    channel = _StubChannel(
        [
            _yes(),  # confirm
            # No initial-batch modal — profile already complete
            _turn_answer("No"),
        ]
    )
    client = _StubClient(start=_start_resp(), turns=[_done_turn()])
    profile = Profile(sex="male", birth_date=date(1990, 1, 1))
    plugin = _make_plugin(eligibility=elig, client=client, profile=profile)
    deps = _deps()
    deps.prompt_channel = channel
    token = apply_context(
        request_id="20260613000000ABCDEFAB", user_id=_USER_ID, language="en"
    )
    try:
        result = await plugin._predict(SimpleNamespace(deps=deps), complaint="ache")
    finally:
        from claritymed.context import reset_context

        reset_context(token)
    assert result["eligible"] is True
    # Two channel calls: confirm + first server-driven question.
    assert len(channel.calls) == 2


async def test_cap_hit_returns_partial_differential() -> None:
    elig = _StubEligibility(
        EligibilityResult(eligible=True, reason="in_scope", confidence=0.7)
    )
    channel = _StubChannel([_yes(), _initial_batch_answer(), _turn_answer("Yes")])
    client = _StubClient(start=_start_resp(), turns=[_cap_turn(severity=2)])
    plugin = _make_plugin(eligibility=elig, client=client)
    deps = _deps()
    deps.prompt_channel = channel
    token = apply_context(
        request_id="20260613000000ABCDEFAB", user_id=_USER_ID, language="en"
    )
    try:
        result = await plugin._predict(
            SimpleNamespace(deps=deps), complaint="severe headache"
        )
    finally:
        from claritymed.context import reset_context

        reset_context(token)
    assert result["hit_cap"] is True
    assert result["partial_differential"][0]["condition_id"] == "meningitis"
    assert result["turn_count"] == 8


async def test_cancel_mid_loop_with_confidence_returns_partial() -> None:
    elig = _StubEligibility(
        EligibilityResult(eligible=True, reason="in_scope", confidence=0.7)
    )
    # First turn answers, second turn the user cancels.
    channel = _StubChannel(
        [_yes(), _initial_batch_answer(), _turn_answer("Yes"), UserDeclinedAnswer()]
    )
    client = _StubClient(
        start=_start_resp(),
        turns=[_next_question_turn()],
        cancel=_cancel_resp(meets=True),
    )
    plugin = _make_plugin(eligibility=elig, client=client)
    deps = _deps()
    deps.prompt_channel = channel
    token = apply_context(
        request_id="20260613000000ABCDEFAB", user_id=_USER_ID, language="en"
    )
    try:
        result = await plugin._predict(
            SimpleNamespace(deps=deps), complaint="my chest hurts"
        )
    finally:
        from claritymed.context import reset_context

        reset_context(token)
    assert result["cancelled"] is True
    assert result["meets_confidence_threshold"] is True
    assert result["partial_differential"][0]["condition_id"] == "cluster_headache"


async def test_cancel_without_confidence_drops_differential() -> None:
    elig = _StubEligibility(
        EligibilityResult(eligible=True, reason="in_scope", confidence=0.7)
    )
    channel = _StubChannel([_yes(), _initial_batch_answer(), UserDeclinedAnswer()])
    client = _StubClient(start=_start_resp(), cancel=_cancel_resp(meets=False))
    plugin = _make_plugin(eligibility=elig, client=client)
    deps = _deps()
    deps.prompt_channel = channel
    token = apply_context(
        request_id="20260613000000ABCDEFAB", user_id=_USER_ID, language="en"
    )
    try:
        result = await plugin._predict(
            SimpleNamespace(deps=deps), complaint="my chest hurts"
        )
    finally:
        from claritymed.context import reset_context

        reset_context(token)
    assert result["cancelled"] is True
    assert result["meets_confidence_threshold"] is False
    assert result["partial_differential"] is None


async def test_server_unreachable_returns_server_error() -> None:
    elig = _StubEligibility(
        EligibilityResult(eligible=True, reason="in_scope", confidence=0.7)
    )
    channel = _StubChannel([_yes(), _initial_batch_answer()])
    client = _StubClient(start=SymptomsServerUnreachableError("down"))
    plugin = _make_plugin(eligibility=elig, client=client)
    deps = _deps()
    deps.prompt_channel = channel
    token = apply_context(
        request_id="20260613000000ABCDEFAB", user_id=_USER_ID, language="en"
    )
    try:
        result = await plugin._predict(
            SimpleNamespace(deps=deps), complaint="chest pain"
        )
    finally:
        from claritymed.context import reset_context

        reset_context(token)
    assert result == {"eligible": True, "server_error": True}


async def test_headless_no_channel_returns_no_interactive_channel() -> None:
    elig = _StubEligibility(
        EligibilityResult(eligible=True, reason="in_scope", confidence=0.7)
    )
    plugin = _make_plugin(eligibility=elig, client=_StubClient(start=_start_resp()))
    deps = _deps()
    deps.prompt_channel = None
    token = apply_context(
        request_id="20260613000000ABCDEFAB", user_id=_USER_ID, language="en"
    )
    try:
        result = await plugin._predict(
            SimpleNamespace(deps=deps), complaint="chest pain"
        )
    finally:
        from claritymed.context import reset_context

        reset_context(token)
    assert result == {"eligible": False, "reason": "no_interactive_channel"}


# --- post_process ----------------------------------------------------------


async def test_post_process_returns_text_unchanged_when_no_stash() -> None:
    """No prior tool call for this request_id → audit-only no-op."""
    plugin = _make_plugin(
        eligibility=_StubEligibility(
            EligibilityResult(eligible=False, reason="out_of_scope")
        ),
        client=_StubClient(start=_start_resp()),
    )
    token = apply_context(
        request_id="20260613000000ABCDEFAB", user_id=_USER_ID, language="en"
    )
    try:
        out = await plugin.post_process("any reply text", {})
    finally:
        from claritymed.context import reset_context

        reset_context(token)
    assert out == "any reply text"


async def test_post_process_critical_with_keyword_no_audit() -> None:
    plugin = _make_plugin(
        eligibility=_StubEligibility(
            EligibilityResult(eligible=True, reason="in_scope")
        ),
        client=_StubClient(start=_start_resp()),
    )
    token = apply_context(
        request_id="20260613000000ABCDEFAB", user_id=_USER_ID, language="en"
    )
    try:
        # Simulate a completed sub-session with a severity-1 disease.
        plugin._stash["20260613000000ABCDEFAB"] = {"max_severity": 1}
        out = await plugin.post_process(
            "Call 911 right now — these symptoms could be a heart attack.", {}
        )
    finally:
        from claritymed.context import reset_context

        reset_context(token)
    assert "Call 911" in out
    # text returned unchanged
    assert out.startswith("Call 911")


async def test_post_process_critical_missing_keyword_emits_audit_unchanged_text(
    caplog,
) -> None:
    plugin = _make_plugin(
        eligibility=_StubEligibility(
            EligibilityResult(eligible=True, reason="in_scope")
        ),
        client=_StubClient(start=_start_resp()),
    )
    token = apply_context(
        request_id="20260613000000ABCDEFAB", user_id=_USER_ID, language="en"
    )
    try:
        plugin._stash["20260613000000ABCDEFAB"] = {"max_severity": 1}
        # Reply describes the differential but never mentions the
        # emergency keywords — Critical tier audit must fire.
        out = await plugin.post_process(
            "You might want to see a doctor about that.", {}
        )
    finally:
        from claritymed.context import reset_context

        reset_context(token)
    # Text always returned unchanged.
    assert out == "You might want to see a doctor about that."


async def test_post_process_moderate_tier_skipped() -> None:
    """Tier ≥3 is not in the audit set; no scan, text unchanged."""
    plugin = _make_plugin(
        eligibility=_StubEligibility(
            EligibilityResult(eligible=True, reason="in_scope")
        ),
        client=_StubClient(start=_start_resp()),
    )
    token = apply_context(
        request_id="20260613000000ABCDEFAB", user_id=_USER_ID, language="en"
    )
    try:
        plugin._stash["20260613000000ABCDEFAB"] = {"max_severity": 4}
        out = await plugin.post_process("Just rest at home.", {})
    finally:
        from claritymed.context import reset_context

        reset_context(token)
    assert out == "Just rest at home."


# --- helper coverage -------------------------------------------------------


def test_resolve_initial_batch_uses_modal_values() -> None:
    result = AskUserQuestionResult(
        answers={"Biological sex (for the differential model)?": "Female"},
        numeric_values={"How old are you, in years?": 42},
    )
    wire = _resolve_initial_batch(result, profile=Profile(), language="en")
    assert wire == {"age_years": 42, "sex": "F"}


def test_resolve_initial_batch_falls_back_to_profile_when_blank() -> None:
    from datetime import date

    profile = Profile(sex="male", birth_date=date(1985, 6, 1))
    result = AskUserQuestionResult(answers={}, numeric_values={})
    wire = _resolve_initial_batch(result, profile=profile, language="en")
    # Profile.sex "male" → wire "M" via _map_sex.
    assert wire["sex"] == "M"
    assert wire["age_years"] >= 30


def test_validate_prompts_raises_when_missing(tmp_path) -> None:
    """A registry missing the symptoms YAMLs must fail loud at plugin
    construct so the regression surfaces at startup, not at first call."""
    from claritymed.core.prompts.registry import PromptRegistry
    from claritymed.orchestrator.features.symptoms_plugin import (
        _validate_symptoms_prompts,
    )

    # Point the registry at an empty directory.
    empty_registry = PromptRegistry(store_dir=tmp_path)
    with pytest.raises(RuntimeError, match="Missing symptoms prompts"):
        _validate_symptoms_prompts(empty_registry)
