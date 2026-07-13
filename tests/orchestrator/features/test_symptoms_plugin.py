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

_REQUEST_ID = "20260613000000ABCDEFAB"


def _dataset_spec(*, id_: str = "ddxplus") -> DatasetSpec:
    return DatasetSpec(
        id=id_,
        enabled=True,
        model_ids=["typed_basd_v1"],
    )


def _symptoms_config(
    *,
    input_source: str = "complaint",
) -> SymptomsConfig:
    return SymptomsConfig(
        datasets=[_dataset_spec()],
        models=[
            ModelSpec(
                id="typed_basd_v1",
                algorithm_module="typed_basd",
                weights_subpath="ddxplus/typed_basd_v1",
                manifest_sha256="0" * 64,
                maxstep=8,
            )
        ],
        eligibility=EligibilityCatalogConfig(
            active="direct",
            catalog=[DirectEligibilityEntry(id="direct", kind="direct")],
            input_source=input_source,  # type: ignore[arg-type]
        ),
    )


class _StubEligibility:
    def __init__(self, result: EligibilityResult) -> None:
        self.result = result
        self.calls = 0
        self.last_complaint: str | None = None

    async def check(self, complaint, language, profile, dataset) -> EligibilityResult:
        self.calls += 1
        self.last_complaint = complaint
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
        request_id=None,
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
        self,
        dataset_id,
        session_id,
        answer,
        *,
        answer_value=None,
        language="en",
        request_id=None,
    ):
        self.calls.append(("turn", {"answer": answer}))
        if not self._turns:
            raise AssertionError("StubClient.turn called with no canned responses left")
        nxt = self._turns.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    async def cancel(self, dataset_id, session_id, *, request_id=None):
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
    input_source: str = "complaint",
) -> SymptomsFeature:
    config = _symptoms_config(input_source=input_source)
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
        request_id=_REQUEST_ID,
        user_id=_USER_ID,
        language="en",
    )
    return coro


# --- tests ------------------------------------------------------------------


async def test_ineligible_returns_silent_reason(tmp_path) -> None:
    elig = _StubEligibility(EligibilityResult(eligible=False, reason="out_of_scope"))
    plugin = _make_plugin(eligibility=elig)
    deps = _deps()
    token = apply_context(request_id=_REQUEST_ID, user_id=_USER_ID, language="en")
    try:
        result = await plugin._predict(
            SimpleNamespace(deps=deps), complaint="reset password"
        )
    finally:
        from claritymed.context import reset_context

        reset_context(token)
    assert result == {"eligible": False, "reason": "out_of_scope"}
    assert elig.calls == 1
    # input_source default == "complaint": raw user text reaches the strategy.
    assert elig.last_complaint == "reset password"


async def test_eligibility_input_uses_summary_when_configured() -> None:
    """input_source=symptom_summary forwards the LLM-distilled phrase.

    Audit-relevant: the eligibility check now references the summary,
    so any downstream "what did we check?" log entry should hold the
    summary string. Verified by checking the stub strategy's
    ``last_complaint`` capture.
    """
    elig = _StubEligibility(EligibilityResult(eligible=False, reason="out_of_scope"))
    plugin = _make_plugin(eligibility=elig, input_source="symptom_summary")
    deps = _deps()
    token = apply_context(request_id=_REQUEST_ID, user_id=_USER_ID, language="en")
    try:
        await plugin._predict(
            SimpleNamespace(deps=deps),
            complaint="i feel terrible, my chest hurts a lot and i cant breathe",
            symptom_summary="acute chest pain with dyspnea",
        )
    finally:
        from claritymed.context import reset_context

        reset_context(token)
    assert elig.last_complaint == "acute chest pain with dyspnea"


async def test_eligibility_input_falls_back_to_complaint_when_summary_blank() -> None:
    """A blank summary under input_source=symptom_summary falls back.

    Default fallback prevents an LLM that forgot the optional argument
    (or sent whitespace) from accidentally feeding the matcher an
    empty string and getting back ``out_of_scope`` for that reason.
    """
    elig = _StubEligibility(EligibilityResult(eligible=False, reason="out_of_scope"))
    plugin = _make_plugin(eligibility=elig, input_source="symptom_summary")
    deps = _deps()
    token = apply_context(request_id=_REQUEST_ID, user_id=_USER_ID, language="en")
    try:
        await plugin._predict(
            SimpleNamespace(deps=deps),
            complaint="chest pain",
            symptom_summary="   ",
        )
    finally:
        from claritymed.context import reset_context

        reset_context(token)
    assert elig.last_complaint == "chest pain"


async def test_user_declines_confirm_modal() -> None:
    elig = _StubEligibility(
        EligibilityResult(eligible=True, reason="in_scope", confidence=0.7)
    )
    channel = _StubChannel([_no()])
    client = _StubClient(start=_start_resp())
    plugin = _make_plugin(eligibility=elig, client=client)
    deps = _deps()
    deps.prompt_channel = channel
    token = apply_context(request_id=_REQUEST_ID, user_id=_USER_ID, language="en")
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
    token = apply_context(request_id=_REQUEST_ID, user_id=_USER_ID, language="en")
    try:
        result = await plugin._predict(
            SimpleNamespace(deps=deps), complaint="my chest hurts"
        )
    finally:
        from claritymed.context import reset_context

        reset_context(token)
    assert result["eligible"] is True
    assert result["turns_used"] == 1
    assert len(result["differential"]) == 1
    assert result["differential"][0]["condition_name"] == "acute appendicitis"
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
    token = apply_context(request_id=_REQUEST_ID, user_id=_USER_ID, language="en")
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
    token = apply_context(request_id=_REQUEST_ID, user_id=_USER_ID, language="en")
    try:
        result = await plugin._predict(
            SimpleNamespace(deps=deps), complaint="severe headache"
        )
    finally:
        from claritymed.context import reset_context

        reset_context(token)
    assert result["hit_cap"] is True
    assert result["partial_differential"][0]["condition_name"] == "meningitis"
    assert result["turns_used"] == 8


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
    token = apply_context(request_id=_REQUEST_ID, user_id=_USER_ID, language="en")
    try:
        result = await plugin._predict(
            SimpleNamespace(deps=deps), complaint="my chest hurts"
        )
    finally:
        from claritymed.context import reset_context

        reset_context(token)
    assert result["cancelled"] is True
    assert result["meets_confidence_threshold"] is True
    assert result["partial_differential"][0]["condition_name"] == "cluster headache"


async def test_cancel_without_confidence_drops_differential() -> None:
    elig = _StubEligibility(
        EligibilityResult(eligible=True, reason="in_scope", confidence=0.7)
    )
    channel = _StubChannel([_yes(), _initial_batch_answer(), UserDeclinedAnswer()])
    client = _StubClient(start=_start_resp(), cancel=_cancel_resp(meets=False))
    plugin = _make_plugin(eligibility=elig, client=client)
    deps = _deps()
    deps.prompt_channel = channel
    token = apply_context(request_id=_REQUEST_ID, user_id=_USER_ID, language="en")
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
    token = apply_context(request_id=_REQUEST_ID, user_id=_USER_ID, language="en")
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
    token = apply_context(request_id=_REQUEST_ID, user_id=_USER_ID, language="en")
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
    token = apply_context(request_id=_REQUEST_ID, user_id=_USER_ID, language="en")
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
    token = apply_context(request_id=_REQUEST_ID, user_id=_USER_ID, language="en")
    try:
        # Simulate a completed sub-session with a severity-1 disease.
        plugin._stash[_REQUEST_ID] = {"max_severity": 1}
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
    import logging

    plugin = _make_plugin(
        eligibility=_StubEligibility(
            EligibilityResult(eligible=True, reason="in_scope")
        ),
        client=_StubClient(start=_start_resp()),
    )
    token = apply_context(request_id=_REQUEST_ID, user_id=_USER_ID, language="en")
    try:
        plugin._stash[_REQUEST_ID] = {"max_severity": 1}
        # Reply describes the differential but never mentions the
        # emergency keywords — Critical tier audit must fire.
        with caplog.at_level(logging.INFO, logger="claritymed.audit"):
            out = await plugin.post_process(
                "You might want to see a doctor about that.", {}
            )
    finally:
        from claritymed.context import reset_context

        reset_context(token)
    # Text always returned unchanged.
    assert out == "You might want to see a doctor about that."
    # Audit event must have been emitted for the missing safety keyword.
    assert any(
        "symptoms.safety_keywords.missing" in record.message
        for record in caplog.records
    ), "Expected symptoms.safety_keywords.missing audit event in log"


async def test_post_process_moderate_tier_skipped() -> None:
    """Tier ≥3 is not in the audit set; no scan, text unchanged."""
    plugin = _make_plugin(
        eligibility=_StubEligibility(
            EligibilityResult(eligible=True, reason="in_scope")
        ),
        client=_StubClient(start=_start_resp()),
    )
    token = apply_context(request_id=_REQUEST_ID, user_id=_USER_ID, language="en")
    try:
        plugin._stash[_REQUEST_ID] = {"max_severity": 4}
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


def test_validate_safety_keywords_raises_when_missing(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Startup guard fires when an i18n tier list is absent.

    Replaces the old ``SafetyKeywordsByTier`` Pydantic validator. Point
    the loader at an empty i18n dir and confirm the construct-time
    check raises rather than silently degrading the audit signal.
    """
    from claritymed.core.i18n import loader as i18n_loader
    from claritymed.orchestrator.features.symptoms_plugin import (
        _validate_safety_keywords,
    )

    empty_i18n = tmp_path / "i18n"
    empty_i18n.mkdir()
    monkeypatch.setattr(i18n_loader, "I18N_DIR", empty_i18n)
    i18n_loader._reset_for_tests()
    try:
        with pytest.raises(RuntimeError, match="safety_keywords"):
            _validate_safety_keywords()
    finally:
        i18n_loader._reset_for_tests()


class _PerDatasetEligibility:
    """Returns different EligibilityResult per dataset id."""

    def __init__(self, results: "dict[str, EligibilityResult]") -> None:
        self._results = results
        self.calls: dict[str, int] = {}

    async def check(self, complaint, language, profile, dataset) -> EligibilityResult:
        self.calls[dataset.id] = self.calls.get(dataset.id, 0) + 1
        return self._results.get(
            dataset.id, EligibilityResult(eligible=False, reason="out_of_scope")
        )


def _two_dataset_config() -> SymptomsConfig:
    from claritymed.core.symptoms.schemas import ModelSpec

    return SymptomsConfig(
        datasets=[_dataset_spec(id_="ddxplus"), _dataset_spec(id_="ddxplus2")],
        models=[
            ModelSpec(
                id="typed_basd_v1",
                algorithm_module="typed_basd",
                weights_subpath="ddxplus/typed_basd_v1",
                manifest_sha256="0" * 64,
                maxstep=8,
            )
        ],
        eligibility=EligibilityCatalogConfig(
            active="direct",
            catalog=[DirectEligibilityEntry(id="direct", kind="direct")],
        ),
    )


async def test_route_by_eligibility_picks_best_scoring_dataset() -> None:
    """Without dataset_hint and multiple enabled datasets, highest-confidence wins."""
    elig = _PerDatasetEligibility(
        {
            "ddxplus": EligibilityResult(
                eligible=True, reason="in_scope", confidence=0.8
            ),
            "ddxplus2": EligibilityResult(
                eligible=False, reason="out_of_scope", confidence=0.1
            ),
        }
    )
    config = _two_dataset_config()
    registry = DatasetRegistry(config.datasets)
    client = _StubClient(start=_start_resp(), turns=[_done_turn()])
    channel = _StubChannel([_yes(), _initial_batch_answer(), _turn_answer("Yes")])
    plugin = SymptomsFeature(
        config=config,
        registry=registry,
        client=client,
        eligibility=elig,
        profile_loader=lambda uid: Profile(),
    )
    deps = _deps()
    deps.prompt_channel = channel
    token = apply_context(request_id=_REQUEST_ID, user_id=_USER_ID, language="en")
    try:
        result = await plugin._predict(
            SimpleNamespace(deps=deps), complaint="chest pain"
        )
    finally:
        from claritymed.context import reset_context

        reset_context(token)
    assert result["eligible"] is True
    assert result.get("differential") is not None
    assert elig.calls.get("ddxplus", 0) == 1
    assert elig.calls.get("ddxplus2", 0) == 1


async def test_route_by_eligibility_all_out_of_scope_returns_ineligible() -> None:
    """When all datasets score 0, no session is started."""
    elig = _PerDatasetEligibility(
        {
            "ddxplus": EligibilityResult(
                eligible=False, reason="out_of_scope", confidence=0.0
            ),
            "ddxplus2": EligibilityResult(
                eligible=False, reason="out_of_scope", confidence=0.0
            ),
        }
    )
    config = _two_dataset_config()
    registry = DatasetRegistry(config.datasets)
    client = _StubClient(start=_start_resp())
    plugin = SymptomsFeature(
        config=config,
        registry=registry,
        client=client,
        eligibility=elig,
        profile_loader=lambda uid: Profile(),
    )
    deps = _deps()
    deps.prompt_channel = _StubChannel([])
    token = apply_context(request_id=_REQUEST_ID, user_id=_USER_ID, language="en")
    try:
        result = await plugin._predict(
            SimpleNamespace(deps=deps), complaint="reset password"
        )
    finally:
        from claritymed.context import reset_context

        reset_context(token)
    assert result == {"eligible": False, "reason": "out_of_scope"}
    assert client.calls == []


async def test_handle_done_writes_phi_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Completed session writes differential + transcript to the PHI payload."""
    written: list[dict] = []
    monkeypatch.setattr(
        "claritymed.orchestrator.features.symptoms_plugin.write_payload",
        lambda uid, req_id, payload: written.append(payload),
    )
    elig = _StubEligibility(
        EligibilityResult(eligible=True, reason="in_scope", confidence=0.7)
    )
    channel = _StubChannel([_yes(), _initial_batch_answer(), _turn_answer("Yes")])
    client = _StubClient(start=_start_resp(), turns=[_done_turn(severity=1)])
    plugin = _make_plugin(eligibility=elig, client=client)
    deps = _deps()
    deps.prompt_channel = channel
    token = apply_context(request_id=_REQUEST_ID, user_id=_USER_ID, language="en")
    try:
        result = await plugin._predict(
            SimpleNamespace(deps=deps), complaint="chest pain"
        )
    finally:
        from claritymed.context import reset_context

        reset_context(token)
    assert result["eligible"] is True
    assert len(written) == 1
    payload = written[0]
    assert payload["kind"] == "symptoms.session.completed"
    assert len(payload["differential"]) == 1
    assert payload["differential"][0]["condition_id"] == "acute_appendicitis"
    assert isinstance(payload["transcript"], list)
    assert len(payload["transcript"]) >= 1


async def test_handle_done_top_condition_id_picks_argmax_not_slot0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``top_condition_id`` in the audit event must be the argmax by prob.

    v3 subset-parametric datasets ship the differential in fixed slot
    order — ``[Pneumonia, Influenza, Other]`` — so the frontend can
    render N+1 cards without argmax reordering. That means a P(Other)-
    dominant session would still have ``diff[0] == Pneumonia`` even
    when the model is confident this is NEITHER target. Reporting
    ``top_condition_id="pneumonia"`` in that case corrupts every
    downstream dashboard keyed on this field.

    This test locks the fix at the audit-event site: with a
    v3-shaped diff where slot 0 is Pneumonia@0.02 and slot 2 is
    Other@0.96, the audit event must carry ``top_condition_id="other"``.
    """
    captured: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        "claritymed.orchestrator.features.symptoms_plugin.write_payload",
        lambda uid, req_id, payload: None,
    )
    monkeypatch.setattr(
        "claritymed.orchestrator.features.symptoms_plugin.audit_event",
        lambda kind, payload: captured.append((kind, payload)),
    )
    elig = _StubEligibility(
        EligibilityResult(eligible=True, reason="in_scope", confidence=0.7)
    )
    channel = _StubChannel([_yes(), _initial_batch_answer(), _turn_answer("Yes")])
    # v3-shape: fixed slot order Pne / Inf / Other, Other dominates.
    diff = [
        DifferentialRow(
            condition_id="pneumonia",
            condition_idx=0,
            condition_name="Pneumonia",
            probability=0.02,
            severity=3,
        ),
        DifferentialRow(
            condition_id="influenza",
            condition_idx=1,
            condition_name="Influenza",
            probability=0.01,
            severity=3,
        ),
        DifferentialRow(
            condition_id="other",
            condition_idx=None,
            condition_name="Other likely condition",
            probability=0.97,
            severity=3,
        ),
    ]
    done = TurnResponse(
        done=True, differential=diff, evidence_collected=[], turn_count=2
    )
    client = _StubClient(start=_start_resp(), turns=[done])
    plugin = _make_plugin(eligibility=elig, client=client)
    deps = _deps()
    deps.prompt_channel = channel
    token = apply_context(request_id=_REQUEST_ID, user_id=_USER_ID, language="en")
    try:
        await plugin._predict(SimpleNamespace(deps=deps), complaint="chest pain")
    finally:
        from claritymed.context import reset_context

        reset_context(token)
    completed = [
        payload for kind, payload in captured if kind == "symptoms.session.completed"
    ]
    assert len(completed) == 1
    # Bug: previously ``diff[0]["condition_id"] == "pneumonia"`` regardless
    # of probability. Fix: argmax picks the true winner — ``other``.
    assert completed[0]["top_condition_id"] == "other"


async def test_handle_cap_writes_phi_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cap-hit session writes partial_differential + transcript to the PHI payload."""
    written: list[dict] = []
    monkeypatch.setattr(
        "claritymed.orchestrator.features.symptoms_plugin.write_payload",
        lambda uid, req_id, payload: written.append(payload),
    )
    elig = _StubEligibility(
        EligibilityResult(eligible=True, reason="in_scope", confidence=0.7)
    )
    channel = _StubChannel([_yes(), _initial_batch_answer(), _turn_answer("Yes")])
    client = _StubClient(start=_start_resp(), turns=[_cap_turn(severity=2)])
    plugin = _make_plugin(eligibility=elig, client=client)
    deps = _deps()
    deps.prompt_channel = channel
    token = apply_context(request_id=_REQUEST_ID, user_id=_USER_ID, language="en")
    try:
        result = await plugin._predict(
            SimpleNamespace(deps=deps), complaint="chest pain"
        )
    finally:
        from claritymed.context import reset_context

        reset_context(token)
    assert result["hit_cap"] is True
    assert len(written) == 1
    payload = written[0]
    assert payload["kind"] == "symptoms.session.cap_hit"
    assert len(payload["partial_differential"]) == 1
    assert isinstance(payload["transcript"], list)


async def test_cancel_severity_override_stashes_override_severity() -> None:
    """severity_override=True → stash uses max_low_severity_seen, not diff severity."""
    elig = _StubEligibility(
        EligibilityResult(eligible=True, reason="in_scope", confidence=0.7)
    )
    channel = _StubChannel([_yes(), _initial_batch_answer(), UserDeclinedAnswer()])
    cancel_resp = _cancel_resp(meets=False, severity_override=True, max_low=1)
    client = _StubClient(start=_start_resp(), cancel=cancel_resp)
    plugin = _make_plugin(eligibility=elig, client=client)
    deps = _deps()
    deps.prompt_channel = channel
    token = apply_context(request_id=_REQUEST_ID, user_id=_USER_ID, language="en")
    try:
        result = await plugin._predict(
            SimpleNamespace(deps=deps), complaint="chest pain"
        )
    finally:
        from claritymed.context import reset_context

        reset_context(token)
    assert result["severity_override"] is True
    assert result["cancelled"] is True
    assert plugin._stash.get(_REQUEST_ID, {}).get("max_severity") == 1


def test_validate_safety_keywords_raises_on_blank_entry(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A whitespace-only keyword in any tier is rejected.

    Mirrors the old ``SafetyKeywordsLang._no_blank_keywords`` validator
    — empty strings can never satisfy the post_process keyword scan, so
    the operator should hear about them at boot.
    """
    import yaml

    from claritymed.core.i18n import loader as i18n_loader
    from claritymed.orchestrator.features.symptoms_plugin import (
        _validate_safety_keywords,
    )

    i18n_dir = tmp_path / "i18n"
    (i18n_dir / "en").mkdir(parents=True)
    (i18n_dir / "zh").mkdir(parents=True)
    payload = {
        "symptoms": {
            "safety_keywords": {
                "Critical": ["call 911", "  "],
                "Urgent": ["urgent care"],
                "Moderate": ["see your doctor"],
                "Mild": ["rest"],
            }
        }
    }
    for lang in ("en", "zh"):
        (i18n_dir / lang / "symptoms.yaml").write_text(
            yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8"
        )
    monkeypatch.setattr(i18n_loader, "I18N_DIR", i18n_dir)
    i18n_loader._reset_for_tests()
    try:
        with pytest.raises(RuntimeError, match="blank"):
            _validate_safety_keywords()
    finally:
        i18n_loader._reset_for_tests()


# --- ddxplus vocab + sidecar loaders ------------------------------------


def test_load_ddxplus_vocab_returns_empty_when_file_missing(tmp_path) -> None:
    from claritymed.orchestrator.features.symptoms_plugin import _load_ddxplus_vocab

    # No release_evidences.json in the dir → empty dict, no exception.
    assert _load_ddxplus_vocab(str(tmp_path)) == {}


def test_load_ddxplus_vocab_extracts_question_and_value_meanings(tmp_path) -> None:
    """Vocab map contains the EN question + all EN value meanings per evidence."""
    import json as _json
    from claritymed.orchestrator.features.symptoms_plugin import _load_ddxplus_vocab

    payload = {
        "E_001": {
            "question_en": "Do you have a fever?",
            "value_meaning": {
                "1": {"en": "yes"},
                "2": {"en": "no"},
            },
        },
        # Evidence with no phrases at all → filtered out of the map.
        "E_999": {"value_meaning": {"1": "garbage-not-dict"}},
    }
    (tmp_path / "release_evidences.json").write_text(
        _json.dumps(payload), encoding="utf-8"
    )

    vocab = _load_ddxplus_vocab(str(tmp_path))
    assert "E_001" in vocab
    assert vocab["E_001"] == frozenset({"Do you have a fever?", "yes", "no"})
    # Evidence without phrases is silently dropped.
    assert "E_999" not in vocab


def test_load_ddxplus_sidecar_returns_empty_when_missing(tmp_path) -> None:
    from claritymed.orchestrator.features.symptoms_plugin import _load_ddxplus_sidecar

    assert _load_ddxplus_sidecar(str(tmp_path)) == {}


def test_load_ddxplus_sidecar_normalizes_keys_and_values_to_str(tmp_path) -> None:
    """The sidecar may contain non-string ids (legacy); always coerce."""
    import json as _json
    from claritymed.orchestrator.features.symptoms_plugin import _load_ddxplus_sidecar

    (tmp_path / "evidence_concepts.json").write_text(
        _json.dumps({"E_001": "C123", 7: 42}),
        encoding="utf-8",
    )
    sidecar = _load_ddxplus_sidecar(str(tmp_path))
    assert sidecar == {"E_001": "C123", "7": "42"}


def test_load_ddxplus_sidecar_returns_empty_when_payload_is_list(tmp_path) -> None:
    """Non-dict payload (e.g. accidental list) → fall through to empty."""
    import json as _json
    from claritymed.orchestrator.features.symptoms_plugin import _load_ddxplus_sidecar

    (tmp_path / "evidence_concepts.json").write_text(
        _json.dumps(["not", "a", "dict"]), encoding="utf-8"
    )
    assert _load_ddxplus_sidecar(str(tmp_path)) == {}


# --- make_symptoms_factory ----------------------------------------------


def test_make_symptoms_factory_returns_none_when_config_load_fails(monkeypatch):
    """A malformed ``symptoms.yaml`` must silently disable the feature."""
    from claritymed.orchestrator.features import symptoms_plugin as plg
    from claritymed import config as _cfg

    def _boom():
        raise RuntimeError("yaml is invalid")

    monkeypatch.setattr(_cfg, "load_symptoms_config", _boom)
    assert plg.make_symptoms_factory() is None


def test_make_symptoms_factory_returns_none_when_no_datasets_enabled(monkeypatch):
    """Every dataset has ``enabled=False`` → feature disabled."""
    from claritymed.orchestrator.features import symptoms_plugin as plg
    from claritymed import config as _cfg

    disabled_spec = _dataset_spec()
    disabled_cfg = _symptoms_config()
    disabled_cfg = disabled_cfg.model_copy(
        update={"datasets": [disabled_spec.model_copy(update={"enabled": False})]}
    )

    monkeypatch.setattr(_cfg, "load_symptoms_config", lambda: disabled_cfg)
    assert plg.make_symptoms_factory() is None


def test_make_symptoms_factory_returns_none_on_eligibility_config_error(monkeypatch):
    """A misconfigured eligibility strategy is treated as feature-disabled, not crash."""
    from claritymed.orchestrator.features import symptoms_plugin as plg
    from claritymed.core.symptoms.eligibility import factory as _ef
    from claritymed import config as _cfg
    from claritymed.errors import EligibilityStrategyConfigError

    monkeypatch.setattr(_cfg, "load_symptoms_config", lambda: _symptoms_config())

    def _bad(*args, **kwargs):
        raise EligibilityStrategyConfigError("active strategy missing dep")

    monkeypatch.setattr(_ef, "build_eligibility_strategy", _bad)
    assert plg.make_symptoms_factory() is None


def test_make_symptoms_factory_returns_none_on_unknown_eligibility_error(monkeypatch):
    """An unexpected exception in eligibility build → feature disabled (no crash)."""
    from claritymed.orchestrator.features import symptoms_plugin as plg
    from claritymed.core.symptoms.eligibility import factory as _ef
    from claritymed import config as _cfg

    monkeypatch.setattr(_cfg, "load_symptoms_config", lambda: _symptoms_config())

    def _boom(*args, **kwargs):
        raise RuntimeError("unexpected lib failure")

    monkeypatch.setattr(_ef, "build_eligibility_strategy", _boom)
    assert plg.make_symptoms_factory() is None


def test_make_symptoms_factory_happy_path_returns_callable(monkeypatch, tmp_path):
    """Config loads + eligibility builds → factory callable returns a SymptomsFeature."""
    from claritymed.orchestrator.features import symptoms_plugin as plg
    from claritymed.core.symptoms.eligibility import factory as _ef
    from claritymed import config as _cfg

    monkeypatch.setattr(_cfg, "DATA_DIR", tmp_path)
    monkeypatch.setattr(_cfg, "load_symptoms_config", lambda: _symptoms_config())

    sentinel_eligibility = _StubEligibility(
        EligibilityResult(eligible=True, reason="in_scope")
    )

    def _ok(eligibility_cfg, *, vocabs, sidecars, term_service_factory):
        # The factory passes the loaded vocab/sidecar maps through — assert
        # both are dicts (possibly empty when on-disk data is absent).
        assert isinstance(vocabs, dict)
        assert isinstance(sidecars, dict)
        return sentinel_eligibility

    monkeypatch.setattr(_ef, "build_eligibility_strategy", _ok)

    factory = plg.make_symptoms_factory(base_url="http://stub.local")
    assert factory is not None
    feature = factory()
    assert isinstance(feature, plg.SymptomsFeature)
    # The eligibility stub is wired through (private attr — feature does
    # not expose it on the surface).
    assert feature._eligibility is sentinel_eligibility
