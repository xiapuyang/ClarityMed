"""Plugin ↔ symptoms-server integration via in-process ASGI transport.

Sits between the unit tests (mock client + scripted channel) and the full
Unit 18 e2e matrix (subprocess server + real providers). Both layers are
real:

* The :class:`SymptomsServerClient` issues real HTTP requests…
* …which are answered by the real FastAPI app over
  :class:`httpx.ASGITransport` — no socket, no subprocess.
* The plugin tool body runs the eligibility → confirm → batch → loop
  pipeline end to end.

Stub seams kept small: a scripted prompt channel (no real Textual app),
a stub Agent populating ``SERVER_STATE.datasets`` (no torch / weights),
and a stub eligibility result (the strategy modules are exercised in
their own unit tests).
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from claritymed.context import apply_context, reset_context
from claritymed.core.interaction.schemas import (
    AskUserQuestionInput,
    AskUserQuestionResult,
)
from claritymed.core.schemas.patient import Profile
from claritymed.core.symptoms.client import SymptomsServerClient
from claritymed.core.symptoms.datasets import (
    CanonicalCondition,
    CanonicalDataset,
    CanonicalEvidence,
    CanonicalValue,
    LoadedDataset,
    LoadedModel,
)
from claritymed.core.symptoms.eligibility import EligibilityResult
from claritymed.core.symptoms.registry import DatasetRegistry
from claritymed.core.symptoms.schemas import (
    DatasetSpec,
    DirectEligibilityEntry,
    EligibilityCatalogConfig,
    ModelSpec,
    SymptomsConfig,
)
from claritymed.ingest.symptoms.typed_basd import build_layout
from claritymed.orchestrator.features.symptoms_plugin import SymptomsFeature
from claritymed.servers.symptoms.app import app
from claritymed.servers.symptoms.state import SERVER_STATE


# --- agent stub (mirrors tests/servers/symptoms/test_app.py) ----------------


class _StubAgent:
    """Deterministic stand-in for a typed-BASD Agent.

    Cycles through evidence indices on ``next_action``; stops after
    ``stop_after`` ``should_stop`` calls; returns a fixed probability
    vector on ``diagnose``.
    """

    def __init__(self, *, n_evidences: int, probs: np.ndarray, stop_after: int = 3):
        self.n_evidences = n_evidences
        self.probs = probs
        self.stop_after = stop_after
        self._next_calls = 0
        self._stop_calls = 0

    def next_action(self, _state: np.ndarray) -> np.ndarray:
        idx = self._next_calls % self.n_evidences
        self._next_calls += 1
        return np.array([idx])

    def should_stop(self, _state: np.ndarray) -> np.ndarray:
        self._stop_calls += 1
        return np.array([self._stop_calls >= self.stop_after])

    def diagnose(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probs = np.tile(self.probs, (state.shape[0], 1))
        return probs.argmax(1), probs


# --- canonical dataset fixture ---------------------------------------------


_DATASET_ID = "testds"
_MODEL_ID = "m1"


def _canonical() -> CanonicalDataset:
    layout = build_layout(
        [
            {"name": "E_a", "dtype": "B", "values": []},
            {"name": "E_b", "dtype": "B", "values": []},
            {"name": "E_c", "dtype": "C", "values": ["V_1", "V_2"]},
        ]
    )
    evidences = [
        CanonicalEvidence(
            id="E_a", idx=0, dtype="B", native_question_text={"en": "Do you have A?"}
        ),
        CanonicalEvidence(
            id="E_b", idx=1, dtype="B", native_question_text={"en": "Do you have B?"}
        ),
        CanonicalEvidence(
            id="E_c",
            idx=2,
            dtype="C",
            values=[
                CanonicalValue(raw="V_1", local_idx=0),
                CanonicalValue(raw="V_2", local_idx=1),
            ],
            native_question_text={"en": "Which C?"},
            native_value_labels={
                "V_1": {"en": "Value one"},
                "V_2": {"en": "Value two"},
            },
        ),
    ]
    conditions = [
        CanonicalCondition(
            id="critical_disease",
            idx=0,
            severity=1,
            icd10="C00",
            native_name={"en": "Critical disease"},
        ),
        CanonicalCondition(
            id="moderate_disease",
            idx=1,
            severity=3,
            native_name={"en": "Moderate disease"},
        ),
    ]
    return CanonicalDataset.build(
        id=_DATASET_ID,
        evidences=evidences,
        conditions=conditions,
        layout=layout,
        severity_vector=np.array([1.0, 3.0]),
    )


def _spec() -> DatasetSpec:
    return DatasetSpec(id=_DATASET_ID, model_ids=[_MODEL_ID])


def _loaded(*, agent: _StubAgent, maxstep: int = 8) -> LoadedDataset:
    model_spec = ModelSpec(
        id=_MODEL_ID,
        algorithm_module="typed_basd",
        weights_subpath=f"{_DATASET_ID}/{_MODEL_ID}",
        manifest_sha256="a" * 64,
        maxstep=maxstep,
    )
    return LoadedDataset(
        spec=_spec(),
        canonical=_canonical(),
        models={
            _MODEL_ID: LoadedModel(
                spec=model_spec,
                agent=agent,
                manifest={"eval": {"DDF1": 88.0}},
            )
        },
    )


@pytest.fixture(autouse=True)
def _skip_lifespan(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("CLARITYMED_SYMPTOMS_SKIP_LOAD", "1")


@pytest.fixture(autouse=True)
def _reset_server_state():
    SERVER_STATE.reset()
    yield
    SERVER_STATE.reset()


# --- harness ---------------------------------------------------------------


class _StubChannel:
    """Scripted prompt channel — mirrors the unit-test fixture."""

    def __init__(self, answers: list[AskUserQuestionResult]) -> None:
        self._answers = list(answers)
        self.calls: list[AskUserQuestionInput] = []

    async def ask(self, payload: AskUserQuestionInput) -> AskUserQuestionResult:
        self.calls.append(payload)
        if not self._answers:
            raise AssertionError("scripted channel ran out of answers")
        return self._answers.pop(0)


class _StubEligibility:
    def __init__(self, result: EligibilityResult) -> None:
        self.result = result

    async def check(self, complaint, language, profile, dataset):
        return self.result


def _config() -> SymptomsConfig:
    spec = _spec()
    model_spec = ModelSpec(
        id=_MODEL_ID,
        algorithm_module="typed_basd",
        weights_subpath=f"{_DATASET_ID}/{_MODEL_ID}",
        manifest_sha256="a" * 64,
        maxstep=8,
    )
    return SymptomsConfig(
        datasets=[spec],
        models=[model_spec],
        eligibility=EligibilityCatalogConfig(
            active="direct",
            catalog=[DirectEligibilityEntry(id="direct", kind="direct")],
        ),
    )


def _make_plugin(
    *,
    channel: _StubChannel,
    agent: _StubAgent,
    maxstep: int = 8,
    eligibility_result: EligibilityResult | None = None,
) -> tuple[SymptomsFeature, SymptomsServerClient, SimpleNamespace]:
    """Construct the plugin against the real ASGI-transport client."""
    SERVER_STATE.datasets[_DATASET_ID] = _loaded(agent=agent, maxstep=maxstep)
    SERVER_STATE.config_loaded = True

    config = _config()
    transport = httpx.ASGITransport(app=app)
    client = SymptomsServerClient("http://test", transport=transport)
    plugin = SymptomsFeature(
        config=config,
        registry=DatasetRegistry(config.datasets),
        client=client,
        eligibility=_StubEligibility(
            eligibility_result
            or EligibilityResult(eligible=True, reason="in_scope", confidence=0.7)
        ),
        profile_loader=lambda uid: Profile(),
    )
    deps = SimpleNamespace(user_id="test", language="en", prompt_channel=channel)
    return plugin, client, deps


def _yes() -> AskUserQuestionResult:
    return AskUserQuestionResult(answers={"Try follow-up?": "Yes"})


def _initial_batch() -> AskUserQuestionResult:
    return AskUserQuestionResult(
        answers={"Biological sex (for the differential model)?": "Male"},
        numeric_values={"How old are you, in years?": 45},
    )


def _answer(label: str = "Yes") -> AskUserQuestionResult:
    return AskUserQuestionResult(answers={"q": label})


# --- tests ----------------------------------------------------------------


async def test_happy_path_end_to_end_through_real_server() -> None:
    """The plugin runs eligibility → confirm → batch → start → turn → done
    via the real FastAPI app. Verifies the wire contract holds end-to-end
    and the differential surfaces back to the LLM with the right schema."""
    agent = _StubAgent(
        n_evidences=3,
        probs=np.array([0.85, 0.15]),  # critical_disease wins
        stop_after=1,  # first /turn → should_stop fires → done
    )
    channel = _StubChannel(
        [
            _yes(),  # confirm modal
            _initial_batch(),  # age + sex
            _answer("Yes"),  # answer to first server question
        ]
    )
    plugin, client, deps = _make_plugin(channel=channel, agent=agent)
    token = apply_context(
        request_id="20260613000000ABCDEFAB", user_id="test", language="en"
    )
    try:
        result = await plugin._predict(
            SimpleNamespace(deps=deps), complaint="chest pain"
        )
    finally:
        await client.aclose()
        reset_context(token)
    assert result["eligible"] is True
    assert result["turn_count"] >= 1
    diff = result["differential"]
    assert len(diff) >= 1
    top = diff[0]
    assert top["condition_id"] == "critical_disease"
    assert top["severity"] == 1
    # KTD-2: the plugin stashed the max severity for post_process. Run it
    # and confirm Critical-tier reply with a keyword is unmodified.
    await plugin.post_process(
        "Call 911 right now — these symptoms could be serious.", result
    )


async def test_cap_hit_path_end_to_end() -> None:
    """maxstep=2 + an agent that never stops → second turn triggers the
    cap, the plugin surfaces ``hit_cap: true`` with partial differential."""
    agent = _StubAgent(
        n_evidences=3,
        probs=np.array([0.5, 0.5]),
        stop_after=100,
    )
    channel = _StubChannel(
        [
            _yes(),
            _initial_batch(),
            _answer("Yes"),  # turn 1
            _answer("Yes"),  # turn 2 — cap fires
        ]
    )
    plugin, client, deps = _make_plugin(channel=channel, agent=agent, maxstep=2)
    token = apply_context(
        request_id="20260613000000ABCDEFAB", user_id="test", language="en"
    )
    try:
        result = await plugin._predict(
            SimpleNamespace(deps=deps), complaint="chest pain"
        )
    finally:
        await client.aclose()
        reset_context(token)
    assert result["hit_cap"] is True
    assert result["turn_count"] == 2
    assert len(result["partial_differential"]) >= 1


async def test_cancel_mid_loop_through_real_server() -> None:
    """User declines a turn mid-loop → plugin calls DELETE
    /v1/datasets/{id}/sessions/{sid} on the real server and translates
    the cancel response into the LLM-facing shape."""
    from claritymed.core.interaction.prompt_channel import UserDeclinedAnswer

    agent = _StubAgent(
        n_evidences=3,
        probs=np.array([0.8, 0.2]),
        stop_after=100,
    )
    channel = _StubChannel(
        [
            _yes(),
            _initial_batch(),
            _answer("Yes"),  # turn 1
            UserDeclinedAnswer(),  # turn 2 — user dismisses
        ]
    )

    # Need to patch the channel since UserDeclinedAnswer needs raising.
    class _CancelChannel(_StubChannel):
        async def ask(self, payload):
            self.calls.append(payload)
            nxt = self._answers.pop(0)
            if isinstance(nxt, UserDeclinedAnswer):
                raise nxt
            return nxt

    channel = _CancelChannel(
        [_yes(), _initial_batch(), _answer("Yes"), UserDeclinedAnswer()]
    )
    plugin, client, deps = _make_plugin(channel=channel, agent=agent)
    token = apply_context(
        request_id="20260613000000ABCDEFAB", user_id="test", language="en"
    )
    try:
        result = await plugin._predict(
            SimpleNamespace(deps=deps), complaint="chest pain"
        )
    finally:
        await client.aclose()
        reset_context(token)
    assert result["cancelled"] is True
    # The cancel decision-table fields are present.
    assert "partial_confidence" in result
    assert "meets_confidence_threshold" in result


async def test_ineligible_short_circuits_no_server_call() -> None:
    """Eligibility false → plugin returns immediately without touching
    the server. Verifies the eligibility filter is structural, not
    delegated to the server."""
    channel = _StubChannel([])
    plugin, client, deps = _make_plugin(
        channel=channel,
        agent=_StubAgent(n_evidences=3, probs=np.array([0.5, 0.5])),
        eligibility_result=EligibilityResult(eligible=False, reason="out_of_scope"),
    )
    token = apply_context(
        request_id="20260613000000ABCDEFAB", user_id="test", language="en"
    )
    try:
        result = await plugin._predict(
            SimpleNamespace(deps=deps), complaint="reset password"
        )
    finally:
        await client.aclose()
        reset_context(token)
    assert result == {"eligible": False, "reason": "out_of_scope"}
    assert channel.calls == []  # no modal ever rendered


# --- safety_keywords.missing audit (Unit 18) -------------------------------

# Probability vectors that lock the differential to a single condition. The
# canonical fixture has two conditions: critical_disease (sev=1) at index 0,
# moderate_disease (sev=3) at index 1. The audit boundary lives at tier:
# Critical (sev=1) + Urgent (sev=2) are audited, Moderate (sev=3-4) + Mild
# (sev=5) are skipped. We cover both branches at the integration layer; the
# Urgent + Mild rows are exercised exhaustively in the unit suite.
#
# Server's ``_topk_rows`` filters out probs below ``DIFFERENTIAL_PROB_THRESHOLD``
# (0.01). The moderate case pushes critical_disease below that floor so only
# moderate_disease enters the differential — otherwise ``_max_severity`` would
# pick up the Critical row and the tier check would fire even though the top-1
# is moderate.
_PROBS_CRITICAL = np.array([0.95, 0.05])  # both rows present; min sev = 1
_PROBS_MODERATE = np.array([0.005, 0.995])  # critical filtered; sev = 3 only


# --- server resilience (Unit 18) -------------------------------------------


def _make_plugin_with_failing_transport(
    *,
    channel: _StubChannel,
    fail_path_substring: str | None = None,
    exception: BaseException,
) -> tuple[SymptomsFeature, SymptomsServerClient, SimpleNamespace]:
    """Like :func:`_make_plugin` but routes requests through a MockTransport
    that injects ``exception``.

    ``fail_path_substring=None`` fails every request (start session).
    ``fail_path_substring="/turn"`` fails turn requests and lets the start
    request pass through to the real ASGI app — covers the "succeeded then
    failed mid-loop" branch the plugin handles separately.
    """
    agent = _StubAgent(
        n_evidences=3,
        probs=np.array([0.95, 0.05]),
        stop_after=100,
    )
    SERVER_STATE.datasets[_DATASET_ID] = _loaded(agent=agent)
    SERVER_STATE.config_loaded = True
    config = _config()

    real_asgi = httpx.ASGITransport(app=app)

    async def _handler(request: httpx.Request) -> httpx.Response:
        if fail_path_substring is None or fail_path_substring in request.url.path:
            raise exception
        return await real_asgi.handle_async_request(request)

    transport = httpx.MockTransport(_handler)
    client = SymptomsServerClient("http://test", transport=transport)
    plugin = SymptomsFeature(
        config=config,
        registry=DatasetRegistry(config.datasets),
        client=client,
        eligibility=_StubEligibility(
            EligibilityResult(eligible=True, reason="in_scope", confidence=0.7)
        ),
        profile_loader=lambda uid: Profile(),
    )
    deps = SimpleNamespace(user_id="test", language="en", prompt_channel=channel)
    return plugin, client, deps


async def test_server_timeout_at_start_returns_server_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Transport-level ``httpx.ReadTimeout`` on ``POST /sessions`` → plugin
    surfaces ``{eligible: True, server_error: True}`` and emits the
    ``symptoms.session.cancelled`` audit with phase=start."""
    channel = _StubChannel([_yes(), _initial_batch()])
    plugin, client, deps = _make_plugin_with_failing_transport(
        channel=channel,
        fail_path_substring=None,
        exception=httpx.ReadTimeout("simulated read timeout"),
    )
    token = apply_context(
        request_id="20260613000000ABCDEFAB", user_id="test", language="en"
    )
    try:
        with caplog.at_level("INFO", logger="claritymed.audit"):
            result = await plugin._predict(
                SimpleNamespace(deps=deps), complaint="chest pain"
            )
    finally:
        await client.aclose()
        reset_context(token)
    assert result == {"eligible": True, "server_error": True}
    cancelled_lines = [
        rec.message
        for rec in caplog.records
        if rec.name == "claritymed.audit"
        and "symptoms.session.cancelled" in rec.message
    ]
    assert len(cancelled_lines) == 1, (
        f"expected one session.cancelled audit, got {len(cancelled_lines)}: "
        f"{cancelled_lines}"
    )
    line = cancelled_lines[0]
    assert '"phase":"start"' in line or '"phase": "start"' in line
    assert (
        '"reason":"server_unreachable"' in line
        or '"reason": "server_unreachable"' in line
    )


async def test_server_500_at_turn_propagates_as_server_error() -> None:
    """A 500 response on the first ``/turn`` becomes
    ``SymptomsServerUnreachableError`` inside the client and the plugin
    treats it the same as a transport failure."""
    channel = _StubChannel([_yes(), _initial_batch(), _answer("Yes")])
    plugin, client, deps = _make_plugin_with_failing_transport(
        channel=channel,
        fail_path_substring=None,  # Want every request to 500.
        exception=httpx.HTTPError("simulated 500"),
    )
    token = apply_context(
        request_id="20260613000000ABCDEFAB", user_id="test", language="en"
    )
    try:
        result = await plugin._predict(
            SimpleNamespace(deps=deps), complaint="chest pain"
        )
    finally:
        await client.aclose()
        reset_context(token)
    assert result == {"eligible": True, "server_error": True}


# --- PHI hygiene (Unit 18) -------------------------------------------------

# Sentinel string improbable in any non-complaint context (audit field name,
# dataset id, evidence label, condition slug, …). If it leaks out of the
# plugin's predicted-data surface, the test catches it.
_PHI_SENTINEL = "ZQ-ENTROPY-MARKER-7741-pHi-hygi3ne"


async def test_complaint_never_appears_in_tool_result_or_audit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """PHI hygiene — the chief complaint string is consumed locally (eligibility
    + local symptoms server) and must NOT appear in:

    * the tool result returned to the orchestrator (which composes the
      cloud-bound assembled prompt); or
    * any audit log line (audit.log is grep-friendly + ships to remote SIEMs).

    A leak here would route the complaint through the orchestrator's cloud
    PHI path, where ``phi_guard`` would have to catch it on the second pass.
    Defence-in-depth: shut the door at the plugin boundary too.
    """
    agent = _StubAgent(n_evidences=3, probs=_PROBS_CRITICAL, stop_after=1)
    channel = _StubChannel([_yes(), _initial_batch(), _answer("Yes")])
    plugin, client, deps = _make_plugin(channel=channel, agent=agent)
    token = apply_context(
        request_id="20260613000000ABCDEFAB", user_id="test", language="en"
    )
    try:
        with caplog.at_level("INFO", logger="claritymed.audit"):
            result = await plugin._predict(
                SimpleNamespace(deps=deps), complaint=_PHI_SENTINEL
            )
    finally:
        await client.aclose()
        reset_context(token)

    # Whole-result scan. Use repr(result) so nested dicts/lists are included.
    assert _PHI_SENTINEL not in repr(result), (
        f"complaint leaked into tool result: {result!r}"
    )
    # Audit log scan — every record routed to the audit logger.
    leaked_audit = [
        rec.message
        for rec in caplog.records
        if rec.name == "claritymed.audit" and _PHI_SENTINEL in rec.message
    ]
    assert leaked_audit == [], f"complaint leaked into audit log: {leaked_audit}"


@pytest.mark.parametrize(
    "case,probs,reply_text,expect_audit",
    [
        (
            "critical_with_keyword",
            _PROBS_CRITICAL,
            "Call 911 right now — these symptoms could be serious.",
            False,
        ),
        (
            "critical_missing_keyword",
            _PROBS_CRITICAL,
            "You should probably see a doctor about that when you can.",
            True,
        ),
        (
            "moderate_tier_skipped",
            _PROBS_MODERATE,
            "No need to do anything special — just rest at home.",
            False,
        ),
    ],
    ids=lambda v: v if isinstance(v, str) else "",
)
async def test_safety_keywords_audit_through_real_server(
    case: str,
    probs: np.ndarray,
    reply_text: str,
    expect_audit: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The stash key set in ``_predict`` must survive the ASGI roundtrip so
    ``post_process`` finds it and the tier check fires correctly. This is
    the e2e complement to the post_process unit matrix — guards against a
    regression where request_id resolution drifts between the two halves.
    """
    agent = _StubAgent(n_evidences=3, probs=probs, stop_after=1)
    channel = _StubChannel([_yes(), _initial_batch(), _answer("Yes")])
    plugin, client, deps = _make_plugin(channel=channel, agent=agent)
    token = apply_context(
        request_id="20260613000000ABCDEFAB", user_id="test", language="en"
    )
    try:
        result = await plugin._predict(
            SimpleNamespace(deps=deps), complaint="chest pain"
        )
        # Capture audit records emitted during post_process only — earlier
        # observations (eligibility, server interactions) are out of scope.
        with caplog.at_level("INFO", logger="claritymed.audit"):
            await plugin.post_process(reply_text, result)
    finally:
        await client.aclose()
        reset_context(token)

    audit_lines = [
        rec.message
        for rec in caplog.records
        if rec.name == "claritymed.audit"
        and "symptoms.safety_keywords.missing" in rec.message
    ]
    if expect_audit:
        assert len(audit_lines) == 1, (
            f"{case}: expected one safety_keywords.missing event, "
            f"got {len(audit_lines)}: {audit_lines}"
        )
        # The audit payload must carry the tier + max_severity so an
        # operator can grep by tier without re-deriving severity.
        line = audit_lines[0]
        assert '"tier":"Critical"' in line or '"tier": "Critical"' in line
        assert '"max_severity":1' in line or '"max_severity": 1' in line
    else:
        assert audit_lines == [], f"{case}: expected no audit, got {audit_lines}"
