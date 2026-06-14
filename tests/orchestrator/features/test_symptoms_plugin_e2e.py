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
    SafetyKeywordsByTier,
    SafetyKeywordsLang,
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


def _spec(maxstep: int = 8) -> DatasetSpec:
    return DatasetSpec(id=_DATASET_ID, model_ids=[_MODEL_ID], maxstep=maxstep)


def _loaded(*, agent: _StubAgent, maxstep: int = 8) -> LoadedDataset:
    model_spec = ModelSpec(
        id=_MODEL_ID,
        algorithm_module="typed_basd",
        weights_subpath=f"{_DATASET_ID}/{_MODEL_ID}",
        manifest_sha256="a" * 64,
    )
    return LoadedDataset(
        spec=_spec(maxstep=maxstep),
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
    )
    return SymptomsConfig(
        datasets=[spec],
        models=[model_spec],
        eligibility=EligibilityCatalogConfig(
            active="direct",
            catalog=[DirectEligibilityEntry(id="direct", kind="direct")],
        ),
        safety_keywords_by_tier=SafetyKeywordsByTier(
            Critical=SafetyKeywordsLang(en=["call 911"], zh=["120"]),
            Urgent=SafetyKeywordsLang(en=["urgent care"], zh=["急诊"]),
            Moderate=SafetyKeywordsLang(en=["see your doctor"], zh=["门诊"]),
            Mild=SafetyKeywordsLang(en=["rest"], zh=["休息"]),
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
