"""End-to-end-ish tests for the FastAPI symptoms app.

Uses :class:`httpx.ASGITransport` so the app runs in-process — no socket
binding, no subprocess. ``SERVER_STATE.datasets`` is pre-populated with
a fake :class:`LoadedDataset` whose agent is a deterministic stub, so
the lifecycle (start → turn → done / cap / cancel) can be exercised
without loading torch weights.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from claritymed.core.symptoms.init_matcher import MatchResult

from claritymed.core.symptoms.datasets import (
    CanonicalCondition,
    CanonicalDataset,
    CanonicalEvidence,
    CanonicalValue,
    LoadedDataset,
    LoadedModel,
)
from claritymed.core.symptoms.schemas import DatasetSpec, ModelSpec
from claritymed.ingest.symptoms.typed_basd import build_layout
from claritymed.servers.symptoms.app import HOST, app
from claritymed.servers.symptoms.state import SERVER_STATE

# Resolve the submodule via sys.modules — ``import claritymed.servers.symptoms.app
# as app_mod`` falls afoul of attribute access (``__init__`` re-exports
# the FastAPI instance under the name ``app``, shadowing the submodule).
app_mod = sys.modules["claritymed.servers.symptoms.app"]


# --- agent stubs ----------------------------------------------------------


class _StubAgent:
    """Deterministic agent for app lifecycle tests.

    ``next_action`` cycles through evidences in order; ``should_stop``
    fires after ``stop_after`` turns; ``diagnose`` returns a fixed prob
    vector (and its argmax). The stub records every call to make
    assertions explicit.
    """

    def __init__(
        self,
        *,
        n_evidences: int,
        probs: np.ndarray,
        stop_after: int = 3,
    ) -> None:
        self.n_evidences = n_evidences
        self.probs = probs
        self.stop_after = stop_after
        self.next_calls = 0
        self.should_stop_calls = 0

    def next_action(self, _state: np.ndarray) -> np.ndarray:
        idx = self.next_calls % self.n_evidences
        self.next_calls += 1
        return np.array([idx])

    def should_stop(self, _state: np.ndarray) -> np.ndarray:
        self.should_stop_calls += 1
        return np.array([self.should_stop_calls >= self.stop_after])

    def diagnose(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        probs = np.tile(self.probs, (state.shape[0], 1))
        return probs.argmax(1), probs


def _canonical() -> CanonicalDataset:
    """Two-binary + one-categorical synthetic dataset; 3 conditions."""
    layout = build_layout(
        [
            {"name": "E_a", "dtype": "B", "values": []},
            {"name": "E_b", "dtype": "B", "values": []},
            {"name": "E_c", "dtype": "C", "values": ["V_1", "V_2"]},
        ]
    )
    evidences = [
        CanonicalEvidence(
            id="E_a",
            idx=0,
            dtype="B",
            native_question_text={"en": "Do you have A?"},
        ),
        CanonicalEvidence(
            id="E_b",
            idx=1,
            dtype="B",
            native_question_text={"en": "Do you have B?"},
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
        CanonicalCondition(
            id="mild_disease",
            idx=2,
            severity=5,
            native_name={"en": "Mild disease"},
        ),
    ]
    return CanonicalDataset.build(
        id="testds",
        evidences=evidences,
        conditions=conditions,
        layout=layout,
        severity_vector=np.array([1.0, 3.0, 5.0]),
    )


def _spec(maxstep: int = 8) -> DatasetSpec:
    return DatasetSpec(id="testds", model_ids=["m1"], maxstep=maxstep)


def _loaded_dataset(*, agent: _StubAgent, maxstep: int = 8) -> LoadedDataset:
    spec = _spec(maxstep=maxstep)
    model_spec = ModelSpec(
        id="m1",
        algorithm_module="typed_basd",
        weights_subpath="testds/m1",
        manifest_sha256="a" * 64,
    )
    return LoadedDataset(
        spec=spec,
        canonical=_canonical(),
        models={
            "m1": LoadedModel(
                spec=model_spec,
                agent=agent,
                manifest={"eval": {"DDF1": 88.0}},
            )
        },
    )


@pytest.fixture(autouse=True)
def _skip_lifespan_load(monkeypatch: pytest.MonkeyPatch):
    """Prevent the lifespan from trying to load real datasets in tests."""
    monkeypatch.setenv("CLARITYMED_SYMPTOMS_SKIP_LOAD", "1")


@pytest.fixture(autouse=True)
def _reset_state():
    SERVER_STATE.reset()
    yield
    SERVER_STATE.reset()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# --- /health --------------------------------------------------------------


def test_health_lists_loaded_datasets(client: TestClient) -> None:
    SERVER_STATE.datasets["testds"] = _loaded_dataset(
        agent=_StubAgent(n_evidences=3, probs=np.array([0.9, 0.05, 0.05]))
    )
    SERVER_STATE.config_loaded = True
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["datasets_loaded"] == ["testds"]
    assert body["models_loaded"] == ["m1"]


def test_health_during_loading(client: TestClient) -> None:
    SERVER_STATE.config_loaded = False
    response = client.get("/health")
    assert response.json()["status"] == "loading"


# --- start session --------------------------------------------------------


def _start_payload(language: str = "en") -> dict:
    return {
        "complaint": "chest pain",
        "profile": {"age_years": 45, "sex": "M"},
        "language": language,
    }


def test_start_session_returns_first_question(client: TestClient) -> None:
    agent = _StubAgent(n_evidences=3, probs=np.array([0.9, 0.05, 0.05]))
    SERVER_STATE.datasets["testds"] = _loaded_dataset(agent=agent)
    SERVER_STATE.config_loaded = True

    response = client.post("/v1/datasets/testds/sessions", json=_start_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["session_id"]
    assert body["first_question"]["question"] == "Do you have A?"
    assert agent.next_calls == 1
    assert body["session_id"] in SERVER_STATE.sessions


def test_start_session_unknown_dataset_returns_404(client: TestClient) -> None:
    SERVER_STATE.config_loaded = True
    response = client.post("/v1/datasets/missing/sessions", json=_start_payload())
    assert response.status_code == 404
    assert "not loaded" in response.json()["detail"]


def test_start_session_rejects_invalid_age(client: TestClient) -> None:
    SERVER_STATE.datasets["testds"] = _loaded_dataset(
        agent=_StubAgent(n_evidences=3, probs=np.array([0.9, 0.05, 0.05]))
    )
    SERVER_STATE.config_loaded = True
    bad = _start_payload()
    bad["profile"]["age_years"] = 999  # out of [0, 120]
    response = client.post("/v1/datasets/testds/sessions", json=bad)
    assert response.status_code == 422


# --- turn lifecycle (done / cap / continuing) ----------------------------


def test_turn_completes_when_should_stop_fires(client: TestClient) -> None:
    agent = _StubAgent(
        n_evidences=3,
        probs=np.array([0.85, 0.1, 0.05]),  # critical_disease wins
        stop_after=1,  # first /turn call → should_stop fires → done
    )
    SERVER_STATE.datasets["testds"] = _loaded_dataset(agent=agent)
    SERVER_STATE.config_loaded = True

    start = client.post("/v1/datasets/testds/sessions", json=_start_payload()).json()
    session_id = start["session_id"]
    response = client.post(
        f"/v1/datasets/testds/sessions/{session_id}/turn",
        json={"answer": "Yes", "language": "en"},
    )
    body = response.json()
    assert body["done"] is True
    assert len(body["differential"]) >= 1
    top = body["differential"][0]
    assert top["condition_id"] == "critical_disease"
    assert top["severity"] == 1
    assert top["icd10"] == "C00"
    # Session is dropped after completion.
    assert session_id not in SERVER_STATE.sessions


def test_turn_returns_next_question_when_still_running(client: TestClient) -> None:
    agent = _StubAgent(
        n_evidences=3,
        probs=np.array([0.4, 0.3, 0.3]),
        stop_after=10,  # never within this test
    )
    SERVER_STATE.datasets["testds"] = _loaded_dataset(agent=agent, maxstep=8)
    SERVER_STATE.config_loaded = True

    start = client.post("/v1/datasets/testds/sessions", json=_start_payload()).json()
    session_id = start["session_id"]
    response = client.post(
        f"/v1/datasets/testds/sessions/{session_id}/turn",
        json={"answer": "Yes", "language": "en"},
    )
    body = response.json()
    assert body["done"] is False
    assert body["hit_cap"] is False
    assert body["next_question"] is not None
    assert body["turn_count"] == 1


def test_turn_hits_cap_at_maxstep(client: TestClient) -> None:
    agent = _StubAgent(
        n_evidences=3,
        probs=np.array([0.5, 0.3, 0.2]),
        stop_after=100,
    )
    SERVER_STATE.datasets["testds"] = _loaded_dataset(agent=agent, maxstep=2)
    SERVER_STATE.config_loaded = True

    start = client.post("/v1/datasets/testds/sessions", json=_start_payload()).json()
    session_id = start["session_id"]
    # First turn — still running.
    client.post(
        f"/v1/datasets/testds/sessions/{session_id}/turn",
        json={"answer": "Yes", "language": "en"},
    )
    # Second turn — cap should trigger because turn_count reaches maxstep=2.
    response = client.post(
        f"/v1/datasets/testds/sessions/{session_id}/turn",
        json={"answer": "Yes", "language": "en"},
    )
    body = response.json()
    assert body["hit_cap"] is True
    assert body["turn_count"] == 2
    assert session_id not in SERVER_STATE.sessions


def test_turn_rejects_unknown_session(client: TestClient) -> None:
    SERVER_STATE.datasets["testds"] = _loaded_dataset(
        agent=_StubAgent(n_evidences=3, probs=np.array([0.9, 0.05, 0.05]))
    )
    SERVER_STATE.config_loaded = True
    response = client.post(
        "/v1/datasets/testds/sessions/nope/turn",
        json={"answer": "Yes", "language": "en"},
    )
    assert response.status_code == 404


# --- cancel ---------------------------------------------------------------


def test_cancel_returns_partial_outcome(client: TestClient) -> None:
    """Severity_override fires when a severity-1 disease has prob > 0.1."""
    agent = _StubAgent(
        n_evidences=3,
        probs=np.array([0.25, 0.35, 0.4]),  # severity 1 at prob 0.25 → override
        stop_after=100,
    )
    SERVER_STATE.datasets["testds"] = _loaded_dataset(agent=agent)
    SERVER_STATE.config_loaded = True

    start = client.post("/v1/datasets/testds/sessions", json=_start_payload()).json()
    session_id = start["session_id"]
    response = client.delete(f"/v1/datasets/testds/sessions/{session_id}")
    body = response.json()
    assert body["cancelled"] is True
    assert body["severity_override"] is True
    assert body["max_low_severity_seen"] == 1
    assert session_id not in SERVER_STATE.sessions


def test_cancel_below_threshold_suppresses_partial(client: TestClient) -> None:
    """No severity-override, top-3 mass below threshold → empty partial."""
    agent = _StubAgent(
        n_evidences=3,
        probs=np.array([0.05, 0.04, 0.06]),  # never sums above partial_min_confidence
        stop_after=100,
    )
    SERVER_STATE.datasets["testds"] = _loaded_dataset(agent=agent)
    SERVER_STATE.config_loaded = True

    start = client.post("/v1/datasets/testds/sessions", json=_start_payload()).json()
    session_id = start["session_id"]
    response = client.delete(f"/v1/datasets/testds/sessions/{session_id}")
    body = response.json()
    assert body["cancelled"] is True
    assert body["meets_confidence_threshold"] is False
    assert body["severity_override"] is False
    assert body["partial_differential"] == []


# --- host guard -----------------------------------------------------------


def test_host_constant_is_loopback() -> None:
    """Pre-bind guard: HOST must never drift away from 127.0.0.1."""
    assert HOST == "127.0.0.1"


def test_main_binds_loopback_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """``main`` calls uvicorn.run with host=127.0.0.1 — captured via monkeypatch."""
    captured: dict[str, Any] = {}

    def _fake_run(app_, host, **kwargs):  # noqa: ANN001
        captured["host"] = host
        captured["kwargs"] = kwargs

    monkeypatch.setattr(app_mod.uvicorn, "run", _fake_run)
    app_mod.main()
    assert captured["host"] == "127.0.0.1"


# --- init-symptom matcher injection --------------------------------------


class _StubMatcher:
    """Process-singleton substitute for InitMatcherEmbedder.

    Carries a fixed mapping ``text → MatchResult``. The server's
    start_session calls ``match(text, catalog)``; this stub ignores
    the catalog and returns the canned result so each test can pin
    exactly which evidence (if any) gets injected.
    """

    def __init__(self, responses: dict[str, "MatchResult"]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def match(self, text: str, _catalog) -> "MatchResult":
        self.calls.append(text)
        from claritymed.core.symptoms.init_matcher import MatchResult

        return self._responses.get(
            text.strip(), MatchResult(evidence_idx=None, score=0.0)
        )


def _loaded_dataset_with_catalog(agent: _StubAgent, *, threshold: float = 0.5):
    """LoadedDataset variant with an init catalog wired in.

    The catalog matrix is a dummy 2×4 — the stub matcher returns
    canned results without consulting it, so values don't matter as
    long as the dataclass is well-formed.
    """
    from claritymed.core.symptoms.datasets.canonical import InitSymptomCatalog

    ds = _loaded_dataset(agent=agent)
    catalog = InitSymptomCatalog(
        candidate_idx=[0, 1],  # E_a, E_b — both B-type
        matrix=np.eye(2, 4, dtype=np.float32),
        threshold=threshold,
    )
    return LoadedDataset(
        spec=ds.spec,
        canonical=ds.canonical,
        models=ds.models,
        init_catalog=catalog,
    )


def test_start_session_injects_matched_evidence_into_state(client: TestClient) -> None:
    from claritymed.core.symptoms.init_matcher import MatchResult

    agent = _StubAgent(n_evidences=3, probs=np.array([0.9, 0.05, 0.05]))
    SERVER_STATE.datasets["testds"] = _loaded_dataset_with_catalog(agent)
    SERVER_STATE.config_loaded = True
    SERVER_STATE.init_matcher_model = _StubMatcher(
        {"chest pain summary": MatchResult(evidence_idx=1, score=0.85)}
    )
    payload = _start_payload()
    payload["symptom_summary"] = "chest pain summary"

    response = client.post("/v1/datasets/testds/sessions", json=payload)
    assert response.status_code == 200
    body = response.json()
    session_id = body["session_id"]
    sub = SERVER_STATE.sessions[session_id]

    assert sub.evidence_collected[0]["evidence_id"] == "E_b"
    assert sub.evidence_collected[0]["source"] == "init_matcher"
    # The matched evidence's block-start slot is 1 in state[0] —
    # the layout's `off` array maps idx → slot.
    ds = SERVER_STATE.datasets["testds"]
    block_start = int(ds.canonical.layout["off"][1])
    assert sub.state[0, block_start] == 1.0


def test_start_session_falls_back_to_complaint_when_no_summary(
    client: TestClient,
) -> None:
    from claritymed.core.symptoms.init_matcher import MatchResult

    agent = _StubAgent(n_evidences=3, probs=np.array([0.9, 0.05, 0.05]))
    SERVER_STATE.datasets["testds"] = _loaded_dataset_with_catalog(agent)
    SERVER_STATE.config_loaded = True
    matcher = _StubMatcher({"chest pain": MatchResult(evidence_idx=0, score=0.9)})
    SERVER_STATE.init_matcher_model = matcher

    response = client.post("/v1/datasets/testds/sessions", json=_start_payload())
    assert response.status_code == 200
    # Matcher received the raw complaint, since payload had no summary.
    assert matcher.calls == ["chest pain"]


def test_start_session_no_injection_when_matcher_missing(client: TestClient) -> None:
    agent = _StubAgent(n_evidences=3, probs=np.array([0.9, 0.05, 0.05]))
    SERVER_STATE.datasets["testds"] = _loaded_dataset_with_catalog(agent)
    SERVER_STATE.config_loaded = True
    SERVER_STATE.init_matcher_model = None  # matcher disabled

    response = client.post("/v1/datasets/testds/sessions", json=_start_payload())
    assert response.status_code == 200
    sub = SERVER_STATE.sessions[response.json()["session_id"]]
    assert sub.evidence_collected == []


def test_start_session_no_injection_when_below_threshold(client: TestClient) -> None:
    from claritymed.core.symptoms.init_matcher import MatchResult

    agent = _StubAgent(n_evidences=3, probs=np.array([0.9, 0.05, 0.05]))
    SERVER_STATE.datasets["testds"] = _loaded_dataset_with_catalog(agent)
    SERVER_STATE.config_loaded = True
    # Stub returns a None-idx MatchResult to mimic sub-threshold cosine.
    SERVER_STATE.init_matcher_model = _StubMatcher(
        {"chest pain": MatchResult(evidence_idx=None, score=0.42)}
    )

    response = client.post("/v1/datasets/testds/sessions", json=_start_payload())
    assert response.status_code == 200
    sub = SERVER_STATE.sessions[response.json()["session_id"]]
    assert sub.evidence_collected == []
