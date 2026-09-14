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


def _spec() -> DatasetSpec:
    return DatasetSpec(id="testds", model_ids=["m1"])


def _loaded_dataset(*, agent: _StubAgent, maxstep: int = 8) -> LoadedDataset:
    spec = _spec()
    model_spec = ModelSpec(
        id="m1",
        algorithm_module="typed_basd",
        weights_subpath="testds/m1",
        manifest_sha256="a" * 64,
        maxstep=maxstep,
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
    """Unknown dataset → 404 wrapped in the shared error envelope.

    The error envelope shape is shared across vision, symptoms, and
    medical-clip — a single client parser keys on ``error.code``. The
    pre-envelope behaviour returned bare strings under ``detail`` and
    broke any client that assumed the unified shape.
    """
    SERVER_STATE.config_loaded = True
    response = client.post("/v1/datasets/missing/sessions", json=_start_payload())
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "dataset_not_loaded"
    assert "missing" in body["error"]["message"]
    assert "available" in body["error"]["details"]


def test_start_session_rejects_invalid_age(client: TestClient) -> None:
    """Pydantic 422s are normalized to 400 + standard error envelope.

    Without the ``RequestValidationError`` handler this would surface
    FastAPI's stock ``{"detail": [{...}]}`` 422 — incompatible with
    vision/medical-clip's normalized shape.
    """
    SERVER_STATE.datasets["testds"] = _loaded_dataset(
        agent=_StubAgent(n_evidences=3, probs=np.array([0.9, 0.05, 0.05]))
    )
    SERVER_STATE.config_loaded = True
    bad = _start_payload()
    bad["profile"]["age_years"] = 999  # out of [0, 120]
    response = client.post("/v1/datasets/testds/sessions", json=bad)
    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "bad_request"
    assert isinstance(body["error"]["details"].get("errors"), list)


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


def test_turn_drains_pre_question_queue_before_ig_takes_over(
    client: TestClient,
) -> None:
    """With N init matches, the first N /turn calls should serve queued
    pre-questions (not IG picks) and the user's Yes/No writes to state.
    Once the queue drains, control passes to the agent's next_action."""
    from claritymed.core.symptoms.init_matcher import MatchResult
    from claritymed.core.symptoms.schemas import InitMatcherConfig

    agent = _StubAgent(
        n_evidences=3,
        probs=np.array([0.5, 0.3, 0.2]),
        stop_after=100,
    )
    SERVER_STATE.datasets["testds"] = _loaded_dataset_with_catalog(agent)
    SERVER_STATE.config_loaded = True

    class _MultiStub:
        def match(self, text, catalog):  # noqa: ARG002
            return MatchResult(evidence_idx=None, score=0.0)

        def match_topk(self, text, catalog, *, k=3, min_score=None):  # noqa: ARG002
            return [
                MatchResult(evidence_idx=0, score=0.90),
                MatchResult(evidence_idx=1, score=0.75),
            ][:k]

    SERVER_STATE.init_matcher_model = _MultiStub()
    SERVER_STATE.init_matcher_cfg = InitMatcherConfig(max_matches=3)

    start = client.post(
        "/v1/datasets/testds/sessions",
        json={**_start_payload(), "symptom_summary": "multi symptom summary"},
    ).json()
    session_id = start["session_id"]
    # First question is the top match (E_a).
    assert start["first_question"]["header"] == "E_a"

    # Answer Yes to E_a → server should apply it, then serve E_b from
    # the pre-question queue (NOT run IG next_action).
    r1 = client.post(
        f"/v1/datasets/testds/sessions/{session_id}/turn",
        json={"answer": "Yes", "language": "en"},
    ).json()
    assert r1["done"] is False
    assert r1["next_question"]["header"] == "E_b"
    sub = SERVER_STATE.sessions[session_id]
    # After the /turn, evidence_collected has ONE entry tagged as init.
    assert len(sub.evidence_collected) == 1
    assert sub.evidence_collected[0]["evidence_id"] == "E_a"
    assert sub.evidence_collected[0]["source"] == "init_matcher"
    assert sub.evidence_collected[0]["answer"] == "Yes"
    # Queue is drained now.
    assert sub.pending_init_confirmations == []

    # Answer No to E_b → queue empty, so next question comes from IG
    # (the stub agent's next_action returns evidence 0 by default; that
    # was already asked so it stays at 0 as fallback, but the important
    # thing is the source tag is NOT init_matcher).
    r2 = client.post(
        f"/v1/datasets/testds/sessions/{session_id}/turn",
        json={"answer": "No", "language": "en"},
    ).json()
    assert r2["done"] is False
    sub = SERVER_STATE.sessions[session_id]
    # Second collected entry is the E_b confirmation with "No".
    assert sub.evidence_collected[1]["evidence_id"] == "E_b"
    assert sub.evidence_collected[1]["source"] == "init_matcher"
    assert sub.evidence_collected[1]["answer"] == "No"


def test_turn_rejects_unknown_session(client: TestClient) -> None:
    """Unknown session → 404 in the shared envelope."""
    SERVER_STATE.datasets["testds"] = _loaded_dataset(
        agent=_StubAgent(n_evidences=3, probs=np.array([0.9, 0.05, 0.05]))
    )
    SERVER_STATE.config_loaded = True
    response = client.post(
        "/v1/datasets/testds/sessions/nope/turn",
        json={"answer": "Yes", "language": "en"},
    )
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "session_not_found"
    assert "session" in body["error"]["message"]


def test_error_envelope_carries_request_id_when_header_set(client: TestClient) -> None:
    """The ``X-Request-ID`` header round-trips through the error envelope
    so client-side traces can correlate failures end-to-end.

    Locks the contract that vision/medical-clip clients depend on:
    every 4xx response from a claritymed server includes
    ``error.request_id`` whenever the request carried the header.
    """
    SERVER_STATE.config_loaded = True
    response = client.post(
        "/v1/datasets/missing/sessions",
        json=_start_payload(),
        headers={"X-Request-ID": "20260621123456ABCDEF12"},
    )
    body = response.json()
    assert body["error"]["request_id"] == "20260621123456ABCDEF12"


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

    def match_topk(
        self,
        text: str,
        catalog,
        *,
        k: int = 3,
        min_score: float | None = None,  # noqa: ARG002
    ) -> list["MatchResult"]:
        """Same canned map, wrapped as a list. Sub-threshold matches (idx=None)
        are filtered so the server sees the same "no injection" branch it
        would with a real matcher below-threshold path."""
        result = self.match(text, catalog)
        if result.evidence_idx is None:
            return []
        return [result][:k]


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


def test_start_session_asks_matched_evidence_as_first_pre_question(
    client: TestClient,
) -> None:
    """Under the pre-question flow, a SapBERT match becomes the first
    Yes/No question the user sees. State is NOT pre-written — the user
    confirms before we treat the evidence as positive. This closes the
    "SapBERT semantic overreach + no negation detection" hole."""
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

    # The matched evidence (E_b, idx=1) is the first question asked.
    assert body["first_question"]["header"] == "E_b"
    # State is NOT pre-written; the user must confirm.
    ds = SERVER_STATE.datasets["testds"]
    block_start = int(ds.canonical.layout["off"][1])
    assert sub.state[0, block_start] == 0.0
    # evidence_collected is empty at session start; it gets filled on
    # the user's answer.
    assert sub.evidence_collected == []
    # Only one match → no more pending confirmations.
    assert sub.pending_init_confirmations == []
    # The pending source tag is recorded so the next /turn labels the
    # collected evidence as originating from the init matcher.
    assert sub.profile.get("_current_source") == "init_matcher"


def test_start_session_queues_secondary_matches_for_later_turns(
    client: TestClient,
) -> None:
    """When SapBERT returns multiple matches, first becomes first_question,
    the rest sit in ``pending_init_confirmations`` and get rendered one
    at a time on subsequent /turn calls."""
    from claritymed.core.symptoms.init_matcher import MatchResult
    from claritymed.core.symptoms.schemas import InitMatcherConfig

    agent = _StubAgent(n_evidences=3, probs=np.array([0.9, 0.05, 0.05]))
    SERVER_STATE.datasets["testds"] = _loaded_dataset_with_catalog(agent)
    SERVER_STATE.config_loaded = True

    class _MultiStub:
        def match(self, text, catalog):  # noqa: ARG002
            return MatchResult(evidence_idx=None, score=0.0)

        def match_topk(self, text, catalog, *, k=3, min_score=None):  # noqa: ARG002
            # E_a, E_b are binary in the stub — E_c is categorical and
            # would be filtered out by the B-only guard, so this test
            # only queues the two binaries.
            return [
                MatchResult(evidence_idx=0, score=0.90),
                MatchResult(evidence_idx=1, score=0.75),
            ][:k]

    SERVER_STATE.init_matcher_model = _MultiStub()
    SERVER_STATE.init_matcher_cfg = InitMatcherConfig(max_matches=3)

    payload = _start_payload()
    payload["symptom_summary"] = "multi symptom summary"

    response = client.post("/v1/datasets/testds/sessions", json=payload)
    assert response.status_code == 200
    body = response.json()
    sub = SERVER_STATE.sessions[body["session_id"]]

    assert body["first_question"]["header"] == "E_a"  # idx=0
    # One remaining binary match queued (E_c categorical was filtered).
    assert [q["ev_idx"] for q in sub.pending_init_confirmations] == [1]
    # Neither binary match has been written to state yet — the user
    # confirms via /turn before we treat them as positives.
    for ev_i in (0, 1):
        block = int(SERVER_STATE.datasets["testds"].canonical.layout["off"][ev_i])
        assert sub.state[0, block] == 0.0


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


def test_prune_expired_sessions_handles_concurrent_eviction() -> None:
    """prune_expired_sessions uses pop() so a session deleted between
    list() and pop() does not raise KeyError."""
    import time

    from claritymed.servers.symptoms.state import (
        SubSessionState,
        prune_expired_sessions,
    )

    agent = _StubAgent(n_evidences=3, probs=np.array([0.9, 0.05, 0.05]))
    SERVER_STATE.datasets["testds"] = _loaded_dataset(agent=agent)
    SERVER_STATE.config_loaded = True

    # Session TTL defaults to 1800 s — set started_at well in the past.
    past = time.time() - 1801
    for sid in ("s1", "s2"):
        SERVER_STATE.sessions[sid] = SubSessionState(
            session_id=sid,
            dataset_id="testds",
            model_id="m1",
            state=np.zeros((1, 10)),
            turn_count=0,
            started_at=past,
            last_ev_idx=None,
        )

    # Simulate another code path (e.g. a cancel request) evicting s1
    # before the purge loop processes it.
    del SERVER_STATE.sessions["s1"]

    # pop(sid, None) must tolerate the already-gone entry without raising.
    purged = prune_expired_sessions()
    assert purged == 1  # only s2 was actually popped by prune
    assert "s2" not in SERVER_STATE.sessions
    assert "s1" not in SERVER_STATE.sessions


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


# --- INFO logging for spot-check / audit workflows -----------------------


def test_turn_done_log_carries_probs_field(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """The ``session done`` INFO log must include the top-K posterior.

    Operators grep for ``session done: session=<id>`` to spot-check
    Pneumonia / Influenza replays against the offline verifier; without
    the ``probs=`` field, the log tells them a session finished but not
    what the model actually decided.
    """
    agent = _StubAgent(
        n_evidences=3,
        probs=np.array([0.85, 0.1, 0.05]),
        stop_after=1,
    )
    SERVER_STATE.datasets["testds"] = _loaded_dataset(agent=agent)
    SERVER_STATE.config_loaded = True

    start = client.post("/v1/datasets/testds/sessions", json=_start_payload()).json()
    session_id = start["session_id"]
    with caplog.at_level("INFO", logger="claritymed.servers.symptoms"):
        client.post(
            f"/v1/datasets/testds/sessions/{session_id}/turn",
            json={"answer": "Yes", "language": "en"},
        )
    done_lines = [
        r for r in caplog.records if r.getMessage().startswith("session done:")
    ]
    assert done_lines, "expected exactly one 'session done' log line"
    msg = done_lines[0].getMessage()
    # Format: id:prob,id:prob,id:prob — leading class must be the argmax.
    assert "probs=critical_disease:0.850" in msg
    assert "moderate_disease:0.100" in msg
    assert "mild_disease:0.050" in msg


def test_turn_next_evidence_logged_at_info_level(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """Per-turn ``next_evidence`` must land at INFO for spot-check greps.

    Previously logged at DEBUG, which most operator log pipelines drop.
    Upgraded so a full session's asked-question sequence is recoverable
    from the standard log stream without turning on verbose debug.
    """
    agent = _StubAgent(
        n_evidences=3,
        probs=np.array([0.4, 0.3, 0.3]),
        stop_after=10,
    )
    SERVER_STATE.datasets["testds"] = _loaded_dataset(agent=agent, maxstep=8)
    SERVER_STATE.config_loaded = True

    start = client.post("/v1/datasets/testds/sessions", json=_start_payload()).json()
    session_id = start["session_id"]
    with caplog.at_level("INFO", logger="claritymed.servers.symptoms"):
        client.post(
            f"/v1/datasets/testds/sessions/{session_id}/turn",
            json={"answer": "Yes", "language": "en"},
        )
    turn_lines = [
        r
        for r in caplog.records
        if "session turn" in r.getMessage() and r.levelname == "INFO"
    ]
    assert turn_lines, "per-turn next_evidence log must fire at INFO"
    assert "next_evidence=E_" in turn_lines[0].getMessage()


def test_cancel_log_carries_probs_field(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """``session cancelled`` INFO log includes probs + threshold_met + override."""
    agent = _StubAgent(
        n_evidences=3,
        probs=np.array([0.25, 0.35, 0.4]),  # severity-1 has prob 0.25 → override
        stop_after=100,
    )
    SERVER_STATE.datasets["testds"] = _loaded_dataset(agent=agent)
    SERVER_STATE.config_loaded = True
    start = client.post("/v1/datasets/testds/sessions", json=_start_payload()).json()
    session_id = start["session_id"]
    with caplog.at_level("INFO", logger="claritymed.servers.symptoms"):
        client.delete(f"/v1/datasets/testds/sessions/{session_id}")
    cancel_lines = [
        r for r in caplog.records if r.getMessage().startswith("session cancelled:")
    ]
    assert cancel_lines
    msg = cancel_lines[0].getMessage()
    # Argmax is mild_disease (idx 2, prob 0.4) → leads the probs list.
    assert "probs=mild_disease:0.400" in msg
    assert "moderate_disease:0.350" in msg
    assert "critical_disease:0.250" in msg


def test_cap_log_carries_probs_field(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    """``session cap`` INFO log includes probs alongside the confidence field."""
    agent = _StubAgent(
        n_evidences=3,
        probs=np.array([0.5, 0.3, 0.2]),
        stop_after=100,
    )
    SERVER_STATE.datasets["testds"] = _loaded_dataset(agent=agent, maxstep=2)
    SERVER_STATE.config_loaded = True
    start = client.post("/v1/datasets/testds/sessions", json=_start_payload()).json()
    session_id = start["session_id"]
    client.post(
        f"/v1/datasets/testds/sessions/{session_id}/turn",
        json={"answer": "Yes", "language": "en"},
    )
    with caplog.at_level("INFO", logger="claritymed.servers.symptoms"):
        client.post(
            f"/v1/datasets/testds/sessions/{session_id}/turn",
            json={"answer": "Yes", "language": "en"},
        )
    cap_lines = [r for r in caplog.records if r.getMessage().startswith("session cap:")]
    assert cap_lines
    msg = cap_lines[0].getMessage()
    assert "probs=critical_disease:0.500" in msg
    assert "moderate_disease:0.300" in msg
    assert "mild_disease:0.200" in msg


def test_format_probs_topk_uses_target_slugs_for_v3_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """v3 subset-parametric datasets render the trailing bucket as ``other``.

    ``target_condition_ids`` on the spec means the classifier emits
    ``N+1`` classes: the first N are targets by slug order, the last is
    a synthetic Other bucket. The log formatter must use the slug list
    (not ``condition_by_idx``, which would resolve to a real disease at
    that pidx position and mislabel Other as e.g. ``pneumonia``).
    """
    ds = _loaded_dataset(
        agent=_StubAgent(n_evidences=3, probs=np.array([0.9, 0.05, 0.05]))
    )
    # Retrofit the spec with a v3 target list — mirrors what the
    # ddxplus_pneumonia_flu dataset config declares in configs/symptoms.yaml.
    monkeypatch.setattr(
        ds.spec.__class__,
        "target_condition_ids",
        ("critical_disease", "moderate_disease"),
        raising=False,
    )
    object.__setattr__(
        ds.spec, "target_condition_ids", ("critical_disease", "moderate_disease")
    )
    probs = np.array([0.05, 0.10, 0.85])  # Other dominates
    formatted = app_mod._format_probs_topk(ds, probs)
    # Argmax is the trailing Other bucket at index 2.
    assert formatted.startswith("other:0.850")
    assert "moderate_disease:0.100" in formatted
    assert "critical_disease:0.050" in formatted
