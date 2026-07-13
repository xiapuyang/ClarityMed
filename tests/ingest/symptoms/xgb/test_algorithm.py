"""End-to-end tests for :class:`XgbAgent` — train a mini XGBoost on a
synthetic 2-class 2-evidence problem, verify the agent's methods behave.

Kept off the DDXPlus data path: the fixtures below build a schema, fake
patients, and a tiny XGBoost classifier from scratch so this suite
runs in milliseconds without any dataset downloaded.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from claritymed.ingest.symptoms.typed_basd import TypedEnv, build_layout
from claritymed.ingest.symptoms.xgb.algorithm import XgbAgent, build_xgb_agent
from claritymed.ingest.symptoms.xgb.encoding import (
    encode_patient_batch,
    feature_columns_from_schema,
)


def _mini_schema() -> dict:
    """Two binary evidences: E_A (positive → class 1), E_B (noise)."""
    evs = [
        {"name": "E_A", "dtype": "B", "values": []},
        {"name": "E_B", "dtype": "B", "values": []},
    ]
    return build_layout(evs)


def _mini_patients(n: int = 200) -> tuple[list[dict], np.ndarray]:
    """Half positive-E_A (class 1), half not (class 0). E_B is 50/50 noise."""
    rng = np.random.default_rng(42)
    patients: list[dict] = []
    y: list[int] = []
    for i in range(n):
        cls = i % 2
        has_a = cls == 1
        has_b = rng.random() < 0.5
        bin_pos: set[int] = set()
        if has_a:
            bin_pos.add(0)
        if has_b:
            bin_pos.add(1)
        patients.append(
            dict(
                bin_pos=bin_pos,
                cat_val={},
                multi_val={},
                pos=bin_pos.copy(),
                init=0 if has_a else 1,
                d=cls,
                age=3,
                sex=0,
                diff=np.array([1.0 - cls, float(cls)]),
            )
        )
        y.append(cls)
    return patients, np.asarray(y)


def _train_mini_agent() -> XgbAgent:
    """Fit an XGBoost classifier on the mini corpus, return a live agent."""
    import xgboost as xgb

    schema = _mini_schema()
    columns, _labels, columns_idx = feature_columns_from_schema(schema)
    patients, y = _mini_patients(400)
    x = encode_patient_batch(patients, schema, columns_idx)
    clf = xgb.XGBClassifier(
        n_estimators=20,
        max_depth=3,
        learning_rate=0.3,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=1,
        tree_method="hist",
    )
    clf.fit(x, y)
    global_marginals = x.mean(axis=0)
    return build_xgb_agent(
        classifier=clf,
        ev_marginals=None,
        schema=schema,
        columns=columns,
        columns_idx=columns_idx,
        thres=0.9,
        global_marginals=global_marginals,
    )


def test_agent_diagnose_returns_valid_probabilities():
    agent = _train_mini_agent()
    env = TypedEnv(_mini_patients(4)[0], _mini_schema(), n_dis=2)
    s, _ = env.initialize_state(4)
    argmax, probs = agent.diagnose(s)
    assert argmax.shape == (4,)
    assert probs.shape == (4, 2)
    # Rows sum to 1 (approx) and lie in [0, 1].
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_agent_next_action_returns_legal_unasked_evidence():
    agent = _train_mini_agent()
    # First patient (i=0) is class 0, has no E_A → init is E_B (idx 1).
    patients = _mini_patients(1)[0]
    env = TypedEnv(patients, _mini_schema(), n_dis=2)
    s, _ = env.initialize_state(1)
    a = agent.next_action(s)
    assert a.shape == (1,)
    # E_B is the init/asked evidence; next_action must pick the unasked
    # E_A (idx 0) — never re-picking an already-asked evidence.
    assert a[0] == 0


def test_agent_should_stop_is_bool_batch():
    agent = _train_mini_agent()
    env = TypedEnv(_mini_patients(4)[0], _mini_schema(), n_dis=2)
    s, _ = env.initialize_state(4)
    stop = agent.should_stop(s)
    assert stop.shape == (4,)
    assert stop.dtype == bool


def test_agent_should_stop_respects_threshold():
    """With thres=1.01 (impossible), should_stop must never fire."""
    agent = _train_mini_agent()
    agent.thres = 1.01
    env = TypedEnv(_mini_patients(4)[0], _mini_schema(), n_dis=2)
    s, _ = env.initialize_state(4)
    stop = agent.should_stop(s)
    assert not stop.any()


def test_agent_save_load_roundtrips():
    agent = _train_mini_agent()
    schema = _mini_schema()
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "weights.pkl"
        agent.save(path)
        reloaded = XgbAgent.load(path, schema)
    # Predictions should match on the same input.
    env = TypedEnv(_mini_patients(4)[0], schema, n_dis=2)
    s, _ = env.initialize_state(4)
    _, p_orig = agent.diagnose(s)
    _, p_reloaded = reloaded.diagnose(s)
    assert np.allclose(p_orig, p_reloaded, atol=1e-6)
    assert reloaded.thres == agent.thres
    assert reloaded.temp == agent.temp
    assert reloaded.mode == agent.mode


def test_agent_next_action_batch_of_1d_state_unwraps():
    """Callers passing a 1-D state (single row) should get a length-1 result."""
    agent = _train_mini_agent()
    env = TypedEnv(_mini_patients(1)[0], _mini_schema(), n_dis=2)
    s, _ = env.initialize_state(1)
    a = agent.next_action(s[0])
    assert a.shape == (1,)


def test_agent_falls_back_to_evidence_zero_when_all_asked():
    """Interactive-eval invariant: next_action never returns -1."""
    agent = _train_mini_agent()
    # Fabricate a state where every block-start slot is set (all asked).
    schema = _mini_schema()
    s = np.zeros((1, schema["sym_size"]))
    for ev_i in range(schema["n_ev"]):
        s[0, schema["off"][ev_i]] = 1.0
    a = agent.next_action(s)
    assert a[0] == 0  # sentinel; caller's stop-gate should fire this turn


@pytest.mark.parametrize("thres", [0.5, 0.7, 0.9])
def test_should_stop_monotonic_in_threshold(thres: float):
    """Higher threshold → fewer stops on the same states."""
    agent = _train_mini_agent()
    env = TypedEnv(_mini_patients(16)[0], _mini_schema(), n_dis=2)
    s, _ = env.initialize_state(16)
    agent.thres = thres
    n_stops = agent.should_stop(s).sum()
    agent.thres = 0.99
    assert agent.should_stop(s).sum() <= n_stops
