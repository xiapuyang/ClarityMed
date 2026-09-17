"""Smoke tests for the XGBoost training pipeline.

Exercises the pure functions (:func:`train_xgb_agent`,
:func:`build_mask_pairs`, :func:`_write_manifest`) on synthetic data
without shelling out to the CLI or touching the DDXPlus corpus. The
real CLI-level e2e is covered by ``tests/e2e/`` when real data is
available.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("xgboost")


from claritymed.ingest.symptoms.typed_basd import (
    TypedEnv,
    build_layout,
    interactive_eval,
)
from claritymed.ingest.symptoms.xgb.algorithm import XgbAgent
from claritymed.ingest.symptoms.xgb.encoding import feature_columns_from_schema
from claritymed.ingest.symptoms.xgb.train import (
    _feature_importance_top_k,
    _write_manifest,
    build_mask_pairs,
    train_xgb_agent,
)


def _tiny_schema() -> dict:
    """Four binary evidences — two carry class signal, two are noise."""
    evs = [
        {"name": "E_disc_a", "dtype": "B", "values": []},
        {"name": "E_disc_b", "dtype": "B", "values": []},
        {"name": "E_noise_1", "dtype": "B", "values": []},
        {"name": "E_noise_2", "dtype": "B", "values": []},
    ]
    return build_layout(evs)


def _tiny_patients(n: int, seed: int) -> tuple[list[dict], np.ndarray]:
    """Class 0 patients tend to have E_disc_a; class 1 tend to have E_disc_b."""
    rng = np.random.default_rng(seed)
    patients: list[dict] = []
    y: list[int] = []
    for i in range(n):
        cls = i % 2
        has_disc_a = rng.random() < (0.9 if cls == 0 else 0.1)
        has_disc_b = rng.random() < (0.1 if cls == 0 else 0.9)
        has_noise_1 = rng.random() < 0.5
        has_noise_2 = rng.random() < 0.5
        bin_pos: set[int] = set()
        if has_disc_a:
            bin_pos.add(0)
        if has_disc_b:
            bin_pos.add(1)
        if has_noise_1:
            bin_pos.add(2)
        if has_noise_2:
            bin_pos.add(3)
        init = next(iter(bin_pos)) if bin_pos else 0
        patients.append(
            dict(
                bin_pos=bin_pos,
                cat_val={},
                multi_val={},
                pos=bin_pos.copy(),
                init=init,
                d=cls,
                age=3,
                sex=0,
                diff=np.array([1.0 - cls, float(cls)]),
            )
        )
        y.append(cls)
    return patients, np.asarray(y)


def test_build_mask_pairs_zeros_a_fraction_of_set_columns():
    x = np.zeros((4, 8), dtype=np.float32)
    # Row 0 has all 8 cols set; the others have 4 set.
    x[0] = 1.0
    x[1, :4] = 1.0
    x[2, 2:6] = 1.0
    x[3, 4:] = 1.0
    rng = np.random.default_rng(0)
    masked, true = build_mask_pairs(x, keep_lo=0.5, keep_hi=0.5, rng=rng)
    assert np.array_equal(true, x)  # x_true unchanged
    for i in range(4):
        set_count = int(x[i].sum())
        if set_count == 0:
            continue
        kept = int(masked[i].sum())
        assert 1 <= kept <= set_count  # at least 1 kept, at most all


def test_build_mask_pairs_keep_1_leaves_all_set():
    x = np.ones((3, 5), dtype=np.float32)
    rng = np.random.default_rng(0)
    masked, _ = build_mask_pairs(x, keep_lo=1.0, keep_hi=1.0, rng=rng)
    assert np.all(masked == x)


def test_train_xgb_agent_produces_working_agent():
    schema = _tiny_schema()
    columns, _labels, columns_idx = feature_columns_from_schema(schema)
    train_pats, _ = _tiny_patients(200, seed=1)
    val_pats, _ = _tiny_patients(60, seed=2)
    agent = train_xgb_agent(
        schema=schema,
        train_patients=train_pats,
        val_patients=val_pats,
        columns=columns,
        columns_idx=columns_idx,
        n_classes=2,
        max_depth=3,
        n_estimators=30,
        lr=0.3,
        calibration="platt",
        mask_policy="random",
        keep_lo=0.3,
        keep_hi=0.7,
        stop_thres=0.9,
        ig_smoothing=0.05,
        seed=0,
    )
    assert isinstance(agent, XgbAgent)
    assert agent.ev_marginals is not None
    assert agent.classifier is not None
    # The agent should classify test patients reasonably well.
    test_pats, y_test = _tiny_patients(50, seed=3)
    env = TypedEnv(list(test_pats), schema, n_dis=2)
    s, _ = env.initialize_state(50)
    argmax, _ = agent.diagnose(s)
    # 200 training patients with clear signal → we expect >70% accuracy.
    assert (argmax == y_test).mean() > 0.7


def test_train_xgb_agent_with_full_mask_policy_skips_ev_marginals():
    schema = _tiny_schema()
    columns, _labels, columns_idx = feature_columns_from_schema(schema)
    train_pats, _ = _tiny_patients(100, seed=1)
    agent = train_xgb_agent(
        schema=schema,
        train_patients=train_pats,
        val_patients=[],
        columns=columns,
        columns_idx=columns_idx,
        n_classes=2,
        max_depth=3,
        n_estimators=20,
        lr=0.3,
        calibration="none",
        mask_policy="full",
        keep_lo=0.3,
        keep_hi=0.7,
        stop_thres=0.9,
        ig_smoothing=0.05,
        seed=0,
    )
    assert agent.ev_marginals is None
    # Agent should still be usable — falls back to global_marginals.
    assert agent.global_marginals is not None


def _ddxplus_like_schema(n_evs: int = 30) -> dict:
    """N binary evidences — 6 discriminative, rest weak/noise. Approximates
    the DDXPlus subset regime where most evidences carry little signal."""
    from claritymed.ingest.symptoms.typed_basd import build_layout

    evs = [{"name": f"E_{i:03d}", "dtype": "B", "values": []} for i in range(n_evs)]
    return build_layout(evs)


def _ddxplus_like_patients(
    n: int, seed: int, n_evs: int = 30
) -> tuple[list[dict], np.ndarray]:
    """Soft-signal 2-class corpus: 6 discriminative features + noise.

    Class prior on each discriminative feature: ~60% for its own class,
    ~40% for the other. No single feature carries enough signal for a
    calibrated classifier to hit 90% confidence on its own — you need
    multiple features answered before the posterior firms up. Matches
    the DDXPlus Pneumonia+Flu shared-symptom regime.
    """
    rng = np.random.default_rng(seed)
    patients: list[dict] = []
    y: list[int] = []
    for i in range(n):
        cls = i % 2
        bin_pos: set[int] = set()
        for j in range(n_evs):
            if j < 3:
                p_on = 0.60 if cls == 0 else 0.40  # class-0 leaning
            elif j < 6:
                p_on = 0.40 if cls == 0 else 0.60  # class-1 leaning
            else:
                p_on = 0.30  # class-independent noise
            if rng.random() < p_on:
                bin_pos.add(j)
        init = next(iter(bin_pos)) if bin_pos else 0
        patients.append(
            dict(
                bin_pos=bin_pos,
                cat_val={},
                multi_val={},
                pos=bin_pos.copy(),
                init=init,
                d=cls,
                age=3,
                sex=0,
                diff=np.array([1.0 - cls, float(cls)]),
            )
        )
        y.append(cls)
    return patients, np.asarray(y)


def test_random_mask_policy_prevents_premature_stop():
    """Regression guard: with mask_policy=random, the classifier must NOT
    fire should_stop at the initial state (single init evidence + zeros).

    Repro of the 2026-07-08 IL=0 failure: training on full signatures only
    made the classifier confidently classify near-empty states, so
    interactive_eval terminated every game before asking anything. With
    multi-band masked training, initial-state posteriors sit below the
    default stop threshold so IL > 0.
    """
    schema = _ddxplus_like_schema()
    columns, _labels, columns_idx = feature_columns_from_schema(schema)
    train_pats, _ = _ddxplus_like_patients(600, seed=1)
    val_pats, _ = _ddxplus_like_patients(150, seed=2)
    agent = train_xgb_agent(
        schema=schema,
        train_patients=train_pats,
        val_patients=val_pats,
        columns=columns,
        columns_idx=columns_idx,
        n_classes=2,
        max_depth=4,
        n_estimators=80,
        lr=0.1,
        calibration="platt",
        mask_policy="random",
        keep_lo=0.3,
        keep_hi=0.7,
        stop_thres=0.9,
        ig_smoothing=0.05,
        seed=0,
    )
    from claritymed.ingest.symptoms.typed_basd import TypedEnv

    test_pats, _ = _ddxplus_like_patients(50, seed=99)
    env = TypedEnv(list(test_pats), schema, n_dis=2)
    s, _ = env.initialize_state(50)
    stop = agent.should_stop(s)
    # Most rows should NOT stop at turn 0 — otherwise interactive_eval
    # sees IL=0. Target: <50% early-stop rate; higher indicates the
    # classifier is over-confident on init-only state (D4 regression).
    assert stop.mean() < 0.5, (
        f"should_stop fired on {stop.mean() * 100:.0f}% of initial states — "
        f"classifier is over-confident on partial input (D4 regression)."
    )


def test_train_xgb_agent_integrates_with_interactive_eval():
    """The agent must plug directly into typed-BASD's interactive_eval loop."""
    schema = _tiny_schema()
    columns, _labels, columns_idx = feature_columns_from_schema(schema)
    train_pats, _ = _tiny_patients(200, seed=1)
    val_pats, _ = _tiny_patients(40, seed=2)
    test_pats, _ = _tiny_patients(40, seed=3)
    agent = train_xgb_agent(
        schema=schema,
        train_patients=train_pats,
        val_patients=val_pats,
        columns=columns,
        columns_idx=columns_idx,
        n_classes=2,
        max_depth=3,
        n_estimators=30,
        lr=0.3,
        calibration="platt",
        mask_policy="random",
        keep_lo=0.3,
        keep_hi=0.7,
        stop_thres=0.9,
        ig_smoothing=0.05,
        seed=0,
    )
    # Severity vector — both classes non-severe (>=3) so DSR path skips.
    severity = np.array([3.0, 3.0])
    metrics = interactive_eval(
        TypedEnv(list(test_pats), schema, n_dis=2),
        agent,
        maxstep=4,
        games=40,
        severity=severity,
    )
    # We don't assert accuracy floors; the point is that the loop
    # completes without raising and returns a well-formed metric block.
    assert 0.0 <= metrics.ACC <= 100.0
    assert metrics.IL >= 0.0


def test_feature_importance_top_k_returns_ordered_pairs():
    schema = _tiny_schema()
    columns, _labels, columns_idx = feature_columns_from_schema(schema)
    train_pats, _ = _tiny_patients(200, seed=1)
    agent = train_xgb_agent(
        schema=schema,
        train_patients=train_pats,
        val_patients=[],
        columns=columns,
        columns_idx=columns_idx,
        n_classes=2,
        max_depth=3,
        n_estimators=30,
        lr=0.3,
        calibration="none",
        mask_policy="full",
        keep_lo=0.3,
        keep_hi=0.7,
        stop_thres=0.9,
        ig_smoothing=0.05,
        seed=0,
    )
    top = _feature_importance_top_k(agent.classifier, columns, k=3)
    assert len(top) <= 3
    # Every entry is a name → gain float pair.
    for entry in top:
        assert entry["name"] in columns
        assert isinstance(entry["gain"], float)
    # Sorted descending by gain.
    gains = [e["gain"] for e in top]
    assert gains == sorted(gains, reverse=True)


def test_write_manifest_emits_all_expected_fields(tmp_path: Path):
    schema = _tiny_schema()
    columns, _labels, columns_idx = feature_columns_from_schema(schema)
    train_pats, _ = _tiny_patients(80, seed=1)
    agent = train_xgb_agent(
        schema=schema,
        train_patients=train_pats,
        val_patients=[],
        columns=columns,
        columns_idx=columns_idx,
        n_classes=2,
        max_depth=3,
        n_estimators=20,
        lr=0.3,
        calibration="none",
        mask_policy="full",
        keep_lo=0.3,
        keep_hi=0.7,
        stop_thres=0.9,
        ig_smoothing=0.05,
        seed=0,
    )
    weights_path = tmp_path / "weights.pkl"
    agent.save(weights_path)
    metrics = interactive_eval(
        TypedEnv(list(train_pats), schema, n_dis=2),
        agent,
        maxstep=3,
        games=len(train_pats),
        severity=np.array([3.0, 3.0]),
    )
    manifest_path = _write_manifest(
        tmp_path,
        model_id="xgb_test",
        weights_sha="abc" * 21 + "d",  # 64 chars for schema
        train_params={"seed": 0},
        diseases_trained=["ClassA", "ClassB"],
        columns=columns,
        feature_importance_top_k=[{"name": "E_disc_a", "gain": 12.5}],
        metrics=metrics,
        maxstep=3,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["algorithm_module"] == "xgb"
    assert manifest["algorithm_module_version"] == 1
    assert manifest["training_target"] == "pathology"
    assert manifest["feature_columns"] == columns
    assert manifest["diseases_trained"] == ["ClassA", "ClassB"]
    assert "eval" in manifest
    assert manifest["eval"]["maxstep"] == 3
    assert "IL" in manifest["eval"]
    assert manifest["feature_importance_top_k"][0]["name"] == "E_disc_a"
