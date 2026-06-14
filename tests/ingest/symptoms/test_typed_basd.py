"""Typed-BASD algorithm-body tests.

Mirrors the demo's pure-numpy ``test_env`` self-test and adds coverage
for ``build_layout`` edge cases, ``build_basd`` input validation, and a
small smoke run of ``interactive_eval`` against synthetic data.

The torch-using paths (``build_basd``, ``Agent.diagnose``,
``interactive_eval``) skip on hosts without torch — the algorithm module
is importable without torch so non-NN harnesses can exercise
``TypedEnv`` standalone.
"""

from __future__ import annotations

import numpy as np
import pytest

from claritymed.ingest.symptoms.typed_basd import (
    AGE_BUCKETS,
    EvalMetrics,
    TypedEnv,
    age_bucket,
    build_basd,
    build_layout,
    interactive_eval,
    parse_list,
    seed_everything,
)

# Shared fixture: 4-evidence schema (binary / nominal-cat / multi / numeric-cat).
_FIXTURE_EVS = [
    {"name": "E0", "dtype": "B", "values": []},
    {"name": "E1", "dtype": "C", "values": ["a", "b", "c"]},
    {"name": "E2", "dtype": "M", "values": ["x", "y"]},
    {"name": "E3", "dtype": "C", "values": ["0", "5", "10"]},
]


def _patient() -> dict:
    """Fixture patient — full positive evidences across all 4 types."""
    return dict(
        bin_pos={0},
        cat_val={1: 1, 3: 1},
        multi_val={2: [0, 1]},
        pos={0, 1, 2, 3},
        init=0,
        age=3,
        sex=0,
        d=0,
        diff=np.array([0.7, 0.2, 0.1]),
    )


# --- build_layout --------------------------------------------------------


def test_build_layout_without_ordinal_packs_numeric_as_onehot() -> None:
    """Preserves the demo's ``# default: E3 numeric is plain one-hot`` assert."""
    schema = build_layout(_FIXTURE_EVS)
    assert schema["sym_size"] == 12, schema["sym_size"]
    assert schema["has_ord"] == [False, False, False, False]


def test_build_layout_with_ordinal_adds_scalar_slot() -> None:
    """Preserves the demo's ``--ordinal sym_size 13`` + offset asserts."""
    schema = build_layout(_FIXTURE_EVS, use_ordinal=True)
    assert schema["sym_size"] == 13, schema["sym_size"]
    assert list(schema["off"]) == [0, 1, 5, 8]
    assert schema["has_ord"] == [False, False, False, True]
    # E3's ordinal scalar is parked at offset 8 + 1 (asked) + 3 (one-hot) = 12.
    assert schema["ord_at"][3] == 12


def test_build_layout_numeric_cat_with_string_values_not_treated_as_ordinal() -> None:
    """Non-numeric strings keep the plain one-hot path."""
    schema = build_layout(
        [{"name": "E0", "dtype": "C", "values": ["mild", "severe"]}],
        use_ordinal=True,
    )
    assert schema["has_ord"] == [False]


def test_build_layout_handles_empty_evidence_list() -> None:
    schema = build_layout([])
    assert schema["sym_size"] == 0
    assert schema["n_ev"] == 0


# --- TypedEnv ------------------------------------------------------------


def test_typed_env_initialize_state_writes_binary_positive() -> None:
    schema = build_layout(_FIXTURE_EVS, use_ordinal=True)
    p = _patient()
    env = TypedEnv([p, p], schema, n_dis=3)
    env.reset()
    env.order = np.array([0, 1])
    s, _ = env.initialize_state(2)
    assert s[0, 0] == 1.0, "binary positive should encode +1"


def test_typed_env_reveal_writes_nominal_onehot() -> None:
    """E1=b → asked@1 + one-hot at 1+1+1=3 (b is local index 1)."""
    schema = build_layout(_FIXTURE_EVS, use_ordinal=True)
    p = _patient()
    env = TypedEnv([p, p], schema, n_dis=3)
    env.reset()
    env.order = np.array([0, 1])
    s, _ = env.initialize_state(2)
    s2 = env.reveal(s, np.array([1, 1]), np.array([False, False]))
    assert s2[0, 1] == 1.0
    assert s2[0, 3] == 1.0
    assert s2[0, 2] == 0.0
    assert s2[0, 4] == 0.0


def test_typed_env_reveal_writes_multi_hot() -> None:
    schema = build_layout(_FIXTURE_EVS, use_ordinal=True)
    p = _patient()
    env = TypedEnv([p, p], schema, n_dis=3)
    env.reset()
    env.order = np.array([0, 1])
    s, _ = env.initialize_state(2)
    s2 = env.reveal(s, np.array([1, 1]), np.array([False, False]))
    s3 = env.reveal(s2, np.array([2, 2]), np.array([False, False]))
    # E2 starts at offset 5; asked@5 + multi-hot at 6, 7
    assert s3[0, 5] == 1.0
    assert s3[0, 6] == 1.0
    assert s3[0, 7] == 1.0


def test_typed_env_reveal_writes_ordinal_scalar() -> None:
    schema = build_layout(_FIXTURE_EVS, use_ordinal=True)
    p = _patient()
    env = TypedEnv([p, p], schema, n_dis=3)
    env.reset()
    env.order = np.array([0, 1])
    s, _ = env.initialize_state(2)
    s_after = env.reveal(s, np.array([3, 3]), np.array([False, False]))
    # E3 value "5" at one-hot local idx 1 → slot 8+1+1=10; ordinal scalar at 12.
    assert s_after[0, 8] == 1.0
    assert s_after[0, 10] == 1.0
    assert abs(s_after[0, 12] - 0.5) < 1e-9


def test_typed_env_reveal_skips_done_rows() -> None:
    schema = build_layout(_FIXTURE_EVS, use_ordinal=True)
    p = _patient()
    env = TypedEnv([p, p], schema, n_dis=3)
    env.reset()
    env.order = np.array([0, 1])
    s, _ = env.initialize_state(2)
    s_after = env.reveal(s, np.array([1, 1]), np.array([False, True]))
    # Row 0 advances; row 1 stays at initialize_state output.
    assert s_after[0, 1] == 1.0
    assert s_after[1, 1] == 0.0


def test_typed_env_asked_mask_after_full_reveal() -> None:
    schema = build_layout(_FIXTURE_EVS, use_ordinal=True)
    p = _patient()
    env = TypedEnv([p, p], schema, n_dis=3)
    env.reset()
    env.order = np.array([0, 1])
    s, _ = env.initialize_state(2)
    s2 = env.reveal(s, np.array([1, 1]), np.array([False, False]))
    s3 = env.reveal(s2, np.array([2, 2]), np.array([False, False]))
    s4 = env.reveal(s3, np.array([3, 3]), np.array([False, False]))
    assert env.asked_mask(s4)[0].all()


def test_typed_env_reset_is_deterministic_under_seed() -> None:
    """Seeded reset → identical patient order across two fresh envs."""
    schema = build_layout(_FIXTURE_EVS)
    pats = [_patient() for _ in range(10)]
    seed_everything(42)
    env_a = TypedEnv(list(pats), schema, n_dis=3)
    env_a.reset()
    seed_everything(42)
    env_b = TypedEnv(list(pats), schema, n_dis=3)
    env_b.reset()
    assert np.array_equal(env_a.order, env_b.order)


# --- helpers ------------------------------------------------------------


def test_age_bucket_boundaries() -> None:
    assert age_bucket(0) == 0
    assert age_bucket(1) == 1
    assert age_bucket(4) == 1
    assert age_bucket(5) == 2
    assert age_bucket(35) == 4
    assert age_bucket(200) == len(AGE_BUCKETS) - 1


def test_age_bucket_above_max_clamps_to_last_bucket() -> None:
    assert age_bucket(99999) == len(AGE_BUCKETS) - 1


def test_parse_list_handles_string_input() -> None:
    assert parse_list("[1, 2, 3]") == [1, 2, 3]


def test_parse_list_handles_passthrough_list() -> None:
    assert parse_list([1, 2]) == [1, 2]


def test_parse_list_none_returns_empty() -> None:
    """Falsy non-string → ``[]``. Empty string isn't supported (matches demo)."""
    assert parse_list(None) == []
    assert parse_list(0) == []
    assert parse_list([]) == []


def test_seed_everything_produces_deterministic_np_random() -> None:
    seed_everything(7)
    a = np.random.rand(5)
    seed_everything(7)
    b = np.random.rand(5)
    assert np.allclose(a, b)


# --- build_basd validation ---------------------------------------------


def test_build_basd_rejects_zero_hidden() -> None:
    schema = build_layout(_FIXTURE_EVS)
    env = TypedEnv([_patient()], schema, n_dis=3)
    with pytest.raises(ValueError, match="hidden"):
        build_basd(env, n_dis=3, hidden=0, lr=1e-3, device="cpu", stop_thres=0.1)


def test_build_basd_rejects_negative_lr() -> None:
    schema = build_layout(_FIXTURE_EVS)
    env = TypedEnv([_patient()], schema, n_dis=3)
    with pytest.raises(ValueError, match="lr"):
        build_basd(env, n_dis=3, hidden=16, lr=-1e-3, device="cpu", stop_thres=0.1)


def test_build_basd_rejects_unknown_stop_mode() -> None:
    schema = build_layout(_FIXTURE_EVS)
    env = TypedEnv([_patient()], schema, n_dis=3)
    with pytest.raises(ValueError, match="stop_mode"):
        build_basd(
            env,
            n_dis=3,
            hidden=16,
            lr=1e-3,
            device="cpu",
            stop_thres=0.1,
            stop_mode="bogus",
        )


# --- torch-path smoke (skip if torch missing) ---------------------------

torch = pytest.importorskip("torch")


def test_build_basd_returns_agent_with_required_methods() -> None:
    schema = build_layout(_FIXTURE_EVS)
    env = TypedEnv([_patient() for _ in range(4)], schema, n_dis=3)
    seed_everything(1)
    agent = build_basd(env, n_dis=3, hidden=8, lr=1e-3, device="cpu", stop_thres=0.1)
    for name in ("next_action", "should_stop", "diagnose", "train_step"):
        assert hasattr(agent, name)


def test_interactive_eval_returns_eval_metrics_with_finite_values() -> None:
    """Smoke: tiny synthetic env round-trips through interactive_eval."""
    schema = build_layout(_FIXTURE_EVS)
    pats = [_patient() for _ in range(20)]
    env = TypedEnv(pats, schema, n_dis=3)
    seed_everything(1)
    agent = build_basd(
        env,
        n_dis=3,
        hidden=8,
        lr=1e-3,
        device="cpu",
        stop_thres=0.1,
        stop_mode="heuristic",
    )
    severity = np.array([1.0, 3.0, 5.0])
    metrics = interactive_eval(env, agent, maxstep=4, games=5, severity=severity)
    assert isinstance(metrics, EvalMetrics)
    for field in ("IL", "ACC", "GTPA", "DDR", "DDP", "DDF1"):
        val = getattr(metrics, field)
        assert np.isfinite(val), f"{field}={val!r} not finite"
    # DSR may be NaN when no severe cases hit, but n_severe must reflect.
    if metrics.n_severe == 0:
        assert np.isnan(metrics.DSR)
    else:
        assert np.isfinite(metrics.DSR)
    # Fixture declares no antecedents, so antecedent-side metrics must be
    # NaN (no patient contributed) while symptom-side stays finite.
    for field in ("PSR", "PSP", "PSF1"):
        assert np.isfinite(getattr(metrics, field)), field
    for field in ("PAR", "PAP", "PAF1"):
        assert np.isnan(getattr(metrics, field)), field


def test_build_layout_propagates_is_antecedent_flag() -> None:
    """``is_antecedent`` defaults to False and round-trips when set."""
    schema = build_layout(_FIXTURE_EVS)
    assert schema["is_antecedent"].dtype == bool
    assert schema["is_antecedent"].shape == (4,)
    assert not schema["is_antecedent"].any()
    mixed = [
        {**_FIXTURE_EVS[0], "is_antecedent": False},
        {**_FIXTURE_EVS[1], "is_antecedent": True},
        _FIXTURE_EVS[2],
        _FIXTURE_EVS[3],
    ]
    schema2 = build_layout(mixed)
    assert schema2["is_antecedent"].tolist() == [False, True, False, False]


def test_interactive_eval_splits_symptom_and_antecedent_metrics() -> None:
    """One antecedent + one symptom → both PSR and PAR are finite."""
    # E0 is now an antecedent, the rest stay symptoms. Patient still has
    # all four evidences positive so both gt_sym and gt_atcd are non-empty.
    evs = [
        {**_FIXTURE_EVS[0], "is_antecedent": True},
        _FIXTURE_EVS[1],
        _FIXTURE_EVS[2],
        _FIXTURE_EVS[3],
    ]
    schema = build_layout(evs)
    pats = [_patient() for _ in range(20)]
    env = TypedEnv(pats, schema, n_dis=3)
    seed_everything(2)
    agent = build_basd(
        env,
        n_dis=3,
        hidden=8,
        lr=1e-3,
        device="cpu",
        stop_thres=0.1,
        stop_mode="heuristic",
    )
    severity = np.array([1.0, 3.0, 5.0])
    metrics = interactive_eval(env, agent, maxstep=4, games=5, severity=severity)
    for field in ("PSR", "PSP", "PSF1", "PAR", "PAP", "PAF1"):
        val = getattr(metrics, field)
        assert np.isfinite(val), f"{field}={val!r} not finite"
        assert 0.0 <= val <= 100.0, f"{field}={val!r} out of [0, 100]"
