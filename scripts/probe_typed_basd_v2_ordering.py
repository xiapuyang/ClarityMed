"""Probe typed_basd_v2 question ordering.

Answers three questions empirically by driving the loaded model directly
(no HTTP layer, no init-matcher):

  1. Is Q1 fixed given the same (age, sex)? — determinism check.
  2. Does the answer to Q_k influence Q_{k+1}? — state-dependency check.
  3. Does Q1 change when age / sex change? — profile-dependency check.

Run with:
    uv run python scripts/probe_typed_basd_v2_ordering.py
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from claritymed.config import load_symptoms_config
from claritymed.core.device import resolve_device
from claritymed.core.symptoms.datasets import build_dataset
from claritymed.ingest.symptoms import ddxplus as _register_adapters  # noqa: F401
from claritymed.servers.symptoms.app import _initial_state, _writer_env
from claritymed.servers.symptoms.questions import build_question, synth_patient


def _load(dataset_id: str) -> Any:
    """Load the first enabled dataset that matches ``dataset_id``."""
    cfg = load_symptoms_config()
    ds_spec = next(d for d in cfg.datasets if d.id == dataset_id and d.enabled)
    device = resolve_device("auto")
    return build_dataset(ds_spec, cfg.models, device=device, init_matcher=None)


def _next_ev_idx(ds: Any, state: np.ndarray) -> int:
    return int(ds.select_model().agent.next_action(state)[0])


def _apply(
    ds: Any, state: np.ndarray, ev_idx: int, answer: Any, answer_value: Any = None
) -> None:
    """Encode an answer into ``state`` in place."""
    synth = synth_patient(
        ds.canonical,
        ds.spec,
        ev_idx,
        answer,
        answer_value=answer_value,
        language="en",
    )
    _writer_env(ds)._write(state[0], ev_idx, synth)


def _describe(ds: Any, ev_idx: int) -> str:
    ev = ds.canonical.evidence_by_idx(ev_idx)
    q = build_question(ds.canonical, ds.spec, ev_idx, language="en")
    n_opts = len(q.options)
    dtype = ev.dtype
    return f"[{ev.id} dtype={dtype} opts={n_opts}] {q.question!r}"


def _first_answer(ds: Any, ev_idx: int) -> tuple[str, str | None]:
    """Return a valid (answer, answer_value) for the first option.

    B evidences take yes/no; C/M take the first raw value.
    """
    ev = ds.canonical.evidence_by_idx(ev_idx)
    if ev.dtype == "B":
        return "yes", None
    q = build_question(ds.canonical, ds.spec, ev_idx, language="en")
    if not q.options:  # numeric fallback path
        return "1", None
    opt = q.options[0]
    return opt.label, opt.value


# ---------------------------------------------------------------------------
# Probe 1: determinism given fixed profile — run twice, expect identical Q1.
# ---------------------------------------------------------------------------


def probe_determinism(ds: Any, age: int, sex: str) -> None:
    print("\n=== Probe 1: determinism (same profile → same Q1) ===")
    q1_runs = []
    for run in range(2):
        state = _initial_state(ds, age, sex)
        q1_runs.append(_next_ev_idx(ds, state))
        print(f"  run {run + 1}: Q1 = {_describe(ds, q1_runs[-1])}")
    assert q1_runs[0] == q1_runs[1], "determinism broken"
    print("  → identical, deterministic ✓")


# ---------------------------------------------------------------------------
# Probe 2: state dependency — vary Q1 answer, watch Q2 diverge.
# ---------------------------------------------------------------------------


def probe_state_dependency(ds: Any, age: int, sex: str) -> None:
    print("\n=== Probe 2: does Q1 answer influence Q2? ===")
    state0 = _initial_state(ds, age, sex)
    q1_idx = _next_ev_idx(ds, state0)
    q1_ev = ds.canonical.evidence_by_idx(q1_idx)
    print(f"  Q1 = {_describe(ds, q1_idx)}")

    branches: dict[str, int] = {}

    if q1_ev.dtype == "B":
        answers = [("yes", None), ("no", None)]
    else:
        q1 = build_question(ds.canonical, ds.spec, q1_idx, language="en")
        # For multi-select: use single-element lists to be safe.
        # Try first + last option to maximize the chance of state divergence.
        picks = [q1.options[0], q1.options[-1]] if len(q1.options) >= 2 else q1.options
        answers = []
        for opt in picks:
            if q1_ev.dtype == "M":
                answers.append(([opt.label], [opt.value]))
            else:
                answers.append((opt.label, opt.value))

    for answer, answer_value in answers:
        s = state0.copy()
        _apply(ds, s, q1_idx, answer, answer_value)
        q2_idx = _next_ev_idx(ds, s)
        key = str(answer)
        branches[key] = q2_idx
        print(f"  Q1={key!r} → Q2 = {_describe(ds, q2_idx)}")

    distinct = len(set(branches.values()))
    if distinct > 1:
        print(
            f"  → Q2 varies across Q1 answers ({distinct} distinct) ✓ state-dependent"
        )
    else:
        print("  → Q2 is the SAME across every Q1 answer we tried ⚠")


# ---------------------------------------------------------------------------
# Probe 3: profile dependency — vary (age, sex), watch Q1.
# ---------------------------------------------------------------------------


def probe_profile_dependency(ds: Any) -> None:
    print("\n=== Probe 3: does (age, sex) change Q1? ===")
    profiles = [(6, "M"), (25, "M"), (25, "F"), (55, "M"), (55, "F"), (85, "F")]
    results: dict[tuple[int, str], int] = {}
    for age, sex in profiles:
        state = _initial_state(ds, age, sex)
        results[(age, sex)] = _next_ev_idx(ds, state)
        print(f"  age={age:>2} sex={sex} → Q1 = {_describe(ds, results[(age, sex)])}")
    distinct = len(set(results.values()))
    print(f"  → {distinct} distinct Q1 across {len(results)} profiles")


# ---------------------------------------------------------------------------
# Probe 4: fixed-answer 3-turn walk — reproduce the user's UI screenshots.
# ---------------------------------------------------------------------------


def probe_three_turn_walk(ds: Any, age: int, sex: str) -> None:
    print("\n=== Probe 4: 3-turn walk with 'first option' answers ===")
    state = _initial_state(ds, age, sex)
    for turn in range(1, 4):
        ev_idx = _next_ev_idx(ds, state)
        print(f"  Q{turn} = {_describe(ds, ev_idx)}")
        answer, value = _first_answer(ds, ev_idx)
        _apply(ds, state, ev_idx, answer, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ddxplus")
    parser.add_argument("--age", type=int, default=30)
    parser.add_argument("--sex", default="M", choices=["M", "F"])
    args = parser.parse_args()

    ds = _load(args.dataset)
    model = ds.select_model()
    print(f"Loaded {args.dataset} / model={model.spec.id}")
    print(f"Fixed profile for probes 1,2,4: age={args.age} sex={args.sex}")
    print("(init-matcher deliberately not injected — pure agent behaviour)")

    probe_determinism(ds, args.age, args.sex)
    probe_state_dependency(ds, args.age, args.sex)
    probe_profile_dependency(ds)
    probe_three_turn_walk(ds, args.age, args.sex)


if __name__ == "__main__":
    main()
