"""Probe the promoted Pneumonia+Influenza subset model.

Three probes:

  1. Determinism: is Q1 fixed given (age, sex)?
  2. Decision boundary: walk maxstep questions, enumerate binary-answer
     combinations, record which flip the argmax between Pneumonia and
     Influenza. Prints the two extreme answer sequences (most-Pneumonia
     and most-Influenza) and any single-question flip points.
  3. Confidence sweep: for the "all-yes" and "all-no" answer paths,
     print P(Pneumonia) after each question so you can see how
     confidence evolves.

The probe skips init_matcher (pure agent behaviour) to isolate the
model's own decision logic from chief-complaint pre-writing.

Run with:
    uv run python scripts/probe_pneumonia_flu_ordering.py
"""

from __future__ import annotations

import argparse
from itertools import product
from typing import Any

import numpy as np

from claritymed.config import load_symptoms_config
from claritymed.core.device import resolve_device
from claritymed.core.symptoms.datasets import build_dataset
from claritymed.ingest.symptoms import ddxplus as _register_adapters  # noqa: F401
from claritymed.servers.symptoms.app import _initial_state, _writer_env
from claritymed.servers.symptoms.questions import build_question, synth_patient

DATASET_ID = "ddxplus_pneumonia_flu"


def _load(dataset_id: str) -> Any:
    """Load the enabled subset dataset with its config-time overrides."""
    cfg = load_symptoms_config()
    ds_spec = next(d for d in cfg.datasets if d.id == dataset_id and d.enabled)
    device = resolve_device("auto")
    return build_dataset(ds_spec, cfg.models, device=device, init_matcher=None)


def _next_ev_idx(ds: Any, state: np.ndarray) -> int:
    return int(ds.select_model().agent.next_action(state)[0])


def _diagnose(ds: Any, state: np.ndarray) -> tuple[int, np.ndarray]:
    """Return (argmax_idx, prob_vector)."""
    argmax, probs = ds.select_model().agent.diagnose(state)
    return int(argmax[0]), probs[0]


def _condition_name(ds: Any, idx: int) -> str:
    return ds.canonical.conditions[idx].native_name["en"]


def _apply_binary(ds: Any, state: np.ndarray, ev_idx: int, yes: bool) -> None:
    """Write a Yes/No answer for a binary evidence."""
    synth = synth_patient(
        ds.canonical,
        ds.spec,
        ev_idx,
        "yes" if yes else "no",
        answer_value=None,
        language="en",
    )
    _writer_env(ds)._write(state[0], ev_idx, synth)


def _apply_default(ds: Any, state: np.ndarray, ev_idx: int) -> None:
    """Write the first option for a non-binary evidence — deterministic default."""
    ev = ds.canonical.evidence_by_idx(ev_idx)
    q = build_question(ds.canonical, ds.spec, ev_idx, language="en")
    if not q.options:
        # numeric fallback
        synth = synth_patient(ds.canonical, ds.spec, ev_idx, "1", language="en")
    else:
        opt = q.options[0]
        answer = [opt.label] if ev.dtype == "M" else opt.label
        answer_value = [opt.value] if ev.dtype == "M" else opt.value
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
    return f"[{ev.id} dtype={ev.dtype}] {q.question}"


# ---------------------------------------------------------------------------
# Probe 1: determinism
# ---------------------------------------------------------------------------


def probe_determinism(ds: Any, age: int, sex: str) -> None:
    print("\n=== Probe 1: same profile → same Q1 ===")
    q1_runs = []
    for run in range(2):
        state = _initial_state(ds, age, sex)
        q1_runs.append(_next_ev_idx(ds, state))
    for i, ev_idx in enumerate(q1_runs, 1):
        print(f"  run {i}: Q1 = {_describe(ds, ev_idx)}")
    assert q1_runs[0] == q1_runs[1], "determinism broken"
    print("  → deterministic ✓")


# ---------------------------------------------------------------------------
# Probe 2: decision boundary — enumerate binary-answer combinations
# ---------------------------------------------------------------------------


def _walk_with_answers(
    ds: Any,
    age: int,
    sex: str,
    binary_answers: list[bool],
    maxstep: int,
) -> tuple[list[int], list[str], np.ndarray]:
    """Walk maxstep questions, answering binary evidences per ``binary_answers``
    (in the order encountered), non-binary evidences with the first option.

    Returns (ev_idx_sequence, ev_id_sequence, final_probs).
    """
    state = _initial_state(ds, age, sex)
    ev_seq: list[int] = []
    id_seq: list[str] = []
    bin_cursor = 0
    for _ in range(maxstep):
        ev_idx = _next_ev_idx(ds, state)
        ev = ds.canonical.evidence_by_idx(ev_idx)
        ev_seq.append(ev_idx)
        id_seq.append(ev.id)
        if ev.dtype == "B":
            yes = (
                binary_answers[bin_cursor]
                if bin_cursor < len(binary_answers)
                else False
            )
            bin_cursor += 1
            _apply_binary(ds, state, ev_idx, yes)
        else:
            _apply_default(ds, state, ev_idx)
    _, probs = _diagnose(ds, state)
    return ev_seq, id_seq, probs


def probe_decision_boundary(ds: Any, age: int, sex: str, maxstep: int) -> None:
    print(f"\n=== Probe 2: decision boundary over {maxstep} questions ===")

    # First pass: run with all-yes to discover the question sequence and
    # count how many binary evidences appear.
    all_yes = [True] * maxstep
    seq_yes, ids_yes, probs_yes = _walk_with_answers(ds, age, sex, all_yes, maxstep)
    all_no_answers = [False] * maxstep
    seq_no, ids_no, probs_no = _walk_with_answers(ds, age, sex, all_no_answers, maxstep)

    # Print sequences under both extreme answer paths.
    print("\n  Under all-YES answers to binary questions:")
    for i, ev_idx in enumerate(seq_yes, 1):
        print(f"    Q{i}: {_describe(ds, ev_idx)}")
    print(f"  Final probs: {_prob_line(ds, probs_yes)}")

    print("\n  Under all-NO answers to binary questions:")
    for i, ev_idx in enumerate(seq_no, 1):
        print(f"    Q{i}: {_describe(ds, ev_idx)}")
    print(f"  Final probs: {_prob_line(ds, probs_no)}")

    # Count binary evidences in the all-yes path — this bounds the enumeration.
    bin_count = sum(1 for e in seq_yes if ds.canonical.evidence_by_idx(e).dtype == "B")
    if bin_count == 0:
        print("\n  (no binary evidences asked — enumeration skipped)")
        return
    if bin_count > 8:
        print(
            f"\n  {bin_count} binary evidences: enumeration would be 2^{bin_count} — "
            "truncating to first 8 for the sweep."
        )
        bin_count = 8

    # Enumerate all 2^bin_count binary answer combinations.
    combos: list[tuple[tuple[bool, ...], int, float]] = []
    for combo in product([False, True], repeat=bin_count):
        _, _, probs = _walk_with_answers(ds, age, sex, list(combo), maxstep)
        argmax = int(probs.argmax())
        combos.append((combo, argmax, float(probs[argmax])))

    # Group by final prediction.
    by_pred: dict[int, list[tuple[tuple[bool, ...], float]]] = {}
    for combo, argmax, conf in combos:
        by_pred.setdefault(argmax, []).append((combo, conf))

    print(
        f"\n  Enumerated {len(combos)} binary answer combinations over "
        f"{bin_count} binary questions:"
    )
    for pred, entries in sorted(by_pred.items()):
        name = _condition_name(ds, pred)
        avg_conf = np.mean([c for _, c in entries])
        print(
            f"    → {name}: {len(entries)} combos ({100 * len(entries) / len(combos):.1f}%), "
            f"avg confidence {avg_conf:.3f}"
        )

    if len(by_pred) < 2:
        print(
            "  ⚠ every answer combination gave the SAME prediction. "
            "Binary answers don't flip the class under this default-non-binary path."
        )
        return

    # Find single-bit flips that change prediction.
    print(
        "\n  Single-question flip points (change one binary answer → different class):"
    )
    reference = tuple(False for _ in range(bin_count))
    _, ref_argmax, _ = (
        _walk_with_answers(
            ds,
            age,
            sex,
            list(reference),
            maxstep,
        ),
        *(None,),
        None,
    )  # dummy unpack
    _, _, ref_probs = _walk_with_answers(ds, age, sex, list(reference), maxstep)
    ref_argmax = int(ref_probs.argmax())
    print(f"    reference (all-NO) → {_condition_name(ds, ref_argmax)}")

    flip_count = 0
    binary_positions = [
        i for i, e in enumerate(seq_yes) if ds.canonical.evidence_by_idx(e).dtype == "B"
    ][:bin_count]
    for bit in range(bin_count):
        flipped = list(reference)
        flipped[bit] = True
        _, _, probs = _walk_with_answers(ds, age, sex, flipped, maxstep)
        argmax = int(probs.argmax())
        if argmax != ref_argmax:
            q_pos = binary_positions[bit]
            ev_id = seq_yes[q_pos]
            flip_count += 1
            print(
                f"    Q{q_pos + 1} ({ds.canonical.evidence_by_idx(ev_id).id}) "
                f"YES → {_condition_name(ds, argmax)} (conf {probs[argmax]:.3f})"
            )
    if flip_count == 0:
        print("    (none — single flips don't shift class from the all-NO baseline)")


# ---------------------------------------------------------------------------
# Probe 3: confidence trajectory
# ---------------------------------------------------------------------------


def probe_confidence_trajectory(ds: Any, age: int, sex: str, maxstep: int) -> None:
    print("\n=== Probe 3: confidence trajectory (P(Pneumonia)) ===")
    for label, all_yes in [
        ("all-YES to binaries", True),
        ("all-NO to binaries", False),
    ]:
        state = _initial_state(ds, age, sex)
        print(f"\n  {label}:")
        _, probs = _diagnose(ds, state)
        p_pneu = _p_of(ds, probs, "Pneumonia")
        print(f"    turn 0 (init only): P(Pneumonia)={p_pneu:.3f}")
        for turn in range(1, maxstep + 1):
            ev_idx = _next_ev_idx(ds, state)
            ev = ds.canonical.evidence_by_idx(ev_idx)
            if ev.dtype == "B":
                _apply_binary(ds, state, ev_idx, all_yes)
                ans = "YES" if all_yes else "NO"
            else:
                _apply_default(ds, state, ev_idx)
                ans = "first-option"
            _, probs = _diagnose(ds, state)
            p_pneu = _p_of(ds, probs, "Pneumonia")
            print(
                f"    Q{turn} {ev.id} ({ev.dtype}) ans={ans} → P(Pneumonia)={p_pneu:.3f}"
            )


def _p_of(ds: Any, probs: np.ndarray, name: str) -> float:
    for idx, cond in enumerate(ds.canonical.conditions):
        if cond.native_name["en"] == name:
            return float(probs[idx])
    return float("nan")


def _prob_line(ds: Any, probs: np.ndarray) -> str:
    parts = []
    for idx, cond in enumerate(ds.canonical.conditions):
        parts.append(f"{cond.native_name['en']}={probs[idx]:.3f}")
    argmax = int(probs.argmax())
    parts.append(f"→ {_condition_name(ds, argmax)}")
    return ", ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--age", type=int, default=35)
    parser.add_argument("--sex", default="M", choices=["M", "F"])
    args = parser.parse_args()

    ds = _load(args.dataset)
    model = ds.select_model()
    maxstep = model.spec.maxstep
    print(f"Loaded {args.dataset} / {model.spec.id}")
    print(
        f"  patho_temp={model.spec.patho_temp} maxstep={maxstep} "
        f"stop_thres={model.spec.stop_thres}"
    )
    print(f"  classes = {[c.native_name['en'] for c in ds.canonical.conditions]}")
    print(f"  profile: age={args.age} sex={args.sex}")

    probe_determinism(ds, args.age, args.sex)
    probe_decision_boundary(ds, args.age, args.sex, maxstep)
    probe_confidence_trajectory(ds, args.age, args.sex, maxstep)


if __name__ == "__main__":
    main()
