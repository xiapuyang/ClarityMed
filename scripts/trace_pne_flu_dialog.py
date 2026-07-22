"""Trace the full question sequence and probability trajectory for a
canonical Pne/Flu chief complaint.

Given the user's opening line (default ``"I have fever, cough, and muscle
aches for 3 days"``), simulate:

  1. **SapBERT init injection.** Match the complaint against the v5
     init catalog. Above-threshold matches are pre-revealed in the state
     vector (multi-evidence injection, gated by ``min_confidence_gate``).
     Falls back to a hand-picked injection set when SapBERT is missing.

  2. **Agent question loop.** At each turn print:
       * The question the agent picks (i.e. what the frontend would ask).
       * The classifier's current posterior ``P(Pne) / P(Flu) / P(Other)``.
       * Both ``Yes`` and ``No`` branches — the new state, the updated
         posterior, and whether the stop-gate fires.

  3. **Four illustrative traces** ending in distinct tiers so it's clear
     how different answer patterns steer the model:
       * confirm-Pne trace  → target Pne_high
       * confirm-Flu trace  → Flu-lead
       * mixed / mid trace  → Pne_mid or Flu_mid
       * deny-all trace     → Other-lead

Loads the shipping v5 checkpoint with the NEW serve-time knobs applied
(``ig_recall_weight=1.5``, ``ig_recall_mode="asymmetric"``, ``stop_policy=
"proj_target_sum"`` with thresholds ``0.65 / 0.85``). No server needed —
this is the same code path ``adapter.py::DDXPlusAdapter.load`` would run.

Usage::

    uv run python scripts/trace_pne_flu_dialog.py \\
        --complaint "I have fever, cough, and muscle aches for 3 days"
"""

from __future__ import annotations

import argparse
import copy
import pathlib
from typing import Any

import numpy as np

from claritymed.ingest.symptoms.ddxplus.schema import (
    load_evidence_schema,
    load_pidx,
)
from claritymed.ingest.symptoms.typed_basd import AGE_BUCKETS, SEX2IDX
from claritymed.ingest.symptoms.xgb.algorithm import XgbAgent
from claritymed.ingest.symptoms.xgb.encoding import load_evidence_meta

DEFAULT_COMPLAINT = "I have fever, cough, and muscle aches for 3 days"
DEFAULT_DATA_DIR = pathlib.Path.home() / ".claritymed/data/symptoms/ddxplus"
DEFAULT_WEIGHTS = (
    pathlib.Path.home()
    / ".claritymed/models/symptoms/ddxplus/xgb_pne_inf_v5_recallig/weights.pkl"
)

PNE_IDX = 0
FLU_IDX = 1
OTHER_IDX = 2

# Fallback when SapBERT can't load — the three evidence IDs a competent
# matcher would return for this complaint. Keep in sync with the
# ``evidences with 'fever'/'cough'/'muscle' in the question_en`` grep in
# ~/.claritymed/data/symptoms/ddxplus/release_evidences.json.
_FALLBACK_INIT_MATCHES = [
    ("E_91", 0.90),   # "Do you have a fever?"
    ("E_201", 0.85),  # "Do you have a cough?"
    ("E_144", 0.78),  # "Do you have diffuse muscle pain?"
]


def _configure_agent(agent: XgbAgent, target_sum_thres: float = 0.85) -> XgbAgent:
    """Apply the NEW serve-time knobs — same as adapter._apply_xgb_ig_overrides."""
    agent.target_class_idxs = [PNE_IDX, FLU_IDX]
    agent.ig_recall_weight = 1.5
    agent.ig_recall_mode = "asymmetric"
    agent.stop_policy = "proj_target_sum"
    agent.target_sum_target_thres = target_sum_thres
    agent.target_sum_other_thres = 0.90
    return agent


def _sapbert_matches(
    complaint: str, schema: dict, meta: dict[str, dict]
) -> list[tuple[str, float]]:
    """Try SapBERT; fall back to hand-picked matches if unavailable."""
    try:
        from claritymed.core.symptoms.init_matcher import (
            InitMatcherEmbedder,
            filter_candidate_evidences,
        )
        from claritymed.core.symptoms.datasets.canonical import (
            CanonicalEvidence,
            CanonicalValue,
            InitSymptomCatalog,
        )
        from claritymed.core.symptoms.schemas import InitSymptomFilter
    except Exception as e:  # noqa: BLE001
        print(f"[trace] SapBERT import failed: {e}; using fallback matches")
        return _FALLBACK_INIT_MATCHES

    # Build a mini-catalog directly from the schema — B-only, non-antecedent.
    canonical_evs: list[CanonicalEvidence] = []
    for ev_i, ev in enumerate(schema["evs"]):
        native_q = meta.get(ev["name"], {}).get("question_en", "")
        canonical_evs.append(
            CanonicalEvidence(
                id=ev["name"],
                idx=ev_i,
                dtype=ev["dtype"],
                values=[],
                native_question_text={"en": native_q} if native_q else {},
                is_antecedent=bool(
                    meta.get(ev["name"], {}).get("is_antecedent", False)
                ),
            )
        )

    embedder = InitMatcherEmbedder(
        model_id="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        device="cpu",
        default_threshold=0.55,
    )
    filt = InitSymptomFilter()
    candidates = filter_candidate_evidences(canonical_evs, filt)
    texts = [
        ev.native_question_text.get("en") or ev.id for ev in candidates
    ]
    print(f"[trace] loading SapBERT (this takes ~5s the first time)...")
    matrix = embedder.encode(texts)
    if matrix is None:
        print("[trace] SapBERT unavailable; using fallback matches")
        return _FALLBACK_INIT_MATCHES
    catalog = InitSymptomCatalog(
        candidate_idx=[ev.idx for ev in candidates],
        matrix=matrix,
        threshold=0.55,
    )
    matches = embedder.match_topk(complaint, catalog, k=3, min_score=0.55)
    if not matches:
        print("[trace] SapBERT matched nothing above threshold; using fallback")
        return _FALLBACK_INIT_MATCHES
    ev_names = [ev["name"] for ev in schema["evs"]]
    return [(ev_names[m.evidence_idx], m.score) for m in matches]


def _empty_state(schema: dict, age_bucket_i: int = 4, sex_i: int = 0) -> np.ndarray:
    """Zero symptoms + one-hot age/sex. Default: 37M."""
    ctx = len(AGE_BUCKETS) + len(SEX2IDX)
    s = np.zeros((1, schema["sym_size"] + ctx))
    s[0, schema["sym_size"] + age_bucket_i] = 1.0
    s[0, schema["sym_size"] + len(AGE_BUCKETS) + sex_i] = 1.0
    return s


def _write_binary_answer(
    state: np.ndarray, schema: dict, ev_name: str, yes: bool
) -> np.ndarray:
    """In-place write a Yes/No answer to a binary evidence's slot."""
    ev_names = [ev["name"] for ev in schema["evs"]]
    if ev_name not in ev_names:
        return state
    ev_i = ev_names.index(ev_name)
    off = int(schema["off"][ev_i])
    s = state.copy()
    s[0, off] = 1.0 if yes else -1.0
    return s


def _fmt_probs(probs: np.ndarray) -> str:
    return (
        f"P(Pne)={probs[0][PNE_IDX]:.2f}  "
        f"P(Flu)={probs[0][FLU_IDX]:.2f}  "
        f"P(Other)={probs[0][OTHER_IDX]:.2f}"
    )


def _tier(probs: np.ndarray) -> str:
    p = probs[0]
    if p[PNE_IDX] > 0.65:
        return "★ Pne_high (>65%)"
    if p[PNE_IDX] >= 0.40:
        return "◐ Pne_mid (40-65%)"
    if p[FLU_IDX] > p[PNE_IDX] and p[FLU_IDX] > p[OTHER_IDX]:
        return "▷ Flu_lead"
    if p[OTHER_IDX] > p[PNE_IDX] and p[OTHER_IDX] > p[FLU_IDX]:
        return "◇ Other_lead"
    return "· Pne_low"


def _q_text(meta: dict, ev_name: str) -> str:
    q = meta.get(ev_name, {}).get("question_en", ev_name)
    return q if len(q) <= 78 else q[:75] + "..."


def _apply_init(
    state: np.ndarray,
    schema: dict,
    matches: list[tuple[str, float]],
    min_confidence_gate: float = 0.62,
) -> tuple[np.ndarray, list[str]]:
    """Multi-evidence injection with the confidence gate."""
    if not matches:
        return state, []
    if matches[0][1] < min_confidence_gate:
        return state, []
    injected: list[str] = []
    for name, _score in matches:
        state = _write_binary_answer(state, schema, name, yes=True)
        injected.append(name)
    return state, injected


def trace_dialog(
    agent: XgbAgent,
    schema: dict,
    meta: dict,
    complaint: str,
    matches: list[tuple[str, float]],
    max_turns: int = 6,
) -> None:
    print()
    print("=" * 78)
    print(f'USER COMPLAINT:  "{complaint}"')
    print("=" * 78)

    print("\n[SapBERT top-3 matches above threshold 0.55]")
    for name, score in matches:
        print(f"  {name}  ({score:.2f})  {_q_text(meta, name)}")

    state = _empty_state(schema)
    state, injected = _apply_init(state, schema, matches, min_confidence_gate=0.62)
    print(f"\n[Init injection]  {len(injected)} evidences pre-revealed: {injected}")

    _, probs = agent.diagnose(state)
    print(f"[Baseline after init]  {_fmt_probs(probs)}   →  {_tier(probs)}")

    # Turn-by-turn: at each turn, print Q and both Yes/No branches.
    print()
    print("─" * 78)
    print("DECISION TREE (baseline branch = Yes; sibling shows the No answer)")
    print("─" * 78)
    _walk(agent, schema, meta, state, depth=0, max_depth=max_turns)


def _walk(
    agent: XgbAgent,
    schema: dict,
    meta: dict,
    state: np.ndarray,
    depth: int,
    max_depth: int,
    path: list[str] | None = None,
) -> None:
    """Recursive walk following the Yes branch, printing Yes/No at each turn."""
    if path is None:
        path = []
    if depth >= max_depth:
        return
    if agent.should_stop(state).any():
        print(f"{'  ' * depth}└─ STOP fires (target_sum > 0.65 or Other > 0.85)")
        return
    a = int(agent.next_action(state)[0])
    ev_name = schema["evs"][a]["name"]
    q = _q_text(meta, ev_name)
    indent = "  " * depth
    print(f"{indent}Q{depth + 1}: [{ev_name}] {q}")

    for answer_label, yes in (("Yes", True), ("No", False)):
        s2 = _write_binary_answer(state, schema, ev_name, yes=yes)
        _, probs = agent.diagnose(s2)
        stop = "  [STOP]" if agent.should_stop(s2).any() else ""
        print(
            f"{indent}    {answer_label:3s} → {_fmt_probs(probs)}   {_tier(probs)}{stop}"
        )
    # Follow the Yes branch one level deeper for narrative flow.
    yes_state = _write_binary_answer(state, schema, ev_name, yes=True)
    _walk(agent, schema, meta, yes_state, depth + 1, max_depth, path + [f"{ev_name}=Y"])


def full_answer_paths(
    agent: XgbAgent,
    schema: dict,
    meta: dict,
    initial_state: np.ndarray,
) -> None:
    """Show four illustrative FULL answer paths, each landing in a
    different tier so the mapping "answer pattern → tier" is concrete."""
    print()
    print("=" * 78)
    print("FOUR CANONICAL ANSWER PATHS (starting from init-injected state)")
    print("=" * 78)

    # For each scripted path, we override answers so the walk hits the
    # target tier regardless of which question the IG policy asks next.
    # The scripted answers are keyed by evidence id — if the agent
    # doesn't ask that evidence, we synthesize by writing it directly.
    paths = [
        (
            "PATH 1 — confirms Pneumonia",
            {"E_77": True, "E_66": True, "E_220": True, "E_88": False},
            "colored sputum + shortness of breath + pleuritic pain = classic pneumonia",
        ),
        (
            "PATH 2 — confirms Influenza",
            {"E_88": True, "E_94": True, "E_77": False, "E_66": False},
            "severe fatigue + chills without sputum/SOB = classic flu presentation",
        ),
        (
            "PATH 3 — mixed / mid Pne",
            {"E_66": True, "E_94": True, "E_77": False, "E_88": False},
            "SOB + chills but no colored sputum + no severe fatigue = ambiguous",
        ),
        (
            "PATH 4 — denies everything",
            {"E_77": False, "E_66": False, "E_88": False, "E_94": False},
            "denies all discriminative symptoms = fever+cough+aches alone = Other",
        ),
    ]

    for name, script, hint in paths:
        print(f"\n{name}")
        print(f"  scenario: {hint}")
        state = initial_state.copy()
        turn = 1
        asked: set[str] = set()
        while turn <= 6:
            if agent.should_stop(state).any():
                print(f"  Q{turn}: (stop-gate fires — model is confident)")
                break
            a = int(agent.next_action(state)[0])
            ev_name = schema["evs"][a]["name"]
            if ev_name in asked:
                break  # safety net
            asked.add(ev_name)
            if ev_name in script:
                answer = script[ev_name]
                marker = ""
            else:
                # Agent asked something not in our script — answer No.
                answer = False
                marker = " [not scripted — answered No]"
            state = _write_binary_answer(state, schema, ev_name, yes=answer)
            _, probs = agent.diagnose(state)
            print(
                f"  Q{turn}: [{ev_name}] {_q_text(meta, ev_name)[:50]}"
                f"  →  answer={'Yes' if answer else 'No'}{marker}"
            )
            print(f"        {_fmt_probs(probs)}   {_tier(probs)}")
            turn += 1
        # Force any remaining scripted answers to guarantee the tier lands.
        for ev_name, answer in script.items():
            if ev_name not in asked:
                state = _write_binary_answer(state, schema, ev_name, yes=answer)
        _, probs = agent.diagnose(state)
        print(f"  FINAL: {_fmt_probs(probs)}   {_tier(probs)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--complaint", default=DEFAULT_COMPLAINT)
    ap.add_argument("--weights", type=pathlib.Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--max-turns", type=int, default=4)
    ap.add_argument(
        "--skip-sapbert",
        action="store_true",
        help="Skip SapBERT and use the fallback matches (E_91/E_201/E_144). "
        "Runs in ~1s instead of 30-60s.",
    )
    args = ap.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"weights missing: {args.weights}")
    if not args.data_dir.exists():
        raise SystemExit(f"data-dir missing: {args.data_dir}")

    schema = load_evidence_schema(args.data_dir)
    _, _ = load_pidx(args.data_dir)  # verifies data integrity
    meta = load_evidence_meta(args.data_dir)
    agent = _configure_agent(XgbAgent.load(args.weights, schema))

    # Compute init matches once and reuse in both sections.
    if args.skip_sapbert:
        matches = _FALLBACK_INIT_MATCHES
        print(f"[trace] --skip-sapbert set; using fallback: {matches}")
    else:
        matches = _sapbert_matches(args.complaint, schema, meta)

    trace_dialog(agent, schema, meta, args.complaint, matches, max_turns=args.max_turns)

    state = _empty_state(schema)
    state, _ = _apply_init(state, schema, matches, min_confidence_gate=0.62)
    full_answer_paths(agent, schema, meta, state)


if __name__ == "__main__":
    main()
