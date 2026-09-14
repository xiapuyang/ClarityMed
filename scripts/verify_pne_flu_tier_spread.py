"""Verify the v5 XGB model's Pne/Flu tier spread and target recall.

Answers two questions the tune / interactive_eval harness can't cleanly:

  1. **Tier spread on a fixed patient.** With demographics + init evidence
     held fixed, enumerate 2^K yes/no answers on the top-K discriminative
     binary evidences. Bin the resulting P(Pneumonia) into five tiers
     (``>0.65``, ``0.40-0.65``, ``<0.40``, ``Flu-lead``, ``Other-lead``).
     A well-behaved model shows non-zero mass in EVERY tier — different
     answer patterns produce genuinely different outcomes. A misbehaving
     model bunches most permutations into ``Other-lead``.

  2. **Target-class recall on real patients.** Simulate the interactive
     agent loop on true Pne + true Flu + true Other test patients under
     both the LEGACY IG policy (symmetric recall + ``proj_max`` stop)
     and the NEW policy (asymmetric recall + ``proj_target_sum`` stop).
     Report:
       * ``target_top1_recall`` — P(argmax ∈ {Pne, Flu} | true ∈ {Pne, Flu})
       * ``target_prob_recall`` — P(P_pne + P_flu > 0.5 | true target)
       * ``mean_IL_on_targets`` — cost paid for target patients
       * ``mean_IL_on_others`` — cost paid for non-target patients

Success criteria for the new policy:
    - Tier histogram: every tier > 0 with visible spread (not a spike).
    - Recall: ``target_top1_recall`` and ``target_prob_recall`` non-decreasing
      vs LEGACY, with ``mean_IL_on_targets`` no worse than +1 turn.

Usage::

    uv run python scripts/verify_pne_flu_tier_spread.py \\
        --weights ~/.claritymed/models/symptoms/ddxplus/xgb_pne_inf_v5_recallig/weights.pkl \\
        --n-test 2000 --n-per-class 30 --enum-k 8
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from claritymed.ingest.symptoms.ddxplus.schema import (
    load_evidence_schema,
    load_patients,
    load_pidx,
)
from claritymed.ingest.symptoms.typed_basd import (
    AGE_BUCKETS,
    SEX2IDX,
    TypedEnv,
    seed_everything,
)
from claritymed.ingest.symptoms.xgb.algorithm import XgbAgent

DEFAULT_DATA_DIR = pathlib.Path.home() / ".claritymed/data/symptoms/ddxplus"
DEFAULT_WEIGHTS = (
    pathlib.Path.home()
    / ".claritymed/models/symptoms/ddxplus/xgb_pne_inf_v5_recallig/weights.pkl"
)
DEFAULT_OUT_DIR = pathlib.Path("runs/verify_pne_flu_tier_spread")

PNE_IDX = 0
FLU_IDX = 1
OTHER_IDX = 2

# Cough+fever is the canonical Pne/Flu chief complaint SapBERT most often
# resolves to. E_144 (cough), E_91 (fever) — matches the v5 init_noise_common_evs.
DEFAULT_INIT_EVS = ("E_144", "E_91")

# Question set for the enumeration. All binary, drawn from the mock's
# question sequence + the two most predictive additional evidences from
# the v5 manifest feature importance. K=8 → 256 permutations, fast on CPU.
DEFAULT_ENUM_EVS = (
    "E_77",  # colored/abundant sputum — Pne-specific
    "E_88",  # severe fatigue — Flu-specific
    "E_94",  # chills/shivers — either
    "E_66",  # shortness of breath — Pne-leaning
    "E_161",  # appetite loss — mild
    "E_220",  # pleuritic pain — Pne-specific
    "E_1",  # chest pain — general
    "E_53",  # sore throat — Flu-leaning
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _relabel_to_subset(patients: list[dict], full_pidx: dict[str, int]) -> None:
    """Pne → 0, Flu → 1, everything else → 2 (Other)."""
    subset = {full_pidx["Pneumonia"]: PNE_IDX, full_pidx["Influenza"]: FLU_IDX}
    for p in patients:
        p["d"] = subset.get(p["d"], OTHER_IDX)


def _load_agent(weights_path: pathlib.Path, schema: dict) -> XgbAgent:
    agent = XgbAgent.load(weights_path, schema)
    # Target projection MUST be active for the recall bonus + variant stops.
    agent.target_class_idxs = [PNE_IDX, FLU_IDX]
    return agent


def _initial_state(schema: dict, init_ev_names: tuple[str, ...]) -> np.ndarray:
    """Empty typed state with init evidences pre-revealed on turn 0."""
    context_size = len(AGE_BUCKETS) + len(SEX2IDX)
    state = np.zeros((1, schema["sym_size"] + context_size))
    # Demographics: 37yo, male — matches DDXPlus default demo case.
    state[0, schema["sym_size"] + 4] = 1.0  # age bucket 4 (30-44)
    state[0, schema["sym_size"] + len(AGE_BUCKETS) + 0] = 1.0  # sex M
    ev_names = [ev["name"] for ev in schema["evs"]]
    for name in init_ev_names:
        if name in ev_names:
            ev_i = ev_names.index(name)
            off = schema["off"][ev_i]
            state[0, off] = 1.0  # binary yes
    return state


# ---------------------------------------------------------------------------
# Section 1 — answer-permutation tier spread
# ---------------------------------------------------------------------------


@dataclass
class TierHistogram:
    """Bins P(Pne) across 2^K answer permutations for one config."""

    config_name: str
    n_permutations: int
    tier_counts: dict[str, int] = field(default_factory=dict)
    example_by_tier: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add(self, tier: str, probs: np.ndarray, answers: list[str]) -> None:
        self.tier_counts[tier] = self.tier_counts.get(tier, 0) + 1
        if tier not in self.example_by_tier:
            self.example_by_tier[tier] = {
                "P_pne": float(probs[PNE_IDX]),
                "P_flu": float(probs[FLU_IDX]),
                "P_other": float(probs[OTHER_IDX]),
                "answers": answers,
            }

    def as_dict(self) -> dict[str, Any]:
        return {
            "config_name": self.config_name,
            "n_permutations": self.n_permutations,
            "tier_counts": dict(self.tier_counts),
            "example_by_tier": dict(self.example_by_tier),
        }


def _classify_tier(probs: np.ndarray) -> str:
    """Five-tier bin. Order matters — first hit wins."""
    p_pne = probs[PNE_IDX]
    p_flu = probs[FLU_IDX]
    p_other = probs[OTHER_IDX]
    if p_pne > 0.65:
        return "Pne_high>0.65"
    if p_pne >= 0.40:
        return "Pne_mid_0.40-0.65"
    if p_flu > p_pne and p_flu > p_other:
        return "Flu_lead"
    if p_other > p_pne and p_other > p_flu:
        return "Other_lead"
    return "Pne_low<0.40"


def compute_tier_spread(
    agent: XgbAgent,
    schema: dict,
    enum_evs: tuple[str, ...],
    init_evs: tuple[str, ...],
    config_name: str,
) -> TierHistogram:
    """Enumerate 2^K yes/no answers on the enum-evidence set, bin by tier.

    Every permutation shares the same demographic + init-evidence state;
    the only variation is the yes/no answers on the K enumerated
    evidences. Direct measure of "does answer variation actually move the
    posterior across tiers, or does it collapse to Other".
    """
    ev_names = [ev["name"] for ev in schema["evs"]]
    enum_indices: list[int] = []
    for name in enum_evs:
        if name in ev_names:
            enum_indices.append(ev_names.index(name))
    if not enum_indices:
        raise SystemExit(f"None of {enum_evs} found in schema — bad ev ids")
    k = len(enum_indices)
    n_perms = 2**k
    hist = TierHistogram(config_name=config_name, n_permutations=n_perms)
    for bits in itertools.product([0, 1], repeat=k):
        state = _initial_state(schema, init_evs)
        answers = []
        for ev_i, bit in zip(enum_indices, bits):
            off = schema["off"][ev_i]
            state[0, off] = 1.0 if bit else -1.0
            answers.append(f"{ev_names[ev_i]}={'Yes' if bit else 'No'}")
        _, probs = agent.diagnose(state)
        tier = _classify_tier(probs[0])
        hist.add(tier, probs[0], answers)
    return hist


# ---------------------------------------------------------------------------
# Section 2 — real-patient recall harness
# ---------------------------------------------------------------------------


@dataclass
class RecallMetrics:
    config_name: str
    n_pne: int = 0
    n_flu: int = 0
    n_other: int = 0
    target_top1_hit: int = 0  # argmax ∈ {Pne, Flu} on target patients
    target_prob_recall_hit: int = 0  # P_target sum > 0.5 on target patients
    other_top1_hit: int = 0  # argmax == Other on other patients
    il_on_targets: list[int] = field(default_factory=list)
    il_on_others: list[int] = field(default_factory=list)

    def record(
        self,
        true_class: int,
        argmax: int,
        probs: np.ndarray,
        il: int,
    ) -> None:
        p_target = float(probs[PNE_IDX] + probs[FLU_IDX])
        if true_class in (PNE_IDX, FLU_IDX):
            if true_class == PNE_IDX:
                self.n_pne += 1
            else:
                self.n_flu += 1
            if argmax in (PNE_IDX, FLU_IDX):
                self.target_top1_hit += 1
            if p_target > 0.5:
                self.target_prob_recall_hit += 1
            self.il_on_targets.append(il)
        else:
            self.n_other += 1
            if argmax == OTHER_IDX:
                self.other_top1_hit += 1
            self.il_on_others.append(il)

    def as_dict(self) -> dict[str, Any]:
        n_targets = self.n_pne + self.n_flu
        return {
            "config_name": self.config_name,
            "n_pne": self.n_pne,
            "n_flu": self.n_flu,
            "n_other": self.n_other,
            "target_top1_recall": (
                self.target_top1_hit / n_targets if n_targets else 0.0
            ),
            "target_prob_recall_at_0.5": (
                self.target_prob_recall_hit / n_targets if n_targets else 0.0
            ),
            "other_specificity": (
                self.other_top1_hit / self.n_other if self.n_other else 0.0
            ),
            "mean_IL_on_targets": (
                float(np.mean(self.il_on_targets)) if self.il_on_targets else 0.0
            ),
            "mean_IL_on_others": (
                float(np.mean(self.il_on_others)) if self.il_on_others else 0.0
            ),
        }


def run_interactive_loop(
    agent: XgbAgent,
    schema: dict,
    patients: list[dict],
    maxstep: int,
    config_name: str,
) -> RecallMetrics:
    """Simulate the agent loop on a list of patients, return recall metrics.

    Uses the SAME writer as the training env so state semantics match.
    Each patient starts from an empty state (no init injection — this is
    the "worst case" the model must survive when SapBERT gives us
    nothing). init from ground-truth is deliberately omitted so we
    measure the raw IG policy's ability to drive to target confidence.
    """
    env = TypedEnv(patients, schema, n_dis=agent.classifier.classes_.shape[0])
    metrics = RecallMetrics(config_name=config_name)
    context_size = len(AGE_BUCKETS) + len(SEX2IDX)
    for p in patients:
        # Fresh empty-symptom state per patient, correct demographics only —
        # no init injection so we measure raw question-driven recall.
        state = np.zeros((1, schema["sym_size"] + context_size))
        state[0, schema["sym_size"] + p["age"]] = 1.0
        state[0, schema["sym_size"] + len(AGE_BUCKETS) + p["sex"]] = 1.0
        env.batch = [p]  # env.reveal reads env.batch[i] as ground truth
        il = 0
        done = agent.should_stop(state)
        for _ in range(maxstep):
            if done.all():
                break
            a = agent.next_action(state)
            state = env.reveal(state, a, done)
            il += 1
            done = done | agent.should_stop(state)
        argmax, probs = agent.diagnose(state)
        metrics.record(
            true_class=int(p["d"]),
            argmax=int(argmax[0]),
            probs=probs[0],
            il=il,
        )
    return metrics


# ---------------------------------------------------------------------------
# CLI + orchestration
# ---------------------------------------------------------------------------


def _stratified_sample(patients: list[dict], n_per_class: int, seed: int) -> list[dict]:
    """Return n_per_class Pne + n_per_class Flu + n_per_class Other patients."""
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[dict]] = {PNE_IDX: [], FLU_IDX: [], OTHER_IDX: []}
    for p in patients:
        by_class[p["d"]].append(p)
    out: list[dict] = []
    for cls in (PNE_IDX, FLU_IDX, OTHER_IDX):
        pool = by_class[cls]
        if not pool:
            continue
        sample_ids = rng.choice(
            len(pool), size=min(n_per_class, len(pool)), replace=False
        )
        for i in sample_ids:
            out.append(pool[i])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", type=pathlib.Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-per-class", type=int, default=30)
    ap.add_argument(
        "--maxstep",
        type=int,
        default=10,
        help="Horizon for the interactive loop. Policy differences don't "
        "show at very short horizons because both symmetric/asymmetric IG "
        "agree on the top pick when P(target) is low; the split appears "
        "around turn 5+.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--skip-recall",
        action="store_true",
        help="Skip the real-patient recall harness (tier spread only)",
    )
    args = ap.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"weights missing: {args.weights}")
    if not args.data_dir.exists():
        raise SystemExit(f"data-dir missing: {args.data_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    t0 = time.time()

    schema = load_evidence_schema(args.data_dir)
    pidx, _sev = load_pidx(args.data_dir)
    agent = _load_agent(args.weights, schema)

    # --- Section 1: tier spread (both configs) ---
    print(f"[verify] loading test patients (n={args.n_test})...")
    test_pats = load_patients(args.data_dir, args.n_test, "test", schema, pidx)
    _relabel_to_subset(test_pats, pidx)

    configs = [
        # Legacy: symmetric abs + proj_max — reproduce v5 pre-fix behavior.
        dict(
            name="LEGACY_symmetric_projmax",
            ig_recall_weight=0.5,
            ig_recall_mode="symmetric",
            stop_policy="proj_max",
        ),
        # New: asymmetric + target-sum stop.
        dict(
            name="NEW_asymmetric_target_sum",
            ig_recall_weight=1.5,
            ig_recall_mode="asymmetric",
            stop_policy="proj_target_sum",
        ),
    ]

    tier_reports: list[dict[str, Any]] = []
    recall_reports: list[dict[str, Any]] = []
    # Two starting scenarios: "user gave suggestive complaint" (cough+fever
    # pre-injected as init) and "user gave nothing" (empty state). The
    # tier histogram differs sharply between them — the empty-state variant
    # shows whether the answers alone (not the complaint priming) can move
    # the posterior into all five tiers.
    scenarios = [
        ("with_init_cough_fever", DEFAULT_INIT_EVS),
        ("empty_init", tuple()),
    ]
    for cfg in configs:
        # Rebuild agent per config so we're comparing the same weights under
        # different serve-time knobs — mirrors what apply_xgb_ig_overrides does.
        a = copy.copy(agent)
        a.ig_recall_weight = cfg["ig_recall_weight"]
        a.ig_recall_mode = cfg["ig_recall_mode"]
        a.stop_policy = cfg["stop_policy"]
        a.target_sum_target_thres = 0.65
        a.target_sum_other_thres = 0.85

        for scenario_name, init_evs in scenarios:
            label = f"{cfg['name']}|{scenario_name}"
            print(
                f"[verify] {label}: tier spread ({2 ** len(DEFAULT_ENUM_EVS)} perms)..."
            )
            hist = compute_tier_spread(a, schema, DEFAULT_ENUM_EVS, init_evs, label)
            tier_reports.append(hist.as_dict())

        if not args.skip_recall:
            print(f"[verify] {cfg['name']}: interactive recall...")
            sample = _stratified_sample(test_pats, args.n_per_class, seed=args.seed)
            metrics = run_interactive_loop(a, schema, sample, args.maxstep, cfg["name"])
            recall_reports.append(metrics.as_dict())

    # --- Emit reports ---
    tier_path = args.out_dir / "tier_spread.json"
    tier_path.write_text(json.dumps(tier_reports, indent=2, ensure_ascii=False))
    print(f"[verify] wrote {tier_path}")

    if not args.skip_recall:
        recall_path = args.out_dir / "recall.json"
        recall_path.write_text(json.dumps(recall_reports, indent=2, ensure_ascii=False))
        print(f"[verify] wrote {recall_path}")

    # --- Console summary ---
    print()
    print("=" * 72)
    print("TIER SPREAD (per config, 2^K answer permutations, base state fixed)")
    print("=" * 72)
    for rep in tier_reports:
        print(f"\n[{rep['config_name']}]  n_perms={rep['n_permutations']}")
        for tier in (
            "Pne_high>0.65",
            "Pne_mid_0.40-0.65",
            "Pne_low<0.40",
            "Flu_lead",
            "Other_lead",
        ):
            n = rep["tier_counts"].get(tier, 0)
            pct = 100 * n / rep["n_permutations"]
            bar = "#" * int(pct / 2)
            print(f"  {tier:20s} {n:4d}  {pct:5.1f}%  {bar}")

    if recall_reports:
        print()
        print("=" * 72)
        print("REAL-PATIENT RECALL")
        print("=" * 72)
        header = (
            f"{'config':32s} {'top1_R':>7s} {'prob_R':>7s} {'other_S':>8s} "
            f"{'IL_tgt':>7s} {'IL_oth':>7s}"
        )
        print(header)
        print("-" * len(header))
        for rep in recall_reports:
            print(
                f"{rep['config_name']:32s} "
                f"{rep['target_top1_recall']:7.3f} "
                f"{rep['target_prob_recall_at_0.5']:7.3f} "
                f"{rep['other_specificity']:8.3f} "
                f"{rep['mean_IL_on_targets']:7.2f} "
                f"{rep['mean_IL_on_others']:7.2f}"
            )

    print()
    print(f"[verify] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
