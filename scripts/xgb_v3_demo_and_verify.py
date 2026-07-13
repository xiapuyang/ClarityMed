"""XGB v3 demo-case picker + interactive-replay verifier.

Three sections, one script — targets the ``xgb_pne_inf_v3`` 3-class
native model (``{Pneumonia, Influenza, Other}``):

1. **Real-patient bucketing.** For every DDXPlus test patient, encode
   the full evidence signature and score ``(P(Pne), P(Inf), P(Other))``.
   Sort by P(Pneumonia); sample ``--per-bucket`` patients from each of
   the three bands (``P(Pne) > --high-thres``, in
   ``[--low-thres, --high-thres]``, and ``< --low-thres``). Output:
   concrete demo cases you can paste in a slide, with chief complaint,
   full evidence list, and the ground-truth label alongside the model
   probabilities.

2. **Synthetic enumeration.** Fix demographics + chief complaint (both
   configurable via CLI). Enumerate yes/no answers on the top-K binary
   evidences from the manifest's feature-importance ranking. Sample the
   top ``--per-bucket`` answer sequences per band. Output: a controlled
   sweep of the decision surface — makes it obvious which evidence
   combinations push the posterior across the boundaries.

3. **Interactive replay.** Sample ``--n-per-class`` Pneumonia +
   ``--n-per-class`` Influenza patients from the test set. Run the
   agent's own IG question-selection loop with ``--maxstep`` +
   ``--stop-thres`` (defaults match the v3 config: 18 / 0.95). Log every
   question asked, the patient's ground-truth answer, and the running
   posterior ``(P(Pne), P(Inf), P(Other))``. Compare final argmax to the
   ground-truth label. Output: JSONL trace + per-class accuracy so the
   pipeline can be spot-checked without staging a server.

Outputs land under ``--out-dir`` (default ``runs/xgb_v3_demo_and_verify``):

    report.md        — human-readable summary of all three sections.
    typical.jsonl    — one line per typical-patient case (Section 1).
    synthetic.jsonl  — one line per synthetic answer sequence (Section 2).
    replay.jsonl     — one line per interactive-replay step (Section 3).

Usage::

    uv run python scripts/xgb_v3_demo_and_verify.py \\
        --weights ~/.claritymed/models/symptoms/ddxplus/xgb_pne_inf_v3/weights.pkl \\
        --n-test 5000 --n-per-class 20 --enum-top-k 8
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from claritymed.ingest.symptoms.ddxplus.schema import (
    load_evidence_schema,
    load_patients,
    load_pidx,
)
from claritymed.ingest.symptoms.typed_basd import TypedEnv, seed_everything
from claritymed.ingest.symptoms.xgb.algorithm import XgbAgent
from claritymed.ingest.symptoms.xgb.encoding import (
    encode_patient_batch,
    feature_columns_from_schema,
    load_evidence_meta,
)

DEFAULT_DATA_DIR = pathlib.Path.home() / ".claritymed/data/symptoms/ddxplus"
DEFAULT_WEIGHTS = (
    pathlib.Path.home()
    / ".claritymed/models/symptoms/ddxplus/xgb_pne_inf_v3/weights.pkl"
)
DEFAULT_OUT_DIR = pathlib.Path("runs/xgb_v3_demo_and_verify")

DEFAULT_N_TEST = 5000
DEFAULT_N_PER_CLASS = 20
DEFAULT_PER_BUCKET = 5
DEFAULT_HIGH_THRES = 0.65
DEFAULT_LOW_THRES = 0.40
DEFAULT_MAXSTEP = 18
DEFAULT_STOP_THRES = 0.95
DEFAULT_ENUM_TOP_K = 8

TARGET_NAMES = ("Pneumonia", "Influenza")
OTHER_NAME = "Other"

# Column indices in the v3 posterior. Same order as the manifest's
# ``diseases_trained`` list.
PNE_IDX = 0
INF_IDX = 1
OTHER_IDX = 2


# ---------------------------------------------------------------------------
# Small helpers — patient → readable evidence list.
# ---------------------------------------------------------------------------


def _evidence_labels_for_patient(
    patient: dict, schema: dict, meta: dict[str, dict]
) -> list[str]:
    """Human-readable ``E_X: question`` list for a patient's positive evidences.

    Renders binary positives + categorical answered value + multi-value
    positives in a single flat list. Uses the evidence JSON's
    ``question_en`` and per-value ``value_meaning.en`` where available;
    falls back to raw IDs so a schema drift doesn't crash the report.
    """
    ev_names = [ev["name"] for ev in schema["evs"]]
    lines: list[str] = []
    for ev_i in patient["bin_pos"]:
        ev_id = ev_names[ev_i]
        q = meta.get(ev_id, {}).get("question_en", ev_id)
        lines.append(f"{ev_id}: {q}")
    for ev_i, lv in patient["cat_val"].items():
        ev_id = ev_names[ev_i]
        raw = schema["evs"][ev_i]["values"][lv]
        q = meta.get(ev_id, {}).get("question_en", ev_id)
        vm = (
            meta.get(ev_id, {})
            .get("value_meaning", {})
            .get(str(raw), {})
            .get("en", raw)
        )
        lines.append(f"{ev_id}={vm}: {q}")
    for ev_i, lvs in patient["multi_val"].items():
        ev_id = ev_names[ev_i]
        q = meta.get(ev_id, {}).get("question_en", ev_id)
        for lv in lvs:
            raw = schema["evs"][ev_i]["values"][lv]
            vm = (
                meta.get(ev_id, {})
                .get("value_meaning", {})
                .get(str(raw), {})
                .get("en", raw)
            )
            lines.append(f"{ev_id}={vm}: {q}")
    return lines


def _init_evidence_label(patient: dict, schema: dict, meta: dict[str, dict]) -> str:
    """Chief-complaint-equivalent label for a patient's ``init`` evidence."""
    ev_i = patient["init"]
    ev_id = schema["evs"][ev_i]["name"]
    return meta.get(ev_id, {}).get("question_en", ev_id)


def _relabel_patients_to_subset(
    patients: list[dict], full_pidx: dict[str, int]
) -> None:
    """In-place relabel ``p["d"]`` from full 49-class pidx to v3 subset space.

    Pneumonia → 0, Influenza → 1, every other class → 2 (Other). The
    classifier already emits N+1-class probabilities in this order, so
    this makes ground-truth aligned with the model's output layout.
    """
    subset = {full_pidx[name]: k for k, name in enumerate(TARGET_NAMES)}
    for p in patients:
        p["d"] = subset.get(p["d"], len(TARGET_NAMES))


# ---------------------------------------------------------------------------
# Section 1 — real-patient bucketing.
# ---------------------------------------------------------------------------


@dataclass
class TypicalCase:
    """One typical-patient demo card with bucket + probs + evidence."""

    bucket: str
    ground_truth: str
    p_pneumonia: float
    p_influenza: float
    p_other: float
    age_bucket: int
    sex: int
    chief_complaint: str
    evidences: list[str]


def _pick_typical_indices(
    p_pne: np.ndarray, high: float, low: float, per_bucket: int
) -> dict[str, list[int]]:
    """Pick ``per_bucket`` typical patient indices per band.

    * ``high``: highest ``P(Pne)`` first — the "most confidently Pneumonia".
    * ``low``: lowest ``P(Pne)`` first — the "most confidently NOT
      Pneumonia".
    * ``mid``: closest to the band midpoint — the ambiguous cases the
      LLM should flag as "consider a broader differential".
    """
    all_idxs = np.arange(len(p_pne))
    high_mask = p_pne >= high
    low_mask = p_pne < low
    mid_mask = (~high_mask) & (~low_mask)
    mid_center = (high + low) / 2.0
    picks: dict[str, list[int]] = {}
    high_idxs = all_idxs[high_mask]
    picks["high"] = list(high_idxs[np.argsort(-p_pne[high_mask])][:per_bucket])
    low_idxs = all_idxs[low_mask]
    picks["low"] = list(low_idxs[np.argsort(p_pne[low_mask])][:per_bucket])
    mid_idxs = all_idxs[mid_mask]
    picks["mid"] = list(
        mid_idxs[np.argsort(np.abs(p_pne[mid_mask] - mid_center))][:per_bucket]
    )
    return picks


def section_typical_cases(
    agent: XgbAgent,
    patients: list[dict],
    schema: dict,
    meta: dict[str, dict],
    columns_idx: dict[str, int],
    high_thres: float,
    low_thres: float,
    per_bucket: int,
) -> tuple[list[TypicalCase], dict[str, int]]:
    """Score every patient, bucket by P(Pne), return sampled TypicalCases."""
    x_full = encode_patient_batch(patients, schema, columns_idx)
    probs = agent.classifier.predict_proba(x_full)  # (N, 3)
    p_pne = probs[:, PNE_IDX]
    picks = _pick_typical_indices(p_pne, high_thres, low_thres, per_bucket)
    label_names = [*TARGET_NAMES, OTHER_NAME]
    cases: list[TypicalCase] = []
    for bucket, idxs in picks.items():
        for i in idxs:
            i = int(i)
            p = patients[i]
            cases.append(
                TypicalCase(
                    bucket=bucket,
                    ground_truth=label_names[p["d"]],
                    p_pneumonia=float(probs[i, PNE_IDX]),
                    p_influenza=float(probs[i, INF_IDX]),
                    p_other=float(probs[i, OTHER_IDX]),
                    age_bucket=int(p["age"]),
                    sex=int(p["sex"]),
                    chief_complaint=_init_evidence_label(p, schema, meta),
                    evidences=_evidence_labels_for_patient(p, schema, meta),
                )
            )
    band_counts = {
        "high (>{:.2f})".format(high_thres): int((p_pne >= high_thres).sum()),
        "mid [{:.2f},{:.2f})".format(low_thres, high_thres): int(
            ((p_pne >= low_thres) & (p_pne < high_thres)).sum()
        ),
        "low (<{:.2f})".format(low_thres): int((p_pne < low_thres).sum()),
    }
    return cases, band_counts


# ---------------------------------------------------------------------------
# Section 2 — synthetic enumeration on top-K binary evidences.
# ---------------------------------------------------------------------------


@dataclass
class SyntheticCase:
    """One enumerated answer sequence with bucket + probs + answer flags."""

    bucket: str
    p_pneumonia: float
    p_influenza: float
    p_other: float
    answers: dict[str, str]  # {ev_id: "yes" | "no"}


def _top_k_binary_evidences(
    manifest_path: pathlib.Path, schema: dict, k: int
) -> list[str]:
    """Return the top-K binary evidence IDs by manifest feature importance.

    Categorical / multi-value evidences are excluded — the enumeration
    grid needs 2^K binary axes, and expanding a K-value categorical to
    K binary axes muddies the semantics of "answer yes to E_X". Falls
    back to the first K binary evidences in schema order if the manifest
    has no ``feature_importance_top_k`` field.
    """
    typ_by_name = {ev["name"]: ev["dtype"] for ev in schema["evs"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ranked = manifest.get("feature_importance_top_k", [])
    picked: list[str] = []
    seen: set[str] = set()
    for entry in ranked:
        # column name like ``E_88`` (binary) or ``E_55__V_54`` (cat slot)
        col = entry["name"].split("__", 1)[0]
        if col in seen:
            continue
        if typ_by_name.get(col) != "B":
            continue
        picked.append(col)
        seen.add(col)
        if len(picked) >= k:
            break
    if len(picked) < k:
        for ev in schema["evs"]:
            if ev["dtype"] != "B" or ev["name"] in seen:
                continue
            picked.append(ev["name"])
            seen.add(ev["name"])
            if len(picked) >= k:
                break
    return picked


def _enumerate_binary_grid(
    agent: XgbAgent,
    schema: dict,
    columns_idx: dict[str, int],
    chief_complaint_ev: str,
    age_bucket: int,
    sex: int,
    top_k_binary: list[str],
) -> tuple[np.ndarray, list[dict[str, str]]]:
    """Enumerate 2^K yes/no combos on ``top_k_binary`` — return (probs, answers).

    The chief-complaint evidence is treated as "yes" in every row; the
    enumeration grid only varies the K binary features. Categorical /
    multi-value columns stay at 0 across all rows (baseline: unanswered).
    Demographics are baked into the encoded matrix separately since the
    XGBoost feature layout has no age/sex column — the model only sees
    the evidence one-hot vector; the demographic call-out in the report
    is descriptive, not encoded.

    Returns ``(probs[2^K, 3], answers[2^K])``. Each ``answers[i]`` maps
    every ``top_k_binary`` id to ``"yes"`` / ``"no"``.
    """
    n_features = len(columns_idx)
    k = len(top_k_binary)
    n_combos = 1 << k
    x = np.zeros((n_combos, n_features), dtype=np.float32)
    cc_col = columns_idx.get(chief_complaint_ev)
    if cc_col is not None:
        x[:, cc_col] = 1.0
    for i in range(n_combos):
        for bit, ev_id in enumerate(top_k_binary):
            if (i >> bit) & 1:
                col = columns_idx.get(ev_id)
                if col is not None:
                    x[i, col] = 1.0
    probs = agent.classifier.predict_proba(x)
    answers: list[dict[str, str]] = []
    for i in range(n_combos):
        answers.append(
            {
                ev_id: ("yes" if (i >> bit) & 1 else "no")
                for bit, ev_id in enumerate(top_k_binary)
            }
        )
    _ = (age_bucket, sex)  # kept for report metadata only; not encoded
    return probs, answers


def section_synthetic_enumeration(
    agent: XgbAgent,
    schema: dict,
    columns_idx: dict[str, int],
    weights_path: pathlib.Path,
    high_thres: float,
    low_thres: float,
    per_bucket: int,
    chief_complaint_ev: str,
    age_bucket: int,
    sex: int,
    enum_top_k: int,
) -> tuple[list[SyntheticCase], list[str]]:
    """Run the synthetic 2^K sweep + bucket samples per band."""
    manifest_path = weights_path.with_name("manifest.json")
    top_k_binary = _top_k_binary_evidences(manifest_path, schema, enum_top_k)
    probs, answers = _enumerate_binary_grid(
        agent, schema, columns_idx, chief_complaint_ev, age_bucket, sex, top_k_binary
    )
    p_pne = probs[:, PNE_IDX]
    picks = _pick_typical_indices(p_pne, high_thres, low_thres, per_bucket)
    cases: list[SyntheticCase] = []
    for bucket, idxs in picks.items():
        for i in idxs:
            i = int(i)
            cases.append(
                SyntheticCase(
                    bucket=bucket,
                    p_pneumonia=float(probs[i, PNE_IDX]),
                    p_influenza=float(probs[i, INF_IDX]),
                    p_other=float(probs[i, OTHER_IDX]),
                    answers=answers[i],
                )
            )
    return cases, top_k_binary


# ---------------------------------------------------------------------------
# Section 3 — interactive replay verification.
# ---------------------------------------------------------------------------


@dataclass
class ReplayStep:
    """One IG-loop step: question asked + patient's ground-truth answer + probs after."""

    step: int
    question_ev: str
    question_text: str
    answer: str
    probs: list[float]  # [P(Pne), P(Inf), P(Other)]


@dataclass
class ReplayTrace:
    """Full replay of one patient through the interactive IG loop."""

    patient_idx: int
    ground_truth: str
    steps: list[ReplayStep] = field(default_factory=list)
    stopped_at_step: int | None = None
    final_probs: list[float] = field(default_factory=list)
    final_argmax: str = ""
    correct: bool = False


def _answer_for_patient(
    patient: dict, ev_i: int, schema: dict, meta: dict[str, dict]
) -> str:
    """Render the patient's ground-truth answer for evidence ``ev_i``."""
    ev = schema["evs"][ev_i]
    ev_id = ev["name"]
    dtype = ev["dtype"]
    if dtype == "B":
        return "yes" if ev_i in patient["bin_pos"] else "no"
    if dtype == "C":
        lv = patient["cat_val"].get(ev_i)
        if lv is None:
            return "not answered"
        raw = ev["values"][lv]
        vm = (
            meta.get(ev_id, {})
            .get("value_meaning", {})
            .get(str(raw), {})
            .get("en", raw)
        )
        return str(vm)
    lvs = patient["multi_val"].get(ev_i, [])
    if not lvs:
        return "none"
    parts: list[str] = []
    for lv in lvs:
        raw = ev["values"][lv]
        vm = (
            meta.get(ev_id, {})
            .get("value_meaning", {})
            .get(str(raw), {})
            .get("en", raw)
        )
        parts.append(str(vm))
    return ", ".join(parts)


def _replay_one_patient(
    agent: XgbAgent,
    patient: dict,
    initial_row: np.ndarray,
    patient_idx: int,
    disease_idx: int,
    schema: dict,
    meta: dict[str, dict],
    maxstep: int,
) -> ReplayTrace:
    """Run the IG loop on a single patient; capture every step + final probs.

    ``initial_row`` is a ``(1, state_dim)`` view already seeded with the
    patient's ``init`` evidence + demographic. Each step: diagnose,
    check stop, pick next evidence via IG, reveal the patient's ground-
    truth answer for that evidence, re-diagnose. A single-patient
    :class:`TypedEnv` is used only for its :meth:`reveal` helper — the
    outer sampling loop keeps the batched ``TypedEnv`` for the initial
    state build so we don't re-initialize N patients per replay.
    """
    label_names = [*TARGET_NAMES, OTHER_NAME]
    ev_names = [ev["name"] for ev in schema["evs"]]
    row = initial_row.copy()
    trace = ReplayTrace(patient_idx=patient_idx, ground_truth=label_names[disease_idx])
    single_env = TypedEnv([patient], schema, n_dis=3)
    single_env.batch = [patient]
    for step in range(1, maxstep + 1):
        if bool(agent.should_stop(row)[0]):
            trace.stopped_at_step = step - 1
            break
        next_ev = int(agent.next_action(row)[0])
        answer = _answer_for_patient(patient, next_ev, schema, meta)
        row = single_env.reveal(row, np.array([next_ev]), np.array([False]))
        _, probs_after = agent.diagnose(row)
        trace.steps.append(
            ReplayStep(
                step=step,
                question_ev=ev_names[next_ev],
                question_text=meta.get(ev_names[next_ev], {}).get(
                    "question_en", ev_names[next_ev]
                ),
                answer=answer,
                probs=[float(x) for x in probs_after[0]],
            )
        )
    _, probs_final = agent.diagnose(row)
    trace.final_probs = [float(x) for x in probs_final[0]]
    trace.final_argmax = label_names[int(np.argmax(probs_final[0]))]
    trace.correct = trace.final_argmax == trace.ground_truth
    return trace


def section_interactive_replay(
    agent: XgbAgent,
    patients: list[dict],
    schema: dict,
    meta: dict[str, dict],
    n_per_class: int,
    maxstep: int,
    seed: int,
) -> list[ReplayTrace]:
    """Sample N Pneumonia + N Influenza patients, replay each through the loop.

    Configures the agent for v3 projected stop (``target_class_idxs=[0,1]``,
    ``proj_max`` policy) so both ``should_stop`` and ``next_action`` behave
    the way the production server does when serving the ``xgb_pne_inf_v3``
    checkpoint against a ``target_condition_ids: [pneumonia, influenza]``
    dataset config.
    """
    # v3 native model has 3 classes; the projected policy on that layout
    # is a no-op (see XgbAgent.should_stop's early return for
    # projection=None-equivalent). Kept for explicitness.
    agent.target_class_idxs = [PNE_IDX, INF_IDX]
    agent.stop_policy = "proj_max"
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {PNE_IDX: [], INF_IDX: []}
    for i, p in enumerate(patients):
        if p["d"] in by_class:
            by_class[p["d"]].append(i)
    sampled: list[int] = []
    for cls_idx, idxs in by_class.items():
        rng.shuffle(idxs)
        if len(idxs) < n_per_class:
            print(
                f"[replay] only {len(idxs)} {'Pneumonia' if cls_idx == PNE_IDX else 'Influenza'} "
                f"patients in the sampled test set — using all of them (< {n_per_class})"
            )
        sampled.extend(idxs[:n_per_class])
    sampled_patients = [patients[i] for i in sampled]
    env = TypedEnv(sampled_patients, schema, n_dis=3)
    initial_state, disease = env.initialize_state(len(sampled_patients))
    traces: list[ReplayTrace] = []
    for slot, patient_idx in enumerate(sampled):
        traces.append(
            _replay_one_patient(
                agent,
                sampled_patients[slot],
                initial_state[slot : slot + 1],
                patient_idx,
                int(disease[slot]),
                schema,
                meta,
                maxstep,
            )
        )
    return traces


# ---------------------------------------------------------------------------
# Reporting.
# ---------------------------------------------------------------------------


def _fmt_probs(probs: list[float] | tuple[float, ...]) -> str:
    return (
        f"P(Pne)={probs[PNE_IDX]:.3f}  P(Inf)={probs[INF_IDX]:.3f}  "
        f"P(Other)={probs[OTHER_IDX]:.3f}"
    )


def _render_report(
    out_path: pathlib.Path,
    typical: list[TypicalCase],
    typical_counts: dict[str, int],
    synthetic: list[SyntheticCase],
    synthetic_axes: list[str],
    replay: list[ReplayTrace],
    args: argparse.Namespace,
    total_patients_scored: int,
) -> None:
    """Write the Markdown report combining all three sections."""
    lines: list[str] = []
    lines.append("# XGB v3 demo cases & replay verification")
    lines.append("")
    lines.append(f"- Weights: `{args.weights}`")
    lines.append(
        f"- Test patients scored: {total_patients_scored} (from `{args.data_dir}`)"
    )
    lines.append(
        f"- Bands: high P(Pne) ≥ {args.high_thres:.2f}, "
        f"mid [{args.low_thres:.2f}, {args.high_thres:.2f}), "
        f"low < {args.low_thres:.2f}"
    )
    lines.append(f"- Loop: maxstep={args.maxstep}, stop_thres={args.stop_thres}")
    lines.append("")

    # Section 1
    lines.append("## Section 1 — Real-patient demo cases")
    lines.append("")
    lines.append("Band population over the full scored test set:")
    lines.append("")
    for band, n in typical_counts.items():
        lines.append(f"- {band}: {n} patients")
    lines.append("")
    grouped: dict[str, list[TypicalCase]] = {"high": [], "mid": [], "low": []}
    for case in typical:
        grouped[case.bucket].append(case)
    for bucket in ("high", "mid", "low"):
        header = {
            "high": f"P(Pneumonia) ≥ {args.high_thres:.2f}",
            "mid": f"{args.low_thres:.2f} ≤ P(Pneumonia) < {args.high_thres:.2f}",
            "low": f"P(Pneumonia) < {args.low_thres:.2f}",
        }[bucket]
        lines.append(f"### {header}")
        lines.append("")
        for i, case in enumerate(grouped[bucket], start=1):
            lines.append(
                f"**Case {i}** — ground truth: `{case.ground_truth}`  "
                f"({_fmt_probs([case.p_pneumonia, case.p_influenza, case.p_other])})"
            )
            lines.append("")
            lines.append(f"- Age bucket: {case.age_bucket}, sex: {case.sex}")
            lines.append(f"- Chief complaint: {case.chief_complaint}")
            lines.append(f"- Evidences ({len(case.evidences)}):")
            for ev in case.evidences:
                lines.append(f"    - {ev}")
            lines.append("")

    # Section 2
    lines.append("## Section 2 — Synthetic enumeration (controlled sweep)")
    lines.append("")
    lines.append(
        f"Chief complaint fixed to `{args.chief_complaint_ev}`. Enumerating "
        f"2^{len(synthetic_axes)} = {1 << len(synthetic_axes)} yes/no combos "
        f"on the top-{len(synthetic_axes)} high-gain binary evidences:"
    )
    lines.append("")
    for ev in synthetic_axes:
        lines.append(f"- {ev}")
    lines.append("")
    syn_grouped: dict[str, list[SyntheticCase]] = {"high": [], "mid": [], "low": []}
    for case in synthetic:
        syn_grouped[case.bucket].append(case)
    for bucket in ("high", "mid", "low"):
        header = {
            "high": f"Combos with P(Pne) ≥ {args.high_thres:.2f}",
            "mid": f"Combos with {args.low_thres:.2f} ≤ P(Pne) < {args.high_thres:.2f}",
            "low": f"Combos with P(Pne) < {args.low_thres:.2f}",
        }[bucket]
        lines.append(f"### {header}")
        lines.append("")
        if not syn_grouped[bucket]:
            lines.append("_(no combos landed in this band — sweep may be uniform)_")
            lines.append("")
            continue
        for i, case in enumerate(syn_grouped[bucket], start=1):
            yeses = [ev for ev, a in case.answers.items() if a == "yes"]
            nos = [ev for ev, a in case.answers.items() if a == "no"]
            lines.append(
                f"**Combo {i}** — "
                f"{_fmt_probs([case.p_pneumonia, case.p_influenza, case.p_other])}"
            )
            lines.append("")
            lines.append(f"- YES: {', '.join(yeses) if yeses else '(none)'}")
            lines.append(f"- NO: {', '.join(nos) if nos else '(none)'}")
            lines.append("")

    # Section 3
    lines.append("## Section 3 — Interactive replay verification")
    lines.append("")
    by_truth: dict[str, list[ReplayTrace]] = {"Pneumonia": [], "Influenza": []}
    for tr in replay:
        by_truth.setdefault(tr.ground_truth, []).append(tr)
    for truth, traces in by_truth.items():
        if not traces:
            continue
        correct = sum(1 for tr in traces if tr.correct)
        lines.append(
            f"- **{truth}**: {correct}/{len(traces)} correct "
            f"({100 * correct / max(1, len(traces)):.1f}%)"
        )
    lines.append("")
    lines.append("### Per-patient traces (first 5 per class)")
    lines.append("")
    for truth in ("Pneumonia", "Influenza"):
        lines.append(f"#### Ground truth: {truth}")
        lines.append("")
        for tr in by_truth.get(truth, [])[:5]:
            hit = "✓" if tr.correct else "✗"
            lines.append(
                f"**Patient #{tr.patient_idx}** — final argmax `{tr.final_argmax}` {hit}  "
                f"({_fmt_probs(tr.final_probs)}), "
                f"stopped at step {tr.stopped_at_step if tr.stopped_at_step is not None else 'maxstep'}"
            )
            lines.append("")
            for step in tr.steps:
                lines.append(
                    f"  - Q{step.step} `{step.question_ev}`: {step.question_text} "
                    f"→ **{step.answer}** → {_fmt_probs(step.probs)}"
                )
            lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _dump_jsonl(path: pathlib.Path, items: list[Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI entry.
# ---------------------------------------------------------------------------


def _find_default_chief_complaint(schema: dict, meta: dict[str, dict]) -> str:
    """Pick the first schema binary evidence whose question mentions 'cough'.

    Falls back to schema-order first binary if no match — gives the CLI
    a sensible default so ``--chief-complaint-ev`` is not required, but
    the operator can override for other scenarios.
    """
    for ev in schema["evs"]:
        if ev["dtype"] != "B":
            continue
        q = meta.get(ev["name"], {}).get("question_en", "").lower()
        if "cough" in q:
            return ev["name"]
    for ev in schema["evs"]:
        if ev["dtype"] == "B":
            return ev["name"]
    raise RuntimeError("schema has no binary evidences — cannot pick chief complaint")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--weights", type=pathlib.Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    ap.add_argument("--n-per-class", type=int, default=DEFAULT_N_PER_CLASS)
    ap.add_argument("--per-bucket", type=int, default=DEFAULT_PER_BUCKET)
    ap.add_argument("--high-thres", type=float, default=DEFAULT_HIGH_THRES)
    ap.add_argument("--low-thres", type=float, default=DEFAULT_LOW_THRES)
    ap.add_argument("--maxstep", type=int, default=DEFAULT_MAXSTEP)
    ap.add_argument("--stop-thres", type=float, default=DEFAULT_STOP_THRES)
    ap.add_argument("--enum-top-k", type=int, default=DEFAULT_ENUM_TOP_K)
    ap.add_argument(
        "--chief-complaint-ev",
        default=None,
        help="Binary evidence id to fix as 'yes' across the synthetic sweep. "
        "Defaults to the first cough-related binary evidence in the schema.",
    )
    ap.add_argument("--age-bucket", type=int, default=3)
    ap.add_argument("--sex", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_everything(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    schema = load_evidence_schema(args.data_dir)
    meta = load_evidence_meta(args.data_dir)
    agent = XgbAgent.load(args.weights, schema)
    n_classes = int(agent.classifier.classes_.shape[0])
    if n_classes != len(TARGET_NAMES) + 1:
        raise SystemExit(
            f"expected v3 native model (n_classes={len(TARGET_NAMES) + 1}), "
            f"got n_classes={n_classes}. This script is v3-specific."
        )
    columns, _labels, columns_idx = feature_columns_from_schema(schema, meta)
    # Override agent thresholds with the CLI-declared operating point.
    agent.thres = args.stop_thres

    full_pidx, _sev = load_pidx(args.data_dir, whitelist=None)
    for name in TARGET_NAMES:
        if name not in full_pidx:
            raise SystemExit(
                f"target {name!r} not in DDXPlus corpus — check --data-dir"
            )
    patients = load_patients(args.data_dir, args.n_test, "test", schema, full_pidx)
    _relabel_patients_to_subset(patients, full_pidx)
    print(f"scored {len(patients)} test patients")

    if args.chief_complaint_ev is None:
        args.chief_complaint_ev = _find_default_chief_complaint(schema, meta)
        print(f"chief complaint auto-picked: {args.chief_complaint_ev}")

    # Section 1
    print("running section 1 — real-patient bucketing...")
    typical, band_counts = section_typical_cases(
        agent,
        patients,
        schema,
        meta,
        columns_idx,
        args.high_thres,
        args.low_thres,
        args.per_bucket,
    )
    _dump_jsonl(args.out_dir / "typical.jsonl", typical)

    # Section 2
    print("running section 2 — synthetic enumeration...")
    synthetic, synthetic_axes = section_synthetic_enumeration(
        agent,
        schema,
        columns_idx,
        args.weights,
        args.high_thres,
        args.low_thres,
        args.per_bucket,
        args.chief_complaint_ev,
        args.age_bucket,
        args.sex,
        args.enum_top_k,
    )
    _dump_jsonl(args.out_dir / "synthetic.jsonl", synthetic)

    # Section 3
    print(
        f"running section 3 — interactive replay ({args.n_per_class} per class, "
        f"maxstep={args.maxstep})..."
    )
    replay = section_interactive_replay(
        agent,
        patients,
        schema,
        meta,
        args.n_per_class,
        args.maxstep,
        args.seed,
    )
    _dump_jsonl(args.out_dir / "replay.jsonl", replay)

    _render_report(
        args.out_dir / "report.md",
        typical,
        band_counts,
        synthetic,
        synthetic_axes,
        replay,
        args,
        total_patients_scored=len(patients),
    )
    print(f"wrote report to {args.out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
