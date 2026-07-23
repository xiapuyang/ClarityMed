"""Production-config eval for the active symptoms model.

Runs the same interactive BASD loop production uses (config-driven maxstep,
stop policy, IG overrides, antecedent penalty), then reports:

Per-class (Pneumonia / Influenza / Other):
  * precision, recall, F1
  * IL — mean questions asked, grouped by TRUE class

Dataset-level calibration:
  * ECE — Expected Calibration Error on top-label confidence, 10 equal-width bins
  * Brier — multi-class Brier score, ``mean(sum_c (p_c - y_c)^2)`` where ``y``
    is one-hot ground truth
  * Brier R² — skill score vs the empirical-prior baseline: ``1 - Brier(model) /
    Brier(prior)``. 1 = perfect, 0 = no better than always predicting the class
    frequencies, negative = worse than the prior.

Class order is fixed by the model checkpoint's ``classifier.classes_`` and by
``configs/symptoms.yaml`` (target_condition_ids first, then Other). For
``xgb_pne_inf_v5_recallig`` that is ``[Pneumonia, Influenza, Other]``.

Usage::

    uv run python scripts/eval_symptoms_metrics.py                    # defaults
    uv run python scripts/eval_symptoms_metrics.py --n-test 5000      # faster
    uv run python scripts/eval_symptoms_metrics.py --output eval.json # custom path

Output: pretty stdout table + JSON dump to ``data/bench/symptoms/eval_metrics_<ts>.json``.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import yaml

from claritymed.core.symptoms.datasets.canonical import slugify_condition
from claritymed.ingest.symptoms.ddxplus.schema import (
    load_evidence_schema,
    load_patients,
    load_pidx,
)
from claritymed.ingest.symptoms.typed_basd import TypedEnv, build_basd, seed_everything
from claritymed.ingest.symptoms.xgb.algorithm import XgbAgent

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "symptoms.yaml"
DEFAULT_DATA_DIR = pathlib.Path.home() / ".claritymed" / "data" / "symptoms" / "ddxplus"
DEFAULT_MODELS_ROOT = pathlib.Path.home() / ".claritymed" / "models" / "symptoms"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "bench" / "symptoms"
DEFAULT_MODEL_ID = "xgb_pne_inf_v5_recallig"
DEFAULT_DATASET_ID = "ddxplus_pneumonia_flu"
DEFAULT_N_TEST = 20_000
DEFAULT_SEED = 42
ECE_BINS = 10


# ---------------------------------------------------------------------------
# Config loading — replicate production overrides without pulling in the
# full pydantic ModelSpec surface (the ddxplus adapter does manifest / SHA
# / whitelist verification we don't need for a local eval).
# ---------------------------------------------------------------------------


@dataclass
class ResolvedModel:
    """Everything the interactive loop needs from configs/symptoms.yaml."""

    model_id: str
    algorithm_module: str  # "xgb" | "typed_basd"
    weights_subpath: str
    class_names: list[str]  # ordered: targets first, then "Other"
    maxstep: int
    stop_thres: float
    patho_temp: float
    target_condition_ids: list[str]  # names in the dataset's pidx space
    # xgb-only knobs — ignored for typed_basd.
    antecedent_penalty: float | None = None
    ig_recall_weight: float | None = None
    ig_recall_mode: str | None = None
    stop_policy: str | None = None
    stop_target_thres: float | None = None
    stop_other_thres: float | None = None


def _resolve_model(model_id: str, dataset_id: str) -> ResolvedModel:
    """Look up the model + its dataset in ``configs/symptoms.yaml``.

    Fails loud on missing fields — the goal is to eval production behavior,
    not a partially-defaulted approximation.
    """
    with CONFIG_PATH.open() as fh:
        cfg = yaml.safe_load(fh)
    dataset_specs = {d["id"]: d for d in cfg["datasets"]}
    if dataset_id not in dataset_specs:
        raise SystemExit(f"dataset {dataset_id!r} not in {list(dataset_specs)}")
    dataset = dataset_specs[dataset_id]
    target_ids: list[str] = list(dataset["target_condition_ids"])
    if model_id not in dataset["model_ids"]:
        # Warn but proceed — allows A/B testing checkpoints (e.g. an
        # experimental calibrated variant) without wiring them into the
        # production dataset entry, which would force the server to load
        # them at startup.
        print(
            f"[warn] model {model_id!r} not in dataset {dataset_id!r}.model_ids; "
            f"proceeding for A/B eval (production chat still uses "
            f"{dataset['model_ids']})",
            file=sys.stderr,
        )
    model_specs = {m["id"]: m for m in cfg["models"]}
    if model_id not in model_specs:
        raise SystemExit(f"model {model_id!r} not in models[] of {CONFIG_PATH}")
    m = model_specs[model_id]
    algo = m.get("algorithm_module")
    if algo not in {"xgb", "typed_basd"}:
        raise SystemExit(
            f"model {model_id!r} algorithm_module={algo!r}; supported: xgb, typed_basd"
        )
    class_names = [n.capitalize() for n in target_ids] + ["Other"]
    # xgb has additional IG + stop-policy knobs; typed_basd is neural-net
    # and reads its stop threshold as a symptom-prob cutoff instead
    # (KTD-D3: same field, opposite direction — see servers/symptoms/loader.py).
    common = dict(
        model_id=model_id,
        algorithm_module=algo,
        weights_subpath=m["weights_subpath"],
        class_names=class_names,
        maxstep=int(m["maxstep"]),
        stop_thres=float(m["stop_thres"]),
        patho_temp=float(m.get("patho_temp", 1.0)),
        target_condition_ids=target_ids,
    )
    if algo == "xgb":
        return ResolvedModel(
            **common,
            antecedent_penalty=float(m["antecedent_penalty"]),
            ig_recall_weight=float(m["ig_recall_weight"]),
            ig_recall_mode=str(m["ig_recall_mode"]),
            stop_policy=str(m["stop_policy"]),
            stop_target_thres=float(m["stop_target_thres"]),
            stop_other_thres=float(m["stop_other_thres"]),
        )
    return ResolvedModel(**common)


def _project_probs(
    probs_full: np.ndarray, target_idxs: list[int], n_classes: int
) -> np.ndarray:
    """Reduce ``(batch, n_classes)`` posterior to ``(batch, N+1)``.

    ``target_idxs[k]`` becomes projected class ``k``; the trailing column
    is the sum over all non-target classes ('Other'). Sum-to-1 is
    preserved because we're summing a disjoint partition of the raw
    class space. Copied from scripts/eval_xgb_subset_metrics.py so both
    scripts stay independent — no cross-import.
    """
    if probs_full.ndim == 1:
        probs_full = probs_full[np.newaxis, :]
    n_targets = len(target_idxs)
    out = np.zeros((probs_full.shape[0], n_targets + 1), dtype=probs_full.dtype)
    for k, idx in enumerate(target_idxs):
        out[:, k] = probs_full[:, idx]
    other_mask = np.ones(n_classes, dtype=bool)
    other_mask[list(target_idxs)] = False
    out[:, n_targets] = probs_full[:, other_mask].sum(axis=1)
    return out


class _ProjectingAgent:
    """Wrap a native-49-class agent, expose 3-class diagnose() to callers.

    ``next_action`` / ``should_stop`` pass through — question selection
    and termination stay in the base agent's native space (which is
    correct: BASD's next_action picks the highest-prob unresolved
    symptom; that's independent of the projection).

    ``diagnose`` returns ``(hard_pred, soft_probs)`` where:

    * ``hard_pred``: **argmax-then-bucket** on the native 49-class
      posterior. Picks the single most-likely disease first, then
      buckets it into ``(target..., Other)``. Reflects how a deployed
      49-class model would route a decision to the 3-class UI — the
      model's own top pick wins. Avoids the systematic "Other beats
      any single target" bias that sum-projected argmax suffers from
      when 47 non-target competitors each hold a small slice of mass.
    * ``soft_probs``: sum-projected 3-class probabilities. Non-target
      classes are summed into Other, sum-to-1 preserved. Used for ECE
      / Brier / Brier R² — those metrics want a full calibrated 3-way
      distribution.

    The two can disagree (a patient may have hard_pred=Pne while
    soft_probs.argmax()=Other). That's intentional: they measure
    different things.
    """

    def __init__(self, base_agent, target_idxs: list[int], n_classes: int) -> None:
        self._base = base_agent
        self._target_idxs = target_idxs
        self._n_classes = n_classes
        self._target_set = set(target_idxs)
        self._other_bucket_idx = len(target_idxs)  # 3-class layout: [..., Other]
        # Map native class idx -> bucket idx (targets first, then Other)
        self._bucket_of = {t: k for k, t in enumerate(target_idxs)}

    def diagnose(self, state):
        _, probs_full = self._base.diagnose(state)
        native_argmax = probs_full.argmax(axis=1)
        hard_pred = np.array(
            [
                self._bucket_of.get(int(c), self._other_bucket_idx)
                for c in native_argmax
            ],
            dtype=np.int64,
        )
        probs_proj = _project_probs(probs_full, self._target_idxs, self._n_classes)
        return hard_pred, probs_proj

    def next_action(self, state):
        return self._base.next_action(state)

    def should_stop(self, state):
        return self._base.should_stop(state)


def _load_xgb_agent(resolved: ResolvedModel, schema: dict, models_root: pathlib.Path):
    """Load XGBoost checkpoint + apply serve-time overrides.

    Mirrors ``ingest/symptoms/ddxplus/adapter.py::_apply_xgb_ig_overrides``
    plus ``apply_model_overrides`` from ``servers/symptoms/loader.py``.
    """
    weights_path = models_root / resolved.weights_subpath / "weights.pkl"
    if not weights_path.exists():
        raise SystemExit(f"weights not found: {weights_path}")
    agent = XgbAgent.load(weights_path, schema)

    n_target = len(resolved.target_condition_ids)
    n_classes = int(agent.classifier.classes_.shape[0])
    if n_classes != n_target + 1:
        raise SystemExit(
            f"xgb checkpoint has {n_classes} classes but config declares "
            f"{n_target} targets + Other = {n_target + 1}. Refusing to eval "
            f"a legacy 49-class xgb model here — this script requires native "
            f"subset xgb."
        )

    agent.thres = resolved.stop_thres
    agent.antecedent_penalty = resolved.antecedent_penalty
    agent.ig_recall_weight = resolved.ig_recall_weight
    agent.ig_recall_mode = resolved.ig_recall_mode
    agent.stop_policy = resolved.stop_policy
    agent.variant_a_target_thres = resolved.stop_target_thres
    agent.target_sum_target_thres = resolved.stop_target_thres
    agent.variant_a_other_thres = resolved.stop_other_thres
    agent.target_sum_other_thres = resolved.stop_other_thres
    agent.target_class_idxs = list(range(n_target))
    return agent


def _load_typed_basd_agent(
    resolved: ResolvedModel,
    schema: dict,
    models_root: pathlib.Path,
    n_dis: int,
    target_full_idxs: list[int],
):
    """Load a torch typed_basd checkpoint and wrap it for projection.

    typed_basd_v2 is trained on the full 49-class DDXPlus corpus. To
    compare against the 3-class xgb baseline we project its posterior
    at diagnose time: target classes stay 1-to-1, the rest sum into
    Other. next_action / should_stop are untouched — question selection
    and stop logic operate in the native space.
    """
    import torch

    weights_path = models_root / resolved.weights_subpath / "weights.pt"
    if not weights_path.exists():
        raise SystemExit(f"weights not found: {weights_path}")

    # Follow servers/symptoms/loader.py::load_torch_agent: read hidden
    # width from the checkpoint so we don't have to duplicate the tune
    # sweep's hyperparameters in the eval script.
    device = "cpu"  # eval is single-shot; cpu avoids device-move overhead
    state = torch.load(weights_path, map_location=device, weights_only=True)
    hidden = state["trunk"]["0.weight"].shape[0]
    stop_thres_ckpt = state.get("thres", 0.1)
    # Read the checkpoint's native class count — do NOT assume the caller's
    # ``n_dis`` (which is len(full_pidx)==49). A native (N+1)-class subset
    # checkpoint stores a patho head of shape (N+1, hidden); building the
    # Agent with n_dis=49 triggers a state_dict size-mismatch on load.
    n_native = int(state["patho"]["weight"].shape[0])

    seed_env = TypedEnv([], schema, n_native)
    base = build_basd(
        seed_env,
        n_dis=n_native,
        hidden=hidden,
        lr=1e-4,
        device=device,
        stop_thres=stop_thres_ckpt,
        stop_mode="heuristic",
    )
    base.trunk.load_state_dict(state["trunk"])
    base.sym.load_state_dict(state["sym"])
    base.patho.load_state_dict(state["patho"])
    if base.stop is not None and state.get("stop") is not None:
        base.stop.load_state_dict(state["stop"])
    base.thres = resolved.stop_thres  # config override wins over checkpoint
    base.temp = resolved.patho_temp

    if n_native == len(resolved.target_condition_ids) + 1:
        # Native (N+1)-class subset — no projection needed.
        print(f"typed_basd is native {n_native}-class subset — no projection")
        return base
    print(
        f"typed_basd native class count = {n_native}; projecting "
        f"to {len(resolved.target_condition_ids)} targets + Other"
    )
    return _ProjectingAgent(base, target_full_idxs, n_native)


def _load_agent(
    resolved: ResolvedModel,
    schema: dict,
    models_root: pathlib.Path,
    n_dis: int,
    target_full_idxs: list[int],
):
    """Dispatch to per-algorithm loader; return an agent with the shared
    ``diagnose / next_action / should_stop`` interface."""
    if resolved.algorithm_module == "xgb":
        return _load_xgb_agent(resolved, schema, models_root)
    if resolved.algorithm_module == "typed_basd":
        return _load_typed_basd_agent(
            resolved, schema, models_root, n_dis, target_full_idxs
        )
    raise SystemExit(f"unhandled algorithm_module: {resolved.algorithm_module!r}")


# ---------------------------------------------------------------------------
# Interactive rollout — mirrors _mini_interactive from the reference eval,
# using the agent's own should_stop() so the stop_policy / thresholds we
# just wrote onto the agent are the source of truth.
# ---------------------------------------------------------------------------


def _rollout(
    agent,
    patients: list[dict],
    schema: dict,
    n_dis: int,
    maxstep: int,
    verify_target_idx: int | None = None,
    verify_turns: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the IG loop; return ``(final_probs [N, 3], il [N])``.

    ``il[i]`` counts the number of evidence-reveal steps for patient i.
    Patients that satisfy ``should_stop`` at step 0 (e.g. SapBERT init
    injection already pushes P(target) past the threshold in production)
    get IL=0 — matching how the server counts "questions asked".

    ``verify_target_idx`` enables post-hoc verification: after the main
    loop terminates, for every patient whose argmax landed on that
    target class, one extra IG-picked question is asked and the model
    re-diagnosed. Purpose is asymmetric precision boost on that class
    (targets a specific over-confidence pattern) without touching the
    main stop policy or requiring model retraining. IL is incremented
    by 1 for each verified patient regardless of whether the answer
    changed the argmax.
    """
    env = TypedEnv(list(patients), schema, n_dis)
    n = len(patients)
    state, _done_env = env.initialize_state(n)
    il = np.zeros(n, dtype=np.int64)
    stopped = np.zeros(n, dtype=bool)
    for _step in range(maxstep):
        stop_now = agent.should_stop(state)
        stopped |= stop_now
        if stopped.all():
            break
        next_evs = agent.next_action(state)
        state = env.reveal(state, next_evs, stopped)
        il[~stopped] += 1
    hard_pred, final_probs = agent.diagnose(state)
    final_probs = final_probs.astype(np.float64)
    hard_pred = np.asarray(hard_pred, dtype=np.int64)

    if verify_target_idx is not None:
        # Bias IG toward "target-vs-rest" discrimination — the default 3-way
        # projection picks questions optimal for Pne/Flu/Other simultaneously,
        # which after stop is rarely the best question to *refute* the current
        # target prediction. Restoring the original list at the end keeps the
        # agent stateless from the caller's perspective.
        original_target_idxs = getattr(agent, "target_class_idxs", None)
        agent.target_class_idxs = [verify_target_idx]

        il = il.copy()
        final_probs = final_probs.copy()
        hard_pred = hard_pred.copy()
        try:
            for _turn in range(verify_turns):
                verify_mask = hard_pred == verify_target_idx
                if not verify_mask.any():
                    break
                freeze = ~verify_mask
                next_evs = agent.next_action(state)
                state = env.reveal(state, next_evs, freeze)
                new_hard, new_probs = agent.diagnose(state)
                new_probs = new_probs.astype(np.float64)
                new_hard = np.asarray(new_hard, dtype=np.int64)
                final_probs[verify_mask] = new_probs[verify_mask]
                hard_pred[verify_mask] = new_hard[verify_mask]
                il[verify_mask] += 1
        finally:
            agent.target_class_idxs = original_target_idxs

    return final_probs, il, hard_pred


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass
class PerClassRow:
    name: str
    support: int
    precision: float
    recall: float
    f1: float
    il_mean: float
    il_median: float


@dataclass
class CalibrationRow:
    ece: float
    brier: float
    brier_baseline: float
    brier_r2: float


def _per_class_prf(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str], il: np.ndarray
) -> list[PerClassRow]:
    """Per-class precision/recall/F1 + IL grouped by true class.

    Multi-class one-vs-rest formulation. ``support`` is the count of
    ground-truth positives for the class. All zero-denominator cases
    return 0.0 (sklearn's ``zero_division=0`` convention) rather than
    NaN so the JSON output stays serializable and the intent is
    unambiguous — "we saw zero of this, so nothing to divide by".
    """
    rows: list[PerClassRow] = []
    for k, name in enumerate(class_names):
        tp = int(((y_pred == k) & (y_true == k)).sum())
        fp = int(((y_pred == k) & (y_true != k)).sum())
        fn = int(((y_pred != k) & (y_true == k)).sum())
        support = int((y_true == k).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        mask = y_true == k
        il_slice = il[mask]
        il_mean = float(il_slice.mean()) if il_slice.size else 0.0
        il_median = float(np.median(il_slice)) if il_slice.size else 0.0
        rows.append(
            PerClassRow(
                name=name,
                support=support,
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                il_mean=il_mean,
                il_median=il_median,
            )
        )
    return rows


def _top_label_ece(
    y_true: np.ndarray, probs: np.ndarray, n_bins: int = ECE_BINS
) -> float:
    """Multi-class ECE using the confidence of the predicted (top) class.

    Split ``max(probs)`` into ``n_bins`` equal-width buckets; each bucket
    contributes ``|acc_in_bin - mean_conf_in_bin| * (bin_size / N)``.
    Standard Guo et al. formulation.
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not mask.any():
            continue
        acc = correct[mask].mean()
        avg_conf = conf[mask].mean()
        ece += (mask.sum() / n) * abs(acc - avg_conf)
    return float(ece)


def _brier_multiclass(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Multi-class Brier: ``mean_i sum_c (p_ic - y_ic)^2`` with one-hot y."""
    n, k = probs.shape
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), y_true] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def _calibration(y_true: np.ndarray, probs: np.ndarray) -> CalibrationRow:
    """Bundle ECE + Brier + Brier R².

    Baseline for Brier R² is the empirical class prior: predict the
    training-set class frequencies for every sample. That's the standard
    "no-information" reference for Brier skill score.
    """
    ece = _top_label_ece(y_true, probs)
    brier = _brier_multiclass(y_true, probs)
    k = probs.shape[1]
    _, counts = np.unique(y_true, return_counts=True)
    prior = np.zeros(k, dtype=np.float64)
    for cls_idx, c in zip(*np.unique(y_true, return_counts=True)):
        prior[cls_idx] = c
    prior = prior / prior.sum() if prior.sum() > 0 else np.full(k, 1.0 / k)
    baseline_probs = np.broadcast_to(prior, probs.shape)
    brier_baseline = _brier_multiclass(y_true, baseline_probs)
    brier_r2 = 1.0 - (brier / brier_baseline) if brier_baseline > 0 else float("nan")
    return CalibrationRow(
        ece=ece,
        brier=brier,
        brier_baseline=float(brier_baseline),
        brier_r2=float(brier_r2),
    )


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    m = np.zeros((k, k), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_report(
    resolved: ResolvedModel,
    n_patients: int,
    per_class: list[PerClassRow],
    calib: CalibrationRow,
    confusion: np.ndarray,
    macro_f1: float,
    overall_acc: float,
    overall_il_mean: float,
    elapsed_s: float,
) -> None:
    print(f"\n=== {resolved.model_id} ===")
    print(f"  patients (test): {n_patients}")
    print(
        f"  config: maxstep={resolved.maxstep}  stop_thres={resolved.stop_thres}  "
        f"stop_policy={resolved.stop_policy}  target={resolved.stop_target_thres}  "
        f"other={resolved.stop_other_thres}"
    )
    print(
        f"  ig: recall_weight={resolved.ig_recall_weight}  "
        f"mode={resolved.ig_recall_mode}  antecedent_penalty={resolved.antecedent_penalty}"
    )
    print(f"  elapsed: {elapsed_s:.1f}s")

    print("\n  Per-class metrics:")
    header = (
        f"    {'class':<12s}  {'support':>7s}  {'prec':>6s}  {'recall':>6s}  "
        f"{'F1':>6s}  {'IL_mean':>8s}  {'IL_med':>7s}"
    )
    print(header)
    for row in per_class:
        print(
            f"    {row.name:<12s}  {row.support:>7d}  {row.precision:>6.3f}  "
            f"{row.recall:>6.3f}  {row.f1:>6.3f}  {row.il_mean:>8.2f}  "
            f"{row.il_median:>7.1f}"
        )
    print(f"    {'macro-F1':<12s}  {'—':>7s}  {'':>6s}  {'':>6s}  {macro_f1:>6.3f}")

    print("\n  Overall:")
    print(f"    accuracy: {overall_acc:.4f}")
    print(f"    IL mean:  {overall_il_mean:.2f}")

    print("\n  Calibration (dataset-level):")
    print(f"    ECE       : {calib.ece:.4f}")
    print(f"    Brier     : {calib.brier:.4f}")
    print(f"    Brier(pr) : {calib.brier_baseline:.4f}   [empirical-prior baseline]")
    print(f"    Brier R²  : {calib.brier_r2:+.4f}       [1=perfect, 0=prior, <0=worse]")

    print("\n  Confusion (rows=true, cols=pred):")
    labels = resolved.class_names
    print("    " + "".join(f"{n[:10]:>11s}" for n in labels))
    for i, row in enumerate(confusion):
        print(f"    {labels[i][:10]:<10s} " + "".join(f"{v:>11d}" for v in row))


def _dump_json(
    path: pathlib.Path,
    resolved: ResolvedModel,
    n_patients: int,
    per_class: list[PerClassRow],
    calib: CalibrationRow,
    confusion: np.ndarray,
    macro_f1: float,
    overall_acc: float,
    overall_il_mean: float,
    seed: int,
    elapsed_s: float,
) -> None:
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": asdict(resolved),
        "n_patients": n_patients,
        "seed": seed,
        "elapsed_s": round(elapsed_s, 2),
        "per_class": [asdict(r) for r in per_class],
        "macro_f1": macro_f1,
        "overall_accuracy": overall_acc,
        "overall_il_mean": overall_il_mean,
        "calibration": asdict(calib),
        "confusion_matrix": {
            "labels": resolved.class_names,
            "matrix": confusion.tolist(),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwrote JSON → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _target_slugs_to_pidx(
    target_slugs: list[str], full_pidx: dict[str, int]
) -> dict[str, int]:
    """Resolve config's slug-form target_condition_ids to raw pidx indices.

    ``full_pidx`` is keyed by DDXPlus display name (e.g. ``"Pneumonia"``);
    the config uses canonical slugs (``"pneumonia"``). Use the same
    ``slugify_condition`` helper the runtime uses so slug drift fails loud
    here rather than silently mismatching downstream.
    """
    slug_to_full_idx: dict[str, int] = {}
    for display, idx in full_pidx.items():
        slug_to_full_idx[slugify_condition(display)] = idx
    missing = [s for s in target_slugs if s not in slug_to_full_idx]
    if missing:
        raise SystemExit(
            f"target slugs {missing} have no pidx match. Available (first 20): "
            f"{sorted(slug_to_full_idx)[:20]}..."
        )
    return {slug: slug_to_full_idx[slug] for slug in target_slugs}


def _relabel_to_subset(
    patients: list[dict], slug_to_full_idx: dict[str, int], target_ids: list[str]
) -> None:
    """Rewrite each patient's ``d`` from full-pidx index to subset index.

    Targets take indices ``0..N-1`` in the order declared by
    ``target_condition_ids``; everything else becomes ``N`` (Other).
    Mutates in place because ``TypedEnv`` reads ``d`` directly.
    """
    mapping = {slug_to_full_idx[name]: k for k, name in enumerate(target_ids)}
    other_idx = len(target_ids)
    for p in patients:
        p["d"] = mapping.get(p["d"], other_idx)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    ap.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--models-root", type=pathlib.Path, default=DEFAULT_MODELS_ROOT)
    ap.add_argument(
        "--n-test",
        type=int,
        default=DEFAULT_N_TEST,
        help="Number of test patients to eval on. Full test split is ~130k.",
    )
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="JSON output path. Default: data/bench/symptoms/eval_metrics_<ts>.json",
    )
    ap.add_argument("--no-json", action="store_true", help="Skip JSON dump.")
    ap.add_argument(
        "--verify-class",
        default=None,
        help="Post-hoc verification: after main rollout, ask up to "
        "--verify-turns extra IG questions (target-vs-rest projection) "
        "for patients whose argmax equals this class name (e.g. 'Pneumonia'), "
        "re-diagnose after each. Precision-boost fix for an over-confident "
        "target without retraining. Class name must match "
        "resolved.class_names exactly.",
    )
    ap.add_argument(
        "--verify-turns",
        type=int,
        default=1,
        help="Max extra questions per verified patient (default 1). Loop "
        "stops early per-patient once argmax leaves the target class.",
    )
    args = ap.parse_args()

    seed_everything(args.seed)
    resolved = _resolve_model(args.model_id, args.dataset_id)
    print(f"model: {resolved.model_id}   classes: {resolved.class_names}")

    schema = load_evidence_schema(args.data_dir)
    full_pidx, _severity = load_pidx(args.data_dir, whitelist=None)
    slug_to_idx = _target_slugs_to_pidx(resolved.target_condition_ids, full_pidx)

    patients = load_patients(args.data_dir, args.n_test, "test", schema, full_pidx)
    print(f"loaded {len(patients)} test patients (requested {args.n_test})")
    _relabel_to_subset(patients, slug_to_idx, resolved.target_condition_ids)

    n_dis = len(full_pidx)
    target_full_idxs = [slug_to_idx[slug] for slug in resolved.target_condition_ids]
    agent = _load_agent(resolved, schema, args.models_root, n_dis, target_full_idxs)

    verify_idx: int | None = None
    if args.verify_class is not None:
        if args.verify_class not in resolved.class_names:
            raise SystemExit(
                f"--verify-class {args.verify_class!r} not in {resolved.class_names}"
            )
        verify_idx = resolved.class_names.index(args.verify_class)
        print(
            f"post-hoc verification enabled for class: {args.verify_class} (idx={verify_idx})"
        )

    t0 = time.perf_counter()
    probs, il, y_pred = _rollout(
        agent,
        patients,
        schema,
        n_dis,
        resolved.maxstep,
        verify_target_idx=verify_idx,
        verify_turns=args.verify_turns,
    )
    elapsed = time.perf_counter() - t0

    y_true = np.asarray([p["d"] for p in patients], dtype=np.int64)

    per_class = _per_class_prf(y_true, y_pred, resolved.class_names, il)
    calib = _calibration(y_true, probs)
    confusion = _confusion(y_true, y_pred, k=len(resolved.class_names))
    macro_f1 = float(np.mean([r.f1 for r in per_class]))
    overall_acc = float((y_pred == y_true).mean())
    overall_il_mean = float(il.mean())

    _print_report(
        resolved,
        len(patients),
        per_class,
        calib,
        confusion,
        macro_f1,
        overall_acc,
        overall_il_mean,
        elapsed,
    )

    if not args.no_json:
        out_path = args.output or (
            DEFAULT_OUT_DIR
            / f"eval_metrics_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
        )
        _dump_json(
            out_path,
            resolved,
            len(patients),
            per_class,
            calib,
            confusion,
            macro_f1,
            overall_acc,
            overall_il_mean,
            args.seed,
            elapsed,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
