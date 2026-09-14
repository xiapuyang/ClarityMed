"""Pneumonia-focused DDXPlus subset selector via baseline confusion matrix.

Reads a trained 49-class typed-BASD checkpoint, runs inference on the
validate split, extracts the target disease's confusion row+column, and
emits a whitelist YAML consumed by second-round subset training.

Selection is deliberately baseline-confusion-driven — symptom Jaccard /
IDF / ICD proximity are all proxies. The model's actual decision surface
is the ground truth for "what confuses my classifier"; this script
reads it out and combines it with a fixed clinical cannot-miss list
(so classes the baseline already separates well are still trained on
in the subset, avoiding a 15-class model that has never seen PE).

See ``docs/symptoms-pneumonia-subset-selection.md`` for the full
methodology + output-YAML schema. Companion to
``ablate_maxstep.py`` — same checkpoint-loading pattern.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from claritymed.core.device import resolve_device
from claritymed.ingest.symptoms.ddxplus.schema import (
    DDXPLUS_SPLIT_FILES,
    load_evidence_schema,
    load_patients,
    load_pidx,
)
from claritymed.ingest.symptoms.typed_basd import (
    TypedEnv,
    build_basd,
    seed_everything,
)

DEFAULT_TARGET = "Pneumonia"
DEFAULT_EVAL_N = 20_000
DEFAULT_GAMES = 200
DEFAULT_MAXSTEP = 18
DEFAULT_TOP_CONFUSED = 8
DEFAULT_MIN_SAMPLES_TRAIN = 5_000
DEFAULT_MAX_SAMPLES_RATIO = 2.0
DEFAULT_COMBINED_FLOOR = 0.01
DEFAULT_STOP_MODE = "heuristic"
DEFAULT_STOP_THRES = 0.1

# Clinical cannot-miss list — see docs/symptoms-pneumonia-subset-selection.md
# Step 4. These stay in the subset regardless of confusion ranking because
# missing one has decision-boundary consequences the baseline hides (the
# baseline already learned to separate them; a subset that drops them
# would silently regress).
CANNOT_MISS_PNEUMONIA = {
    "Pulmonary embolism": "anticoagulation_vs_antibiotics_pathway",
    "Tuberculosis": "chronic_mimic_public_health_reporting",
    "Acute COPD exacerbation / infection": "steroid_pathway_SOB_cough",
    "Bronchospasm / acute asthma exacerbation": "bronchodilator_pathway",
    "Spontaneous pneumothorax": "chest_tube_acute_chest_pain_SOB",
    "Acute pulmonary edema": "cardiac_vs_infectious_diuretic_vs_abx",
}


def _load_agent(weights_path: Path, env: TypedEnv, n_dis: int, device: str) -> Any:
    """Rebuild an agent from checkpoint; infer hidden from tensor shape."""
    import torch

    state = torch.load(weights_path, map_location=device)
    hidden = state["trunk"]["0.weight"].shape[0]
    stop_mode = state.get("mode", DEFAULT_STOP_MODE)
    stop_thres = state.get("thres", DEFAULT_STOP_THRES)
    agent = build_basd(
        env,
        n_dis=n_dis,
        hidden=hidden,
        lr=1e-4,
        device=device,
        stop_thres=stop_thres,
        stop_mode=stop_mode,
    )
    agent.trunk.load_state_dict(state["trunk"])
    agent.sym.load_state_dict(state["sym"])
    agent.patho.load_state_dict(state["patho"])
    if agent.stop is not None and state.get("stop") is not None:
        agent.stop.load_state_dict(state["stop"])
    agent.temp = state.get("temp", agent.temp)
    return agent


def _collect_confusion(
    env: TypedEnv, agent: Any, n_dis: int, games: int, maxstep: int
) -> np.ndarray:
    """Run interactive inference; return [n_dis, n_dis] confusion counts."""
    env.reset()
    env.order = np.arange(len(env.patients))
    conf = np.zeros((n_dis, n_dis), dtype=np.int64)
    while env.idx + games <= len(env.patients):
        s, _ = env.initialize_state(games)
        done = agent.should_stop(s)
        for _ in range(maxstep):
            a = agent.next_action(s)
            s = env.reveal(s, a, done)
            done = done | agent.should_stop(s)
            if done.all():
                break
        pred, _ = agent.diagnose(s)
        for t, p in zip(env.disease.tolist(), pred.tolist()):
            conf[t, p] += 1
    return conf


def _train_sample_counts(data_dir: Path) -> dict[str, int]:
    """Count patients per pathology in the train split (cheap PATHOLOGY-only scan)."""
    import pandas as pd

    path = data_dir / DDXPLUS_SPLIT_FILES["train"]
    if not path.exists():
        raise FileNotFoundError(f"missing train zip: {path}")
    counts = pd.read_csv(path, usecols=["PATHOLOGY"])["PATHOLOGY"].value_counts()
    return counts.to_dict()


def _top_confused(conf: np.ndarray, target_id: int, k: int, floor: float) -> list[dict]:
    """Return top-k confused peers with recall+precision decomposition."""
    row_sum = conf[target_id, :].sum()
    col_sum = conf[:, target_id].sum()
    if row_sum == 0 or col_sum == 0:
        raise RuntimeError(
            "target class never appeared as true or predicted — increase --eval-n"
        )
    recall = conf[target_id, :] / row_sum
    precision = conf[:, target_id] / col_sum
    combined = recall + precision
    combined[target_id] = -1.0  # exclude the target itself
    order = np.argsort(-combined)
    picks: list[dict] = []
    for j in order:
        if combined[j] < floor:
            break
        if len(picks) >= k:
            break
        picks.append(
            {
                "id": int(j),
                "recall_share": float(recall[j]),
                "precision_share": float(precision[j]),
                "combined_score": float(combined[j]),
            }
        )
    return picks


def _apply_sample_guardrails(
    diseases: list[dict],
    counts: dict[str, int],
    target_name: str,
    min_train: int,
    max_ratio: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split into (kept, dropped_low, capped_high). Mutates each dict's sampling_cap."""
    target_n = counts.get(target_name, 0)
    cap = int(target_n * max_ratio) if target_n else None
    kept: list[dict] = []
    dropped: list[dict] = []
    capped: list[dict] = []
    for d in diseases:
        n = counts.get(d["name"], 0)
        d["train_samples"] = n
        if (
            n < min_train
            and d["name"] != target_name
            and d.get("tier") != "cannot_miss"
        ):
            d["reason"] = "below_min_samples_train"
            dropped.append(d)
            continue
        if cap and n > cap and d["name"] != target_name:
            d["sampling_cap"] = cap
            capped.append(
                {
                    "name": d["name"],
                    "train_samples": n,
                    "sampling_cap": cap,
                    "reason": "exceeds_max_samples_ratio",
                }
            )
        else:
            d["sampling_cap"] = None
        kept.append(d)
    return kept, dropped, capped


def _build_yaml(
    target_name: str,
    target_id: int,
    kept: list[dict],
    dropped: list[dict],
    capped: list[dict],
    id2name: dict[int, str],
    args: argparse.Namespace,
) -> dict:
    """Assemble the whitelist YAML structure per docs schema."""
    tiers: dict[str, list[dict]] = {
        "target": [],
        "from_confusion": [],
        "cannot_miss": [],
    }
    for d in kept:
        entry = {
            "name": d["name"],
            "train_samples": d["train_samples"],
            "sampling_cap": d.get("sampling_cap"),
        }
        if d["name"] == target_name:
            tiers["target"].append(entry)
            continue
        # Attach confusion metrics when available.
        for k in ("recall_share", "precision_share", "combined_score"):
            if k in d:
                entry[k] = round(d[k], 4)
        if d.get("tier") == "cannot_miss":
            entry["reason"] = CANNOT_MISS_PNEUMONIA.get(d["name"], "clinical_priority")
            tiers["cannot_miss"].append(entry)
        else:
            tiers["from_confusion"].append(entry)
    return {
        "version": 1,
        "target": target_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "baseline_weights": str(args.weights),
            "data_dir": str(args.data_dir),
            "n_eval_patients": args.eval_n,
            "eval_split": "validate",
        },
        "selection_params": {
            "target_disease": target_name,
            "top_confused_n": args.top_confused_n,
            "min_samples_train": args.min_samples_train,
            "max_samples_ratio": args.max_samples_ratio,
            "combined_score_floor": args.combined_score_floor,
        },
        "diseases": tiers,
        "dropped_low_sample": dropped,
        "capped_high_sample": capped,
    }


def _print_report(bundle: dict) -> None:
    """Print a markdown summary to stdout for quick eyeballing."""
    print()
    print(f"# Subset for target: {bundle['target']}\n")
    tiers = bundle["diseases"]
    total = sum(len(v) for v in tiers.values())
    print(
        f"Kept **{total}** classes (target + {len(tiers['from_confusion'])} "
        f"from-confusion + {len(tiers['cannot_miss'])} cannot-miss)\n"
    )
    print(
        "| tier | name | train_samples | recall_share | precision_share | combined | cap |"
    )
    print(
        "|------|------|---------------|--------------|-----------------|----------|-----|"
    )
    for tier_name in ("target", "from_confusion", "cannot_miss"):
        for d in tiers[tier_name]:
            cap = d.get("sampling_cap") or "-"
            r = d.get("recall_share", "-")
            p = d.get("precision_share", "-")
            c = d.get("combined_score", "-")
            print(
                f"| {tier_name} | {d['name']} | {d['train_samples']} | "
                f"{r} | {p} | {c} | {cap} |"
            )
    if bundle["dropped_low_sample"]:
        print(f"\n**Dropped** ({len(bundle['dropped_low_sample'])}):")
        for d in bundle["dropped_low_sample"]:
            print(f"- {d['name']} (n={d['train_samples']}) — {d['reason']}")
    if bundle["capped_high_sample"]:
        print(f"\n**Capped** ({len(bundle['capped_high_sample'])}):")
        for d in bundle["capped_high_sample"]:
            print(f"- {d['name']} (n={d['train_samples']}) → cap {d['sampling_cap']}")


def main() -> None:
    """CLI: ``uv run python scripts/select_ddxplus_subset.py ...``."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument(
        "--weights",
        required=True,
        type=Path,
        help="Trained 49-class baseline weights.pt from train.py.",
    )
    ap.add_argument(
        "--out", required=True, type=Path, help="Output whitelist YAML path."
    )
    ap.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        help="Target disease name (must match release_conditions.json).",
    )
    ap.add_argument(
        "--eval-n",
        type=int,
        default=DEFAULT_EVAL_N,
        help="Validate patients to load. Below 5k the confusion "
        "row for the target class becomes noisy.",
    )
    ap.add_argument(
        "--games",
        type=int,
        default=DEFAULT_GAMES,
        help="Batch size for interactive inference.",
    )
    ap.add_argument("--maxstep", type=int, default=DEFAULT_MAXSTEP)
    ap.add_argument("--top-confused-n", type=int, default=DEFAULT_TOP_CONFUSED)
    ap.add_argument("--min-samples-train", type=int, default=DEFAULT_MIN_SAMPLES_TRAIN)
    ap.add_argument(
        "--max-samples-ratio", type=float, default=DEFAULT_MAX_SAMPLES_RATIO
    )
    ap.add_argument(
        "--combined-score-floor", type=float, default=DEFAULT_COMBINED_FLOOR
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.weights.exists():
        print(f"weights not found: {args.weights}", file=sys.stderr)
        raise SystemExit(2)
    if args.eval_n < 5_000:
        print(
            f"WARNING: --eval-n {args.eval_n} < 5000; confusion row will be noisy",
            file=sys.stderr,
        )

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit("torch not installed") from exc

    seed_everything(args.seed)
    device = resolve_device(args.device)
    schema = load_evidence_schema(args.data_dir)
    pidx, sev = load_pidx(args.data_dir)
    n_dis = len(pidx)
    id2name = {i: n for n, i in pidx.items()}

    if args.target not in pidx:
        raise SystemExit(
            f"target {args.target!r} not in release_conditions.json. "
            f"Available: {sorted(pidx)[:5]}..."
        )
    target_id = pidx[args.target]

    print(f"loading {args.eval_n} validate patients …", file=sys.stderr)
    val_pats = load_patients(args.data_dir, args.eval_n, "validate", schema, pidx)
    env = TypedEnv(val_pats, schema, n_dis)
    print(f"loading baseline weights: {args.weights}", file=sys.stderr)
    agent = _load_agent(args.weights, env, n_dis, device)

    print(
        f"running inference on {len(val_pats)} patients "
        f"(games={args.games}, maxstep={args.maxstep}) …",
        file=sys.stderr,
    )
    conf = _collect_confusion(env, agent, n_dis, args.games, args.maxstep)
    target_row_n = int(conf[target_id, :].sum())
    target_col_n = int(conf[:, target_id].sum())
    print(f"target seen: true={target_row_n} pred={target_col_n}", file=sys.stderr)

    picks = _top_confused(
        conf, target_id, args.top_confused_n, args.combined_score_floor
    )

    # Assemble the pre-guardrail disease list.
    diseases: list[dict] = [{"name": args.target}]
    for p in picks:
        p["name"] = id2name[p["id"]]
        diseases.append(p)
    for name in CANNOT_MISS_PNEUMONIA:
        if (
            name in pidx
            and name != args.target
            and name not in {d["name"] for d in diseases}
        ):
            j = pidx[name]
            entry = {
                "id": j,
                "name": name,
                "tier": "cannot_miss",
                "recall_share": float(conf[target_id, j] / max(target_row_n, 1)),
                "precision_share": float(conf[j, target_id] / max(target_col_n, 1)),
            }
            entry["combined_score"] = entry["recall_share"] + entry["precision_share"]
            diseases.append(entry)

    counts = _train_sample_counts(args.data_dir)
    kept, dropped, capped = _apply_sample_guardrails(
        diseases,
        counts,
        args.target,
        args.min_samples_train,
        args.max_samples_ratio,
    )

    bundle = _build_yaml(args.target, target_id, kept, dropped, capped, id2name, args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        yaml.safe_dump(bundle, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"wrote {args.out}", file=sys.stderr)
    _print_report(bundle)


if __name__ == "__main__":
    main()
