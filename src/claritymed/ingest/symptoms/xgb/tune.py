"""Joint sweep of XGBoost-specific runtime knobs on a trained checkpoint.

Different sweep space than :mod:`.ddxplus.tune` — patho_temp is
meaningless for XGBoost (posterior calibration is applied at training
time via :class:`CalibratedClassifierCV`). We sweep:

* ``--maxsteps`` — per-session question budget cap (same semantics as
  typed-BASD).
* ``--stop-thres`` — max class prob at which to stop asking. This is
  the field with type-shift vs typed-BASD (see D3 in the plan): for
  XGBoost, higher = more confidence required = ask more questions.
* ``--ig-smoothings`` — additive smoothing on ``P(col=1 | state)`` in
  the IG policy. Non-zero smoothing prevents the IG rank from ignoring
  rare-positive columns whose training marginal is close to zero.

Selection reuses the same two-stage DSR-floor + IL-elbow gate as
:mod:`.ddxplus.tune`: eligible points require DSR ≥ 92 (or NaN when
the subset pidx has no severe diseases) and maxstep at or beyond the
IL saturation elbow. Score = ``DDF1 − IL_PENALTY · IL`` — one extra
question must buy at least IL_PENALTY DDF1 points to earn its keep.

Usage::

    uv run --extra symptoms-server claritymed-symptoms-xgb-tune-ddxplus \\
        --data-dir ~/.claritymed/data/symptoms/ddxplus \\
        --weights ~/.claritymed/models/symptoms/ddxplus/xgb_pneumonia_flu_v1/weights.pkl \\
        --maxsteps 4,6,8,10,12 \\
        --stop-thres 0.85,0.90,0.95,0.99 \\
        --ig-smoothings 0.0,0.05,0.10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from claritymed.ingest.symptoms.ddxplus.schema import (
    load_evidence_schema,
    load_patients,
    load_pidx,
)
from claritymed.ingest.symptoms.mlflow_utils import log_eval_metrics, symptom_run
from claritymed.ingest.symptoms.typed_basd import (
    TypedEnv,
    interactive_eval,
    seed_everything,
)
from claritymed.ingest.symptoms.xgb.algorithm import XgbAgent

DEFAULT_MAXSTEPS = "4,6,8,10,12"
DEFAULT_STOP_THRES = "0.85,0.90,0.95,0.99"
DEFAULT_IG_SMOOTHINGS = "0.0,0.05,0.10"
DSR_FLOOR = 92.0
SATURATION_RATE = 0.3
IL_PENALTY = 1.0


def main() -> None:
    """CLI entry — sweeps (maxstep × stop_thres × ig_smoothing) and logs to MLflow."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument(
        "--weights",
        required=True,
        type=Path,
        help="Path to weights.pkl produced by claritymed-symptoms-xgb-train-ddxplus.",
    )
    ap.add_argument("--maxsteps", default=DEFAULT_MAXSTEPS)
    ap.add_argument("--stop-thres", default=DEFAULT_STOP_THRES)
    ap.add_argument("--ig-smoothings", default=DEFAULT_IG_SMOOTHINGS)
    ap.add_argument("--games", type=int, default=None)
    ap.add_argument("--eval-n", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--diseases",
        default=None,
        help="Comma-separated disease names matching the subset the "
        "checkpoint was trained on. MUST match manifest.diseases_trained.",
    )
    ap.add_argument("--quick", action="store_true", help="100-patient smoke sweep.")
    args = ap.parse_args()

    if not args.weights.exists():
        print(
            f"weights not found: {args.weights} — train first via "
            f"`claritymed-symptoms-xgb-train-ddxplus`.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    maxsteps = [int(s) for s in args.maxsteps.split(",")]
    stop_thres_list = [float(s) for s in args.stop_thres.split(",")]
    smoothings = [float(s) for s in args.ig_smoothings.split(",")]

    whitelist: set[str] | None = None
    if args.diseases:
        whitelist = {d.strip() for d in args.diseases.split(",") if d.strip()}

    if args.quick:
        args.eval_n = 100

    if args.seed is not None:
        seed_everything(args.seed)

    schema = load_evidence_schema(args.data_dir)
    pidx, sev = load_pidx(args.data_dir, whitelist=whitelist)
    n_dis = len(pidx)
    if whitelist:
        print(f"[xgb-tune] subset pidx = {pidx}")
    test_pats = load_patients(args.data_dir, args.eval_n, "test", schema, pidx)

    agent = XgbAgent.load(args.weights, schema)
    games = args.games if args.games is not None else len(test_pats)

    import mlflow

    parent_params = {
        "maxsteps": args.maxsteps,
        "stop_thres": args.stop_thres,
        "ig_smoothings": args.ig_smoothings,
        "eval_n": args.eval_n,
        "games": games,
        "seed": args.seed,
        "weights": str(args.weights),
    }

    # results[(thres, smoothing)][maxstep] = EvalMetrics
    results: dict[tuple[float, float], dict[int, object]] = {}

    run_name = f"xgb-tune-{args.weights.parent.name}"
    with symptom_run(
        "ddxplus",
        run_name=run_name,
        run_type="xgb_tune",
        params=parent_params,
    ):
        for smoothing in smoothings:
            agent.ig_smoothing = smoothing
            for thres in stop_thres_list:
                agent.thres = thres
                results[(thres, smoothing)] = {}
                for maxstep in maxsteps:
                    child_name = f"th{thres}-sm{smoothing}-ms{maxstep}"
                    child_params = {
                        "stop_thres": thres,
                        "ig_smoothing": smoothing,
                        "maxstep": maxstep,
                    }
                    with symptom_run(
                        "ddxplus",
                        run_name=child_name,
                        run_type="xgb_tune_child",
                        params=child_params,
                        nested=True,
                    ):
                        env = TypedEnv(list(test_pats), schema, n_dis)
                        m = interactive_eval(
                            env,
                            agent,
                            maxstep=maxstep,
                            games=games,
                            severity=sev,
                        )
                        log_eval_metrics(m)
                        mlflow.log_metric("score", m.DDF1 - IL_PENALTY * m.IL)
                    results[(thres, smoothing)][maxstep] = m

    _print_results(stop_thres_list, smoothings, maxsteps, results)


def _elbow_maxstep(maxsteps: list[int], il_by_maxstep: dict[int, float]) -> int:
    """Return the smallest maxstep at the IL saturation elbow.

    Same semantics as :func:`.ddxplus.tune._elbow_maxstep`: walks sorted
    maxsteps and returns the first ms where the *next* step's marginal
    IL gain/step drops below :data:`SATURATION_RATE`. Falls back to the
    smallest maxstep with a warning when no saturation is detected.
    """
    sorted_ms = sorted(maxsteps)
    for i in range(len(sorted_ms) - 1):
        delta_ms = sorted_ms[i + 1] - sorted_ms[i]
        delta_il = il_by_maxstep[sorted_ms[i + 1]] - il_by_maxstep[sorted_ms[i]]
        if delta_il / delta_ms < SATURATION_RATE:
            return sorted_ms[i]
    print(
        f"[xgb-tune] warning: no IL saturation elbow found across maxsteps "
        f"{sorted_ms}. Stop gate is not firing naturally — falling back to "
        f"smallest maxstep ({sorted_ms[0]}). Inspect the table and pick "
        f"the point yourself if this recommendation looks off.",
        file=sys.stderr,
    )
    return sorted_ms[0]


def _print_results(
    stop_thres_list: list[float],
    smoothings: list[float],
    maxsteps: list[int],
    results: dict[tuple[float, float], dict[int, object]],
) -> None:
    print("\n## Joint tune results (DDXPlus / XGBoost)\n")
    print(
        f"{'thres':>6} | {'smth':>5} | {'ms':>4} | {'IL':>6} | {'DDR':>6} | "
        f"{'DDP':>6} | {'DDF1':>6} | {'DSR':>6} | {'score':>7} |"
    )
    print("-" * 76)

    # IL is invariant to smoothing/thres for a fixed maxstep only when the
    # stop gate never fires; more generally we compute the elbow using the
    # first (thres, smoothing) tuple's IL curve. Operators should re-inspect
    # the table when the elbow varies visibly across sweeps.
    first_key = (stop_thres_list[0], smoothings[0])
    il_by_maxstep = {ms: results[first_key][ms].IL for ms in maxsteps}
    elbow_ms = _elbow_maxstep(maxsteps, il_by_maxstep)

    qualifying: list[tuple[float, int, float, float]] = []
    for smoothing in smoothings:
        for thres in stop_thres_list:
            for maxstep in maxsteps:
                m = results[(thres, smoothing)][maxstep]
                dsr = m.DSR if not np.isnan(m.DSR) else float("nan")
                passes_dsr = np.isnan(dsr) or dsr >= DSR_FLOOR
                at_elbow = maxstep >= elbow_ms
                score = m.DDF1 - IL_PENALTY * m.IL
                flag = ""
                if passes_dsr:
                    flag += " *"
                if maxstep == elbow_ms:
                    flag += " ←elbow"
                print(
                    f"{thres:>6.2f} | {smoothing:>5.2f} | {maxstep:>4} | "
                    f"{m.IL:>6.2f} | {m.DDR:>6.2f} | {m.DDP:>6.2f} | "
                    f"{m.DDF1:>6.2f} | {dsr:>6.2f} | {score:>7.2f} |{flag}"
                )
                if passes_dsr and at_elbow:
                    qualifying.append((-score, maxstep, thres, smoothing))

    print(
        f"\n* = DSR ≥ {DSR_FLOOR}   ←elbow = IL saturation point"
        f"   score = DDF1 − {IL_PENALTY} × IL"
        f"   (SATURATION_RATE={SATURATION_RATE})"
    )
    if not qualifying:
        print(
            f"BLOCKED: no qualifying (DSR ≥ {DSR_FLOOR}) point at or beyond "
            f"the IL elbow (maxstep ≥ {elbow_ms}) — retrain required.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    qualifying.sort()
    neg_score, best_ms, best_thres, best_smoothing = qualifying[0]
    best_m = results[(best_thres, best_smoothing)][best_ms]
    print(
        f"\nRecommended: stop_thres={best_thres}  ig_smoothing={best_smoothing}  "
        f"maxstep={best_ms}  (score={-neg_score:.2f}  IL={best_m.IL:.2f}  "
        f"DDF1={best_m.DDF1:.2f}  elbow={elbow_ms})"
    )
    print(
        f"Apply: set in configs/symptoms.yaml under the xgb model entry:\n"
        f"  maxstep: {best_ms}\n"
        f"  stop_thres: {best_thres}\n"
        f"  # ig_smoothing is baked into weights.pkl; retrain to change."
    )


if __name__ == "__main__":
    main()
