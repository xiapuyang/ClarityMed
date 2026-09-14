"""Joint (maxstep × patho_temp) tuning for typed-BASD on DDXPlus.

Sweeps every (maxstep, patho_temp) combination and logs each as a nested
MLflow child run under a single parent run.  After the sweep, prints a
results matrix and recommends the (T, maxstep) pair via two-stage selection:

1. IL saturation elbow gate — find the smallest maxstep where the next
   step's marginal IL gain/step drops below SATURATION_RATE, indicating the
   stop gate is already terminating games naturally.  Only points at or
   beyond the elbow are eligible (below the elbow the agent is truncated
   every game, so IL ≈ maxstep and DDF1 is artificially depressed).

2. Among eligible DSR ≥ 92 points, maximise ``score = DDF1 − IL_PENALTY × IL``
   (default IL_PENALTY = 1.0).  This trades diagnostic accuracy against
   conversation length: one extra question must buy at least IL_PENALTY DDF1
   points to be worthwhile.  Ties on score favour the smaller maxstep.

Usage::

    uv run --extra symptoms-server claritymed-symptoms-tune-ddxplus \\
        --data-dir ~/.claritymed/data/symptoms/ddxplus \\
        --weights ~/.claritymed/models/symptoms/ddxplus/typed_basd_v2/weights.pt \\
        --maxsteps 6,8,10,12,14,16,18,24,30 \\
        --temps 1.0,0.7,0.5 \\
        --games 1000

MLflow UI::

    mlflow ui --port 5000
    # then open http://localhost:5000, filter experiment claritymed-symptoms-ddxplus,
    # run_type=tune.

Operator script — failure modes surface via ``SystemExit``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from claritymed.core.device import resolve_device
from claritymed.ingest.symptoms.ddxplus.schema import (
    load_evidence_schema,
    load_patients,
    load_pidx,
)
from claritymed.ingest.symptoms.mlflow_utils import log_eval_metrics, symptom_run
from claritymed.ingest.symptoms.typed_basd import (
    TypedEnv,
    build_basd,
    interactive_eval,
    seed_everything,
)

DEFAULT_MAXSTEPS = "6,8,10,12,14,16,18,24,30"
DEFAULT_TEMPS = "1.0,0.7,0.5"
DSR_FLOOR = 92.0
# Marginal IL gain per maxstep unit below this → stop gate is saturating.
# At this threshold the agent already terminates most games before the cap.
SATURATION_RATE = 0.3
# DDF1 points credited per unit of IL saved.  score = DDF1 − IL_PENALTY × IL.
# 1.0 means one extra question must buy at least 1 DDF1 point to be worthwhile.
# Raise to prefer shorter conversations; lower to favour accuracy.
IL_PENALTY = 1.0


def _load_weights(agent, weights_path: Path, device: str) -> None:
    import torch

    state = torch.load(weights_path, map_location=device)
    agent.trunk.load_state_dict(state["trunk"])
    agent.sym.load_state_dict(state["sym"])
    agent.patho.load_state_dict(state["patho"])
    if agent.stop is not None and state.get("stop") is not None:
        agent.stop.load_state_dict(state["stop"])
    agent.thres = state.get("thres", agent.thres)
    agent.temp = state.get("temp", agent.temp)


def main() -> None:
    """CLI entry — sweeps (maxstep × patho_temp) and logs to MLflow."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument(
        "--maxsteps",
        default=DEFAULT_MAXSTEPS,
        help="Comma-separated maxstep values to sweep.",
    )
    ap.add_argument(
        "--temps",
        default=DEFAULT_TEMPS,
        help="Comma-separated patho_temp values to sweep.",
    )
    ap.add_argument(
        "--games",
        type=int,
        default=None,
        help="Patients per eval call. Defaults to all loaded (--eval-n).",
    )
    ap.add_argument("--eval-n", type=int, default=5_000)
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Omit to let runs vary naturally.",
    )
    ap.add_argument(
        "--stop-mode", choices=["learned", "heuristic"], default="heuristic"
    )
    ap.add_argument("--stop-thres", type=float, default=0.1)
    ap.add_argument(
        "--diseases",
        default=None,
        help="Comma-separated disease names, matching the subset the "
        "checkpoint was trained on (e.g. 'Pneumonia,Influenza'). MUST match "
        "manifest.diseases_trained — mismatch triggers a shape error at "
        "weight load. Leave unset for full-corpus checkpoints.",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="100-patient subset smoke test.",
    )
    args = ap.parse_args()

    if not args.weights.exists():
        print(
            f"weights not found: {args.weights} — train first via "
            f"`claritymed-symptoms-train-ddxplus`.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    maxsteps = [int(s) for s in args.maxsteps.split(",")]
    temps = [float(t) for t in args.temps.split(",")]

    whitelist: set[str] | None = None
    if args.diseases:
        whitelist = {d.strip() for d in args.diseases.split(",") if d.strip()}

    if args.quick:
        args.eval_n = 100

    if args.seed is not None:
        seed_everything(args.seed)
    device = resolve_device(args.device)
    schema = load_evidence_schema(args.data_dir)
    pidx, sev = load_pidx(args.data_dir, whitelist=whitelist)
    n_dis = len(pidx)
    if whitelist:
        print(f"[tune] subset pidx = {pidx}")
    test_pats = load_patients(args.data_dir, args.eval_n, "test", schema, pidx)

    import mlflow
    import torch

    ckpt_state = torch.load(args.weights, map_location=device)
    hidden = ckpt_state["trunk"]["0.weight"].shape[0]

    seed_env = TypedEnv(test_pats[:1], schema, n_dis)
    agent = build_basd(
        seed_env,
        n_dis=n_dis,
        hidden=hidden,
        lr=1e-4,
        device=device,
        stop_thres=args.stop_thres,
        stop_mode=args.stop_mode,
    )
    _load_weights(agent, args.weights, device)

    games = args.games if args.games is not None else len(test_pats)
    parent_params = {
        "maxsteps": args.maxsteps,
        "temps": args.temps,
        "eval_n": args.eval_n,
        "games": games,
        "seed": args.seed,
        "weights": str(args.weights),
    }

    # results[temp][maxstep] = EvalMetrics
    results: dict[float, dict[int, object]] = {}

    run_name = f"tune-{args.weights.parent.name}"
    with symptom_run(
        "ddxplus", run_name=run_name, run_type="tune", params=parent_params
    ):
        for temp in temps:
            agent.temp = temp
            results[temp] = {}
            for maxstep in maxsteps:
                child_name = f"T{temp}-ms{maxstep}"
                child_params = {"patho_temp": temp, "maxstep": maxstep}
                with symptom_run(
                    "ddxplus",
                    run_name=child_name,
                    run_type="tune_child",
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
                results[temp][maxstep] = m

    _print_results(temps, maxsteps, results)


def _elbow_maxstep(maxsteps: list[int], il_by_maxstep: dict[int, float]) -> int:
    """Return the smallest maxstep at the IL saturation elbow.

    Walks sorted maxsteps and returns the first ms where the *next* step's
    marginal IL gain/step drops below SATURATION_RATE — the point at which
    the stop gate is already terminating most games before the cap.

    Falls back to the SMALLEST maxstep when no saturation is detected: this
    covers the "stop gate never fires" case (e.g. a 2-class subset where
    max symptom prob stays > stop_thres for every state along the way).
    In that regime IL == maxstep everywhere and there is no principled way
    to prefer any point over the shortest one; the alternative — picking
    the largest maxstep as before — silently recommends the WORST point
    in the sweep by cost per DDF1. A warning is printed so operators
    notice the degraded selection semantics.
    """
    sorted_ms = sorted(maxsteps)
    for i in range(len(sorted_ms) - 1):
        delta_ms = sorted_ms[i + 1] - sorted_ms[i]
        delta_il = il_by_maxstep[sorted_ms[i + 1]] - il_by_maxstep[sorted_ms[i]]
        if delta_il / delta_ms < SATURATION_RATE:
            return sorted_ms[i]
    print(
        f"[tune] warning: no IL saturation elbow found across maxsteps "
        f"{sorted_ms} (marginal IL/step never dropped below "
        f"{SATURATION_RATE}). Stop gate is not firing naturally — falling "
        f"back to smallest maxstep ({sorted_ms[0]}). Common cause: subset "
        f"model where max symptom prob stays > stop_thres for every "
        f"question. Inspect the table and pick the (T, ms) point yourself "
        f"if this recommendation looks off.",
        file=sys.stderr,
    )
    return sorted_ms[0]


def _print_results(
    temps: list[float],
    maxsteps: list[int],
    results: dict,
) -> None:
    print("\n## Joint tune results (DDXPlus / typed-BASD)\n")
    print(
        f"{'T':>5} | {'ms':>4} | {'IL':>6} | {'DDR':>6} | {'DDP':>6} | "
        f"{'DDF1':>6} | {'DSR':>6} | {'score':>7} |"
    )
    print("-" * 66)

    # IL doesn't vary with temp; use first temp to compute the elbow.
    il_by_maxstep = {ms: results[temps[0]][ms].IL for ms in maxsteps}
    elbow_ms = _elbow_maxstep(maxsteps, il_by_maxstep)

    # score = DDF1 − IL_PENALTY × IL.  Qualifying: DSR ≥ floor, maxstep ≥ elbow.
    # Sort key: (-score, maxstep) so ties on score favour fewer questions.
    qualifying: list[tuple[float, int, float]] = []  # (-score, maxstep, temp)
    for temp in temps:
        for maxstep in maxsteps:
            m = results[temp][maxstep]
            dsr = m.DSR if not np.isnan(m.DSR) else float("nan")
            # NaN DSR = subset pidx has no severe (severity < 3) diseases.
            # Not a training failure; skip the floor check for that case.
            passes_dsr = np.isnan(dsr) or dsr >= DSR_FLOOR
            at_elbow = maxstep >= elbow_ms
            score = m.DDF1 - IL_PENALTY * m.IL
            flag = ""
            if passes_dsr:
                flag += " *"
            if maxstep == elbow_ms:
                flag += " ←elbow"
            print(
                f"{temp:>5.2f} | {maxstep:>4} | {m.IL:>6.2f} | "
                f"{m.DDR:>6.2f} | {m.DDP:>6.2f} | {m.DDF1:>6.2f} | "
                f"{dsr:>6.2f} | {score:>7.2f} |{flag}"
            )
            if passes_dsr and at_elbow:
                qualifying.append((-score, maxstep, temp))

    print(
        f"\n* = DSR ≥ {DSR_FLOOR}   ←elbow = IL saturation point"
        f"   score = DDF1 − {IL_PENALTY} × IL"
        f"   (SATURATION_RATE={SATURATION_RATE})"
    )
    print()
    if not qualifying:
        print(
            f"BLOCKED: no qualifying (DSR ≥ {DSR_FLOOR}) point at or beyond "
            f"the IL elbow (maxstep ≥ {elbow_ms}) — retrain required.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    qualifying.sort()  # ascending -score (best first), then ascending maxstep
    neg_score, best_ms, best_t = qualifying[0]
    best_m = results[best_t][best_ms]
    print(
        f"Recommended: patho_temp={best_t}  maxstep={best_ms}  "
        f"(score={-neg_score:.2f}  IL={best_m.IL:.2f}  DDF1={best_m.DDF1:.2f}  elbow={elbow_ms})"
    )
    print(
        f"Apply: set in configs/symptoms.yaml under models[typed_basd_v2]:\n"
        f"  maxstep: {best_ms}\n"
        f"  patho_temp: {best_t}"
    )


if __name__ == "__main__":
    main()
