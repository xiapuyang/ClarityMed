"""Maxstep ablation for typed-BASD on DDXPlus (gating script for OQ-2).

Sweeps ``maxstep ∈ {6, 8, 10, 12, 18}`` against an existing checkpoint
and reports IL / DDR / DDP / DDF1 / DSR per maxstep. The plan's Phase 0
selects the smallest ``maxstep`` where DSR ≥ 92 and pins it to
``configs/symptoms.yaml.datasets[*].maxstep``.

Usage::

    uv run python scripts/symptoms_ablate_maxstep.py \\
        --data-dir ./demo/ddxplus_demo/ddxplus \\
        --weights ~/.claritymed/models/symptoms/ddxplus/typed_basd_v1/weights.pt \\
        --maxsteps 6,8,10,12,18 \\
        --games 1000

Operator script — failure modes surface via ``SystemExit`` rather than
pytest fixtures. The companion decision doc lives at
``docs/solutions/2026-06-13-001-maxstep-ablation-decision.md`` (written
by ``/ce:compound`` after the run lands).
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
from claritymed.ingest.symptoms.typed_basd import (
    TypedEnv,
    build_basd,
    interactive_eval,
    seed_everything,
)

DEFAULT_MAXSTEPS = "6,8,10,12,18"
DEFAULT_GAMES = 200
DEFAULT_HIDDEN = 2048
DSR_FLOOR = 92.0


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
    """CLI entry — prints a markdown results table + recommended maxstep."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--maxsteps", default=DEFAULT_MAXSTEPS)
    ap.add_argument("--games", type=int, default=DEFAULT_GAMES)
    ap.add_argument("--eval-n", type=int, default=5_000)
    ap.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--stop-mode", choices=["learned", "heuristic"], default="heuristic"
    )
    ap.add_argument("--stop-thres", type=float, default=0.1)
    ap.add_argument(
        "--patho-temp",
        type=float,
        default=None,
        help="Classifier softmax temperature (T<1 sharpens differential → DDP up, DDR down). "
        "Defaults to the value stored in the checkpoint.",
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="100-patient subset smoke test; not a real ablation.",
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
    if args.quick:
        args.eval_n = 100
        args.games = 50

    seed_everything(args.seed)
    device = resolve_device(args.device)
    schema = load_evidence_schema(args.data_dir)
    pidx, sev = load_pidx(args.data_dir)
    n_dis = len(pidx)
    test_pats = load_patients(args.data_dir, args.eval_n, "test", schema, pidx)

    # Detect hidden size from checkpoint so --hidden doesn't need to match.
    import torch

    ckpt_state = torch.load(args.weights, map_location=device)
    detected_hidden = ckpt_state["trunk"]["0.weight"].shape[0]
    if args.hidden != DEFAULT_HIDDEN and args.hidden != detected_hidden:
        print(
            f"WARNING: --hidden {args.hidden} overrides checkpoint hidden "
            f"{detected_hidden}; load will fail.",
            file=sys.stderr,
        )
    hidden = detected_hidden

    # Build a one-shot env to instantiate the agent (sizes derive from env);
    # interactive_eval rebuilds a fresh env per maxstep below.
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
    if args.patho_temp is not None:
        agent.temp = args.patho_temp

    print(f"\n## Maxstep ablation (DDXPlus / typed-BASD)  patho_temp={agent.temp}\n")
    print("| maxstep | IL | DDR | DDP | DDF1 | DSR | n_severe |")
    print("|---------|-----|-----|-----|------|-----|---------|")
    rows: list[tuple[int, float]] = []
    for m in maxsteps:
        env = TypedEnv(list(test_pats), schema, n_dis)
        metrics = interactive_eval(
            env,
            agent,
            maxstep=m,
            games=min(args.games, len(test_pats)),
            severity=sev,
        )
        dsr = metrics.DSR if not np.isnan(metrics.DSR) else float("nan")
        rows.append((m, dsr))
        print(
            f"| {m:>7} | {metrics.IL:.2f} | {metrics.DDR:.2f} | "
            f"{metrics.DDP:.2f} | {metrics.DDF1:.2f} | "
            f"{dsr:.2f} | {metrics.n_severe} |"
        )

    qualifying = [m for m, dsr in rows if not np.isnan(dsr) and dsr >= DSR_FLOOR]
    print()
    if not qualifying:
        print(
            f"BLOCKED: no maxstep in {maxsteps} cleared the DSR ≥ "
            f"{DSR_FLOOR} floor — retrain required before shipping.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    selected = min(qualifying)
    print(f"Selected maxstep = {selected} (smallest passing DSR ≥ {DSR_FLOOR}).")
    print(
        f"Apply: set `datasets[*].maxstep: {selected}` in configs/symptoms.yaml "
        f"and commit the decision to docs/solutions/."
    )


if __name__ == "__main__":
    main()
