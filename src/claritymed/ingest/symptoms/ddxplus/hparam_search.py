"""Optuna hyperparameter search for typed-BASD on DDXPlus (hidden + lr).

Searches hidden layer width and learning rate using early stopping on the
validate split. Objective: maximise DDF1 subject to DSR >= 92.0. Trials
below the DSR floor are penalised so Optuna's sampler can still learn from
them rather than treating them as noise.

HyperbandPruner cuts unpromising trials after a minimum of 3 epochs so the
budget concentrates on configurations that show early signal.

Usage
-----
    uv run python scripts/symptoms_hparam_search.py \
        --data-dir ./demo/ddxplus_demo/ddxplus \
        --n-trials 20

    # Persist across interruptions (trial weights survive in .typed_basd_hparam_trials/):
    uv run python scripts/symptoms_hparam_search.py \
        --data-dir ./demo/ddxplus_demo/ddxplus \
        --storage sqlite:///hparam.db \
        --n-trials 40
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

from claritymed import config as _cfg
from claritymed.core.device import resolve_device
from claritymed.ingest.symptoms.ddxplus.schema import (
    load_evidence_schema,
    load_patients,
    load_pidx,
)
from claritymed.ingest.symptoms.typed_basd import (
    EvalMetrics,
    TypedEnv,
    build_basd,
    interactive_eval,
    seed_everything,
)


def _run_dir() -> Path:
    return _cfg.CLARITYMED_HOME / "models" / "symptoms" / "ddxplus" / "run"


DEFAULT_MAX_EPOCHS = 40
DEFAULT_PATIENCE = 4
DEFAULT_GAMES = 200
DEFAULT_EVAL_N_VAL = 2_000
DEFAULT_N_TRIALS = 20
DEFAULT_MAXSTEP = 18
DEFAULT_TRAIN_N = 200_000
DSR_FLOOR = 92.0
HIDDEN_CHOICES = [512, 1024, 2048, 4096]


def _train_epoch(
    agent, patients: list, games: int, target: str, idx: np.ndarray
) -> None:
    np.random.shuffle(idx)
    for b in range(0, len(idx) - games, games):
        agent.train_step([patients[j] for j in idx[b : b + games]], target)


def _eval(
    agent,
    patients: list,
    schema: dict,
    n_dis: int,
    sev: np.ndarray,
    games: int,
    maxstep: int,
) -> EvalMetrics:
    env = TypedEnv(list(patients), schema, n_dis)
    return interactive_eval(
        env, agent, maxstep=maxstep, games=min(games, len(patients)), severity=sev
    )


def _score(m: EvalMetrics) -> float:
    return m.DDF1 if m.DSR >= DSR_FLOOR else m.DSR - 200.0


def _save_weights(agent, path: Path) -> None:
    import torch

    torch.save(
        {
            "trunk": agent.trunk.state_dict(),
            "sym": agent.sym.state_dict(),
            "patho": agent.patho.state_dict(),
            "stop": agent.stop.state_dict() if agent.stop is not None else None,
            "thres": agent.thres,
            "mode": agent.mode,
            "temp": agent.temp,
        },
        path,
    )


def make_objective(
    *,
    train_pats: list,
    val_pats: list,
    schema: dict,
    n_dis: int,
    sev: np.ndarray,
    device: str,
    games: int,
    maxstep: int,
    max_epochs: int,
    patience: int,
    target: str,
    stop_mode: str,
    stop_thres: float,
    trial_dir: Path,
    base_seed: int,
):
    import optuna

    def objective(trial: optuna.Trial) -> float:
        hidden = trial.suggest_categorical("hidden", HIDDEN_CHOICES)
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

        seed_everything(base_seed + trial.number * 1000)
        env = TypedEnv(list(train_pats), schema, n_dis)
        agent = build_basd(
            env,
            n_dis=n_dis,
            hidden=hidden,
            lr=lr,
            device=device,
            stop_thres=stop_thres,
            stop_mode=stop_mode,
        )

        idx = np.arange(len(train_pats))
        best_score = float("-inf")
        no_improve = 0
        best_path = trial_dir / f"trial_{trial.number}.pt"

        for ep in range(max_epochs):
            _train_epoch(agent, train_pats, games, target, idx)
            m = _eval(agent, val_pats, schema, n_dis, sev, games, maxstep)
            score = _score(m)

            # Update before reporting so the attr is set even if pruned next.
            if score > best_score:
                best_score = score
                no_improve = 0
                _save_weights(agent, best_path)
                trial.set_user_attr("best_weights", str(best_path))
            else:
                no_improve += 1

            print(
                f"  [trial {trial.number} ep {ep + 1:02d}] "
                f"hidden={hidden} lr={lr:.1e} "
                f"DDF1={m.DDF1:.2f} DSR={m.DSR:.2f} score={score:.2f}"
            )

            trial.report(score, ep)
            if trial.should_prune():
                raise optuna.TrialPruned()

            if no_improve >= patience:
                print(f"  early stop ep {ep + 1} (no improve × {patience})")
                break

        return best_score

    return objective


def main() -> None:
    """CLI entry — runs Optuna search and writes the best checkpoint."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Destination for best weights.pt "
        "(default: CLARITYMED_HOME/models/symptoms/ddxplus/run/hparam_best.pt).",
    )
    ap.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    ap.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    ap.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    ap.add_argument("--games", type=int, default=DEFAULT_GAMES)
    ap.add_argument(
        "--eval-n-val",
        type=int,
        default=DEFAULT_EVAL_N_VAL,
        help="Validate patients to load. Smaller is faster per epoch but noisier.",
    )
    ap.add_argument("--train-n", type=int, default=DEFAULT_TRAIN_N)
    ap.add_argument("--maxstep", type=int, default=DEFAULT_MAXSTEP)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--target", choices=["pathology", "differential"], default="differential"
    )
    ap.add_argument(
        "--stop-mode", choices=["learned", "heuristic"], default="heuristic"
    )
    ap.add_argument("--stop-thres", type=float, default=0.1)
    ap.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL. Defaults to the shared "
        "CLARITYMED_HOME/tracking/optuna.db so trials from every "
        "feature live in one DB (disambiguated by --study-name).",
    )
    ap.add_argument("--study-name", default="typed_basd_hparam")
    args = ap.parse_args()

    try:
        import optuna
    except ImportError as exc:
        raise SystemExit("optuna not installed — run: uv add optuna") from exc
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit("torch not installed") from exc

    run_dir = _run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    out = args.out or run_dir / "hparam_best.pt"
    if args.storage is None:
        from claritymed.ingest.mlflow_utils import optuna_storage_uri

        args.storage = optuna_storage_uri()
    # Persistent so weights survive across --storage resumptions.
    trial_dir = run_dir / f".{args.study_name}_trials"
    trial_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    seed_everything(args.seed)

    schema = load_evidence_schema(args.data_dir)
    pidx, sev = load_pidx(args.data_dir)
    n_dis = len(pidx)

    print(f"Loading patients — train:{args.train_n}  val:{args.eval_n_val}")
    train_pats = load_patients(args.data_dir, args.train_n, "train", schema, pidx)
    val_pats = load_patients(args.data_dir, args.eval_n_val, "validate", schema, pidx)
    print(
        f"train={len(train_pats)}  val={len(val_pats)}  "
        f"diseases={n_dis}  device={device}"
    )

    pruner = optuna.pruners.HyperbandPruner(
        min_resource=3, max_resource=args.max_epochs, reduction_factor=3
    )
    study = optuna.create_study(
        direction="maximize",
        pruner=pruner,
        storage=args.storage,
        study_name=args.study_name,
        load_if_exists=True,
    )

    objective = make_objective(
        train_pats=train_pats,
        val_pats=val_pats,
        schema=schema,
        n_dis=n_dis,
        sev=sev,
        device=device,
        games=args.games,
        maxstep=args.maxstep,
        max_epochs=args.max_epochs,
        patience=args.patience,
        target=args.target,
        stop_mode=args.stop_mode,
        stop_thres=args.stop_thres,
        trial_dir=trial_dir,
        base_seed=args.seed,
    )
    study.optimize(objective, n_trials=args.n_trials)

    best = study.best_trial
    print(f"\nBest trial #{best.number}  score={best.value:.4f}")
    print(f"  hidden={best.params['hidden']}  lr={best.params['lr']:.2e}")

    best_weights = best.user_attrs.get("best_weights")
    if best_weights and Path(best_weights).exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, out)
        print(f"Weights → {out}")
    else:
        print(
            "WARNING: best trial weights not found — re-run with the best params manually:\n"
            f"  hidden={best.params['hidden']}  lr={best.params['lr']:.2e}"
        )

    print("\n## Trial summary\n")
    print("| # | hidden | lr | score |")
    print("|---|--------|-----|-------|")
    for t in sorted(
        [t for t in study.trials if t.value is not None],
        key=lambda t: t.value,
        reverse=True,
    ):
        print(
            f"| {t.number} | {t.params.get('hidden', '?')} | "
            f"{t.params.get('lr', 0):.2e} | {t.value:.4f} |"
        )


if __name__ == "__main__":
    main()
