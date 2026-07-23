"""Training entry: ``claritymed-symptoms-train-ddxplus``.

Drives typed-BASD against DDXPlus and writes weights + manifest under
``CLARITYMED_HOME/models/symptoms/<weights_subpath>/``. The manifest
records the eval numbers + the weights sha256 so the server's startup
check can refuse a tampered checkpoint (KTD-6).

This is an operator script; failure modes are surfaced via
``SystemExit`` with remediation hints rather than wrapped in pytest.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from claritymed import config as _cfg
from claritymed.core.device import resolve_device
from claritymed.ingest.symptoms.ddxplus.schema import (
    load_evidence_schema,
    load_patients,
    load_pidx,
)
from claritymed.ingest.symptoms.mlflow_utils import (
    log_epoch,
    log_eval_metrics,
    symptom_run,
)
from claritymed.ingest.symptoms.typed_basd import (
    EvalMetrics,
    TypedEnv,
    build_basd,
    interactive_eval,
    seed_everything,
)

MANIFEST_VERSION = 1
DEFAULT_HIDDEN = 2048
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 20
DEFAULT_PATIENCE = 5
DEFAULT_BATCH_SIZE = 200
DEFAULT_MAXSTEP = 30
DEFAULT_EVAL_N_VAL = 2_000
DSR_FLOOR = 92.0


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _models_dir() -> Path:
    return _cfg.CLARITYMED_HOME / "models" / "symptoms"


def _run_dir() -> Path:
    return _models_dir() / "ddxplus" / "run"


def main() -> None:
    """CLI: ``uv run --extra symptoms-server claritymed-symptoms-train-ddxplus ...``."""
    ap = argparse.ArgumentParser(description="Train typed-BASD on DDXPlus.")
    ap.add_argument("--data-dir", required=True, help="DDXPlus release dir.")
    ap.add_argument(
        "--out-subpath",
        default=None,
        help="Output path under CLARITYMED_HOME/models/symptoms/. "
        "Defaults to ddxplus/typed_basd_v2_<timestamp>; promote to "
        "typed_basd_v2 manually after evaluation.",
    )
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=200_000)
    ap.add_argument("--eval-n", type=int, default=5_000)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help="Early-stop after this many epochs with no val improvement.",
    )
    ap.add_argument(
        "--eval-n-val",
        type=int,
        default=DEFAULT_EVAL_N_VAL,
        help="Validate patients to load for early stopping.",
    )
    ap.add_argument(
        "--patho-temp",
        type=float,
        default=1.0,
        help="Classifier softmax temperature saved into the checkpoint.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Patients per training step (mini-batch size).",
    )
    ap.add_argument("--maxstep", type=int, default=DEFAULT_MAXSTEP)
    ap.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument(
        "--stop-mode", choices=["learned", "heuristic"], default="heuristic"
    )
    ap.add_argument("--stop-thres", type=float, default=None)
    ap.add_argument(
        "--target",
        choices=["pathology", "differential"],
        default=None,
        help="Loss target. Defaults to 'differential' (full 49-class) or "
        "'pathology' when --diseases is set (subset-conditional differentials "
        "degrade to near-one-hot; see docs).",
    )
    ap.add_argument(
        "--diseases",
        default=None,
        help="Comma-separated disease names for subset training "
        "(e.g. 'Pneumonia,Influenza'). When set: pidx shrinks to this list "
        "in name-sorted order, out-of-scope patients are dropped, and the "
        "trained model handles a subset-conditional differential only. "
        "Names must match release_conditions.json exactly.",
    )
    ap.add_argument(
        "--targets",
        default=None,
        help="Comma-separated target disease names for 3-class native "
        "subset training (e.g. 'Pneumonia,Influenza'). Non-target patients "
        "are RELABELED to a synthetic 'Other' class instead of being "
        "dropped (unlike --diseases). Produces a native (N+1)-class classifier. "
        "Mutually exclusive with --diseases. Names must match "
        "release_conditions.json exactly. Order defines subset class indices.",
    )
    ap.add_argument("--ordinal", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="Tiny run for sanity testing.")
    args = ap.parse_args()

    whitelist: set[str] | None = None
    whitelist_sorted: list[str] = []
    if args.diseases:
        whitelist_sorted = sorted(
            d.strip() for d in args.diseases.split(",") if d.strip()
        )
        whitelist = set(whitelist_sorted)
        if len(whitelist) < 2:
            raise SystemExit(
                f"--diseases needs at least 2 names, got {whitelist_sorted!r}. "
                f"A 1-class model is not a classifier."
            )

    # v3 subset-parametric (3-class native) — mutually exclusive with v2
    # whitelist mode. Names preserve the order the user typed them (defines
    # subset class indices, mirrors xgb/train.py --targets semantics).
    target_names: list[str] = []
    if args.targets:
        if args.diseases:
            raise SystemExit(
                "--diseases (v2 whitelist) and --targets (v3 native subset) "
                "are mutually exclusive."
            )
        target_names = [n.strip() for n in args.targets.split(",") if n.strip()]
        if not target_names:
            raise SystemExit("--targets is empty after parsing.")

    if args.target is None:
        args.target = "pathology" if (whitelist or target_names) else "differential"
    elif whitelist and args.target == "differential":
        print(
            "[train] WARNING: --target differential with --diseases set. "
            "Under a small subset the renormalized differential collapses to "
            "near one-hot (see docs). --target pathology is the honest choice."
        )

    if args.out_subpath is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if target_names:
            slug = "_".join(n.replace(" ", "-").lower()[:6] for n in target_names[:3])
            args.out_subpath = f"ddxplus/run/typed_basd_native_{slug}_other_{ts}"
        elif whitelist:
            # Slug from initials + count for readability at the filesystem level.
            slug = "_".join(
                n.replace(" ", "-").lower()[:6] for n in whitelist_sorted[:3]
            )
            args.out_subpath = f"ddxplus/run/typed_basd_subset_{slug}_{ts}"
        else:
            args.out_subpath = f"ddxplus/run/typed_basd_v2_{ts}"
    if args.stop_thres is None:
        args.stop_thres = 0.7 if args.stop_mode == "learned" else 0.1
    if args.smoke:
        args.episodes = min(args.episodes, 4_000)
        args.eval_n = min(args.eval_n, 1_000)
        args.epochs = 2
        args.batch_size = 100
        args.hidden = 256

    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("training needs PyTorch; install the project deps") from exc

    seed_everything(args.seed)
    device = resolve_device(args.device)
    t0 = time.time()
    schema = load_evidence_schema(args.data_dir, use_ordinal=args.ordinal)

    # Three loading modes:
    # 1. --targets: load FULL pidx, then in-place relabel non-targets to Other class.
    #    Produces a native (len(targets)+1)-class model, no data dropped.
    # 2. --diseases: load whitelist-filtered pidx, drop out-of-scope patients.
    #    Legacy 2-class subset-conditional model.
    # 3. Neither: load full 49-class pidx (default).
    subset_mapping: dict[int, int] | None = None
    if target_names:
        # Import lazily so a torch-only failure doesn't happen at module load —
        # the xgb train.py path depends on xgboost which is an optional extra.
        from claritymed.ingest.symptoms.xgb.train import (
            _relabel_patients,
            _resolve_subset_mapping,
        )

        pidx_full, sev_full = load_pidx(args.data_dir, whitelist=None)
        subset_mapping, n_dis = _resolve_subset_mapping(target_names, pidx_full)
        pidx = pidx_full  # load_patients reads pidx to validate patient labels
        # Rebuild severity vector in subset order: targets take their original
        # severity, Other bucket gets 3.0 (moderate — matches xgb subset default).
        sev = np.zeros(n_dis, dtype=np.float32)
        for k, name in enumerate(target_names):
            sev[k] = sev_full[pidx_full[name]]
        sev[len(target_names)] = 3.0
        print(
            f"[train] native subset training: targets={target_names} "
            f"→ classes {[*target_names, 'Other']} (n_dis={n_dis}); "
            f"non-target patients are relabeled to Other, not dropped."
        )
    else:
        pidx, sev = load_pidx(args.data_dir, whitelist=whitelist)
        n_dis = len(pidx)
        if whitelist:
            print(
                f"[train] subset training: pidx = {pidx} "
                f"(target={args.target}; --episodes reads N raw rows before "
                f"filtering — bump it if the kept-count is too low)"
            )
    train_pats = load_patients(args.data_dir, args.episodes, "train", schema, pidx)
    if subset_mapping is not None:
        _relabel_patients(train_pats, subset_mapping)
    env = TypedEnv(train_pats, schema, n_dis)
    print(
        f"typed: questions(actions)={schema['n_ev']} "
        f"state_sym={schema['sym_size']} diseases={n_dis} device={device} "
        f"[{time.time() - t0:.1f}s]"
    )

    agent = build_basd(
        env,
        n_dis=n_dis,
        hidden=args.hidden,
        lr=args.lr,
        device=device,
        stop_thres=args.stop_thres,
        stop_mode=args.stop_mode,
    )
    agent.temp = args.patho_temp

    val_pats = load_patients(args.data_dir, args.eval_n_val, "validate", schema, pidx)
    if subset_mapping is not None:
        _relabel_patients(val_pats, subset_mapping)
    out_dir = _models_dir() / args.out_subpath
    out_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = out_dir / "weights.pt"
    model_id = Path(args.out_subpath).name

    # Sorted pidx names — this ordering is what the patho head learns.
    # Recorded in the manifest so the server adapter can fail-loud when
    # the runtime whitelist config drifts from what was trained. Under
    # --targets mode the ordering is [target_0, ..., target_N-1, Other] —
    # matches the subset mapping's class layout so the adapter's
    # target_condition_ids check (ddxplus/adapter.py:_assert_target_...) passes.
    if target_names:
        diseases_trained = [*target_names, "Other"]
    else:
        diseases_trained = sorted(pidx, key=lambda n: pidx[n])
    train_params = {
        "hidden": args.hidden,
        "lr": args.lr,
        "episodes": args.episodes,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "maxstep": args.maxstep,
        "eval_n": args.eval_n,
        "eval_n_val": args.eval_n_val,
        "seed": args.seed,
        "patho_temp": args.patho_temp,
        "stop_mode": args.stop_mode,
        "stop_thres": args.stop_thres,
        "target": args.target,
        "ordinal": args.ordinal,
        "device": str(device),
        "whitelist_active": whitelist is not None,
        "targets": target_names or None,
        "native_subset": bool(target_names),
        "train_pats_kept": len(train_pats),
    }

    with symptom_run(
        "ddxplus", run_name=model_id, run_type="train", params=train_params
    ):
        idx = np.arange(len(train_pats))
        best_score = float("-inf")
        no_improve = 0
        for ep in range(args.epochs):
            np.random.shuffle(idx)
            sym_loss = pat_loss = stop_loss = 0.0
            nb = 0
            for b in range(0, len(idx) - args.batch_size, args.batch_size):
                sl, pl, stl = agent.train_step(
                    [train_pats[j] for j in idx[b : b + args.batch_size]],
                    target=args.target,
                )
                sym_loss += sl
                pat_loss += pl
                stop_loss += stl
                nb += 1
            if not nb:
                continue

            val_m: EvalMetrics = interactive_eval(
                TypedEnv(list(val_pats), schema, n_dis),
                agent,
                maxstep=args.maxstep,
                games=len(val_pats),
                severity=sev,
            )
            # DSR is NaN when the pidx has no severe (severity < 3) diseases —
            # a subset case, not a training failure. Fall back to DDF1 as the
            # improvement signal so the gate doesn't lock out every epoch.
            if np.isnan(val_m.DSR):
                score = val_m.DDF1
            elif val_m.DSR >= DSR_FLOOR:
                score = val_m.DDF1
            else:
                score = val_m.DSR - 200.0
            improved = score > best_score
            if improved:
                best_score = score
                no_improve = 0
                _save_weights(agent, best_weights_path)
            else:
                no_improve += 1
            log_epoch(ep, sym_loss / nb, pat_loss / nb, stop_loss / nb, val_m, score)
            print(
                f"[typed-basd] epoch {ep + 1}: "
                f"sym={sym_loss / nb:.3f} pat={pat_loss / nb:.3f} "
                f"| val DDF1={val_m.DDF1:.2f} DSR={val_m.DSR:.2f} score={score:.2f}"
                + (" ✓" if improved else f" (no improve ×{no_improve})")
            )
            if no_improve >= args.patience:
                print(f"early stop at epoch {ep + 1}")
                break

        # Reload best checkpoint before test eval so manifest records best-epoch metrics.
        import torch

        best_state = torch.load(best_weights_path, map_location=device)
        agent.trunk.load_state_dict(best_state["trunk"])
        agent.sym.load_state_dict(best_state["sym"])
        agent.patho.load_state_dict(best_state["patho"])

        test_pats = load_patients(args.data_dir, args.eval_n, "test", schema, pidx)
        if subset_mapping is not None:
            _relabel_patients(test_pats, subset_mapping)
        metrics: EvalMetrics = interactive_eval(
            TypedEnv(test_pats, schema, n_dis),
            agent,
            maxstep=args.maxstep,
            games=len(test_pats),
            severity=sev,
        )
        print(
            f"== TEST (maxstep={args.maxstep}) IL={metrics.IL:.2f} "
            f"DDR={metrics.DDR:.2f} DDP={metrics.DDP:.2f} "
            f"DDF1={metrics.DDF1:.2f} DSR={metrics.DSR:.2f} "
            f"PSF1={metrics.PSF1:.2f} PAF1={metrics.PAF1:.2f}"
        )
        log_eval_metrics(metrics, prefix="test/")

        manifest = {
            "manifest_version": MANIFEST_VERSION,
            "dataset_id": "ddxplus",
            "model_id": model_id,
            "algorithm_module": "typed_basd",
            "training_commit": _git_commit(),
            "sha256": _sha256_file(best_weights_path),
            "train_params": train_params,
            # diseases_trained is the class-idx contract of the patho head.
            # Adapter must reject a config whose disease_whitelist doesn't
            # match this list exactly — otherwise class ids silently misalign.
            "diseases_trained": diseases_trained,
            "training_target": args.target,
            "eval": {
                **dataclasses.asdict(metrics),
                "maxstep": args.maxstep,
            },
        }
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        import mlflow

        mlflow.log_artifact(str(best_weights_path))
        mlflow.log_artifact(str(manifest_path))
        print(f"wrote {best_weights_path}\nwrote {manifest_path}")


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


def _git_commit() -> str:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"


if __name__ == "__main__":
    main()
