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
import hashlib
import json
import time
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

MANIFEST_VERSION = 1
DEFAULT_HIDDEN = 2048
DEFAULT_LR = 1e-4
DEFAULT_EPOCHS = 20
DEFAULT_GAMES = 200
DEFAULT_MAXSTEP = 30


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _models_dir() -> Path:
    return _cfg.CLARITYMED_HOME / "models" / "symptoms"


def main() -> None:
    """CLI: ``uv run --extra symptoms-server claritymed-symptoms-train-ddxplus ...``."""
    ap = argparse.ArgumentParser(description="Train typed-BASD on DDXPlus.")
    ap.add_argument("--data-dir", required=True, help="DDXPlus release dir.")
    ap.add_argument("--out-subpath", default="ddxplus/typed_basd_v1")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--episodes", type=int, default=200_000)
    ap.add_argument("--eval-n", type=int, default=5_000)
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--games", type=int, default=DEFAULT_GAMES)
    ap.add_argument("--maxstep", type=int, default=DEFAULT_MAXSTEP)
    ap.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument(
        "--stop-mode", choices=["learned", "heuristic"], default="heuristic"
    )
    ap.add_argument("--stop-thres", type=float, default=None)
    ap.add_argument(
        "--target", choices=["pathology", "differential"], default="differential"
    )
    ap.add_argument("--ordinal", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="Tiny run for sanity testing.")
    args = ap.parse_args()

    if args.stop_thres is None:
        args.stop_thres = 0.7 if args.stop_mode == "learned" else 0.1
    if args.smoke:
        args.episodes = min(args.episodes, 4_000)
        args.eval_n = min(args.eval_n, 1_000)
        args.epochs = 2
        args.games = 100
        args.hidden = 256

    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("training needs PyTorch; install the project deps") from exc

    seed_everything(args.seed)
    device = resolve_device(args.device)
    t0 = time.time()
    schema = load_evidence_schema(args.data_dir, use_ordinal=args.ordinal)
    pidx, sev = load_pidx(args.data_dir)
    n_dis = len(pidx)
    train_pats = load_patients(args.data_dir, args.episodes, "train", schema, pidx)
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

    idx = np.arange(len(train_pats))
    for ep in range(args.epochs):
        np.random.shuffle(idx)
        sym_loss = pat_loss = stop_loss = 0.0
        nb = 0
        for b in range(0, len(idx) - args.games, args.games):
            sl, pl, stl = agent.train_step(
                [train_pats[j] for j in idx[b : b + args.games]],
                target=args.target,
            )
            sym_loss += sl
            pat_loss += pl
            stop_loss += stl
            nb += 1
        if nb:
            print(
                f"[typed-basd] epoch {ep + 1}: "
                f"sym_loss={sym_loss / nb:.3f} patho_loss={pat_loss / nb:.3f} "
                f"stop_loss={stop_loss / nb:.3f}"
            )

    test_pats = load_patients(args.data_dir, args.eval_n, "test", schema, pidx)
    metrics: EvalMetrics = interactive_eval(
        TypedEnv(test_pats, schema, n_dis),
        agent,
        maxstep=args.maxstep,
        games=min(args.games, len(test_pats)),
        severity=sev,
    )
    print(
        f"== TEST (maxstep={args.maxstep}) IL={metrics.IL:.2f} "
        f"DDR={metrics.DDR:.2f} DDP={metrics.DDP:.2f} "
        f"DDF1={metrics.DDF1:.2f} DSR={metrics.DSR:.2f}"
    )

    out_dir = _models_dir() / args.out_subpath
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "weights.pt"
    _save_weights(agent, weights_path)
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_id": "ddxplus",
        "model_id": Path(args.out_subpath).name,
        "algorithm_module": "typed_basd",
        "training_commit": _git_commit(),
        "sha256": _sha256_file(weights_path),
        "eval": {
            "IL": metrics.IL,
            "DDR": metrics.DDR,
            "DDP": metrics.DDP,
            "DDF1": metrics.DDF1,
            "DSR": metrics.DSR,
            "maxstep": args.maxstep,
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {weights_path}\nwrote {out_dir / 'manifest.json'}")


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
