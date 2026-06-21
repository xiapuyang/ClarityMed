"""``claritymed-vision-forge`` — one entry, every dataset, every phase.

Usage:

.. code-block:: bash

    # Full pipeline against BUSI's U-Net:
    claritymed-vision-forge pipeline \\
        --model claritymed.ingest.vision.busi.models.unet_resnet50:UNET_RESNET50

    # Search only, chest CT ResNet-50:
    claritymed-vision-forge hparam \\
        --model claritymed.ingest.vision.chest_ct.models.resnet50_v1:RESNET50_V1 \\
        --trials 30

    # Deploy a specific staging dir:
    claritymed-vision-forge deploy \\
        --model claritymed.ingest.vision.busi.models.unet_resnet50:UNET_RESNET50 \\
        --staging-dir ~/.claritymed/models/vision/breast_cancer_ultrasound/run/...

    # First end-to-end on real data (real dataset + real training,
    # microscopic budgets, gates downgraded; takes minutes, not hours):
    claritymed-vision-forge pipeline \\
        --model claritymed.ingest.vision.chest_ct.models.resnet50_v1:RESNET50_V1 \\
        --quick

The ``--model`` flag is a dotted-path-plus-colon-attribute selector
(``module.path:ATTR``). Forge ``importlib``-loads the module and reads
the named attribute — which must be a
:class:`~claritymed.ingest.vision.forge.spec.ModelSpec` instance. The
single CLI replaces the five per-dataset console scripts of the
original BUSI pipeline.

Pipeline has two short-circuit modes: ``--smoke`` (synthetic everything,
sub-second; verifies CLI / phase wiring without torch or a dataset) and
``--quick`` (real torch + real dataset, microscopic budgets; smallest
honest end-to-end). Use ``--smoke`` for wiring checks; ``--quick`` for
the first run on a new dataset or fresh box.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import resource
import sys
import time
from pathlib import Path

from claritymed.ingest.mlflow_utils import generate_task_id
from claritymed.ingest.vision.forge.common import (
    ALL_PHASES,
    latest_staging_dir,
    parse_phases,
)
from claritymed.ingest.vision.forge.framework import (
    run_deploy,
    run_hparam,
    run_pipeline,
    run_train,
    run_tune,
)
from claritymed.ingest.vision.forge.spec import ModelSpec

logger = logging.getLogger(__name__)

# ---- argparse defaults ---------------------------------------------
DEFAULT_HPARAM_TRIALS = 20
DEFAULT_SEARCH_EPOCHS = 15
DEFAULT_MAX_EPOCHS = 100
DEFAULT_PATIENCE = 15
DEFAULT_TUNE_TRIALS = 100

# ---- pipeline --quick microscopic budgets --------------------------
QUICK_TRIALS = 3
QUICK_SEARCH_EPOCHS = 3
QUICK_MAX_EPOCHS = 5
QUICK_PATIENCE = 3
QUICK_TUNE_TRIALS = 5

_QUICK_OVERRIDES = {
    "trials": QUICK_TRIALS,
    "search_epochs": QUICK_SEARCH_EPOCHS,
    "max_epochs": QUICK_MAX_EPOCHS,
    "patience": QUICK_PATIENCE,
    "tune_trials": QUICK_TUNE_TRIALS,
    "force": True,
    "deploy_force": True,
}


def _resolve_model_spec(target: str) -> ModelSpec:
    """Resolve ``module.path:ATTR`` to a :class:`ModelSpec` instance.

    Fail loud on a bad path / wrong attribute type — silent fallback
    would let an operator train against the wrong spec.
    """
    if ":" not in target:
        raise SystemExit(
            f"--model {target!r}: expected 'module.path:ATTR' (got no ':')."
        )
    module_path, attr = target.split(":", 1)
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise SystemExit(
            f"--model {target!r}: cannot import {module_path!r}: {exc}"
        ) from exc
    spec = getattr(module, attr, None)
    if spec is None:
        raise SystemExit(
            f"--model {target!r}: module {module_path!r} has no attribute {attr!r}."
        )
    if not isinstance(spec, ModelSpec):
        raise SystemExit(
            f"--model {target!r}: {attr!r} is not a ModelSpec instance "
            f"(got {type(spec).__name__})."
        )
    return spec


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="claritymed-vision-forge",
        description=__doc__,
    )
    sub = parser.add_subparsers(dest="phase", required=True)

    # ---- shared model + smoke args ---------------------------------
    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--model",
            required=True,
            help="Dotted-path:ATTR selector resolving to a ModelSpec.",
        )
        p.add_argument("--smoke", action="store_true")
        p.add_argument("--task-id", default=None)

    # ---- hparam ----------------------------------------------------
    sp = sub.add_parser("hparam", help="Optuna HP search")
    add_common(sp)
    sp.add_argument("--trials", type=int, default=DEFAULT_HPARAM_TRIALS)
    sp.add_argument("--epochs", type=int, default=DEFAULT_SEARCH_EPOCHS)

    # ---- train -----------------------------------------------------
    sp = sub.add_parser("train", help="Production training (reads best HP from study)")
    add_common(sp)
    sp.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    sp.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)

    # ---- tune ------------------------------------------------------
    sp = sub.add_parser("tune", help="Inference-param Optuna sweep")
    add_common(sp)
    sp.add_argument("--staging-dir", type=Path, default=None)
    sp.add_argument("--trials", type=int, default=DEFAULT_TUNE_TRIALS)

    # ---- deploy ----------------------------------------------------
    sp = sub.add_parser("deploy", help="Floor + regression gate; promote staging dir")
    add_common(sp)
    sp.add_argument("--staging-dir", type=Path, default=None)
    sp.add_argument(
        "--deploy-force",
        action="store_true",
        help="Skip the floor and regression gates (WARNING: development only).",
    )

    # ---- pipeline --------------------------------------------------
    sp = sub.add_parser("pipeline", help="Orchestrate search → train → tune → deploy")
    add_common(sp)
    sp.add_argument("--phases", default=",".join(ALL_PHASES))
    sp.add_argument("--trials", type=int, default=DEFAULT_HPARAM_TRIALS)
    sp.add_argument("--search-epochs", type=int, default=DEFAULT_SEARCH_EPOCHS)
    sp.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    sp.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    sp.add_argument("--tune-trials", type=int, default=DEFAULT_TUNE_TRIALS)
    sp.add_argument("--staging-dir", type=Path, default=None)
    sp.add_argument(
        "--force",
        action="store_true",
        help="Downgrade inter-phase floor gates to warnings (diagnostic runs).",
    )
    sp.add_argument(
        "--deploy-force",
        action="store_true",
        help="Skip the deploy floor and regression gates (WARNING: development only).",
    )
    sp.add_argument(
        "--quick",
        action="store_true",
        help=(
            f"Real-data shakedown with microscopic budgets: trials={QUICK_TRIALS}, "
            f"search-epochs={QUICK_SEARCH_EPOCHS}, max-epochs={QUICK_MAX_EPOCHS}, "
            f"patience={QUICK_PATIENCE}, tune-trials={QUICK_TUNE_TRIALS}, "
            "auto --force + --deploy-force. Needs real dataset + torch. "
            "Mutually exclusive with --smoke."
        ),
    )

    return parser


def _bump_fd_limit() -> None:
    """Raise RLIMIT_NOFILE so multi-trial sweeps don't trip macOS's 256 default.

    On macOS, processes launched outside an interactive login shell (IDE
    terminals, ``nohup``, launchd agents) inherit launchd's 256 soft FD
    limit instead of the bumped shell ulimit. With 8 search trials +
    ``persistent_workers=True`` DataLoaders + MLflow/Optuna SQLite pools,
    that ceiling gets hit somewhere around the train phase and
    ``os.pipe()`` raises ``EMFILE`` inside ``multiprocessing.spawn``.
    Bumping at process start is belt-and-braces — the leak is still
    worth fixing, but this prevents the silent crash in the meantime.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(hard, 1 << 20)
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        logger.info("bumped RLIMIT_NOFILE %d -> %d", soft, target)


def main(argv: list[str]) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    _bump_fd_limit()
    args = _build_parser().parse_args(argv)
    spec = _resolve_model_spec(args.model)
    task_id = args.task_id or generate_task_id()
    started = time.monotonic()

    if args.phase == "hparam":
        run_hparam(
            spec,
            trials=args.trials,
            epochs=args.epochs,
            smoke=args.smoke,
            task_id=task_id,
        )
        print(f"hparam task_id={task_id}")
        return 0

    if args.phase == "train":
        staging = run_train(
            spec,
            max_epochs=args.max_epochs,
            patience=args.patience,
            smoke=args.smoke,
            task_id=task_id,
        )
        elapsed = time.monotonic() - started
        print(f"wrote {staging} in {elapsed:.1f}s")
        return 0

    if args.phase == "tune":
        target = args.staging_dir or latest_staging_dir(
            dataset_id=spec.dataset.disease_id, model_id=spec.model_id
        )
        out = run_tune(spec, staging_dir=target, trials=args.trials, smoke=args.smoke)
        print(f"tuned {out}")
        return 0

    if args.phase == "deploy":
        target = args.staging_dir or latest_staging_dir(
            dataset_id=spec.dataset.disease_id, model_id=spec.model_id
        )
        stable = run_deploy(
            spec, staging_dir=target, smoke=args.smoke, force=args.deploy_force
        )
        print(f"deployed {stable}")
        return 0

    if args.phase == "pipeline":
        if args.quick and args.smoke:
            raise SystemExit("--quick and --smoke are mutually exclusive.")
        if args.quick:
            for name, value in _QUICK_OVERRIDES.items():
                setattr(args, name, value)
            logger.info(
                "quick mode: trials=%d search_epochs=%d max_epochs=%d "
                "patience=%d tune_trials=%d force=%s deploy_force=%s",
                args.trials,
                args.search_epochs,
                args.max_epochs,
                args.patience,
                args.tune_trials,
                args.force,
                args.deploy_force,
            )
        phases = parse_phases(args.phases)
        stable = run_pipeline(
            spec,
            phases=phases,
            trials=args.trials,
            search_epochs=args.search_epochs,
            max_epochs=args.max_epochs,
            patience=args.patience,
            tune_trials=args.tune_trials,
            smoke=args.smoke,
            staging_dir=args.staging_dir,
            task_id=task_id,
            force=args.force,
            deploy_force=args.deploy_force,
        )
        elapsed = time.monotonic() - started
        if args.smoke and stable is not None:
            print(f"smoke ok in {elapsed:.1f}s — staging at {stable}")
        elif stable is not None:
            print(f"pipeline ok in {elapsed:.1f}s — deployed {stable}")
        else:
            print(f"pipeline ok in {elapsed:.1f}s (no deploy phase)")
        return 0

    raise SystemExit(f"unknown phase {args.phase!r}")


def cli() -> None:  # pragma: no cover
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":  # pragma: no cover
    cli()
