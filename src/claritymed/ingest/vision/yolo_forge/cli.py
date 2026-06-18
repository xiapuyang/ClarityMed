"""``claritymed-vision-yolo-forge`` — one entry, every detection dataset.

Mirrors :mod:`claritymed.ingest.vision.forge.cli`. Phases:
``prepare`` / ``search`` / ``train`` / ``tune`` / ``deploy`` /
``pipeline``. ``eval`` is also exposed as a standalone re-evaluation
subcommand (not part of the pipeline; tune+deploy cover the in-loop
eval).

Usage:

.. code-block:: bash

    # End-to-end with HPO (~1 GPU-hour on RSNA-scale):
    claritymed-vision-yolo-forge pipeline \\
        --model claritymed.ingest.vision.rsna_pneumonia_yolo.models.yolov8n_v1:RSNA_YOLOV8N_V1

    # Quick dry run (skip HPO, 1 train epoch):
    claritymed-vision-yolo-forge pipeline \\
        --model claritymed.ingest.vision.rsna_pneumonia_yolo.models.yolov8n_v1:RSNA_YOLOV8N_V1 \\
        --quick --skip-search

    # Single-phase entry points:
    claritymed-vision-yolo-forge prepare --model <spec>
    claritymed-vision-yolo-forge search  --model <spec> --trials 10 --epochs-per-trial 5
    claritymed-vision-yolo-forge train   --model <spec>
    claritymed-vision-yolo-forge tune    --model <spec> --trials 20
    claritymed-vision-yolo-forge deploy  --model <spec>

    # One-off re-eval at a chosen (conf, iou):
    claritymed-vision-yolo-forge eval    --model <spec> --split val --conf 0.3 --iou 0.5

``--model`` resolves to a :class:`YoloModelSpec` via dotted-path +
colon-attribute, identical to forge.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from claritymed.ingest.vision.yolo_forge.common import (
    generate_task_id,
    parse_phases,
    resolve_model_spec,
    staging_dir,
)
from claritymed.ingest.vision.yolo_forge.framework import (
    DEFAULT_IMAGE_CONF,
    DEFAULT_NMS_IOU,
    DEFAULT_SEARCH_EPOCHS,
    DEFAULT_SEARCH_TRIALS,
    DEFAULT_TUNE_TRIALS,
    run_deploy,
    run_eval,
    run_prepare,
    run_search,
    run_train,
    run_tune,
)

logger = logging.getLogger(__name__)


def _add_model_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model",
        required=True,
        help="Dotted path to YoloModelSpec: 'pkg.mod:ATTR'.",
    )


def _add_staging_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Override the staging dir (default: latest for this model_id).",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="claritymed-vision-yolo-forge",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="Materialise YOLO format on disk.")
    _add_model_arg(p_prepare)

    p_search = sub.add_parser(
        "search", help="Optuna HPO over spec.hparam_space; persist best_hparams.json."
    )
    _add_model_arg(p_search)
    p_search.add_argument(
        "--trials", type=int, default=DEFAULT_SEARCH_TRIALS, help="Optuna trial count."
    )
    p_search.add_argument(
        "--epochs-per-trial",
        type=int,
        default=DEFAULT_SEARCH_EPOCHS,
        help="Epochs each HPO trial trains for.",
    )
    _add_staging_arg(p_search)

    p_train = sub.add_parser("train", help="Train the YOLO model.")
    _add_model_arg(p_train)
    p_train.add_argument(
        "--quick",
        action="store_true",
        help="Override epochs=1 + patience=1 for a fresh-box dry run.",
    )
    p_train.add_argument(
        "--use-search",
        action="store_true",
        help=(
            "Read <staging>/best_hparams.json (must already exist) and overlay "
            "onto the spec defaults before training."
        ),
    )
    _add_staging_arg(p_train)

    p_tune = sub.add_parser(
        "tune",
        help=(
            "Optuna HPO over spec.inference_space on val, then test eval; "
            "writes eval_metrics.json."
        ),
    )
    _add_model_arg(p_tune)
    p_tune.add_argument(
        "--trials", type=int, default=DEFAULT_TUNE_TRIALS, help="Optuna trial count."
    )
    _add_staging_arg(p_tune)

    p_eval = sub.add_parser(
        "eval",
        help="Standalone re-eval at chosen (conf, iou). Not part of the pipeline.",
    )
    _add_model_arg(p_eval)
    _add_staging_arg(p_eval)
    p_eval.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
        help="Which split to evaluate (default: test).",
    )
    p_eval.add_argument("--conf", type=float, default=DEFAULT_IMAGE_CONF)
    p_eval.add_argument("--iou", type=float, default=DEFAULT_NMS_IOU)

    p_deploy = sub.add_parser("deploy", help="Gate + promote a trained model.")
    _add_model_arg(p_deploy)
    _add_staging_arg(p_deploy)

    p_pipeline = sub.add_parser(
        "pipeline",
        help="Run prepare → search → train → tune → deploy in sequence.",
    )
    _add_model_arg(p_pipeline)
    p_pipeline.add_argument(
        "--quick",
        action="store_true",
        help="Pass quick=True down to the train phase.",
    )
    p_pipeline.add_argument(
        "--skip-search",
        action="store_true",
        help="Skip HPO; train with spec-default hparams.",
    )
    p_pipeline.add_argument("--search-trials", type=int, default=DEFAULT_SEARCH_TRIALS)
    p_pipeline.add_argument("--search-epochs", type=int, default=DEFAULT_SEARCH_EPOCHS)
    p_pipeline.add_argument("--tune-trials", type=int, default=DEFAULT_TUNE_TRIALS)
    p_pipeline.add_argument(
        "--phases",
        default=None,
        help=(
            "Comma-separated subset of {prepare,search,train,tune,deploy} "
            "(re-ordered to canonical sequence). Omit to run all five."
        ),
    )

    return parser


def cli(argv: list[str] | None = None) -> int:
    """Console-script entry. Returns process exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    spec = resolve_model_spec(args.model)
    started = time.monotonic()

    if args.command == "prepare":
        run_prepare(spec)
    elif args.command == "search":
        splits = run_prepare(spec)
        run_search(
            spec,
            splits,
            trials=args.trials,
            epochs_per_trial=args.epochs_per_trial,
            staging=args.staging_dir,
        )
    elif args.command == "train":
        splits = run_prepare(spec)
        hparams_override = (
            _load_best_hparams(args.staging_dir, spec) if args.use_search else None
        )
        run_train(
            spec,
            splits,
            hparams_override=hparams_override,
            quick=args.quick,
            staging=args.staging_dir,
        )
    elif args.command == "tune":
        splits = run_prepare(spec)
        run_tune(spec, splits, trials=args.trials, staging=args.staging_dir)
    elif args.command == "eval":
        splits = run_prepare(spec)
        run_eval(
            spec,
            splits,
            staging=args.staging_dir,
            split=args.split,
            conf=args.conf,
            iou=args.iou,
        )
    elif args.command == "deploy":
        run_deploy(spec, staging=args.staging_dir)
    elif args.command == "pipeline":
        _run_pipeline_phases(spec, args)
    else:  # pragma: no cover — argparse enforces required subcommand
        raise SystemExit(f"unknown command: {args.command!r}")

    logger.info(
        "yolo_forge.%s: done in %.1fs", args.command, time.monotonic() - started
    )
    return 0


def _load_best_hparams(staging: Path | None, spec) -> dict:
    """Load ``<staging>/best_hparams.json`` (fail loud when missing)."""
    import json

    from claritymed.ingest.vision.yolo_forge.common import latest_staging_dir

    s = staging or latest_staging_dir(
        dataset_id=spec.dataset.dataset_id, model_id=spec.model_id
    )
    path = s / "best_hparams.json"
    if not path.is_file():
        raise SystemExit(
            f"--use-search: {path} missing; run `search` first or drop the flag."
        )
    blob = json.loads(path.read_text())
    return blob["merged_hparams"]


def _run_pipeline_phases(spec, args) -> None:
    """Apply ``--phases`` filter + sequence the pipeline against one staging dir."""
    phases = parse_phases(args.phases)
    # One staging dir for the whole pipeline so search → train → tune
    # all land under the same UTC-timestamped directory.
    out_dir = staging_dir(dataset_id=spec.dataset.dataset_id, model_id=spec.model_id)
    task_id = generate_task_id()
    logger.info("yolo_forge.pipeline: task_id=%s", task_id)
    splits = None
    hparams_override = None

    if "prepare" in phases:
        splits = run_prepare(spec)
    if "search" in phases:
        if splits is None:
            splits = run_prepare(spec)
        if not args.skip_search and spec.hparam_space:
            hparams_override = run_search(
                spec,
                splits,
                trials=args.search_trials,
                epochs_per_trial=args.search_epochs,
                staging=out_dir,
                task_id=task_id,
            )
    if "train" in phases:
        if splits is None:
            splits = run_prepare(spec)
        run_train(
            spec,
            splits,
            hparams_override=hparams_override,
            quick=args.quick,
            staging=out_dir,
            task_id=task_id,
        )
    if "tune" in phases:
        if splits is None:
            splits = run_prepare(spec)
        run_tune(
            spec, splits, trials=args.tune_trials, staging=out_dir, task_id=task_id
        )
    if "deploy" in phases:
        run_deploy(spec, staging=out_dir, task_id=task_id)


def main() -> None:  # console-script wrapper
    sys.exit(cli())


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = ["cli", "main"]
