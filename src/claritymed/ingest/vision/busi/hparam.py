"""Optuna hyperparameter search for BUSI U-Net.

Search space:

* ``backbone`` — ``"resnet50"`` | ``"efficientnet_b0"`` | ``"custom_unet"``
* ``lr`` — log-uniform in ``[1e-5, 5e-3]``
* ``seg_loss_weight`` — uniform in ``[0.1, 2.0]``

Each trial runs a short training loop (default 5 epochs) and reports
the early-stopping composite score (``val/malignant_recall * 0.6 +
val/dice * 0.4``) — `malignant` recall is the medically-real bar
(see plan §"Unit 6 Verification"). Trials persist to the shared Optuna
storage at ``CLARITYMED_HOME/tracking/optuna.db`` (study disambiguated
by :func:`~claritymed.ingest.mlflow_utils.study_name`) so a Ctrl-C
resume continues from the last completed trial.

This file is a CLI entry; the actual model + dataset wiring lives in
``train.py`` so a one-shot operator pass can also invoke training
directly without going through Optuna.
"""

from __future__ import annotations

import argparse
import logging
import sys

from claritymed.ingest.mlflow_utils import (
    generate_task_id,
    optuna_storage_uri,
    study_name,
)
from claritymed.ingest.vision.busi.train import DATASET_ID

logger = logging.getLogger(__name__)

PHASE = "hparam"


def study_id(task_id: str) -> str:
    """Canonical study name for the BUSI hparam search of one pipeline run.

    One study per ``task_id``: re-running the same task_id resumes
    (Optuna ``load_if_exists=True``), a fresh task_id starts a clean
    study. Keeps prior-run garbage out of ``study.best_trial`` without
    needing a separate ``strict_task`` filter downstream.

    Shares ``DATASET_ID`` with :mod:`~claritymed.ingest.vision.busi.train`
    so the MLflow experiment + Optuna study + on-disk run dir stay
    aligned.
    """
    return f"{study_name('vision', DATASET_ID, PHASE)}-{task_id}"


def run_search(*, trials: int, epochs: int, smoke: bool, task_id: str) -> str:
    """Run the Optuna search; every trial is tagged with ``task_id``.

    Returns the task_id so the calling orchestrator can thread the same
    value into train + tune + deploy. Persists trials to the shared
    Optuna storage so a Ctrl-C resume continues from the last completed
    trial — task_id is carried per-trial, so subsequent runs with a
    different task_id append rather than overwriting.
    """
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "optuna not installed — run `uv sync --extra vision-server`."
        ) from exc

    study = optuna.create_study(
        study_name=study_id(task_id),
        storage=optuna_storage_uri(),
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(
        _build_objective(epochs=epochs, smoke=smoke, task_id=task_id),
        n_trials=trials,
    )
    return task_id


def _build_objective(*, epochs: int, smoke: bool, task_id: str):
    """Construct the Optuna objective.

    Imports torch lazily so the module stays importable on a CPU-only
    box that just wants to read the search-space docstring. The
    ``task_id`` lands on every trial's ``user_attrs`` so downstream
    phases can pivot from the trial back to the owning pipeline run.
    """

    def _objective(trial) -> float:
        from claritymed.ingest.vision.busi.train import run_training_trial

        trial.set_user_attr("task_id", task_id)
        params = {
            "backbone": trial.suggest_categorical(
                "backbone", ["resnet50", "efficientnet_b0", "custom_unet"]
            ),
            "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
            "seg_loss_weight": trial.suggest_float("seg_loss_weight", 0.1, 2.0),
        }
        return run_training_trial(params, epochs=epochs, smoke=smoke, trial=trial)

    return _objective


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--task-id",
        default=None,
        help="Pipeline lineage id. Generated when absent — printed at start so "
        "downstream train/tune/deploy can re-use the same value.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny per-trial subsample (10 images, 1 epoch) — wires the loop without training",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    task_id = args.task_id or generate_task_id()
    print(f"hparam task_id={task_id}")

    run_search(
        trials=args.trials, epochs=args.epochs, smoke=args.smoke, task_id=task_id
    )

    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "optuna not installed — run `uv sync --extra vision-server`."
        ) from exc
    study = optuna.load_study(
        study_name=study_id(task_id), storage=optuna_storage_uri()
    )
    best = study.best_trial
    logger.info(
        "best trial #%d score=%.4f params=%s", best.number, best.value, best.params
    )
    print(
        f"best: trial #{best.number} score={best.value:.4f} "
        f"task_id={best.user_attrs.get('task_id')!r} params={best.params}"
    )
    print(f"hparam task_id={task_id}")
    return 0


def cli() -> None:  # pragma: no cover - thin entry point
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":  # pragma: no cover
    cli()
