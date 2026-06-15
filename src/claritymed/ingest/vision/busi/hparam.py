"""Optuna hyperparameter search for BUSI U-Net.

Search space:

* ``backbone`` — ``"resnet50"`` | ``"efficientnet_b0"`` | ``"custom_unet"``
* ``lr`` — log-uniform in ``[1e-5, 5e-3]``
* ``seg_loss_weight`` — uniform in ``[0.1, 2.0]``

Each trial runs a short training loop (default 5 epochs) and reports
the early-stopping composite score (``val/malignant_recall * 0.6 +
val/dice * 0.4``) — `malignant` recall is the medically-real bar
(see plan §"Unit 6 Verification"). Trials persist to
``CLARITYMED_HOME/models/vision/breast_cancer_ultrasound/run/hparam.db``
so a Ctrl-C resume continues from the last completed trial.

This file is a CLI entry; the actual model + dataset wiring lives in
``train.py`` so a one-shot operator pass can also invoke training
directly without going through Optuna.
"""

from __future__ import annotations

import argparse
import logging
import sys

from claritymed import config as _cfg

logger = logging.getLogger(__name__)

STUDY_NAME = "claritymed-vision-breast-cancer-ultrasound-hparam"


def hparam_db_uri() -> str:
    """SQLite URI under the BUSI run dir, CWD-independent."""
    db = (
        _cfg.CLARITYMED_HOME
        / "models"
        / "vision"
        / "breast_cancer_ultrasound"
        / "run"
        / "hparam.db"
    )
    db.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db}"


def _build_objective(*, epochs: int, smoke: bool):
    """Construct the Optuna objective.

    Imports torch lazily so the module stays importable on a CPU-only
    box that just wants to read the search-space docstring.
    """

    def _objective(trial) -> float:
        from claritymed.ingest.vision.busi.train import run_training_trial

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
        "--smoke",
        action="store_true",
        help="Tiny per-trial subsample (10 images, 1 epoch) — wires the loop without training",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "optuna not installed — run `uv sync --extra vision-server`."
        ) from exc

    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=hparam_db_uri(),
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(
        _build_objective(epochs=args.epochs, smoke=args.smoke),
        n_trials=args.trials,
    )
    best = study.best_trial
    logger.info(
        "best trial #%d score=%.4f params=%s", best.number, best.value, best.params
    )
    print(f"best: trial #{best.number} score={best.value:.4f} params={best.params}")
    return 0


def cli() -> None:  # pragma: no cover - thin entry point
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":  # pragma: no cover
    cli()
