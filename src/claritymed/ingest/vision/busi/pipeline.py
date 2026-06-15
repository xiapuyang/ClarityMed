"""End-to-end orchestrator: search → train → tune → deploy.

Single CLI that runs all four phases in order, passing the staging dir
forward. Each phase is also callable independently (its own ``main``
+ console script), so this orchestrator is **not** the only entry —
operators can resume from any phase if a step needs to re-run.

Design constraints (rationale lives in the brainstorm + plan docs):

* **Single objective at every tuning step** — hparam search and tune
  both optimize the same composite
  (``0.6 * malignant_recall + 0.4 * dice``); the deploy floors + the
  regression-vs-active gate are pass/fail gates, never weighted into a
  score. No magic-constant tradeoff knobs in any phase.
* **All non-HP inference-affecting params live in tune**, not in
  config or in train. Adding a new knob is one entry in
  ``tune.py::_suggest_params`` + one field in
  :class:`~claritymed.core.vision.schemas.TunedInferenceParams`.
* **Train never overwrites a stable model** — the staging dir is
  timestamped, deploy promotes to a versioned sibling, and
  ``configs/vision.yaml`` is patched in place with a minimal diff.
  Old versions stay on disk for rollback.

Smoke mode (``--smoke``) wires every phase end-to-end on synthetic
data so the orchestrator can be verified without GPU + real BUSI.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from claritymed.ingest.mlflow_utils import generate_task_id
from claritymed.ingest.vision.busi import deploy, hparam, train, tune
from claritymed.ingest.vision.busi.tune import _latest_staging_dir

logger = logging.getLogger(__name__)

ALL_PHASES = ("search", "train", "tune", "deploy")


def run_pipeline(
    *,
    phases: tuple[str, ...],
    trials: int,
    max_epochs: int,
    patience: int,
    tune_trials: int,
    smoke: bool,
    staging_dir: Path | None,
    task_id: str | None = None,
) -> Path | None:
    """Run the requested phases in order. Returns the final stable path or None.

    A single ``task_id`` is generated (or passed in) once at the start
    and threaded into every phase — search trials, train MLflow tags,
    tune MLflow tags + Optuna trials, deploy LATEST.jsonl row. From any
    artifact a viewer can reach all the others by filtering on it.
    """
    if task_id is None:
        task_id = generate_task_id()
    logger.info("pipeline task_id=%s", task_id)

    staging = staging_dir
    stable_path: Path | None = None

    if "search" in phases:
        logger.info("=== phase: search (Optuna HP) — task_id=%s ===", task_id)
        if smoke:
            logger.info("smoke: skipping real Optuna study build")
        else:
            hparam.run_search(trials=trials, epochs=5, smoke=False, task_id=task_id)

    if "train" in phases:
        logger.info(
            "=== phase: train (early-stop on val composite) — task_id=%s ===", task_id
        )
        staging = train.run_production_training(
            max_epochs=max_epochs,
            patience=patience,
            smoke=smoke,
            task_id=task_id,
        )
        logger.info("staging dir: %s", staging)

    if "tune" in phases:
        logger.info(
            "=== phase: tune (inference-param Optuna) — task_id=%s ===", task_id
        )
        target = staging or _latest_staging_dir()
        tune.run_tune(staging_dir=target, trials=tune_trials, smoke=smoke)
        staging = target

    if "deploy" in phases:
        logger.info(
            "=== phase: deploy (floor + regression gate + versioned promote) "
            "— task_id=%s ===",
            task_id,
        )
        target = staging or _latest_staging_dir()
        stable_path = deploy.run_deploy(staging_dir=target, smoke=smoke)
        logger.info("deployed to: %s", stable_path)

    return stable_path


def _parse_phases(raw: str) -> tuple[str, ...]:
    """Validate and normalize ``--phases``."""
    items = tuple(x.strip() for x in raw.split(",") if x.strip())
    unknown = [item for item in items if item not in ALL_PHASES]
    if unknown:
        raise SystemExit(
            f"--phases got unknown {unknown!r}; valid: {list(ALL_PHASES)!r}"
        )
    # Re-order to match the canonical pipeline order so accidental
    # "deploy,tune" specs run as "tune,deploy" instead of failing
    # opaquely on a missing manifest.
    return tuple(p for p in ALL_PHASES if p in items)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phases",
        default=",".join(ALL_PHASES),
        help="Comma-separated subset of: " + ",".join(ALL_PHASES),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Optuna trial count for the HP search phase.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Train phase epoch ceiling; early stopping picks the actual stop.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Train phase patience on val composite stagnation.",
    )
    parser.add_argument(
        "--tune-trials",
        type=int,
        default=30,
        help="Optuna trial count for the inference-param tune phase.",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help="Path to a tune-or-train staging dir (skips upstream phases).",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        help=(
            "Pipeline lineage id. Generated when absent and threaded "
            "into every phase (MLflow tags, Optuna trials, "
            "provenance.json, LATEST.jsonl). Pass an existing id to "
            "extend a prior pipeline run."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run every phase on synthetic data; deploy skips configs/vision.yaml edit.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    phases = _parse_phases(args.phases)
    started = time.monotonic()
    stable = run_pipeline(
        phases=phases,
        trials=args.trials,
        max_epochs=args.max_epochs,
        patience=args.patience,
        tune_trials=args.tune_trials,
        smoke=args.smoke,
        staging_dir=args.staging_dir,
        task_id=args.task_id,
    )
    elapsed = time.monotonic() - started
    if stable is not None:
        print(f"pipeline ok in {elapsed:.1f}s — deployed {stable}")
    else:
        print(f"pipeline ok in {elapsed:.1f}s (no deploy phase)")
    return 0


def cli() -> None:  # pragma: no cover
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":  # pragma: no cover
    cli()
