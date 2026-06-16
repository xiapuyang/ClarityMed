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
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from claritymed.ingest.mlflow_utils import generate_task_id, optuna_storage_uri
from claritymed.ingest.vision.busi import deploy, hparam, train, tune
from claritymed.ingest.vision.busi.scoring import (
    SEARCH_FLOORS,
    TRAIN_FLOORS,
    TUNE_FLOORS,
    gate_or_raise,
)
from claritymed.ingest.vision.busi.tune import _latest_staging_dir

logger = logging.getLogger(__name__)

ALL_PHASES = ("search", "train", "tune", "deploy")


def _read_search_winner_breakdown(task_id: str) -> dict[str, float] | None:
    """Load the best Optuna trial for ``task_id`` and return its breakdown.

    Returns ``None`` if the study doesn't exist (smoke runs skip Optuna)
    or the best trial didn't stash a breakdown — the gate skips
    silently in those cases since there's nothing concrete to check.
    """
    try:
        import optuna
    except ImportError:
        return None
    name = hparam.study_id(task_id)
    try:
        study = optuna.load_study(study_name=name, storage=optuna_storage_uri())
    except KeyError:
        return None
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not completed:
        return None
    best = max(completed, key=lambda t: float("-inf") if t.value is None else t.value)
    breakdown = best.user_attrs.get("breakdown")
    if not breakdown:
        return None
    return dict(breakdown)


def _read_staging_breakdown(staging_dir: Path, key: str) -> dict[str, float] | None:
    """Read a specific breakdown dict out of ``eval_metrics.json``.

    ``key`` is the top-level field name (``val_breakdown`` after train,
    ``tuned_test_breakdown`` after tune). Returns ``None`` when the
    file or key is missing — pipeline gates degrade gracefully on
    partial artifacts.
    """
    path = staging_dir / "eval_metrics.json"
    if not path.exists():
        return None
    try:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    block = payload.get(key)
    if not isinstance(block, dict):
        return None
    return {k: float(v) for k, v in block.items() if isinstance(v, (int, float))}


def run_pipeline(
    *,
    phases: tuple[str, ...],
    trials: int,
    search_epochs: int,
    max_epochs: int,
    patience: int,
    tune_trials: int,
    smoke: bool,
    staging_dir: Path | None,
    task_id: str | None = None,
    force: bool = False,
    deploy_force: bool = False,
) -> Path | None:
    """Run the requested phases in order. Returns the final stable path or None.

    A single ``task_id`` is generated (or passed in) once at the start
    and threaded into every phase — search trials, train MLflow tags,
    tune MLflow tags + Optuna trials, deploy LATEST.jsonl row. From any
    artifact a viewer can reach all the others by filtering on it.

    Between phases the winner's breakdown is checked against that
    phase's :class:`~claritymed.ingest.vision.busi.scoring.PhaseFloors`.
    A failure bails before the next phase runs (saving wasted compute);
    ``force=True`` downgrades the gate to a warning so an operator can
    push through for diagnostic runs.

    Smoke mode skips the inter-phase gates — the synthetic numbers from
    the smoke helpers are designed to be feasible by construction, but
    in CI we don't want to depend on that invariant for orchestration
    tests.
    """
    if task_id is None:
        task_id = generate_task_id()
    logger.info("pipeline task_id=%s force=%s", task_id, force)

    staging = staging_dir
    stable_path: Path | None = None

    if "search" in phases:
        logger.info("=== phase: search (Optuna HP) — task_id=%s ===", task_id)
        if smoke:
            logger.info("smoke: skipping real Optuna study build")
        else:
            hparam.run_search(
                trials=trials, epochs=search_epochs, smoke=False, task_id=task_id
            )
            breakdown = _read_search_winner_breakdown(task_id)
            if breakdown is not None:
                gate_or_raise(
                    phase_label="search",
                    breakdown=breakdown,
                    floors=SEARCH_FLOORS,
                    force=force,
                )

    if "train" in phases:
        logger.info(
            "=== phase: train (feasibility-aware early stop) — task_id=%s ===",
            task_id,
        )
        staging = train.run_production_training(
            max_epochs=max_epochs,
            patience=patience,
            smoke=smoke,
            task_id=task_id,
        )
        logger.info("staging dir: %s", staging)
        if not smoke:
            breakdown = _read_staging_breakdown(staging, "val_breakdown")
            if breakdown is not None:
                gate_or_raise(
                    phase_label="train",
                    breakdown=breakdown,
                    floors=TRAIN_FLOORS,
                    force=force,
                )

    if "tune" in phases:
        logger.info(
            "=== phase: tune (inference-param Optuna) — task_id=%s ===", task_id
        )
        target = staging or _latest_staging_dir()
        tune.run_tune(staging_dir=target, trials=tune_trials, smoke=smoke)
        staging = target
        if not smoke:
            # Gate against tuned_test_breakdown (held-out split under
            # the chosen tuned params) — that's what deploy will check
            # next, surfaced one step earlier so a doomed run aborts
            # before the YAML edit + LATEST.jsonl write.
            breakdown = _read_staging_breakdown(staging, "tuned_test_breakdown")
            if breakdown is not None:
                gate_or_raise(
                    phase_label="tune",
                    breakdown=breakdown,
                    floors=TUNE_FLOORS,
                    force=force,
                )

    if "deploy" in phases:
        logger.info(
            "=== phase: deploy (floor + regression gate + versioned promote) "
            "— task_id=%s ===",
            task_id,
        )
        target = staging or _latest_staging_dir()
        stable_path = deploy.run_deploy(
            staging_dir=target, smoke=smoke, force=deploy_force
        )
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
        "--search-epochs",
        type=int,
        default=15,
        help=(
            "Epochs per search trial. Too-low values starve dice (the "
            "mask head needs time to converge) and surface as a "
            "search-phase floor failure on `dice`. 15 is the floor-passing "
            "default; lower it only for diagnostic sweeps."
        ),
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
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Downgrade inter-phase floor gates to warnings. Default off; "
            "use only for diagnostic runs against a known-broken HP combo. "
            "Does not affect the deploy floor gate — use --deploy-force for that."
        ),
    )
    parser.add_argument(
        "--deploy-force",
        action="store_true",
        help=(
            "Skip the deploy floor gate and promote the checkpoint anyway. "
            "WARNING: the deployed model may not meet medical safety thresholds. "
            "Use only for development or pipeline-wiring tests."
        ),
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
        search_epochs=args.search_epochs,
        max_epochs=args.max_epochs,
        patience=args.patience,
        tune_trials=args.tune_trials,
        smoke=args.smoke,
        staging_dir=args.staging_dir,
        task_id=args.task_id,
        force=args.force,
        deploy_force=args.deploy_force,
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
