"""yolo_forge phase implementations: prepare → search → train → tune → deploy.

Each phase function takes a :class:`YoloModelSpec` plus phase-specific
kwargs and produces a side-effect on disk (and a small return value the
next phase consumes). The CLI in :mod:`yolo_forge.cli` is the only
place phases are sequenced; tests import phase functions directly.

Phase semantics mirror :mod:`claritymed.ingest.vision.forge.framework`:

* ``prepare`` — materialise YOLO format on disk (forge folds this into
  the dataset spec's ``build_splits``; we keep it as an observable
  phase because the symlink tree + ``data.yaml`` are user-inspectable).
* ``search`` — Optuna HPO over ``spec.hparam_space``. Each trial trains
  a short epoch budget and scores ``mAP50`` on the val split; the best
  param set persists to ``<staging>/best_hparams.json``.
* ``train`` — full training with best (or default) hparams on the
  train split, early-stopped on val.
* ``tune`` — Optuna HPO over ``spec.inference_space`` (conf / iou) on
  the val split, then a final evaluation on the test split with the
  winning params. Writes ``eval_metrics.json`` containing both the
  tuned params and the test metrics — that is what ``deploy`` reads.
* ``deploy`` — gate on ``spec.eval_thresholds`` + regression check
  against the last ``LATEST.jsonl`` entry for the same ``model_id``.

Heavy imports (``ultralytics``, ``optuna``, ``torch``) are deferred to
function bodies so this module stays importable from the smoke path
without the ``yolo-forge`` extra installed.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from claritymed.ingest.vision.yolo_forge.common import (
    append_entry,
    disease_root,
    generate_task_id,
    latest_jsonl_path,
    latest_staging_dir,
    log_metrics,
    mlflow_phase_run,
    read_last_entry,
    staging_dir,
    version_tag,
)
from claritymed.ingest.vision.yolo_forge.spec import (
    DetectionSplits,
    YoloModelSpec,
)

logger = logging.getLogger(__name__)

PIPELINE_TAG = "yolo_forge"

# Per-image confidence threshold default for the image-level recall
# collapse when no tuned value is on hand (e.g. the standalone
# ``eval`` subcommand). 0.25 matches Ultralytics' predict default.
DEFAULT_IMAGE_CONF: float = 0.25
DEFAULT_NMS_IOU: float = 0.5

# Search phase defaults — kept low so a full ``pipeline`` invocation
# finishes within ~1 GPU-hour on the RSNA-scale dataset. Override at
# the CLI via ``--search-trials`` / ``--search-epochs``.
DEFAULT_SEARCH_TRIALS = 10
DEFAULT_SEARCH_EPOCHS = 5

# Tune phase: cheap (no retraining, just predict + val), so we can
# afford more trials. Each trial is one ``model.val`` pass.
DEFAULT_TUNE_TRIALS = 20

# Shared HPO objective for both search + tune: Ultralytics' canonical
# detection-fitness formula (``ultralytics.utils.metrics.DetMetrics.fitness``).
# Heavily favours the stricter mAP50-95 over mAP50 so HPO doesn't pick
# hparams (or a conf threshold) that look good at IoU=0.5 but localise
# poorly at higher IoU bands. Clinical fail-safe (image-level recall ≥
# floor) stays enforced by the deploy gate's ``eval_thresholds`` —
# keeping the HPO objective purely detection-quality matches how the
# rest of the ecosystem (Ultralytics training fitness, COCO leaderboards)
# ranks models.
FITNESS_MAP50_WEIGHT = 0.1
FITNESS_MAP_WEIGHT = 0.9


def _fitness(map50: float, map50_95: float) -> float:
    """Apply the fitness formula. Same shape used by search + tune."""
    return FITNESS_MAP50_WEIGHT * map50 + FITNESS_MAP_WEIGHT * map50_95


# --- prepare -------------------------------------------------------------


def run_prepare(spec: YoloModelSpec) -> DetectionSplits:
    """Materialise the dataset in YOLO format on disk; return splits info."""
    logger.info(
        "yolo_forge.prepare: dataset_id=%s disease_id=%s",
        spec.dataset.dataset_id,
        spec.dataset.disease_id,
    )
    splits = spec.dataset.prepare_fn()
    splits.assert_non_empty()
    logger.info(
        "yolo_forge.prepare: data.yaml=%s train=%d val=%d test=%d",
        splits.data_yaml_path,
        splits.train_count,
        splits.val_count,
        splits.test_count,
    )
    return splits


# --- search --------------------------------------------------------------


def run_search(
    spec: YoloModelSpec,
    splits: DetectionSplits,
    *,
    trials: int = DEFAULT_SEARCH_TRIALS,
    epochs_per_trial: int = DEFAULT_SEARCH_EPOCHS,
    staging: Path | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Optuna HPO over ``spec.hparam_space``; return best hparams.

    Each trial trains a short (``epochs_per_trial``) run with the
    suggested hparams overlaid on the spec defaults, then scores the
    Ultralytics fitness composite (``0.1·mAP50 + 0.9·mAP50-95``) on
    the val split. The best set persists to
    ``<staging>/best_hparams.json`` and is also returned for the
    train phase to consume.

    No-op (returns the spec defaults) when ``hparam_space`` is empty —
    spec authors can ship a model without HPO and still ride the
    pipeline.
    """
    if not spec.hparam_space:
        logger.info("yolo_forge.search: spec.hparam_space empty — skipping.")
        return asdict(spec.train_hparams)

    import optuna
    from ultralytics import YOLO

    out_dir = staging or staging_dir(
        dataset_id=spec.dataset.dataset_id, model_id=spec.model_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    search_dir = out_dir / "search"
    search_dir.mkdir(exist_ok=True)
    task_id = task_id or generate_task_id()

    with mlflow_phase_run(
        spec=spec,
        phase="search",
        task_id=task_id,
        params={"trials": trials, "epochs_per_trial": epochs_per_trial},
    ):

        def objective(trial: optuna.Trial) -> float:
            suggested = spec.suggest_hparams(trial)
            hparams = asdict(spec.train_hparams)
            hparams.update(suggested)
            hparams["epochs"] = epochs_per_trial
            hparams["patience"] = max(1, epochs_per_trial // 2)

            model = YOLO(spec.base_weights)
            results = model.train(
                data=str(splits.data_yaml_path),
                project=str(search_dir),
                name=f"trial_{trial.number:03d}",
                exist_ok=True,
                verbose=False,
                **hparams,
            )
            map50 = float(getattr(results.box, "map50", 0.0))
            map50_95 = float(getattr(results.box, "map", 0.0))
            fitness = _fitness(map50=map50, map50_95=map50_95)
            # Step = trial number so the MLflow run shows a fitness
            # curve over trials, just like a per-epoch loss curve.
            log_metrics(
                {
                    "search/mAP50": map50,
                    "search/mAP50-95": map50_95,
                    "search/fitness": fitness,
                },
                step=trial.number,
            )
            return fitness

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials, show_progress_bar=False)

        best_hparams = asdict(spec.train_hparams)
        best_hparams.update(study.best_params)
        log_metrics({"search/best_fitness": float(study.best_value)})

    out_path = out_dir / "best_hparams.json"
    out_path.write_text(
        json.dumps(
            {
                "best_value": float(study.best_value),
                "best_params": study.best_params,
                "trials": trials,
                "epochs_per_trial": epochs_per_trial,
                "merged_hparams": best_hparams,
                "task_id": task_id,
            },
            indent=2,
        )
    )
    logger.info(
        "yolo_forge.search: best fitness=%.4f params=%s",
        study.best_value,
        study.best_params,
    )
    return best_hparams


# --- train ---------------------------------------------------------------


def run_train(
    spec: YoloModelSpec,
    splits: DetectionSplits,
    *,
    hparams_override: dict[str, Any] | None = None,
    quick: bool = False,
    staging: Path | None = None,
    task_id: str | None = None,
) -> Path:
    """Train the YOLO model; return the staging dir holding artifacts.

    ``hparams_override`` (typically from :func:`run_search`) supersedes
    :attr:`YoloModelSpec.train_hparams` field-by-field; missing keys
    fall back to spec defaults. ``quick=True`` overrides ``epochs=1``
    and ``patience=1`` after the override merge so a dry run still
    exercises the search-picked hparams.
    """
    from ultralytics import YOLO

    out_dir = staging or staging_dir(
        dataset_id=spec.dataset.dataset_id, model_id=spec.model_id
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    hparams = asdict(spec.train_hparams)
    if hparams_override:
        hparams.update(hparams_override)
    if quick:
        hparams["epochs"] = 1
        hparams["patience"] = 1
    task_id = task_id or generate_task_id()

    logger.info(
        "yolo_forge.train: base_weights=%s staging=%s hparams=%s",
        spec.base_weights,
        out_dir,
        hparams,
    )

    # Stringify all hparam values for MLflow log_params (it can't take
    # non-JSON-scalar types like Path).
    log_params = {k: str(v) for k, v in hparams.items()}
    log_params["base_weights"] = spec.base_weights

    with mlflow_phase_run(spec=spec, phase="train", task_id=task_id, params=log_params):
        model = YOLO(spec.base_weights)
        model.train(
            data=str(splits.data_yaml_path),
            project=str(out_dir),
            name="train",
            exist_ok=True,
            verbose=False,
            **hparams,
        )
        # Replay Ultralytics' per-epoch CSV into MLflow as proper
        # timeseries. We do this once after train rather than via a
        # callback so we don't have to wire into Ultralytics' lifecycle.
        _log_results_csv_to_mlflow(out_dir / "train" / "results.csv")

    _write_train_metadata(out_dir, spec, splits, hparams, task_id=task_id)
    return out_dir


def _log_results_csv_to_mlflow(csv_path: Path) -> None:
    """Read Ultralytics' ``results.csv`` and log each row as a step.

    Silent no-op when the file is missing — keeps the train phase from
    crashing if Ultralytics' output layout changes upstream. ``epoch``
    column drives ``step``; everything else logs as ``train/<col>``.
    """
    if not csv_path.is_file():
        logger.debug("yolo_forge.train: %s missing — skipping mlflow replay", csv_path)
        return
    import csv as _csv

    with csv_path.open() as fh:
        reader = _csv.DictReader(fh)
        for row in reader:
            try:
                step = int(float(row.get("epoch", 0)))
            except (TypeError, ValueError):
                continue
            metrics: dict[str, float] = {}
            for key, raw in row.items():
                if key == "epoch" or raw is None or raw == "":
                    continue
                try:
                    metrics[f"train/{key.strip()}"] = float(raw)
                except (TypeError, ValueError):
                    continue
            if metrics:
                log_metrics(metrics, step=step)


def _write_train_metadata(
    out_dir: Path,
    spec: YoloModelSpec,
    splits: DetectionSplits,
    hparams: dict[str, Any],
    *,
    task_id: str,
) -> None:
    """Drop a forge-style metadata.json beside Ultralytics' own outputs."""
    meta = {
        "pipeline": PIPELINE_TAG,
        "version_tag": version_tag(),
        "task_id": task_id,
        "dataset_id": spec.dataset.dataset_id,
        "disease_id": spec.dataset.disease_id,
        "class_names": list(spec.dataset.class_names),
        "model_id": spec.model_id,
        "model_version": spec.model_version,
        "base_weights": spec.base_weights,
        "train_hparams": hparams,
        "splits": {
            "data_yaml": str(splits.data_yaml_path),
            "train_count": splits.train_count,
            "val_count": splits.val_count,
            "test_count": splits.test_count,
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


# --- evaluate (shared inner helper + standalone CLI entry) ----------------


def _evaluate(
    *,
    model,
    data_yaml: Path,
    split: str,
    conf: float,
    iou: float,
    project_dir: Path,
    name: str,
) -> dict[str, float]:
    """Run ``model.val`` + image-level binary collapse for one (conf, iou) pair.

    Returns a flat metrics dict: ``mAP50``, ``mAP50-95``,
    ``image_recall``, ``image_precision``, ``image_tp``, ``image_fp``,
    ``image_tn``, ``image_fn``, plus the ``conf``/``iou`` it was
    measured at (so a logged dict is self-describing).
    """
    val_results = model.val(
        data=str(data_yaml),
        split=split,
        conf=conf,
        iou=iou,
        project=str(project_dir),
        name=name,
        exist_ok=True,
        verbose=False,
    )
    metrics: dict[str, float] = {
        "mAP50": float(val_results.box.map50),
        "mAP50-95": float(val_results.box.map),
    }
    img = _image_level_metrics(
        model=model, data_yaml=data_yaml, split=split, conf=conf, iou=iou
    )
    metrics.update(img)
    metrics["conf"] = float(conf)
    metrics["iou"] = float(iou)
    return metrics


def run_eval(
    spec: YoloModelSpec,
    splits: DetectionSplits,
    *,
    staging: Path | None = None,
    split: str = "test",
    conf: float = DEFAULT_IMAGE_CONF,
    iou: float = DEFAULT_NMS_IOU,
    task_id: str | None = None,
) -> dict[str, float]:
    """Standalone eval entry — used by the CLI ``eval`` subcommand.

    The ``tune`` phase does its own eval internally with the tuned
    (conf, iou); this function is for one-off re-evaluation against
    arbitrary thresholds without going through tune.
    """
    from ultralytics import YOLO

    staging = staging or latest_staging_dir(
        dataset_id=spec.dataset.dataset_id, model_id=spec.model_id
    )
    weights = staging / "train" / "weights" / "best.pt"
    if not weights.is_file():
        raise SystemExit(f"no trained weights at {weights}. Did `train` complete?")

    model = YOLO(str(weights))
    task_id = task_id or generate_task_id()
    with mlflow_phase_run(
        spec=spec,
        phase="eval",
        task_id=task_id,
        params={"split": split, "conf": conf, "iou": iou},
    ):
        metrics = _evaluate(
            model=model,
            data_yaml=splits.data_yaml_path,
            split=split,
            conf=conf,
            iou=iou,
            project_dir=staging,
            name=f"eval_{split}",
        )
        log_metrics(
            {
                f"eval/{k}": float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            }
        )

    (staging / "eval_metrics.json").write_text(
        json.dumps(
            {
                "split": split,
                "metrics": metrics,
                "task_id": task_id,
                "source": "run_eval",
            },
            indent=2,
        )
    )
    logger.info("yolo_forge.eval: %s", metrics)
    return metrics


def _image_level_metrics(
    *, model, data_yaml: Path, split: str, conf: float, iou: float
) -> dict[str, float]:
    """Binary recall/precision over the split: any-box ≥ conf → positive.

    Reads the YAML to find ``<split>`` image dir + corresponding
    labels dir (Ultralytics convention: parallel ``images/`` and
    ``labels/`` trees). Truth label per image: empty label .txt means
    normal, non-empty means positive.
    """
    data = yaml.safe_load(data_yaml.read_text())
    root = Path(data.get("path", data_yaml.parent)).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    split_rel = data[split]
    img_dir = (
        (root / split_rel).resolve()
        if not Path(split_rel).is_absolute()
        else Path(split_rel)
    )
    label_dir = Path(str(img_dir).replace("/images/", "/labels/"))

    images = sorted(img_dir.glob("*.png")) + sorted(img_dir.glob("*.jpg"))
    if not images:
        raise SystemExit(
            f"yolo_forge.eval: no images found under {img_dir} for split={split!r}."
        )

    tp = fp = tn = fn = 0
    results = model.predict(
        source=[str(p) for p in images],
        conf=conf,
        iou=iou,
        verbose=False,
        stream=False,
    )
    for img_path, result in zip(images, results, strict=True):
        truth_label = label_dir / f"{img_path.stem}.txt"
        truth_positive = truth_label.is_file() and truth_label.stat().st_size > 0
        pred_positive = result.boxes is not None and len(result.boxes) > 0
        if truth_positive and pred_positive:
            tp += 1
        elif truth_positive and not pred_positive:
            fn += 1
        elif not truth_positive and pred_positive:
            fp += 1
        else:
            tn += 1

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return {
        "image_recall": recall,
        "image_precision": precision,
        "image_tp": tp,
        "image_fp": fp,
        "image_tn": tn,
        "image_fn": fn,
    }


# --- tune ---------------------------------------------------------------


def run_tune(
    spec: YoloModelSpec,
    splits: DetectionSplits,
    *,
    trials: int = DEFAULT_TUNE_TRIALS,
    staging: Path | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Optuna over ``inference_space`` on val, then final test eval.

    Two-step:

    1. Optuna walks :attr:`YoloModelSpec.inference_space` for
       ``trials`` rounds, scoring each candidate by Ultralytics' own
       ``fitness`` formula (``0.1·mAP50 + 0.9·mAP50-95``) on the
       **val** split.
    2. With the best params from step 1, run a final ``_evaluate``
       on the **test** split and persist
       ``<staging>/eval_metrics.json`` (consumed by ``deploy``).

    No-op fallback: if ``inference_space`` is empty, picks the
    framework defaults (``DEFAULT_IMAGE_CONF`` / ``DEFAULT_NMS_IOU``)
    and runs only the final test eval.
    """
    from ultralytics import YOLO

    staging = staging or latest_staging_dir(
        dataset_id=spec.dataset.dataset_id, model_id=spec.model_id
    )
    weights = staging / "train" / "weights" / "best.pt"
    if not weights.is_file():
        raise SystemExit(f"no trained weights at {weights}. Did `train` complete?")
    model = YOLO(str(weights))
    task_id = task_id or generate_task_id()

    with mlflow_phase_run(
        spec=spec,
        phase="tune",
        task_id=task_id,
        params={"trials": trials, "has_inference_space": bool(spec.inference_space)},
    ):
        best_params, val_metrics = _run_tune_inner(
            spec=spec, splits=splits, staging=staging, model=model, trials=trials
        )

        test_metrics = _evaluate(
            model=model,
            data_yaml=splits.data_yaml_path,
            split="test",
            conf=float(best_params.get("conf", DEFAULT_IMAGE_CONF)),
            iou=float(best_params.get("iou", DEFAULT_NMS_IOU)),
            project_dir=staging,
            name="test_at_best",
        )

        log_metrics({f"tune/best_{k}": float(v) for k, v in best_params.items()})
        log_metrics(
            {
                f"tune/val_{k}": float(v)
                for k, v in val_metrics.items()
                if isinstance(v, (int, float))
            }
        )
        log_metrics(
            {
                f"tune/test_{k}": float(v)
                for k, v in test_metrics.items()
                if isinstance(v, (int, float))
            }
        )

    out = {
        "split": "test",
        "metrics": test_metrics,
        "val_metrics_at_best": val_metrics,
        "tuned_inference_params": best_params,
        "task_id": task_id,
        "source": "run_tune",
    }
    (staging / "eval_metrics.json").write_text(json.dumps(out, indent=2))
    logger.info("yolo_forge.tune: test metrics=%s", test_metrics)
    return out


def _run_tune_inner(
    *,
    spec: YoloModelSpec,
    splits: DetectionSplits,
    staging: Path,
    model,
    trials: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Optuna-or-defaults block; returns (best_params, val_metrics_at_best).

    Factored out so :func:`run_tune` reads top-to-bottom as
    setup → tune → final test → log + persist, instead of branching
    inline.
    """
    if spec.inference_space:
        import optuna

        tune_dir = staging / "tune"
        tune_dir.mkdir(exist_ok=True)

        def objective(trial: optuna.Trial) -> float:
            suggested = spec.suggest_inference_params(trial)
            conf = float(suggested.get("conf", DEFAULT_IMAGE_CONF))
            iou = float(suggested.get("iou", DEFAULT_NMS_IOU))
            metrics = _evaluate(
                model=model,
                data_yaml=splits.data_yaml_path,
                split="val",
                conf=conf,
                iou=iou,
                project_dir=tune_dir,
                name=f"trial_{trial.number:03d}",
            )
            fitness = _fitness(map50=metrics["mAP50"], map50_95=metrics["mAP50-95"])
            log_metrics(
                {
                    "tune/mAP50": metrics["mAP50"],
                    "tune/mAP50-95": metrics["mAP50-95"],
                    "tune/image_recall": metrics["image_recall"],
                    "tune/fitness": fitness,
                    "tune/conf": conf,
                    "tune/iou": iou,
                },
                step=trial.number,
            )
            return fitness

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        best_params = dict(study.best_params)
        logger.info(
            "yolo_forge.tune: best fitness=%.4f params=%s",
            study.best_value,
            best_params,
        )
        val_metrics = _evaluate(
            model=model,
            data_yaml=splits.data_yaml_path,
            split="val",
            conf=float(best_params.get("conf", DEFAULT_IMAGE_CONF)),
            iou=float(best_params.get("iou", DEFAULT_NMS_IOU)),
            project_dir=staging,
            name="val_at_best",
        )
        return best_params, val_metrics

    logger.info("yolo_forge.tune: inference_space empty — using framework defaults.")
    best_params = {"conf": DEFAULT_IMAGE_CONF, "iou": DEFAULT_NMS_IOU}
    val_metrics = _evaluate(
        model=model,
        data_yaml=splits.data_yaml_path,
        split="val",
        conf=DEFAULT_IMAGE_CONF,
        iou=DEFAULT_NMS_IOU,
        project_dir=staging,
        name="val_default",
    )
    return best_params, val_metrics


# --- deploy --------------------------------------------------------------


def run_deploy(
    spec: YoloModelSpec,
    *,
    staging: Path | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Gate on eval thresholds, promote weights, append audit entry."""
    staging = staging or latest_staging_dir(
        dataset_id=spec.dataset.dataset_id, model_id=spec.model_id
    )
    eval_path = staging / "eval_metrics.json"
    if not eval_path.is_file():
        raise SystemExit(
            f"no eval_metrics.json at {eval_path}. Run the `tune` or `eval` phase first."
        )
    eval_blob = json.loads(eval_path.read_text())
    metrics = eval_blob["metrics"]

    failures = _check_thresholds(metrics, spec.eval_thresholds)
    if failures:
        raise SystemExit(
            f"deploy gate failed for {spec.model_id}: "
            + ", ".join(f"{k}={v} < floor={f}" for k, v, f in failures)
        )

    regression = _regression_check(spec, metrics)
    if regression:
        raise SystemExit(
            f"deploy gate: regression vs last deployed {spec.model_id}: "
            + ", ".join(f"{k}={cur} < prev={prev}" for k, cur, prev in regression)
        )

    src = staging / "train" / "weights" / "best.pt"
    if not src.is_file():
        raise SystemExit(f"no best.pt at {src} — cannot promote.")
    stable = (
        disease_root(spec.dataset.dataset_id)
        / f"{spec.model_id}_{spec.model_version}.pt"
    )
    stable.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, stable)
    task_id = task_id or eval_blob.get("task_id") or generate_task_id()

    entry = {
        "pipeline": PIPELINE_TAG,
        "task_id": task_id,
        "dataset_id": spec.dataset.dataset_id,
        "disease_id": spec.dataset.disease_id,
        "model_id": spec.model_id,
        "model_version": spec.model_version,
        "version_tag": version_tag(),
        "metrics": metrics,
        "tuned_inference_params": eval_blob.get("tuned_inference_params"),
        "weights_path": str(stable),
        "staging_dir": str(staging),
    }
    append_entry(latest_jsonl_path(spec.dataset.dataset_id), entry)

    with mlflow_phase_run(
        spec=spec,
        phase="deploy",
        task_id=task_id,
        extra_tags={"deployed": "true", "weights_path": str(stable)},
    ):
        log_metrics(
            {
                f"deploy/{k}": float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            }
        )

    logger.info("yolo_forge.deploy: promoted weights → %s", stable)
    return entry


def _check_thresholds(
    metrics: dict[str, float], thresholds: dict[str, float]
) -> list[tuple[str, float, float]]:
    """Return ``[(metric, observed, floor)]`` for every unmet threshold."""
    out: list[tuple[str, float, float]] = []
    for name, floor in thresholds.items():
        observed = metrics.get(name)
        if observed is None:
            raise SystemExit(
                f"threshold references unknown metric {name!r}; "
                f"available={sorted(metrics)}"
            )
        if observed < floor:
            out.append((name, float(observed), float(floor)))
    return out


def _regression_check(
    spec: YoloModelSpec, metrics: dict[str, float]
) -> list[tuple[str, float, float]]:
    """Compare current metrics against the last LATEST.jsonl entry."""
    last = read_last_entry(
        latest_jsonl_path(spec.dataset.dataset_id), model_id=spec.model_id
    )
    if last is None:
        return []
    prev_metrics = last.get("metrics", {})
    out: list[tuple[str, float, float]] = []
    for name in spec.eval_thresholds:
        prev = prev_metrics.get(name)
        if prev is None:
            continue
        cur = metrics.get(name, 0.0)
        if cur + 0.01 < prev:
            out.append((name, float(cur), float(prev)))
    return out


# --- pipeline ------------------------------------------------------------


def run_pipeline(
    spec: YoloModelSpec,
    *,
    quick: bool = False,
    search_trials: int = DEFAULT_SEARCH_TRIALS,
    search_epochs: int = DEFAULT_SEARCH_EPOCHS,
    tune_trials: int = DEFAULT_TUNE_TRIALS,
    skip_search: bool = False,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Chain prepare → search → train → tune → deploy.

    ``skip_search=True`` (or an empty ``spec.hparam_space``) jumps
    straight from prepare to train with spec-default hparams — useful
    for quick dry runs where HPO budget would dominate wall time.

    A single ``task_id`` (generated if not provided) tags every phase
    run in MLflow so the four runs can be pivoted as one pipeline
    execution in the UI.
    """
    splits = run_prepare(spec)
    task_id = task_id or generate_task_id()
    logger.info("yolo_forge.pipeline: task_id=%s", task_id)

    out_dir = staging_dir(dataset_id=spec.dataset.dataset_id, model_id=spec.model_id)

    if skip_search or not spec.hparam_space:
        hparams_override = None
    else:
        hparams_override = run_search(
            spec,
            splits,
            trials=search_trials,
            epochs_per_trial=search_epochs,
            staging=out_dir,
            task_id=task_id,
        )

    staging = run_train(
        spec,
        splits,
        hparams_override=hparams_override,
        quick=quick,
        staging=out_dir,
        task_id=task_id,
    )
    run_tune(spec, splits, trials=tune_trials, staging=staging, task_id=task_id)
    return run_deploy(spec, staging=staging, task_id=task_id)


__all__ = [
    "DEFAULT_IMAGE_CONF",
    "DEFAULT_NMS_IOU",
    "DEFAULT_SEARCH_EPOCHS",
    "DEFAULT_SEARCH_TRIALS",
    "DEFAULT_TUNE_TRIALS",
    "PIPELINE_TAG",
    "run_deploy",
    "run_eval",
    "run_pipeline",
    "run_prepare",
    "run_search",
    "run_train",
    "run_tune",
]
