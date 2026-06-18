"""Dataset-agnostic helpers shared across yolo_forge phases.

Mirrors :mod:`claritymed.ingest.vision.forge.common` for the detection
pipeline. Reuses the same on-disk artifact root convention so a future
unified bench (``claritymed-vision-bench``) can iterate both
classification and detection runs under one tree.

This module deliberately does **not** import :mod:`ultralytics` —
keeps the CLI's ``--smoke`` path runnable without the optional extra,
and isolates the heavy import to :mod:`yolo_forge.framework`.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Generator

from claritymed import config as _cfg
from claritymed.ingest.mlflow_utils import (
    generate_task_id,
    mlflow_run as _mlflow_run,
)
from claritymed.ingest.vision.forge.common import log_metrics
from claritymed.ingest.vision.yolo_forge.spec import YoloModelSpec

logger = logging.getLogger(__name__)

# Shared with forge so cross-pipeline MLflow comparisons land in the
# same ``claritymed-vision-<disease_id>`` experiment.
FEATURE = "vision"
# Discriminator tag value used on every run yolo_forge starts. Pairs
# with forge's runs (which omit the tag or set it to ``"forge"``) so
# the MLflow UI can filter cleanly.
PIPELINE_TAG_VALUE = "yolo_forge"

# Canonical phase order — the CLI's ``parse_phases`` re-orders any
# subset to this order so ``--phases deploy,train`` runs as the right
# sequence anyway. Mirrors forge's (search, train, tune, deploy) plus
# a ``prepare`` step (forge folds prepare into the dataset spec's
# ``build_splits``; yolo_forge needs a dedicated step because the
# YOLO-format materialisation is observable on disk).
ALL_PHASES: tuple[str, ...] = ("prepare", "search", "train", "tune", "deploy")


# --- paths ---------------------------------------------------------------


def version_tag() -> str:
    """UTC timestamp used as the immutable staging-dir suffix."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def disease_root(dataset_id: str) -> Path:
    """``~/.claritymed/models/vision/<dataset_id>/``.

    Shared with the classification ``forge`` so a future bench can
    enumerate both pipelines' artifacts under one tree.
    """
    return _cfg.CLARITYMED_HOME / "models" / "vision" / dataset_id


def staging_dir(*, dataset_id: str, model_id: str) -> Path:
    """Fresh timestamped staging dir for one YOLO train run."""
    return disease_root(dataset_id) / "run" / f"{model_id}_{version_tag()}"


def latest_staging_dir(*, dataset_id: str, model_id: str) -> Path:
    """Most recent staging dir for ``(dataset_id, model_id)``.

    Fail loud if no run exists — silent fallback would let an operator
    deploy stale artifacts unaware that the requested model was never
    trained on this box.
    """
    run_dir = disease_root(dataset_id) / "run"
    if not run_dir.is_dir():
        raise SystemExit(
            f"no run/ dir under {disease_root(dataset_id)} — train this model first."
        )
    candidates = sorted(
        (p for p in run_dir.glob(f"{model_id}_*") if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(
            f"no staging dirs under {run_dir} for model_id={model_id!r} — "
            f"train this model first or pass --staging-dir."
        )
    return candidates[0]


# --- audit log ------------------------------------------------------------


def latest_jsonl_path(dataset_id: str) -> Path:
    """Append-only deploy audit log; one file per dataset.

    Shared with classification forge so the file is a single source of
    truth for "what got promoted on this disease, when." Consumers
    filter rows by ``entry["pipeline"] in {"forge", "yolo_forge"}`` if
    they need to separate task types.
    """
    return disease_root(dataset_id) / "LATEST.jsonl"


def read_last_entry(path: Path, *, model_id: str) -> dict[str, Any] | None:
    """Return the last entry for ``model_id`` in the audit log, or ``None``."""
    if not path.exists():
        return None
    last: dict[str, Any] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("model_id") != model_id:
            continue
        last = entry
    return last


def append_entry(path: Path, entry: dict[str, Any]) -> None:
    """Append one JSON line; create the file + parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


# --- phases helper --------------------------------------------------------


def parse_phases(arg: str | None) -> list[str]:
    """Parse a comma-separated phase list into the canonical order.

    ``None`` / empty → run every phase. Unknown tokens raise
    ``SystemExit`` so a typo doesn't silently skip a phase.
    """
    if not arg:
        return list(ALL_PHASES)
    requested = [p.strip() for p in arg.split(",") if p.strip()]
    unknown = [p for p in requested if p not in ALL_PHASES]
    if unknown:
        raise SystemExit(
            f"unknown phase(s): {unknown!r}. Valid phases: {list(ALL_PHASES)}"
        )
    # Re-order to the canonical sequence regardless of how the operator typed it.
    return [p for p in ALL_PHASES if p in requested]


# --- spec resolution ------------------------------------------------------


def resolve_model_spec(target: str) -> YoloModelSpec:
    """Resolve ``module.path:ATTR`` to a :class:`YoloModelSpec` instance.

    Same convention as forge's ``_resolve_model_spec``: fail loud on a
    bad dotted path or wrong attribute type rather than silently
    falling back to a default — the operator named a specific model,
    we'd better train that one.
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
    if not hasattr(module, attr):
        raise SystemExit(
            f"--model {target!r}: {module_path!r} has no attribute {attr!r}"
        )
    spec = getattr(module, attr)
    if not isinstance(spec, YoloModelSpec):
        raise SystemExit(
            f"--model {target!r}: {attr!r} is {type(spec).__name__}, expected YoloModelSpec."
        )
    return spec


# --- MLflow shim --------------------------------------------------------


@contextlib.contextmanager
def mlflow_phase_run(
    *,
    spec: YoloModelSpec,
    phase: str,
    task_id: str,
    params: dict[str, Any] | None = None,
    extra_tags: dict[str, str] | None = None,
) -> Generator[Any, None, None]:
    """Open an MLflow run for one pipeline phase.

    Same experiment as forge (``claritymed-vision-<disease_id>``) so a
    detection + classification model on the same disease land in one
    UI view, separable by the ``pipeline`` tag.

    **mlflow is a hard requirement at pipeline execution.** Module
    import stays cheap (mlflow is only loaded when this context is
    actually entered), so installing the package without the
    ``yolo-forge`` extra is still fine for code that just *reads*
    spec / common / framework. But the first phase that opens this
    context without mlflow installed fails loud with a remediation
    hint — silent no-op here was a bug magnet (training appearing to
    succeed with zero metrics persisted).

    Args:
        spec: The model spec — supplies dataset_id / disease_id /
            model_id / model_version for tags.
        phase: ``"search"`` / ``"train"`` / ``"tune"`` / ``"deploy"``.
            Surfaced as the MLflow ``run_type`` tag.
        task_id: Lineage tag (``claritymed.task_id``) shared across
            every phase run in one pipeline execution.
        params: ``mlflow.log_params(...)`` payload — typically the
            train hparams or tune trial params.
        extra_tags: Caller-supplied tags merged on top of the defaults.
    """
    try:
        import mlflow  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "yolo_forge: mlflow not installed but a pipeline phase tried to log. "
            "Run `uv sync --extra yolo-forge` (or pip install mlflow>=2.0) and retry."
        ) from exc

    tags = {
        "pipeline": PIPELINE_TAG_VALUE,
        "claritymed.task_id": task_id,
        "model_id": spec.model_id,
        "model_version": spec.model_version,
        "dataset_id": spec.dataset.dataset_id,
        "task": "detection",
    }
    if extra_tags:
        tags.update(extra_tags)

    run_name = f"{spec.model_id}_{phase}_{task_id}"
    with _mlflow_run(
        feature=FEATURE,
        dataset_id=spec.dataset.disease_id,  # same key forge uses
        run_name=run_name,
        run_type=phase,
        params=params,
        tags=tags,
    ) as run:
        yield run


__all__ = [
    "ALL_PHASES",
    "FEATURE",
    "PIPELINE_TAG_VALUE",
    "append_entry",
    "disease_root",
    "generate_task_id",
    "latest_jsonl_path",
    "latest_staging_dir",
    "log_metrics",
    "mlflow_phase_run",
    "parse_phases",
    "read_last_entry",
    "resolve_model_spec",
    "staging_dir",
    "version_tag",
]
