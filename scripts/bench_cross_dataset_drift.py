"""Cross-dataset drift bench driver.

Loads trained vision-model artifacts (``manifest.json`` + ``weights.pt``)
from ``~/.claritymed/models/vision/<disease_id>/<model_id>/``, runs
each model over a different dataset's test split, collapses the result
to a binary clinical task via
:func:`~claritymed.core.vision.eval_metrics.binary_clinical_metrics`,
and writes both a JSON record list and a human-readable Markdown table
to ``docs/benchmarks/cross_dataset_drift/``.

Usage::

    uv run python scripts/bench_cross_dataset_drift.py --pair breast_us
    uv run python scripts/bench_cross_dataset_drift.py --pair breast_us --dry-run

The ``--dry-run`` mode parses the registry + the dataset specs but does
no model loading and no forward pass — it's a wiring smoke check.

Design notes:

* The bench is **read-only** against forge artifacts. It never touches
  ``LATEST.jsonl``, ``configs/vision.yaml``, MLflow, or Optuna stores.
  Bench failures do not contaminate the regression gate.
* The bench is **standalone**, not a forge Task — forge phases are
  ``search/train/tune/deploy``; the bench is none of those, conflating
  it with the phase machinery would muddy the abstraction.
* When the model's label set differs from the eval dataset's (BUSI
  3-class model vs breast_us_kaggle 2-class eval set), the bench
  remaps ground-truth indices into the model's label space by **name**.
  Eval-only labels (e.g. BUSI's ``normal`` when evaluating a 2-class
  model) get folded into any non-positive model label — the binary
  collapse means the choice doesn't change the metric.
* Bench output lives in ``docs/benchmarks/cross_dataset_drift/`` as
  committed artifacts. Each run produces a new dated file; we do not
  overwrite prior results. This preserves a longitudinal record.

See ``docs/plans/2026-06-17-001-feat-cross-dataset-drift-bench-plan.md``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Make ``scripts/`` importable so the registry can be loaded as a sibling.
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from cross_dataset_drift_registry import BenchEntry, entries_for_pair  # noqa: E402

from claritymed.core.vision.eval_metrics import (  # noqa: E402
    BinaryMetrics,
    binary_clinical_metrics,
)
from claritymed.core.vision.schemas import Manifest  # noqa: E402

logger = logging.getLogger(__name__)


# --- constants -----------------------------------------------------------

DEFAULT_VISION_ROOT = Path.home() / ".claritymed" / "models" / "vision"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "docs" / "benchmarks" / "cross_dataset_drift"
)
DEFAULT_BATCH_SIZE = 16

# Threshold-kind labels surfaced in the JSON + Markdown. ``natural`` is
# the model's raw argmax decision; ``tuned`` applies the manifest's
# ``classification_thresholds`` for the positive class.
THRESHOLD_NATURAL = "natural"
THRESHOLD_TUNED = "tuned"


# --- result records ------------------------------------------------------


@dataclass(frozen=True)
class BenchCell:
    """One row in the output matrix — combination of bench entry + metric kind."""

    pair_id: str
    model_id: str
    model_artifact_dir: str
    train_dataset_id: str
    eval_dataset_id: str
    threshold_kind: str
    threshold_value: float | None
    positive_labels: tuple[str, ...]
    metrics: BinaryMetrics
    notes: str

    def to_json_dict(self) -> dict[str, Any]:
        """Flatten for JSON serialization — keep arrays / lists, not sets."""
        m = self.metrics
        return {
            "pair_id": self.pair_id,
            "model_id": self.model_id,
            "model_artifact_dir": self.model_artifact_dir,
            "train_dataset": self.train_dataset_id,
            "eval_dataset": self.eval_dataset_id,
            "threshold_kind": self.threshold_kind,
            "threshold_value": self.threshold_value,
            "positive_labels": list(self.positive_labels),
            "sensitivity": m.sensitivity,
            "specificity": m.specificity,
            "accuracy": m.accuracy,
            "auc": m.auc,
            "n_total": m.n_total,
            "n_positive": m.n_positive,
            "notes": self.notes,
        }


# --- artifact + dataset resolution ---------------------------------------


def _resolve_artifact_dir(rel_path: str, *, vision_root: Path) -> Path:
    """Resolve ``<rel_path>`` against the vision artifact root, fail-loud on miss."""
    abs_path = vision_root / rel_path
    if not abs_path.is_dir():
        raise SystemExit(
            f"bench artifact directory not found: {abs_path}. "
            f"Run forge `pipeline` for this model, or fix the registry's "
            f"model_artifact_dir."
        )
    manifest_path = abs_path / "manifest.json"
    weights_path = abs_path / "weights.pt"
    if not manifest_path.is_file():
        raise SystemExit(f"missing manifest.json in {abs_path}")
    if not weights_path.is_file():
        raise SystemExit(f"missing weights.pt in {abs_path}")
    return abs_path


def _resolve_dataset_spec(dotted_path: str) -> Any:
    """Resolve ``module.path:ATTR`` to a :class:`DatasetSpec` instance.

    Mirrors ``ingest/vision/forge/cli.py::_resolve_model_spec``. Fail
    loud on a bad path / wrong attribute type — silent fallback would
    let the bench score against the wrong dataset.
    """
    if ":" not in dotted_path:
        raise SystemExit(
            f"dataset spec dotted_path={dotted_path!r}: expected "
            f"'module.path:ATTR' (no ':')."
        )
    module_path, attr = dotted_path.split(":", 1)
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"failed to import {module_path!r} for dataset spec: {exc!s}"
        ) from exc
    if not hasattr(module, attr):
        raise SystemExit(
            f"module {module_path!r} has no attribute {attr!r} (looking "
            f"for a DatasetSpec)."
        )
    return getattr(module, attr)


# --- model building ------------------------------------------------------


def _build_model_for_manifest(manifest: Manifest, device: str) -> Any:
    """Instantiate the matching torch architecture for ``manifest``.

    Routes by ``manifest.task`` — the same dispatch the runtime adapter
    uses, kept here so the bench has no runtime dependency. Detection
    models are intentionally not supported in this iteration; the
    breast US + chest X-ray pairs are both classification (with optional
    seg head whose output we discard).
    """
    num_classes = len(manifest.labels)
    if manifest.task == "classification":
        from claritymed.servers.vision.adapters.forge_torch import (
            build_classifier_model,
        )

        model = build_classifier_model(
            backbone=manifest.backbone,
            num_classes=num_classes,
            pretrained=False,
        )
    elif manifest.task == "classification+segmentation":
        from claritymed.servers.vision.adapters.forge_torch import build_cls_seg_model

        model = build_cls_seg_model(
            backbone=manifest.backbone,
            num_classes=num_classes,
            pretrained=False,
        )
    else:
        raise SystemExit(
            f"manifest.task={manifest.task!r} not supported by the cross-dataset "
            f"bench yet — add handling in _build_model_for_manifest if you need "
            f"detection drift coverage."
        )
    return model.to(device)


def _load_weights(model: Any, weights_path: Path, device: str) -> None:
    """Load ``weights.pt`` into ``model``. Mirrors the runtime adapter."""
    import torch

    try:
        state = torch.load(weights_path, map_location=device, weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"failed to torch.load weights at {weights_path}: {exc!s}"
        ) from exc
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # ``strict=False`` matches runtime: aux heads or buffers that aren't
    # part of the deployed arch get skipped silently. Real shape
    # mismatches still raise.
    model.load_state_dict(state, strict=False)
    model.eval()


# --- label-space remap ---------------------------------------------------


def _remap_gt_to_model_space(
    *,
    eval_labels: tuple[str, ...],
    model_labels: tuple[str, ...],
    gt_eval_space: list[int],
    positive_labels: frozenset[str],
) -> tuple[list[int], str]:
    """Map ``gt_eval_space`` from the eval dataset's index space to the
    model's. Eval labels missing from the model vocab fall into any
    non-positive model label — the binary collapse renders the choice
    irrelevant for the metric.

    Returns ``(remapped_gt, notes)`` where ``notes`` is a short
    human-readable string describing any fallbacks applied, surfaced in
    the Markdown report so the reader knows what happened.
    """
    model_idx_by_name = {lbl: i for i, lbl in enumerate(model_labels)}
    negative_model_indices = [
        i for i, lbl in enumerate(model_labels) if lbl not in positive_labels
    ]
    if not negative_model_indices:
        raise SystemExit(
            f"model_labels={model_labels!r} has no non-positive label "
            f"relative to positive_labels={sorted(positive_labels)!r}; "
            f"binary collapse needs at least one negative class."
        )
    fallback_negative = negative_model_indices[0]

    fallback_count = 0
    fallback_labels: set[str] = set()
    remapped: list[int] = []
    for eval_idx in gt_eval_space:
        eval_name = eval_labels[eval_idx]
        if eval_name in model_idx_by_name:
            remapped.append(model_idx_by_name[eval_name])
        else:
            remapped.append(fallback_negative)
            fallback_count += 1
            fallback_labels.add(eval_name)

    if fallback_count > 0:
        notes = (
            f"{fallback_count} sample(s) with eval-only labels "
            f"{sorted(fallback_labels)!r} folded into model label "
            f"{model_labels[fallback_negative]!r} (negative class)"
        )
    else:
        notes = ""
    return remapped, notes


# --- forward pass --------------------------------------------------------


def _forward_over_test_split(
    *,
    model: Any,
    dataset_spec: Any,
    device: str,
    batch_size: int,
) -> tuple[Any, list[int]]:
    """Run the model over the dataset's test split, return ``(probs, gt_eval_space)``.

    ``probs`` is a numpy ``(N, num_model_classes)`` array — softmax has
    been applied with whatever temperature the manifest declares. The
    bench reads it back to compute binary-collapsed metrics in numpy.
    """
    import numpy as np
    import torch

    splits = dataset_spec.build_splits()
    loader = torch.utils.data.DataLoader(
        splits.test, batch_size=batch_size, shuffle=False, num_workers=0
    )

    logits_chunks: list[Any] = []
    gt: list[int] = []
    with torch.no_grad():
        for batch in loader:
            # cls dataset yields ``(imgs, labels)``; cls+seg's test split
            # may yield ``(imgs, labels, mask)`` — we only use the first two.
            imgs = batch[0].to(device)
            labels = batch[1]
            raw = model(imgs)
            cls_logits = raw[0] if isinstance(raw, tuple) else raw
            logits_chunks.append(cls_logits.cpu().numpy())
            gt.extend(int(lbl) for lbl in labels.tolist())

    logits = np.concatenate(logits_chunks, axis=0)
    return logits, gt


def _softmax(logits: Any, *, temperature: float) -> Any:
    """Stable softmax with optional temperature scaling. Pure numpy."""
    import numpy as np

    scaled = logits / float(temperature)
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


# --- per-entry runner ----------------------------------------------------


def _run_entry(
    entry: BenchEntry,
    *,
    vision_root: Path,
    device: str,
    batch_size: int,
) -> list[BenchCell]:
    """Execute one bench entry, return ``[natural_cell, tuned_cell]``.

    Two cells per entry — natural (argmax) and tuned (manifest's
    classification_thresholds applied to the positive-class probability
    sum). Tuned cell falls back to the natural cell when the manifest
    has no threshold for any positive label.
    """
    artifact_dir = _resolve_artifact_dir(
        entry.model_artifact_dir, vision_root=vision_root
    )
    manifest = Manifest.model_validate_json(
        (artifact_dir / "manifest.json").read_text()
    )
    dataset_spec = _resolve_dataset_spec(entry.eval_dataset_dotted_path)

    logger.info(
        "bench entry: model=%s eval=%s positive=%s",
        manifest.model_id,
        entry.eval_dataset_id,
        sorted(entry.positive_labels),
    )

    model = _build_model_for_manifest(manifest, device=device)
    _load_weights(model, artifact_dir / "weights.pt", device=device)

    logits, gt_eval_space = _forward_over_test_split(
        model=model,
        dataset_spec=dataset_spec,
        device=device,
        batch_size=batch_size,
    )

    temperature = 1.0
    if manifest.tuned_inference is not None and manifest.tuned_inference.temperature:
        temperature = float(manifest.tuned_inference.temperature)
    probs = _softmax(logits, temperature=temperature)

    model_labels = tuple(manifest.labels)
    eval_labels = tuple(dataset_spec.labels)

    gt_model_space, remap_notes = _remap_gt_to_model_space(
        eval_labels=eval_labels,
        model_labels=model_labels,
        gt_eval_space=gt_eval_space,
        positive_labels=entry.positive_labels,
    )

    # --- natural threshold cell ---
    natural_metrics = binary_clinical_metrics(
        gt_labels=gt_model_space,
        probs=probs,
        label_tuple=model_labels,
        positive_labels=entry.positive_labels,
        threshold=None,
    )

    # --- tuned threshold cell ---
    tuned_threshold = _resolve_tuned_threshold(
        manifest=manifest, positive_labels=entry.positive_labels
    )
    if tuned_threshold is None:
        tuned_metrics = natural_metrics
        tuned_threshold_value: float | None = None
        tuned_notes = (
            "(no tuned threshold in manifest — same as natural)"
            if not remap_notes
            else f"{remap_notes}; (no tuned threshold in manifest — same as natural)"
        )
    else:
        tuned_metrics = binary_clinical_metrics(
            gt_labels=gt_model_space,
            probs=probs,
            label_tuple=model_labels,
            positive_labels=entry.positive_labels,
            threshold=tuned_threshold,
        )
        tuned_threshold_value = tuned_threshold
        tuned_notes = remap_notes

    return [
        BenchCell(
            pair_id=entry.pair_id,
            model_id=manifest.model_id,
            model_artifact_dir=entry.model_artifact_dir,
            train_dataset_id=entry.train_dataset_id,
            eval_dataset_id=entry.eval_dataset_id,
            threshold_kind=THRESHOLD_NATURAL,
            threshold_value=None,
            positive_labels=tuple(sorted(entry.positive_labels)),
            metrics=natural_metrics,
            notes=remap_notes,
        ),
        BenchCell(
            pair_id=entry.pair_id,
            model_id=manifest.model_id,
            model_artifact_dir=entry.model_artifact_dir,
            train_dataset_id=entry.train_dataset_id,
            eval_dataset_id=entry.eval_dataset_id,
            threshold_kind=THRESHOLD_TUNED,
            threshold_value=tuned_threshold_value,
            positive_labels=tuple(sorted(entry.positive_labels)),
            metrics=tuned_metrics,
            notes=tuned_notes,
        ),
    ]


def _resolve_tuned_threshold(
    *, manifest: Manifest, positive_labels: frozenset[str]
) -> float | None:
    """Pick the manifest's threshold for the positive class, if any.

    When positive_labels has one member: return that label's threshold.
    When it spans multiple: take the minimum (most permissive) threshold
    across the positive labels — sum-of-probs ≥ threshold makes the most
    sense at the loosest cutoff. None when no threshold is declared for
    any positive label (vanilla classifier without tuned operating point).
    """
    if manifest.tuned_inference is None:
        return None
    thresholds = manifest.tuned_inference.classification_thresholds
    if thresholds is None:
        return None
    candidates = [thresholds[label] for label in positive_labels if label in thresholds]
    if not candidates:
        return None
    return min(candidates)


# --- output writers ------------------------------------------------------


def _write_json(cells: list[BenchCell], path: Path) -> None:
    """Write cells as a JSON list. One file per pair_id per run date."""
    payload = [c.to_json_dict() for c in cells]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    logger.info("wrote %d cells to %s", len(cells), path)


def _format_metric(value: float | None, *, percent: bool = True) -> str:
    """Render a metric for the Markdown table — ``N/A`` for ``None``."""
    if value is None:
        return "N/A"
    if percent:
        return f"{value * 100:.1f}%"
    return f"{value:.3f}"


def _write_markdown(cells: list[BenchCell], path: Path, *, pair_id: str) -> None:
    """Render cells as a single Markdown table grouped by pair."""
    today = dt.date.today().isoformat()
    positive_labels = sorted({lbl for c in cells for lbl in c.positive_labels})

    lines = [
        f"# Cross-dataset drift — `{pair_id}` ({today})",
        "",
        f"**Binary clinical task** — positive class = `{positive_labels!r}`; "
        f"all other labels collapse to negative.",
        "",
        "Self-eval rows (same train + eval dataset) are the in-distribution "
        "baseline. Cross rows show what happens when the model is fed images "
        "from a different source — the gap is the drift signal.",
        "",
    ]

    # Single table, sorted for deterministic output: by (train, eval, threshold_kind).
    lines.append(
        "| Model | Train dataset | Eval dataset | Threshold | n | n_pos | Sens. | Spec. | Acc. | AUC | Notes |"
    )
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---|")
    sorted_cells = sorted(
        cells,
        key=lambda c: (
            c.model_id,
            c.train_dataset_id,
            c.eval_dataset_id,
            c.threshold_kind,
        ),
    )
    for c in sorted_cells:
        m = c.metrics
        threshold_str = c.threshold_kind
        if c.threshold_value is not None:
            threshold_str = f"{c.threshold_kind} ({c.threshold_value:.2f})"
        lines.append(
            "| `{model}` | {train} | {eval} | {thresh} | {n} | {npos} | "
            "{sens} | {spec} | {acc} | {auc} | {notes} |".format(
                model=c.model_id,
                train=c.train_dataset_id,
                eval=c.eval_dataset_id,
                thresh=threshold_str,
                n=m.n_total,
                npos=m.n_positive,
                sens=_format_metric(m.sensitivity),
                spec=_format_metric(m.specificity),
                acc=_format_metric(m.accuracy),
                auc=_format_metric(m.auc, percent=False),
                notes=c.notes or "—",
            )
        )

    lines.append("")
    lines.append("Generated by `scripts/bench_cross_dataset_drift.py`.")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    logger.info("wrote markdown to %s", path)


# --- CLI -----------------------------------------------------------------


def _select_device() -> str:
    """Pick the best available torch device — same dispatch as forge."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        required=True,
        help="Bench pair id (e.g. 'breast_us', 'chest_xray').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve registry + dataset specs without loading models or running inference.",
    )
    parser.add_argument(
        "--vision-root",
        type=Path,
        default=DEFAULT_VISION_ROOT,
        help=f"Override the vision artifact root (default: {DEFAULT_VISION_ROOT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output dir (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Inference batch size (default: {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device selection (cuda/mps/cpu). Defaults to best available.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the bench for one pair, write JSON + Markdown to output dir."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    args = _parse_args(argv)

    entries = entries_for_pair(args.pair)
    logger.info("bench pair=%s entries=%d", args.pair, len(entries))

    today = dt.date.today().isoformat()
    json_path = args.output_dir / f"{today}-{args.pair}.json"
    md_path = args.output_dir / f"{today}-{args.pair}.md"

    if args.dry_run:
        # Resolve dataset specs to exercise the dotted-path importer but
        # do not call build_splits (which would require the dataset to
        # be downloaded). Write an empty-skeleton output so wiring is visible.
        for entry in entries:
            _resolve_dataset_spec(entry.eval_dataset_dotted_path)
            _ = (
                _resolve_artifact_dir(
                    entry.model_artifact_dir, vision_root=args.vision_root
                )
                if (args.vision_root / entry.model_artifact_dir).is_dir()
                else None
            )
        _write_json([], json_path.with_suffix(".dry_run.json"))
        return 0

    device = args.device or _select_device()
    logger.info("device=%s", device)

    cells: list[BenchCell] = []
    for entry in entries:
        cells.extend(
            _run_entry(
                entry,
                vision_root=args.vision_root,
                device=device,
                batch_size=args.batch_size,
            )
        )

    _write_json(cells, json_path)
    _write_markdown(cells, md_path, pair_id=args.pair)
    return 0


if __name__ == "__main__":
    sys.exit(main())
