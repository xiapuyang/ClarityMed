"""Modality classifier A/B benchmark.

Runs the sampled image pool through each requested classifier and
reports per-modality precision / recall / F1 plus a confusion matrix.

Usage
-----

::

    uv run python -m tests.benchmarks.modality_classifier.run \\
      --classifiers medical_clip,omlx_llm \\
      --per-dataset 20 --seed 0 \\
      --out data/bench/modality_classifier/<ts>/

Single classifier sweep::

    uv run python -m tests.benchmarks.modality_classifier.run \\
      --classifiers medical_clip --datasets busi,chest_ct

Outputs
-------

``trials.jsonl``
    One row per ``(classifier, image)`` — classifier, dataset, sha256,
    expected_modality, predicted_modality, latency_ms, error, path.

``summary.csv``
    One row per ``(classifier, modality)`` — n, tp, fp, fn, tn,
    precision, recall, F1, latency_p50, latency_p95. ``unknown`` is
    treated as a real modality in the matrix so a classifier that
    bails out is *visible* in the numbers rather than silently dropped.

``confusion.csv``
    One row per ``(classifier, expected_modality)`` with one column
    per predicted modality — useful for spotting "histopath gets
    labeled photo" failure clusters.

Skip semantics
--------------

Any classifier that can't be constructed (server unreachable, missing
provider, placeholder not wired) is logged at startup and omitted from
the run. Any dataset that resolves to zero images on disk is logged
and skipped. As long as one ``(classifier, dataset)`` pair is viable,
the benchmark produces output.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from claritymed.config import load_env_file
from tests.benchmarks.modality_classifier.classifiers import (
    KNOWN_MODALITIES,
    DEFAULT_MEDICAL_CLIP_URL,
    aclose_classifiers,
    build_classifiers,
    load_image,
    sha256_bytes,
)
from tests.benchmarks.modality_classifier.samples import (
    SamplePath,
    build_sample_pool,
    list_known_datasets,
)

logger = logging.getLogger("modality_classifier_bench")

SUPPORTED_CLASSIFIERS = ("medical_clip", "omlx_llm", "resnet")
DEFAULT_OUT_ROOT = Path("data") / "bench" / "modality_classifier"


# --- trial -----------------------------------------------------------------


async def _classify_one(
    classifier,
    sample: SamplePath,
) -> tuple[str | None, float, str | None]:
    """Run one classification and return ``(predicted, latency_ms, error)``."""
    try:
        image_bytes = load_image(sample.path)
    except Exception as exc:  # noqa: BLE001
        return None, 0.0, f"read_error: {type(exc).__name__}: {exc}"
    digest = sha256_bytes(image_bytes)
    started = time.perf_counter()
    try:
        predicted = await classifier.classify(image_bytes, sha256=digest)
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - started) * 1000.0
        return None, elapsed, f"{type(exc).__name__}: {exc}"
    elapsed = (time.perf_counter() - started) * 1000.0
    return predicted, elapsed, None


def _trial_row(
    *,
    classifier_id: str,
    sample: SamplePath,
    predicted: str | None,
    latency_ms: float,
    error: str | None,
) -> dict[str, Any]:
    return {
        "classifier": classifier_id,
        "dataset": sample.dataset,
        "expected_modality": sample.modality,
        "predicted_modality": predicted,
        "latency_ms": round(latency_ms, 2),
        "error": error,
        "path": str(sample.path),
    }


# --- aggregation -----------------------------------------------------------


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    return sorted_vals[max(0, int(round(0.95 * (len(sorted_vals) - 1))))]


def aggregate_per_modality(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per ``(classifier, modality)`` — TP/FP/FN/TN + metrics.

    ``modality`` is iterated over the full :data:`KNOWN_MODALITIES`
    list so the summary shape is stable: classifiers that never
    predict ``"histopathology"`` still get a row with FN counts.
    """
    by_clf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_clf[row["classifier"]].append(row)

    out: list[dict[str, Any]] = []
    for classifier, clf_rows in sorted(by_clf.items()):
        for modality in KNOWN_MODALITIES:
            tp = fp = fn = tn = 0
            latencies: list[float] = []
            errored = 0
            for r in clf_rows:
                if r["error"] is not None:
                    if r["expected_modality"] == modality:
                        errored += 1
                    continue
                expected_is = r["expected_modality"] == modality
                predicted_is = r["predicted_modality"] == modality
                if expected_is and predicted_is:
                    tp += 1
                    latencies.append(r["latency_ms"])
                elif not expected_is and predicted_is:
                    fp += 1
                elif expected_is and not predicted_is:
                    fn += 1
                else:
                    tn += 1
            total = tp + fp + fn + tn
            if total == 0 and errored == 0:
                continue  # no rows touched this modality; keep table compact
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            )
            out.append(
                {
                    "classifier": classifier,
                    "modality": modality,
                    "n_expected": tp + fn,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "errored_expected": errored,
                    "precision": round(precision, 4),
                    "recall": round(recall, 4),
                    "f1": round(f1, 4),
                    "latency_p50_ms": (
                        round(statistics.median(latencies), 2) if latencies else 0.0
                    ),
                    "latency_p95_ms": round(_p95(latencies), 2),
                }
            )
    return out


def build_confusion_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per ``(classifier, expected_modality)``, columns per predicted.

    Errored trials get bucketed under a ``__errored__`` column so the
    counts still total to ``n_expected``.
    """
    by_clf: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    for row in rows:
        cells = by_clf[row["classifier"]][row["expected_modality"]]
        if row["error"] is not None:
            cells["__errored__"] += 1
        else:
            cells[row["predicted_modality"] or "unknown"] += 1

    out: list[dict[str, Any]] = []
    columns = [*KNOWN_MODALITIES, "__errored__"]
    for classifier in sorted(by_clf):
        for expected in sorted(by_clf[classifier]):
            cells = by_clf[classifier][expected]
            entry: dict[str, Any] = {
                "classifier": classifier,
                "expected_modality": expected,
            }
            for col in columns:
                entry[col] = cells.get(col, 0)
            entry["n"] = sum(entry[col] for col in columns)
            out.append(entry)
    return out


# --- output ----------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_summary_table(summary: list[dict[str, Any]]) -> None:
    """Pretty per-modality table — same column order as summary.csv."""
    if not summary:
        print("(no rows — every classifier was skipped)")
        return
    cols = (
        "classifier",
        "modality",
        "n_expected",
        "tp",
        "fp",
        "fn",
        "errored_expected",
        "precision",
        "recall",
        "f1",
        "latency_p50_ms",
        "latency_p95_ms",
    )
    widths = {c: max(len(c), max(len(str(r[c])) for r in summary)) for c in cols}

    def _fmt(row: dict[str, Any]) -> str:
        return "  ".join(str(row[c]).rjust(widths[c]) for c in cols)

    print(_fmt({c: c for c in cols}))
    print("  ".join("-" * widths[c] for c in cols))
    for row in summary:
        print(_fmt(row))


# --- entrypoint ------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--classifiers",
        default=",".join(SUPPORTED_CLASSIFIERS),
        help=(
            "comma-separated classifier ids to run (subset of "
            f"{list(SUPPORTED_CLASSIFIERS)!r}; default: all)"
        ),
    )
    p.add_argument(
        "--datasets",
        default=None,
        help=(
            "comma-separated dataset names to include (default: all). "
            f"available: {list_known_datasets()!r}"
        ),
    )
    p.add_argument(
        "--per-dataset",
        type=int,
        default=20,
        help="max samples drawn per dataset (default: 20)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed for reproducible sampling (default: 0)",
    )
    p.add_argument(
        "--medical-clip-url",
        default=DEFAULT_MEDICAL_CLIP_URL,
        help=f"medical-clip-server base URL (default: {DEFAULT_MEDICAL_CLIP_URL})",
    )
    p.add_argument(
        "--omlx-provider-id",
        default="omlx",
        help="provider id for the omlx_llm classifier (default: omlx)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="output directory (default: data/bench/modality_classifier/<ts>/)",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


async def _run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    load_env_file()

    requested = [s.strip() for s in args.classifiers.split(",") if s.strip()]
    dataset_filter = (
        [d.strip() for d in args.datasets.split(",") if d.strip()]
        if args.datasets
        else None
    )

    classifiers = build_classifiers(
        requested,
        medical_clip_url=args.medical_clip_url,
        omlx_provider_id=args.omlx_provider_id,
    )
    if not classifiers:
        logger.error("no classifiers constructed; nothing to run.")
        return 2

    pool = build_sample_pool(
        per_dataset=args.per_dataset,
        seed=args.seed,
        datasets=dataset_filter,
    )
    if not pool:
        logger.error("sample pool is empty (no datasets resolved). nothing to run.")
        return 2
    logger.info(
        "sampled %d images across %d datasets; running %d classifier(s)",
        len(pool),
        len({s.dataset for s in pool}),
        len(classifiers),
    )

    rows: list[dict[str, Any]] = []
    try:
        for classifier_id, classifier in classifiers.items():
            logger.info(
                "running classifier=%s over %d images", classifier_id, len(pool)
            )
            for sample in pool:
                predicted, latency_ms, error = await _classify_one(classifier, sample)
                rows.append(
                    _trial_row(
                        classifier_id=classifier_id,
                        sample=sample,
                        predicted=predicted,
                        latency_ms=latency_ms,
                        error=error,
                    )
                )
    finally:
        await aclose_classifiers(classifiers)

    summary = aggregate_per_modality(rows)
    confusion = build_confusion_matrix(rows)

    out_dir = (
        Path(args.out)
        if args.out
        else DEFAULT_OUT_ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    _write_jsonl(out_dir / "trials.jsonl", rows)
    _write_csv(out_dir / "summary.csv", summary)
    _write_csv(out_dir / "confusion.csv", confusion)
    logger.info(
        "wrote %d trial rows + %d summary rows + %d confusion rows to %s",
        len(rows),
        len(summary),
        len(confusion),
        out_dir,
    )

    print()
    _print_summary_table(summary)
    return 0


def main() -> int:
    args = _parse_args()
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        logger.warning("interrupted; partial outputs (if any) are on disk.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
