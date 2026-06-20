"""Per-modality confidence threshold analyzer for medical-clip.

Reads ``trials.jsonl`` produced by :mod:`tests.benchmarks.modality_classifier.run`
and recommends a confidence cutoff for the vision_plugin soft gate, one
threshold per modality. The threshold is derived from observed
distributions on the bench fixtures — not handpicked — so the gate
firing rate stays defensible.

Why per-modality and not a single global τ
-------------------------------------------

BiomedCLIP is calibrated differently across modalities. CXR shows up
heavily in the pre-training corpus; histopathology and dermoscopy do
not. A global τ tuned to one of them either over-rejects the other
(false negatives) or under-rejects (the soft gate becomes a no-op).
Per-modality numbers cost an extra column in YAML and buy a much
sharper gate.

Decision rule (clamp min_correct to the shippable band)
-------------------------------------------------------

Per modality ``M`` we compute two distributions:

* ``correct``: rows where ``expected == predicted == M``. The
  post-softmax top-1 score on true positives.
* ``wrong``: rows where ``predicted == M`` but ``expected != M``. The
  confidence on false positives.

The recommended τ is::

    recommended_tau = clamp(min_correct, MIN_CONFIDENCE_FLOOR, SHIPPABLE_CAP)

with ``MIN_CONFIDENCE_FLOOR = 0.55`` (the server-side
``min_confidence`` — anything below collapses to ``unknown`` upstream
so we can't gate lower) and ``SHIPPABLE_CAP = 0.70`` (the
``_MIN_MEDICAL_CONFIDENCE_CAP`` enforced by
``servers/medical_clip/app.py::_validate_gating``). Numbers above the
cap are bench artefacts on clean public datasets — shipping them
locks real users out.

Why ``min_correct`` rather than a percentile or midpoint:

* The soft gate is fail-safe (low confidence prompts the user, doesn't
  silently bypass) so the right failure mode is "don't reject TPs",
  not "don't admit FPs".
* ``min_correct`` is the lowest confidence on an observed TP. Setting
  τ here means the bench data shows zero false rejections.
* Outlier sensitivity is bounded by the floor: any min_correct below
  0.55 is clamped to 0.55, because the server-side gate already
  rejects everything below 0.55 (collapses to ``unknown``). A
  pathological TP at 0.10 confidence doesn't drag τ below 0.55.
* Midpoint rules (the previous ``mean(min, max)`` formula) routinely
  recommend τ values above 0.70 on this bench data, which then can't
  be shipped — the ``at_cap`` column surfaces this gap explicitly.

The ``at_cap`` column flags modalities where the uncapped ``min_correct``
exceeds the project cap — these modalities ship at the cap regardless,
and the per-modality dict carries no additional information for them
versus the ``default`` entry.

Calibration warnings (still emitted independently of τ):

* ``CALIBRATION_FAIL``: ``min_correct <= max_wrong``. At least one FP
  has confidence equal to or above the worst TP — the distributions
  cross at the extremes and the rule has no clean wedge to fence
  against.
* ``CALIBRATION_SHAKY``: ``wrong_p95 >= correct_p5``. The bulk of the
  distributions touch (top 5% of wrongs reaches the bottom 5% of
  rights). Softer signal than the extreme test.

Both warnings are surfaced — they disagree when a single outlier
sample pushes the extremes apart from the percentile bulk.

Usage
-----

::

    uv run python -m tests.benchmarks.modality_classifier.analyze_thresholds \\
      --in data/bench/modality_classifier/<ts>/trials.jsonl
    # or, pick the most recent run automatically:
    uv run python -m tests.benchmarks.modality_classifier.analyze_thresholds --latest

Outputs (alongside the trials file)
-----------------------------------

``confidence_distribution.csv``
    One row per ``(modality, group)`` — n, min, P5, P25, P50, P75,
    P95, max. ``group ∈ {correct, wrong}``.

``recommended_thresholds.csv``
    One row per modality — n_correct, n_wrong, correct_p5,
    wrong_p95, recommended_tau, clean_separator (bool), warning.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger("modality_threshold_analyzer")

DEFAULT_OUT_ROOT = Path("data") / "bench" / "modality_classifier"
DEFAULT_CLASSIFIER = "medical_clip"
# Minimum n before quantile recommendations are statistically meaningful.
# Below this we still report the raw distribution but flag the threshold
# as ``n_too_small`` so the operator doesn't ship a τ tuned on 8 samples.
MIN_SAMPLES_FOR_RECOMMENDATION = 20
# Server-side ``gating.min_confidence`` from configs/medical_clip.yaml.
# Recommended τ can't drop below this — the server already collapses
# top1 < MIN_CONFIDENCE_FLOOR to ``unknown`` before the per-modality
# threshold ever fires. Duplicated here (not imported) to keep the
# analyzer free of FastAPI/uvicorn transitive imports; if the server
# config moves, this constant must move with it.
MIN_CONFIDENCE_FLOOR = 0.55
# Server-side ``_MIN_MEDICAL_CONFIDENCE_CAP`` —
# servers/medical_clip/app.py rejects any per-modality value above
# this. Recommended τ is clamped at the cap so the analyzer's output
# is directly pasteable into the YAML.
SHIPPABLE_CAP = 0.70


# --- percentiles ------------------------------------------------------------


def _percentile(values: list[float], pct: float) -> float:
    """Return the ``pct`` percentile (0–100) of ``values`` via nearest-rank.

    Empty input returns ``0.0`` — callers gate on ``n`` separately. The
    nearest-rank method is intentional: we want a value that *actually
    appeared* in the distribution rather than an interpolated point
    that no single image produced. Makes the τ defensible ("the 95th
    percentile wrong-confidence on dataset X was 0.62, so τ=0.65
    rejects all those").
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if pct <= 0:
        return sorted_vals[0]
    if pct >= 100:
        return sorted_vals[-1]
    idx = max(
        0, min(len(sorted_vals) - 1, int(round(pct / 100.0 * (len(sorted_vals) - 1))))
    )
    return sorted_vals[idx]


def _distribution_stats(values: list[float]) -> dict[str, float]:
    """Compute n / min / P5 / P25 / P50 / P75 / P95 / max for ``values``."""
    if not values:
        return {
            "n": 0,
            "min": 0.0,
            "p5": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "n": len(values),
        "min": round(min(values), 4),
        "p5": round(_percentile(values, 5), 4),
        "p25": round(_percentile(values, 25), 4),
        "p50": round(_percentile(values, 50), 4),
        "p75": round(_percentile(values, 75), 4),
        "p95": round(_percentile(values, 95), 4),
        "max": round(max(values), 4),
    }


# --- analysis ---------------------------------------------------------------


def load_trials(path: Path, *, classifier: str) -> list[dict[str, Any]]:
    """Load trial rows from ``path`` filtered to one classifier.

    Errored rows (``error`` non-null) and rows with ``confidence is
    None`` are dropped — the distributions only make sense on
    backends that emit a real score.
    """
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("classifier") != classifier:
                continue
            if row.get("error") is not None:
                continue
            if row.get("confidence") is None:
                continue
            rows.append(row)
    return rows


def split_by_modality(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, list[float]]]:
    """Bucket confidences into ``{modality: {"correct": [...], "wrong": [...]}}``.

    ``correct``: rows where ``expected == predicted == modality``.
    ``wrong``: rows where ``predicted == modality`` but ``expected !=
    modality`` (false positives — the cases the soft gate is meant
    to catch).
    """
    buckets: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"correct": [], "wrong": []}
    )
    for row in rows:
        expected = row["expected_modality"]
        predicted = row["predicted_modality"]
        confidence = float(row["confidence"])
        if predicted is None:
            continue
        if expected == predicted:
            buckets[predicted]["correct"].append(confidence)
        else:
            # FP for the predicted bucket. Also useful to attribute FN
            # to ``expected`` later; for threshold tuning we only need
            # the FP angle since τ gates the *predicted* label.
            buckets[predicted]["wrong"].append(confidence)
    return buckets


def _clamp_to_shippable(value: float) -> float:
    """Clamp ``value`` to ``[MIN_CONFIDENCE_FLOOR, SHIPPABLE_CAP]``."""
    return round(max(MIN_CONFIDENCE_FLOOR, min(value, SHIPPABLE_CAP)), 4)


def recommend_threshold(correct: list[float], wrong: list[float]) -> dict[str, Any]:
    """Compute the recommended τ + diagnostic flags for one modality.

    Returns a dict with ``min_correct`` / ``max_wrong`` (extremes that
    drive the τ rule), ``correct_p5`` / ``correct_p25`` / ``wrong_p95``
    (percentile diagnostics), ``raw_tau`` (uncapped ``min_correct``),
    ``recommended_tau`` (raw_tau clamped to the shippable band; ``None``
    if insufficient data), ``at_cap`` (bool — raw_tau exceeded the
    project cap), ``clean_separator`` (bool — strict min/max test),
    and ``warning`` (string code or empty).
    """
    n_correct = len(correct)
    min_correct = round(min(correct), 4) if correct else None
    max_wrong = round(max(wrong), 4) if wrong else None
    correct_p5 = round(_percentile(correct, 5), 4) if correct else None
    correct_p25 = round(_percentile(correct, 25), 4) if correct else None
    wrong_p95 = round(_percentile(wrong, 95), 4) if wrong else None

    base = {
        "min_correct": min_correct,
        "max_wrong": max_wrong,
        "correct_p5": correct_p5,
        "correct_p25": correct_p25,
        "wrong_p95": wrong_p95,
    }

    if n_correct < MIN_SAMPLES_FOR_RECOMMENDATION:
        return {
            **base,
            "raw_tau": min_correct,
            "recommended_tau": None,
            "at_cap": None,
            "clean_separator": None,
            "warning": "n_too_small",
        }

    raw_tau = min_correct
    recommended_tau = _clamp_to_shippable(raw_tau)
    at_cap = raw_tau > SHIPPABLE_CAP

    # Two warning signals — both surfaced, they may disagree:
    #   * min/max overlap: at least one wrong sample crosses the
    #     lowest correct one. Confidence-based gating cannot wedge
    #     between the two distributions on this data.
    #   * P5/P95 overlap: the *bulk* of the distributions touch. A
    #     softer signal; calibration is shaky even if extremes don't
    #     cross.
    # Both checks are independent of τ — the cap can mask the
    # threshold value but it cannot mask the underlying overlap.
    if wrong:
        extreme_clean = min_correct > max_wrong
        percentile_clean = wrong_p95 < correct_p5
    else:
        # No FPs observed; both checks are vacuously satisfied.
        extreme_clean = True
        percentile_clean = True

    warnings: list[str] = []
    if not extreme_clean:
        warnings.append("CALIBRATION_FAIL: min_correct <= max_wrong")
    if not percentile_clean:
        warnings.append("CALIBRATION_SHAKY: wrong_p95 >= correct_p5")
    return {
        **base,
        "raw_tau": raw_tau,
        "recommended_tau": recommended_tau,
        "at_cap": at_cap,
        "clean_separator": extreme_clean,
        "warning": "; ".join(warnings),
    }


def build_distribution_rows(
    buckets: dict[str, dict[str, list[float]]],
) -> list[dict[str, Any]]:
    """One row per ``(modality, group)`` with the percentile summary."""
    out: list[dict[str, Any]] = []
    for modality in sorted(buckets):
        for group in ("correct", "wrong"):
            stats = _distribution_stats(buckets[modality][group])
            out.append({"modality": modality, "group": group, **stats})
    return out


def build_threshold_rows(
    buckets: dict[str, dict[str, list[float]]],
) -> list[dict[str, Any]]:
    """One row per modality with the recommended τ."""
    out: list[dict[str, Any]] = []
    for modality in sorted(buckets):
        correct = buckets[modality]["correct"]
        wrong = buckets[modality]["wrong"]
        rec = recommend_threshold(correct, wrong)
        out.append(
            {
                "modality": modality,
                "n_correct": len(correct),
                "n_wrong": len(wrong),
                **rec,
            }
        )
    return out


# --- io ---------------------------------------------------------------------


def _resolve_input(args: argparse.Namespace) -> Path:
    """Pick the trials.jsonl path — explicit ``--in`` wins over ``--latest``."""
    if args.input:
        path = Path(args.input)
        if not path.is_file():
            raise SystemExit(f"--in path does not exist: {path}")
        return path
    if args.latest:
        if not DEFAULT_OUT_ROOT.exists():
            raise SystemExit(
                f"no benchmark runs under {DEFAULT_OUT_ROOT} — run the benchmark first"
            )
        candidates = sorted(
            (p for p in DEFAULT_OUT_ROOT.iterdir() if (p / "trials.jsonl").is_file()),
            key=lambda p: p.name,
        )
        if not candidates:
            raise SystemExit(f"no trials.jsonl found under {DEFAULT_OUT_ROOT}")
        return candidates[-1] / "trials.jsonl"
    raise SystemExit("pass --in <trials.jsonl> or --latest")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _print_distribution(dist_rows: list[dict[str, Any]]) -> None:
    if not dist_rows:
        print("(no rows — no confidence-bearing trials found)")
        return
    print("\n=== confidence_distribution ===")
    cols = ("modality", "group", "n", "min", "p5", "p25", "p50", "p75", "p95", "max")
    widths = {c: max(len(c), max(len(str(r[c])) for r in dist_rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in dist_rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


def _print_thresholds(threshold_rows: list[dict[str, Any]]) -> None:
    if not threshold_rows:
        return
    print("\n=== recommended_thresholds ===")
    cols = (
        "modality",
        "n_correct",
        "n_wrong",
        "min_correct",
        "max_wrong",
        "correct_p5",
        "correct_p25",
        "wrong_p95",
        "raw_tau",
        "recommended_tau",
        "at_cap",
        "clean_separator",
        "warning",
    )
    widths = {c: max(len(c), max(len(str(r[c])) for r in threshold_rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for r in threshold_rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


# --- entrypoint -------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--in",
        dest="input",
        default=None,
        help="path to trials.jsonl (default: pick latest with --latest)",
    )
    p.add_argument(
        "--latest",
        action="store_true",
        help=f"pick the most recent run under {DEFAULT_OUT_ROOT}",
    )
    p.add_argument(
        "--classifier",
        default=DEFAULT_CLASSIFIER,
        help=(
            f"classifier id to analyze (default: {DEFAULT_CLASSIFIER}; only "
            "backends that emit a confidence score are useful here)"
        ),
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    trials_path = _resolve_input(args)
    out_dir = trials_path.parent
    logger.info("reading %s", trials_path)

    rows = load_trials(trials_path, classifier=args.classifier)
    if not rows:
        print(
            f"no confidence-bearing rows for classifier={args.classifier!r} "
            f"in {trials_path}",
            file=sys.stderr,
        )
        return 2
    logger.info("loaded %d rows for classifier=%s", len(rows), args.classifier)

    buckets = split_by_modality(rows)
    dist_rows = build_distribution_rows(buckets)
    threshold_rows = build_threshold_rows(buckets)

    _write_csv(out_dir / "confidence_distribution.csv", dist_rows)
    _write_csv(out_dir / "recommended_thresholds.csv", threshold_rows)
    logger.info(
        "wrote confidence_distribution.csv + recommended_thresholds.csv to %s",
        out_dir,
    )

    _print_distribution(dist_rows)
    _print_thresholds(threshold_rows)

    # Surface calibration warnings at exit so a CI run can grep for them.
    # CALIBRATION_FAIL is the hard signal (rule produces τ inside overlap);
    # CALIBRATION_SHAKY is the soft signal (percentile tails touch).
    fails = [r for r in threshold_rows if "CALIBRATION_FAIL" in (r["warning"] or "")]
    shaky = [r for r in threshold_rows if "CALIBRATION_SHAKY" in (r["warning"] or "")]
    if fails:
        print(
            f"\nWARNING: {len(fails)} modality/modalities have min_correct "
            f"<= max_wrong (rule fails): "
            f"{[r['modality'] for r in fails]!r}",
            file=sys.stderr,
        )
    if shaky:
        print(
            f"\nNOTE: {len(shaky)} modality/modalities have wrong_p95 >= "
            f"correct_p5 (percentile overlap; calibration shaky): "
            f"{[r['modality'] for r in shaky]!r}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
