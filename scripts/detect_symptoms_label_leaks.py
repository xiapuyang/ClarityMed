"""Detect label-leak candidate features in a DDXPlus subset.

For any subset of DDXPlus conditions, computes per-column
class-conditional prevalence ``P(f=1 | class)`` and flags columns whose
maximum pairwise absolute gap crosses ``--gap-threshold``. These are the
features an XGBoost classifier (or any margin-based model) will latch
onto as near-deterministic label proxies — the exact failure mode that
produced the E_131 / E_135 leak on the Pneumonia+Influenza subset
(see ``docs/solutions/2026-07-08-001-xgb-label-leak.md``).

Definition of "leak" here is intentionally soft: a large gap means the
column is class-informative, which is what a good feature *should* be.
The tell is when the gap is (a) near ±1.0 AND (b) rides on an evidence
that is clinically irrelevant to the differential (e.g. dermatology
questions on a respiratory-infection subset). This script surfaces (a)
mechanically; (b) is a call the operator makes when reviewing the list.

Rule of thumb: on a well-behaved subset the top-gap features should be
the clinical differentiators. If the top-gap list is dominated by
categorical / multi-value evidences whose IDs don't appear in the
subset's clinical rationale, add them to
``_LABEL_LEAKAGE_BLACKLIST`` in ``ingest/symptoms/xgb/encoding.py``.

Usage::

    uv run python scripts/detect_symptoms_label_leaks.py \\
        --data-dir ~/.claritymed/data/symptoms/ddxplus \\
        --diseases "Pneumonia,Influenza" \\
        --gap-threshold 0.85 \\
        --top-k 30
"""

from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from claritymed.ingest.symptoms.ddxplus.schema import (
    load_evidence_schema,
    load_patients,
    load_pidx,
)
from claritymed.ingest.symptoms.xgb.encoding import (
    encode_patient_batch,
    feature_columns_from_schema,
    load_evidence_meta,
)

DEFAULT_DATA_DIR = Path.home() / ".claritymed" / "data" / "symptoms" / "ddxplus"
DEFAULT_N_TRAIN = 300_000
DEFAULT_GAP_THRESHOLD = 0.85
DEFAULT_TOP_K = 30


def _label_by_col(columns: list[str], labels: list[str]) -> dict[str, str]:
    """Return ``{column_name: human-readable label}``."""
    return dict(zip(columns, labels))


def _max_pairwise_gap(prev_by_class: np.ndarray) -> tuple[float, int, int]:
    """Return ``(max_abs_gap, class_i, class_j)`` for a per-class prevalence row.

    For a 2-class subset the gap is the single ``|p[0] - p[1]|`` value.
    For 3+ classes we scan all pairs and return the largest — a feature
    that separates any two classes near-perfectly is still a leak
    candidate even if the others sit in the middle.
    """
    n = len(prev_by_class)
    best = (0.0, 0, 0)
    for i, j in combinations(range(n), 2):
        gap = abs(float(prev_by_class[i]) - float(prev_by_class[j]))
        if gap > best[0]:
            best = (gap, i, j)
    return best


def _extract_ev_id(column: str) -> str:
    """Return the parent evidence id for a feature column.

    Column shapes: ``E_X`` (binary) or ``E_X__value`` (cat / multi).
    Splitting on ``__`` keeps the pipeline schema-agnostic — anything to
    the left of the first ``__`` is the evidence id.
    """
    return column.split("__", 1)[0]


def _run(args: argparse.Namespace) -> int:
    """Load subset patients, compute per-column prevalence, print top gaps."""
    whitelist = None
    if args.diseases:
        whitelist = {d.strip() for d in args.diseases.split(",") if d.strip()}
        if len(whitelist) < 2:
            print("--diseases needs at least 2 names for a gap to exist.")
            return 2

    schema = load_evidence_schema(args.data_dir)
    meta = load_evidence_meta(args.data_dir)
    pidx, _ = load_pidx(args.data_dir, whitelist=whitelist)
    if len(pidx) < 2:
        print(f"subset pidx has <2 classes: {pidx}. No gap to compute.")
        return 2
    columns, labels, columns_idx = feature_columns_from_schema(schema, meta)
    label_by_col = _label_by_col(columns, labels)

    print(f"data:         {args.data_dir}")
    print(f"subset pidx:  {pidx}")
    print(f"features:     {len(columns)}  (post-blacklist enumeration)")

    patients = load_patients(args.data_dir, args.n_train, "train", schema, pidx)
    y = np.asarray([p["d"] for p in patients])
    balance = Counter(y.tolist())
    print(f"train kept:   {len(patients)}   class balance: {dict(balance)}")

    x = encode_patient_batch(patients, schema, columns_idx)
    # Per-class prevalence P(f=1 | class=k) for every column.
    n_classes = len(pidx)
    prev = np.zeros((n_classes, x.shape[1]), dtype=np.float32)
    for cls_idx in range(n_classes):
        mask = y == cls_idx
        if mask.any():
            prev[cls_idx] = x[mask].mean(axis=0)
    class_names = sorted(pidx, key=lambda n: pidx[n])

    # Rank columns by max pairwise gap.
    ranked: list[tuple[float, int, str, int, int]] = []
    for col_idx, col in enumerate(columns):
        gap, ci, cj = _max_pairwise_gap(prev[:, col_idx])
        ranked.append((gap, col_idx, col, ci, cj))
    ranked.sort(key=lambda t: -t[0])

    # Filter to threshold + de-dup evidences: pick the strongest column per
    # parent evidence. A binary evidence contributes 1 column so this is a
    # no-op; a wide cat/multi evidence with 200 columns collapses to its
    # single most-leaky column so the operator sees each evidence once.
    per_evidence_best: dict[str, tuple[float, str, int, int]] = {}
    for gap, _col_idx, col, ci, cj in ranked:
        if gap < args.gap_threshold:
            break
        ev_id = _extract_ev_id(col)
        existing = per_evidence_best.get(ev_id)
        if existing is None or gap > existing[0]:
            per_evidence_best[ev_id] = (gap, col, ci, cj)

    print(
        f"\n=== Candidate leaks: max |P(f=1|A) − P(f=1|B)| ≥ "
        f"{args.gap_threshold} (top {args.top_k} evidences) ==="
    )
    print("If the evidence is clinically irrelevant to the subset differential,")
    print("add its id to _LABEL_LEAKAGE_BLACKLIST in xgb/encoding.py and retrain.")

    top = sorted(per_evidence_best.items(), key=lambda kv: -kv[1][0])[: args.top_k]
    if not top:
        print(f"\n  (none — no evidence crosses gap {args.gap_threshold})")
        print("  Model should learn on multiple co-occurring signals — healthy state.")
        return 0

    for rank, (ev_id, (gap, col, ci, cj)) in enumerate(top, 1):
        cn_i, cn_j = class_names[ci], class_names[cj]
        print(
            f"  {rank:>2}. gap={gap:.3f}  {col:20s}  "
            f"P(=1|{cn_i})={float(prev[ci, columns_idx[col]]):.3f}  "
            f"P(=1|{cn_j})={float(prev[cj, columns_idx[col]]):.3f}"
        )
        text = label_by_col.get(col, col)
        # Truncate for readability — full text is in release_evidences.json.
        print(f"      → {text[:110]}")

    print(
        f"\nSummary: {len(per_evidence_best)} evidences exceed the gap floor "
        f"({args.gap_threshold}). Review each: if clinical → keep, if artifactual → blacklist."
    )
    return 0


def main() -> None:
    """CLI entry."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument(
        "--diseases",
        default=None,
        help="Comma-separated disease names for subset analysis "
        "(e.g. 'Pneumonia,Influenza'). Omit to scan the full DDXPlus catalog.",
    )
    ap.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN)
    ap.add_argument("--gap-threshold", type=float, default=DEFAULT_GAP_THRESHOLD)
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = ap.parse_args()
    raise SystemExit(_run(args))


if __name__ == "__main__":
    main()
