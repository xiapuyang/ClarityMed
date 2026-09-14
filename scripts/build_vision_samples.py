#!/usr/bin/env python3
"""Build a vision samples library at ``data/vision_samples/<dataset>/<class>/``.

For every dataset registered in :func:`_specs`, this replays the pipeline's
own deterministic stratified split (70/15/15, ``sha256(<seed>|<stem>)``
ranked per class — see each ``ingest/vision/<x>/dataset.py``) and copies
``--per-class`` random images **from the test bucket only** into a per-class
folder under ``data/vision_samples/``.

Pulling from the pipeline's test list (not from on-disk ``test/`` folders,
which are upstream-Kaggle-packaging and bear no relation to the actual
train/val/test the trainer uses) guarantees the samples library contains
only images the model never saw.

Re-runnable. Without ``--force``, a non-empty per-dataset destination is
skipped. With ``--force``, the per-dataset destination is wiped before
copying.

Typical use::

    uv run python scripts/build_vision_samples.py
    uv run python scripts/build_vision_samples.py --datasets busi,chest_ct --force
    uv run python scripts/build_vision_samples.py --per-class 10 --seed 7
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("vision-samples")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = REPO_ROOT / "data" / "vision_samples"


@dataclass(frozen=True)
class _Spec:
    """Per-dataset descriptor.

    ``labels`` order must match the trainer's ``<X>_LABELS`` tuple — that
    is the ground-truth ``label`` int -> name mapping.
    """

    id: str
    labels: tuple[str, ...]
    root_fn: Callable[[], Path]
    subdir: str
    discover_fn: Callable[[Path], list[Any]]
    split_fn: Callable[[list[Any]], dict[str, list[Any]]]


def _specs() -> list[_Spec]:
    """Lazy import so missing torch/PIL doesn't break ``--help``."""
    from claritymed.ingest.vision.breast_us_kaggle import (
        dataset as buk_d,
        download as buk_dl,
    )
    from claritymed.ingest.vision.busi import dataset as busi_d, download as busi_dl
    from claritymed.ingest.vision.chest_ct import (
        dataset as cct_d,
        download as cct_dl,
    )
    from claritymed.ingest.vision.chest_xray_pneumonia import (
        dataset as cxp_d,
        download as cxp_dl,
    )
    from claritymed.ingest.vision.colon_histopath import dataset as ch_d
    from claritymed.ingest.vision.lung_colon_histopath import download as lch_dl
    from claritymed.ingest.vision.lung_histopath import dataset as lh_d
    from claritymed.ingest.vision.rsna_pneumonia import (
        dataset as rsna_d,
        download as rsna_dl,
    )
    from claritymed.ingest.vision.skin_lesion import (
        dataset as skl_d,
        download as skl_dl,
    )

    return [
        _Spec(
            "busi",
            busi_d.BUSI_LABELS,
            busi_dl.busi_data_root,
            busi_dl.DATASET_SUBDIR,
            busi_d.discover,
            busi_d.stratified_split,
        ),
        _Spec(
            "breast_us_kaggle",
            buk_d.BREAST_US_KAGGLE_LABELS,
            buk_dl.breast_us_kaggle_data_root,
            buk_dl.DATASET_SUBDIR,
            buk_d.discover,
            buk_d.stratified_split,
        ),
        _Spec(
            "chest_ct",
            cct_d.CHEST_CT_LABELS,
            cct_dl.chest_ct_data_root,
            cct_dl.DATASET_SUBDIR,
            cct_d.discover,
            cct_d.stratified_split,
        ),
        _Spec(
            "chest_xray_pneumonia",
            cxp_d.CHEST_XRAY_PNEUMONIA_LABELS,
            cxp_dl.chest_xray_pneumonia_data_root,
            cxp_dl.DATASET_SUBDIR,
            cxp_d.discover,
            cxp_d.stratified_split,
        ),
        _Spec(
            "skin_lesion",
            skl_d.SKIN_LESION_LABELS,
            skl_dl.skin_lesion_data_root,
            skl_dl.DATASET_SUBDIR,
            skl_d.discover,
            skl_d.stratified_split,
        ),
        _Spec(
            "lung_histopath",
            lh_d.LUNG_HISTOPATH_LABELS,
            lch_dl.lung_colon_histopath_data_root,
            lch_dl.DATASET_SUBDIR,
            lh_d.discover,
            lh_d.stratified_split,
        ),
        _Spec(
            "colon_histopath",
            ch_d.COLON_HISTOPATH_LABELS,
            lch_dl.lung_colon_histopath_data_root,
            lch_dl.DATASET_SUBDIR,
            ch_d.discover,
            ch_d.stratified_split,
        ),
        _Spec(
            "rsna_pneumonia",
            rsna_d.RSNA_PNEUMONIA_LABELS,
            rsna_dl.rsna_pneumonia_data_root,
            rsna_dl.DATASET_SUBDIR,
            rsna_d.discover,
            rsna_d.stratified_split,
        ),
    ]


def _process(
    spec: _Spec,
    *,
    per_class: int,
    dest: Path,
    rng: random.Random,
    force: bool,
) -> tuple[int, int, bool]:
    """Sample one dataset.

    Returns ``(classes_done, files_copied, skipped)``.
    """
    out_root = dest / spec.id
    if out_root.exists() and any(out_root.iterdir()):
        if not force:
            logger.info(
                "[%s] dest %s non-empty — skipping (pass --force to rebuild)",
                spec.id,
                out_root,
            )
            return 0, 0, True
        shutil.rmtree(out_root)

    root = spec.root_fn() / spec.subdir
    if not root.exists():
        logger.warning("[%s] source root %s missing — skipping", spec.id, root)
        return 0, 0, True

    logger.info("[%s] discover %s", spec.id, root)
    try:
        samples = spec.discover_fn(root)
    except Exception:  # noqa: BLE001 — surface any discover failure as skip
        logger.error("[%s] discover failed", spec.id, exc_info=True)
        return 0, 0, True

    splits = spec.split_fn(samples)
    test_samples = splits["test"]
    by_label: dict[int, list[Any]] = {}
    for s in test_samples:
        by_label.setdefault(s.label, []).append(s)

    classes_done = 0
    files_copied = 0
    for label_idx, label_name in enumerate(spec.labels):
        bucket = by_label.get(label_idx, [])
        if not bucket:
            logger.warning("[%s] '%s' has 0 test samples", spec.id, label_name)
            continue
        k = min(per_class, len(bucket))
        if k < per_class:
            logger.warning(
                "[%s] '%s' has only %d test samples (< %d requested)",
                spec.id,
                label_name,
                len(bucket),
                per_class,
            )
        chosen = rng.sample(bucket, k=k)
        class_dir = out_root / label_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for sample in chosen:
            shutil.copy2(sample.image_path, class_dir / sample.image_path.name)
            files_copied += 1
        classes_done += 1
        logger.info(
            "[%s] %s: copied %d (of %d available in test)",
            spec.id,
            label_name,
            k,
            len(bucket),
        )

    return classes_done, files_copied, False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-class",
        type=int,
        default=5,
        help="Images per class (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for the per-class pick (default: 42)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination root (default: {DEFAULT_DEST.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--datasets",
        help="Comma-separated dataset ids; default all registered",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Wipe and rebuild per-dataset dest if non-empty",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    specs = _specs()
    if args.datasets:
        wanted = {x.strip() for x in args.datasets.split(",") if x.strip()}
        known = {s.id for s in specs}
        unknown = wanted - known
        if unknown:
            parser.error(
                f"unknown dataset(s): {sorted(unknown)}; known: {sorted(known)}"
            )
        specs = [s for s in specs if s.id in wanted]

    rng = random.Random(args.seed)
    args.dest.mkdir(parents=True, exist_ok=True)

    total_classes = 0
    total_files = 0
    skipped: list[str] = []
    for spec in specs:
        c, f, was_skipped = _process(
            spec,
            per_class=args.per_class,
            dest=args.dest,
            rng=rng,
            force=args.force,
        )
        total_classes += c
        total_files += f
        if was_skipped:
            skipped.append(spec.id)

    logger.info(
        "done. classes=%d files=%d dest=%s",
        total_classes,
        total_files,
        args.dest,
    )
    if skipped:
        logger.info("skipped: %s", ", ".join(skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
