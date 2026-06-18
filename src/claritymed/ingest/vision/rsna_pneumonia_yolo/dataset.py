"""RSNA Pneumonia → YOLO-format ingest (bbox-preserving).

Two-step pipeline:

1. Reuse the classification adapter's :func:`discover` purely to
   populate the DICOM→PNG cache under
   ``<raw_root>/png_cache/<patient_id>.png``. The cache is shared
   across both adapters so no duplicate decoding.
2. Re-parse ``stage_2_train_labels.csv`` *keeping the bbox rows*,
   write one YOLO-format label file per image
   (``labels/<split>/<patient_id>.txt``), symlink the PNG into
   ``images/<split>/<patient_id>.png``, and emit ``data.yaml``.

Stratified split uses the same hash-based seed as the classification
adapter so a model trained on the detection split sees the same
patients in train/val/test as a classification model would — keeps
cross-pipeline comparisons honest.

Single class: ``pneumonia`` (id 0). Normal images get an empty label
file (Ultralytics convention: the image exists but has zero ground
truth boxes).
"""

from __future__ import annotations

import csv
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from PIL import Image

from claritymed.ingest.vision.rsna_pneumonia.dataset import (
    discover as _populate_png_cache,
)
from claritymed.ingest.vision.rsna_pneumonia.download import (
    DATASET_SUBDIR,
    rsna_pneumonia_data_root,
)
from claritymed.ingest.vision.yolo_forge.spec import DetectionSplits

logger = logging.getLogger(__name__)

# Class id matches single-class detection convention: 0 = pneumonia.
# Sync this with ``RSNA_PNEUMONIA_YOLO_CLASSES`` in dataset_spec.py;
# class_names there is the source of truth.
PNEUMONIA_CLASS_ID = 0

# On-disk filenames in the extraction root.
_LABELS_CSV = "stage_2_train_labels.csv"
_PNG_CACHE_DIR = "png_cache"

# YOLO-format prepared root lives next to the raw archive so the
# image symlinks remain valid no matter where the YOLO tooling reads
# from.
_PREPARED_SUBDIR = "yolo"

# Same seed string as the classification stratified_split — keeping
# splits aligned across pipelines is the whole point of this fork.
_SPLIT_SEED = "rsna-pneumonia-v1"
_TRAIN_FRAC = 0.7
_VAL_FRAC = 0.15

Split = Literal["train", "val", "test"]


@dataclass(frozen=True)
class _BBox:
    """One ground-truth bounding box in original PNG pixel coordinates."""

    x: float
    y: float
    w: float
    h: float


# --- CSV parsing (bbox-preserving) --------------------------------------


def _parse_bboxes(csv_path: Path) -> dict[str, list[_BBox]]:
    """Parse the labels CSV into ``{patient_id: [bbox, ...]}``.

    Rows with ``Target=0`` map to an entry with an empty list (so the
    downstream code can still iterate every patient_id and know to
    write an empty label file). Rows with ``Target=1`` always have all
    four bbox columns populated per the upstream spec.
    """
    by_patient: dict[str, list[_BBox]] = {}
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            patient_id = row["patientId"]
            by_patient.setdefault(patient_id, [])
            if int(row["Target"]) == 0:
                continue
            try:
                bbox = _BBox(
                    x=float(row["x"]),
                    y=float(row["y"]),
                    w=float(row["width"]),
                    h=float(row["height"]),
                )
            except (KeyError, ValueError) as exc:
                # Target=1 with missing/non-numeric bbox columns is a
                # corrupt row; fail loud so the operator notices.
                raise RuntimeError(
                    f"rsna_pneumonia_yolo: malformed bbox row for {patient_id!r}: {row} ({exc})"
                ) from exc
            by_patient[patient_id].append(bbox)
    if not by_patient:
        raise RuntimeError(f"no rows parsed from {csv_path}")
    return by_patient


# --- stratified split ---------------------------------------------------


def _split_patients(
    by_patient: dict[str, list[_BBox]],
) -> dict[Split, list[str]]:
    """Hash-based stratified split that matches the classification adapter.

    The same ``_SPLIT_SEED`` + same per-class hash ranking + same
    fractions as :func:`rsna_pneumonia.dataset.stratified_split`
    guarantees patient-level parity across pipelines.
    """
    pos = sorted(pid for pid, bboxes in by_patient.items() if bboxes)
    neg = sorted(pid for pid, bboxes in by_patient.items() if not bboxes)
    out: dict[Split, list[str]] = {"train": [], "val": [], "test": []}
    for group in (pos, neg):
        ranked = sorted(
            group,
            key=lambda pid: hashlib.sha256(f"{_SPLIT_SEED}|{pid}".encode()).hexdigest(),
        )
        n = len(ranked)
        n_train = int(n * _TRAIN_FRAC)
        n_val = int(n * _VAL_FRAC)
        out["train"].extend(ranked[:n_train])
        out["val"].extend(ranked[n_train : n_train + n_val])
        out["test"].extend(ranked[n_train + n_val :])
    return out


# --- YOLO label + symlink emit ------------------------------------------


def _write_yolo_label(
    label_path: Path, bboxes: list[_BBox], image_w: int, image_h: int
) -> None:
    """Write a YOLO-format label file (``cls cx cy w h`` per line, normalized).

    Empty list → empty file. Ultralytics treats an existing-but-empty
    label file as "image with zero ground-truth boxes" (a true
    negative); a *missing* label file is treated as "no labels yet"
    and silently skips the image. The distinction is load-bearing so
    we always create the file.
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for bb in bboxes:
        cx = (bb.x + bb.w / 2.0) / image_w
        cy = (bb.y + bb.h / 2.0) / image_h
        nw = bb.w / image_w
        nh = bb.h / image_h
        # Clamp to [0, 1] to defend against off-by-one bboxes flowing
        # in from the upstream CSV; Ultralytics rejects out-of-range
        # coords with a confusing error otherwise.
        cx = min(max(cx, 0.0), 1.0)
        cy = min(max(cy, 0.0), 1.0)
        nw = min(max(nw, 0.0), 1.0)
        nh = min(max(nh, 0.0), 1.0)
        lines.append(f"{PNEUMONIA_CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _symlink_image(src: Path, dst: Path) -> None:
    """Idempotently symlink ``dst → src``.

    Reuses an existing symlink if it already points at the right
    target; replaces a stale one. Falls back to a copy on filesystems
    that don't support symlinks (some Windows configs) — keeps the
    pipeline portable.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        try:
            if dst.resolve() == src.resolve():
                return
        except OSError:
            pass
        dst.unlink()
    try:
        dst.symlink_to(src)
    except OSError:  # pragma: no cover — non-POSIX fallback
        import shutil

        shutil.copy2(src, dst)


# --- data.yaml builder --------------------------------------------------


def _write_data_yaml(prepared_root: Path, class_names: tuple[str, ...]) -> Path:
    """Emit the data.yaml Ultralytics consumes via ``YOLO.train(data=...)``."""
    data = {
        "path": str(prepared_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(class_names)},
    }
    yaml_path = prepared_root / "data.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))
    return yaml_path


# --- main entry point ---------------------------------------------------


def prepare_rsna_pneumonia_yolo(
    *,
    raw_root: Path | None = None,
    class_names: tuple[str, ...] = ("pneumonia",),
    max_samples: int | None = None,
) -> DetectionSplits:
    """Materialise RSNA Pneumonia in YOLO format. Idempotent.

    ``raw_root`` overrides the default extraction path
    (``CLARITYMED_HOME/data/vision/rsna_pneumonia``). ``max_samples``
    caps how many patients flow through — useful for tests / quick
    runs without paying for the full ~28k DICOM decode.
    """
    root = (raw_root or (rsna_pneumonia_data_root() / DATASET_SUBDIR)).resolve()
    csv_path = root / _LABELS_CSV
    if not csv_path.is_file():
        raise SystemExit(
            f"rsna_pneumonia raw archive not present at {root}. Run "
            "`uv run python -m claritymed.ingest.vision.rsna_pneumonia.download` first."
        )

    # Step 1 — populate the shared PNG cache via the classification
    # adapter. We discard the return value; only the side-effect (PNGs
    # on disk under root/png_cache/) matters here.
    _populate_png_cache(root, max_samples=max_samples)

    # Step 2 — re-parse the CSV keeping bboxes per patient.
    by_patient = _parse_bboxes(csv_path)
    if max_samples is not None:
        # Mirror the classification adapter's "first N sorted IDs" cap
        # so both pipelines see the same subset.
        keep = set(sorted(by_patient)[:max_samples])
        by_patient = {pid: bboxes for pid, bboxes in by_patient.items() if pid in keep}

    splits = _split_patients(by_patient)

    prepared_root = root / _PREPARED_SUBDIR
    prepared_root.mkdir(parents=True, exist_ok=True)

    counts: dict[Split, int] = {"train": 0, "val": 0, "test": 0}
    for split, patient_ids in splits.items():
        for patient_id in patient_ids:
            src_png = root / _PNG_CACHE_DIR / f"{patient_id}.png"
            if not src_png.is_file():
                logger.warning(
                    "rsna_pneumonia_yolo: png cache miss for %s; skipping",
                    patient_id,
                )
                continue
            # Read original image dimensions for bbox normalization. PIL
            # opens the header without loading pixels — fast.
            with Image.open(src_png) as im:
                w, h = im.size

            dst_img = prepared_root / "images" / split / f"{patient_id}.png"
            _symlink_image(src_png, dst_img)

            dst_label = prepared_root / "labels" / split / f"{patient_id}.txt"
            _write_yolo_label(dst_label, by_patient[patient_id], w, h)

            counts[split] += 1

    data_yaml = _write_data_yaml(prepared_root, class_names)
    logger.info(
        "rsna_pneumonia_yolo: prepared %s (train=%d val=%d test=%d)",
        prepared_root,
        counts["train"],
        counts["val"],
        counts["test"],
    )
    return DetectionSplits(
        data_yaml_path=data_yaml,
        train_count=counts["train"],
        val_count=counts["val"],
        test_count=counts["test"],
    )


__all__ = [
    "PNEUMONIA_CLASS_ID",
    "prepare_rsna_pneumonia_yolo",
]
