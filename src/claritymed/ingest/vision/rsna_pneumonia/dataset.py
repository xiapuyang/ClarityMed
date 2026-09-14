"""RSNA Pneumonia Detection Challenge dataset wrapper.

The upstream archive ships ~28k adult chest X-ray DICOMs labeled
NORMAL / PNEUMONIA at the image level (with bounding boxes for
positive cases, ignored here — the cross-dataset drift bench is
classification-only).

Two transforms applied at ingest time:

1. **DICOM → PNG conversion** via :mod:`pydicom`. DICOMs decode slowly
   and are not directly consumable by PIL, so we convert each one once
   and cache the PNG under ``<root>/png_cache/`` keyed by ``patientId``.
   Subsequent ``discover()`` runs skip the conversion and read straight
   from the cache.
2. **Image-level label derivation**: ``stage_2_train_labels.csv`` has
   one row per bounding box, so a positive case has multiple rows with
   the same ``patientId``. We collapse on ``patientId`` and take the
   max ``Target`` — any bbox row makes the image positive.

Output shape per item: ``(image: Tensor[C=3, H, W], label: int)``,
matching every other classification ingest module so the bench can
treat datasets uniformly.
"""

from __future__ import annotations

import csv
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.data import Dataset
else:  # pragma: no cover — import-time stubs
    Tensor = "Tensor"  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]

# Canonical labels. Order matches chest_xray_pneumonia so a model
# trained on either can score the other without label-tuple gymnastics.
RSNA_PNEUMONIA_LABELS: tuple[str, ...] = ("normal", "pneumonia")
Split = Literal["train", "val", "test"]
# 128 over the historical 256 — RSNA's ~20k 1024² PNGs were dominating
# wall-clock at the dataset's PIL.BILINEAR resize layer. At 128² the
# pre-resize cache also fits comfortably under ~500MB while the full-
# res png_cache/ stays available for the YOLO adapter.
DEFAULT_INPUT_SIZE = 128

# On-disk filenames in the extraction root.
_LABELS_CSV = "stage_2_train_labels.csv"
_TRAIN_DICOMS_DIR = "stage_2_train_images"
_PNG_CACHE_DIR = "png_cache"
_RESIZED_CACHE_DIR_FMT = "png_cache_{size}"


@dataclass(frozen=True)
class RsnaPneumoniaSample:
    """One on-disk image with its image-level binary label."""

    image_path: Path  # PNG path under png_cache/
    label: int  # index into RSNA_PNEUMONIA_LABELS


# --- label parsing ------------------------------------------------------


def _parse_image_level_labels(csv_path: Path) -> dict[str, int]:
    """Parse ``stage_2_train_labels.csv`` into ``{patientId: image_level_label}``.

    The CSV has one row per bounding box, so a positive case has multiple
    rows. We collapse on ``patientId`` and take the max ``Target``: any
    bbox row makes the image positive. ``Target=0`` rows have empty
    bbox fields, ``Target=1`` rows have at least one bbox.
    """
    by_patient: dict[str, int] = {}
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            patient_id = row["patientId"]
            target = int(row["Target"])
            by_patient[patient_id] = max(by_patient.get(patient_id, 0), target)
    if not by_patient:
        raise RuntimeError(f"no rows parsed from {csv_path}")
    return by_patient


# --- DICOM → PNG conversion --------------------------------------------


def _convert_dicom_to_png(dicom_path: Path, png_path: Path) -> None:
    """Decode one DICOM and write its pixel data as an 8-bit PNG.

    pydicom's ``pixel_array`` returns ``uint16`` for typical chest
    X-rays; we apply VOI LUT (if present) then scale to ``uint8`` so
    PIL can render it as a regular image. RSNA's DICOMs are monochrome
    (PhotometricInterpretation = MONOCHROME2), so we don't bother with
    multi-channel logic.
    """
    import numpy as np
    import pydicom
    from PIL import Image
    from pydicom.pixels import apply_voi_lut

    ds = pydicom.dcmread(str(dicom_path))
    pixels = ds.pixel_array

    # Apply VOI LUT (Window Center + Width) when present to get a more
    # display-faithful image; falls through to raw pixels when absent.
    try:
        pixels = apply_voi_lut(pixels, ds)
    except Exception:  # noqa: BLE001
        # Some DICOMs have malformed LUT tags; fall back to raw pixels
        # rather than failing the whole ingest. Logged so the operator
        # can investigate if it happens often.
        logger.debug("apply_voi_lut failed for %s; using raw pixels", dicom_path)

    # MONOCHROME1 stores inverted intensity (black=high). Flip so PNG
    # looks right under normal viewers.
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        pixels = pixels.max() - pixels

    # Normalize to uint8 for PNG storage. Min-max rescale per image —
    # window levels vary across the dataset, and the runtime adapter
    # also runs its own normalization, so per-image rescale here is
    # fine.
    pmin, pmax = float(pixels.min()), float(pixels.max())
    if pmax > pmin:
        scaled = ((pixels - pmin) / (pmax - pmin) * 255.0).astype(np.uint8)
    else:
        scaled = np.zeros_like(pixels, dtype=np.uint8)

    png_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(scaled, mode="L").save(png_path, format="PNG")


def _png_cache_path(root: Path, patient_id: str) -> Path:
    return root / _PNG_CACHE_DIR / f"{patient_id}.png"


def _resized_cache_path(root: Path, patient_id: str, input_size: int) -> Path:
    return root / _RESIZED_CACHE_DIR_FMT.format(size=input_size) / f"{patient_id}.png"


def _ensure_resized_png(src_png: Path, dst_png: Path, input_size: int) -> None:
    """Resize ``src_png`` once → ``dst_png``; idempotent on existing files."""
    from PIL import Image

    if dst_png.is_file():
        return
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    img = (
        Image.open(src_png)
        .convert("RGB")
        .resize((input_size, input_size), Image.BILINEAR)
    )
    img.save(dst_png, format="PNG")


def ensure_resized_cache(
    samples: list[RsnaPneumoniaSample],
    root: Path,
    *,
    input_size: int,
) -> list[RsnaPneumoniaSample]:
    """Pre-resize each sample's PNG once; return samples pointing at the cache.

    Reads the full-res PNG already cached by :func:`discover`, writes a
    sibling ``png_cache_<size>/`` PNG, and returns samples whose
    ``image_path`` points at the resized version. Subsequent runs short-
    circuit when every resized PNG already exists.

    Per-epoch ``PIL.Image.resize`` on ~14k 1024² → 128² calls was the
    dominant cost in the classification training loop; pre-resizing
    once amortises it across every epoch and every trial. The full-res
    cache stays untouched so the YOLO adapter (which reads
    ``png_cache/`` directly) is unaffected.
    """
    out: list[RsnaPneumoniaSample] = []
    built = 0
    for sample in samples:
        resized = _resized_cache_path(root, sample.image_path.stem, input_size)
        if not resized.is_file():
            _ensure_resized_png(sample.image_path, resized, input_size)
            built += 1
        out.append(RsnaPneumoniaSample(image_path=resized, label=sample.label))
    if built > 0:
        logger.info(
            "rsna_pneumonia: built %d resized PNG(s) at %dx%d",
            built,
            input_size,
            input_size,
        )
    return out


# --- discovery ----------------------------------------------------------


def discover(
    root: Path, *, max_samples: int | None = None
) -> list[RsnaPneumoniaSample]:
    """Walk the dataset root, convert + cache DICOMs, return samples.

    ``max_samples`` caps the total number of converted images — useful
    for smoke-testing the bench without paying for the full ~28k
    conversion. ``None`` (default) processes everything.

    Raises ``FileNotFoundError`` when the labels CSV or the DICOM
    directory is missing — surfaces a clear error before training would
    produce a zero-sample epoch.
    """
    csv_path = root / _LABELS_CSV
    dicom_dir = root / _TRAIN_DICOMS_DIR
    if not csv_path.is_file():
        raise FileNotFoundError(f"{csv_path} missing — did `download` complete?")
    if not dicom_dir.is_dir():
        raise FileNotFoundError(f"{dicom_dir} missing — did `download` complete?")

    by_patient = _parse_image_level_labels(csv_path)
    label_to_idx = {label: idx for idx, label in enumerate(RSNA_PNEUMONIA_LABELS)}

    samples: list[RsnaPneumoniaSample] = []
    # Sort patient_ids so the discovery order is deterministic across
    # filesystems with different directory enumeration order.
    patient_ids = sorted(by_patient.keys())
    if max_samples is not None:
        patient_ids = patient_ids[:max_samples]

    converted = 0
    for patient_id in patient_ids:
        dicom_path = dicom_dir / f"{patient_id}.dcm"
        if not dicom_path.is_file():
            # CSV references an image we don't have on disk — skip and
            # warn. Don't raise, since the upstream sometimes ships
            # extra label rows.
            logger.warning(
                "rsna_pneumonia: csv references missing DICOM %s; skipping",
                dicom_path,
            )
            continue

        png_path = _png_cache_path(root, patient_id)
        if not png_path.is_file():
            _convert_dicom_to_png(dicom_path, png_path)
            converted += 1

        target = by_patient[patient_id]
        label_name = "pneumonia" if target == 1 else "normal"
        samples.append(
            RsnaPneumoniaSample(
                image_path=png_path,
                label=label_to_idx[label_name],
            )
        )

    if converted > 0:
        logger.info("rsna_pneumonia: converted %d new DICOM(s) to PNG cache", converted)
    if not samples:
        raise RuntimeError(f"no rsna_pneumonia samples discovered under {root}")
    return samples


# --- stratified split ---------------------------------------------------


def stratified_split(
    samples: list[RsnaPneumoniaSample],
    *,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: str = "rsna-pneumonia-v1",
) -> dict[Split, list[RsnaPneumoniaSample]]:
    """Deterministic stratified split, same shape as the other modules."""
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError(f"invalid split fractions: train={train_frac} val={val_frac}")
    by_label: dict[int, list[RsnaPneumoniaSample]] = {}
    for s in samples:
        by_label.setdefault(s.label, []).append(s)

    out: dict[Split, list[RsnaPneumoniaSample]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for label_samples in by_label.values():
        ranked = sorted(
            label_samples,
            key=lambda s: hashlib.sha256(
                f"{seed}|{s.image_path.stem}".encode()
            ).hexdigest(),
        )
        n = len(ranked)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        out["train"].extend(ranked[:n_train])
        out["val"].extend(ranked[n_train : n_train + n_val])
        out["test"].extend(ranked[n_train + n_val :])
    return out


# --- torch dataset ------------------------------------------------------


try:
    import numpy as _np
    import torch as _torch
    from PIL import Image as _Image
    from torch.utils.data import Dataset as _DatasetBase

    class RsnaPneumoniaDataset(_DatasetBase):  # type: ignore[misc, valid-type]
        def __init__(self, items: list[RsnaPneumoniaSample]) -> None:
            self._items = items

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, idx: int):
            sample = self._items[idx]
            # PNG is pre-resized to the target size by
            # ``ensure_resized_cache``; no per-batch resize → workers
            # bottleneck on PIL decode + numpy convert only.
            image = _Image.open(sample.image_path).convert("RGB")
            img_arr = _np.asarray(image, dtype=_np.float32) / 255.0
            img_t = _torch.from_numpy(img_arr).permute(2, 0, 1)
            return img_t, sample.label

except ImportError:  # pragma: no cover — torch-less envs hit discover/split only
    RsnaPneumoniaDataset = None  # type: ignore[assignment, misc]


def build_dataset(samples: list[RsnaPneumoniaSample]):
    """Construct an ``RsnaPneumoniaDataset`` for these samples.

    The target image size is baked into the resized cache by
    :func:`ensure_resized_cache`, so this constructor takes no
    ``input_size`` argument — call ``ensure_resized_cache(samples,
    root, input_size=N)`` first so every PNG referenced here is
    already the right shape.
    """
    if RsnaPneumoniaDataset is None:
        raise SystemExit("torch not installed — run `uv sync --extra vision-server`.")
    return RsnaPneumoniaDataset(samples)
