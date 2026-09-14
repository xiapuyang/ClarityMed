"""RSNA Pneumonia dataset wrapper tests.

Synthetic DICOMs constructed in-process via pydicom — no real upstream
archive download required for these unit tests. The heavy end-to-end
conversion test against the real Kaggle archive is operator-driven.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import pydicom.dataset
import pydicom.uid
import pytest

from claritymed.ingest.vision.rsna_pneumonia.dataset import (
    RSNA_PNEUMONIA_LABELS,
    RsnaPneumoniaSample,
    _parse_image_level_labels,
    discover,
    stratified_split,
)


# --- helpers -------------------------------------------------------------


def _write_synthetic_dicom(
    out_path: Path,
    *,
    photometric: str = "MONOCHROME2",
    bits: int = 8,
    rows: int = 32,
    cols: int = 32,
) -> None:
    """Construct a minimal-valid DICOM with monotonically-increasing pixel data.

    Mirrors the upstream RSNA shape (single-channel chest X-ray) but at
    a tiny resolution so tests stay fast. ``photometric`` lets a test
    flip between MONOCHROME1 / MONOCHROME2 to verify the loader's
    intensity-inversion path.
    """
    ds = pydicom.dataset.Dataset()
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
    ds.file_meta = file_meta

    ds.PatientID = out_path.stem
    ds.Modality = "DX"
    ds.PhotometricInterpretation = photometric
    ds.SamplesPerPixel = 1
    ds.PixelRepresentation = 0
    ds.BitsStored = bits
    ds.BitsAllocated = bits
    ds.HighBit = bits - 1
    ds.Rows = rows
    ds.Columns = cols

    dtype = np.uint8 if bits == 8 else np.uint16
    # Monotonically-increasing pixels so MONOCHROME1 inversion is
    # visually verifiable (top-left becomes the brightest, not darkest).
    pixels = np.arange(rows * cols, dtype=dtype).reshape(rows, cols)
    ds.PixelData = pixels.tobytes()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(out_path), write_like_original=False)


def _make_fake_rsna(
    root: Path,
    *,
    n_normal: int,
    n_pneumonia: int,
    photometric: str = "MONOCHROME2",
) -> Path:
    """Build a fake RSNA extraction root with a CSV + DICOM files.

    Positive cases get two CSV rows each (mimicking upstream's
    bbox-per-row shape) — verifies the patientId collapse logic.
    """
    dicom_dir = root / "stage_2_train_images"
    dicom_dir.mkdir(parents=True)

    rows = ["patientId,x,y,width,height,Target"]
    for i in range(n_normal):
        patient_id = f"normal_{i:03d}"
        rows.append(f"{patient_id},,,,,0")
        _write_synthetic_dicom(dicom_dir / f"{patient_id}.dcm", photometric=photometric)
    for i in range(n_pneumonia):
        patient_id = f"pneumonia_{i:03d}"
        # Two bbox rows per positive — exercises the max-Target collapse.
        rows.append(f"{patient_id},100,150,80,60,1")
        rows.append(f"{patient_id},200,200,40,40,1")
        _write_synthetic_dicom(dicom_dir / f"{patient_id}.dcm", photometric=photometric)

    (root / "stage_2_train_labels.csv").write_text("\n".join(rows) + "\n")
    return root


# --- label parsing -------------------------------------------------------


def test_parse_image_level_labels_collapses_bboxes_to_max(tmp_path: Path) -> None:
    """Positive cases ship multiple bbox rows per patientId; the loader
    collapses on patientId and takes the max Target.
    """
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "patientId,x,y,width,height,Target\n"
        "p1,,,,,0\n"
        "p2,10,20,50,60,1\n"
        "p2,200,200,30,30,1\n"  # second bbox for same positive patient
        "p3,,,,,0\n"
    )
    out = _parse_image_level_labels(csv_path)
    assert out == {"p1": 0, "p2": 1, "p3": 0}


def test_parse_image_level_labels_empty_csv_raises(tmp_path: Path) -> None:
    """Empty CSV (header only) → fail-loud."""
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text("patientId,x,y,width,height,Target\n")
    with pytest.raises(RuntimeError, match="no rows parsed"):
        _parse_image_level_labels(csv_path)


# --- DICOM conversion ----------------------------------------------------


def test_discover_converts_dicoms_and_writes_png_cache(tmp_path: Path) -> None:
    """Happy path: DICOMs land in png_cache/ as 8-bit PNGs."""
    root = _make_fake_rsna(tmp_path, n_normal=2, n_pneumonia=3)
    samples = discover(root)
    assert len(samples) == 5
    assert all(s.image_path.suffix == ".png" for s in samples)
    assert all(s.image_path.is_file() for s in samples)
    # Labels are in (normal=0, pneumonia=1) index space.
    labels = sorted(s.label for s in samples)
    assert labels == [0, 0, 1, 1, 1]


def test_discover_skips_conversion_when_png_already_cached(tmp_path: Path) -> None:
    """Re-running discovery on the same root reads from cache instead of
    re-converting. We verify by mutating one cached PNG between calls
    and confirming the loader keeps the mutated bytes.
    """
    root = _make_fake_rsna(tmp_path, n_normal=1, n_pneumonia=1)
    discover(root)
    cached = root / "png_cache" / "normal_000.png"
    original_bytes = cached.read_bytes()
    cached.write_bytes(b"\x89PNG\r\n\x1a\nmutated")
    discover(root)
    # If discover() re-converted, the original bytes would have been
    # restored. We confirm the mutated bytes survived.
    assert cached.read_bytes() != original_bytes
    assert cached.read_bytes().startswith(b"\x89PNG\r\n\x1a\nmutated")


def test_discover_warns_on_missing_dicom_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """CSV row referencing a DICOM not on disk → warn + skip, don't crash."""
    root = _make_fake_rsna(tmp_path, n_normal=1, n_pneumonia=1)
    # Add an extra CSV row pointing at a missing DICOM.
    csv_path = root / "stage_2_train_labels.csv"
    csv_path.write_text(csv_path.read_text() + "ghost_patient,,,,,0\n")

    import logging

    with caplog.at_level(
        logging.WARNING, logger="claritymed.ingest.vision.rsna_pneumonia.dataset"
    ):
        samples = discover(root)
    # ghost_patient excluded from samples; warning logged.
    assert len(samples) == 2
    assert any("ghost_patient" in rec.message for rec in caplog.records)


def test_discover_respects_max_samples_cap(tmp_path: Path) -> None:
    """``max_samples=2`` processes only the first 2 sorted patient_ids."""
    root = _make_fake_rsna(tmp_path, n_normal=3, n_pneumonia=3)
    samples = discover(root, max_samples=2)
    assert len(samples) == 2


# --- DICOM edge cases ----------------------------------------------------


def test_discover_handles_monochrome1_inversion(tmp_path: Path) -> None:
    """MONOCHROME1 stores inverted intensity — the loader must flip
    before writing the PNG so a top-left max value becomes the brightest
    pixel in the saved file.
    """
    from PIL import Image

    root = _make_fake_rsna(
        tmp_path, n_normal=1, n_pneumonia=0, photometric="MONOCHROME1"
    )
    discover(root)
    png_path = root / "png_cache" / "normal_000.png"
    pil = Image.open(png_path)
    arr = np.asarray(pil)
    # Synthetic source: pixels = arange(32*32), so pixel[0,0]=0 and
    # pixel[31,31]=1023 (clamped to 8-bit). MONOCHROME1 inversion +
    # min-max rescale flips the gradient: the original-max corner should
    # become the original-min corner. After our inversion + rescale,
    # pixel[0,0] should be the brightest.
    assert arr[0, 0] > arr[-1, -1]


# --- error paths ---------------------------------------------------------


def test_discover_missing_csv_raises(tmp_path: Path) -> None:
    """No CSV → clear FileNotFoundError pointing the user at download."""
    root = tmp_path / "rsna"
    (root / "stage_2_train_images").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="stage_2_train_labels.csv"):
        discover(root)


def test_discover_missing_dicom_dir_raises(tmp_path: Path) -> None:
    """CSV present but no DICOMs dir → fail-loud."""
    root = tmp_path / "rsna"
    root.mkdir()
    (root / "stage_2_train_labels.csv").write_text(
        "patientId,x,y,width,height,Target\n"
    )
    with pytest.raises(FileNotFoundError, match="stage_2_train_images"):
        discover(root)


# --- stratified split ----------------------------------------------------


def test_stratified_split_proportions_hold_per_class() -> None:
    """Per-class 70/15/15 split matches other ingest modules' shape."""
    samples = [
        RsnaPneumoniaSample(image_path=Path(f"normal_{i}.png"), label=0)
        for i in range(20)
    ] + [
        RsnaPneumoniaSample(image_path=Path(f"pneumonia_{i}.png"), label=1)
        for i in range(20)
    ]
    splits = stratified_split(samples)
    assert len(splits["train"]) == 14 + 14
    assert len(splits["val"]) == 3 + 3
    assert len(splits["test"]) == 3 + 3


def test_labels_tuple_matches_chest_xray_pneumonia() -> None:
    """RSNA labels must match Kermany's tuple exactly so bench-side
    binary collapse stays apples-to-apples across the cross-eval pair.
    """
    from claritymed.ingest.vision.chest_xray_pneumonia.dataset import (
        CHEST_XRAY_PNEUMONIA_LABELS,
    )

    assert RSNA_PNEUMONIA_LABELS == CHEST_XRAY_PNEUMONIA_LABELS
