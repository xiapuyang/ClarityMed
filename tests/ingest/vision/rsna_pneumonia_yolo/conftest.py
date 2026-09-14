"""Synthetic RSNA fixture builder for the yolo_forge detection tests.

Constructs a fake extraction root in-process via :mod:`pydicom` so the
detection-side tests don't need the real (~3 GB) Kaggle archive.
Mirrors the helper in ``tests/ingest/vision/rsna_pneumonia/test_dataset.py``
but emits the bbox rows (``Target=1``) with concrete x/y/w/h values so
the bbox-preserving parser actually has something to read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
import pydicom.dataset
import pydicom.uid


# Tiny but non-trivial DICOM size — keeps tests under a second while
# exercising the bbox normalisation path (256x256 means each pixel
# bbox value rounds to a distinct decimal in normalised form).
FIXTURE_ROWS = 256
FIXTURE_COLS = 256


def _write_synthetic_dicom(
    out_path: Path, *, rows: int = FIXTURE_ROWS, cols: int = FIXTURE_COLS
) -> None:
    """Write one minimal-valid DICOM with a monotonically-increasing pixel grid."""
    ds = pydicom.dataset.Dataset()
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = pydicom.uid.generate_uid()
    ds.file_meta = file_meta

    ds.PatientID = out_path.stem
    ds.Modality = "DX"
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel = 1
    ds.PixelRepresentation = 0
    ds.BitsStored = 8
    ds.BitsAllocated = 8
    ds.HighBit = 7
    ds.Rows = rows
    ds.Columns = cols
    pixels = np.arange(rows * cols, dtype=np.uint8).reshape(rows, cols)
    ds.PixelData = pixels.tobytes()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_as(str(out_path), write_like_original=False)


def make_fake_rsna_root(
    root: Path,
    *,
    n_normal: int,
    n_pneumonia: int,
    bbox_per_positive: int = 1,
) -> Path:
    """Build a fake RSNA extraction root with a labels CSV + DICOM files.

    Each positive patient gets ``bbox_per_positive`` bbox rows in the
    CSV — covers the multi-bbox-per-patient collapse path. The bbox
    coordinates are picked to be normalisable to predictable decimals
    against ``FIXTURE_ROWS``/``FIXTURE_COLS``.
    """
    dicom_dir = root / "stage_2_train_images"
    dicom_dir.mkdir(parents=True, exist_ok=True)

    rows = ["patientId,x,y,width,height,Target"]
    for i in range(n_normal):
        pid = f"normal_{i:03d}"
        rows.append(f"{pid},,,,,0")
        _write_synthetic_dicom(dicom_dir / f"{pid}.dcm")
    for i in range(n_pneumonia):
        pid = f"pneumonia_{i:03d}"
        # Each bbox shifts right by 16 px so multi-bbox rows are
        # distinguishable on inspection. x=32, y=64, w=64, h=64 →
        # normalises to (cx=0.25, cy=0.375, w=0.25, h=0.25) at 256×256.
        for j in range(bbox_per_positive):
            x = 32 + j * 16
            rows.append(f"{pid},{x},64,64,64,1")
        _write_synthetic_dicom(dicom_dir / f"{pid}.dcm")

    (root / "stage_2_train_labels.csv").write_text("\n".join(rows) + "\n")
    return root
