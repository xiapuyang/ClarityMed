"""Tests for the bbox-preserving RSNA Pneumonia → YOLO ingest.

The classification adapter's tests cover DICOM→PNG conversion and
image-level label parsing; this file zeros in on what's new in the
detection path — bbox-row preservation, YOLO label normalisation,
empty-label-file emission for normals, and the data.yaml shape.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from claritymed.ingest.vision.rsna_pneumonia_yolo.dataset import (
    PNEUMONIA_CLASS_ID,
    _parse_bboxes,
    _split_patients,
    prepare_rsna_pneumonia_yolo,
)
from tests.ingest.vision.rsna_pneumonia_yolo.conftest import (
    FIXTURE_COLS,
    FIXTURE_ROWS,
    make_fake_rsna_root,
)


# --- CSV parsing --------------------------------------------------------


def test_parse_bboxes_keeps_one_entry_per_bbox_row(tmp_path: Path) -> None:
    """Positive patient with two CSV rows → two bbox entries (not collapsed)."""
    csv_path = tmp_path / "labels.csv"
    csv_path.write_text(
        "patientId,x,y,width,height,Target\n"
        "p1,,,,,0\n"
        "p2,10,20,50,60,1\n"
        "p2,200,200,30,30,1\n"
        "p3,,,,,0\n"
    )
    out = _parse_bboxes(csv_path)
    assert sorted(out) == ["p1", "p2", "p3"]
    assert out["p1"] == []
    assert out["p3"] == []
    assert len(out["p2"]) == 2
    # Order in the CSV is preserved (first row → index 0).
    assert (out["p2"][0].x, out["p2"][0].y) == (10.0, 20.0)
    assert (out["p2"][1].x, out["p2"][1].y) == (200.0, 200.0)


# --- stratified split ---------------------------------------------------


def test_split_patients_is_stratified_and_deterministic() -> None:
    """Positive/negative ratios in train/val/test track the source distribution."""
    by_patient = {
        **{f"pos_{i:03d}": [object()] for i in range(20)},
        **{f"neg_{i:03d}": [] for i in range(20)},
    }
    splits = _split_patients(by_patient, negative_ratio=None)  # type: ignore[arg-type]
    # 70/15/15 of 20 per class.
    assert sum(1 for p in splits["train"] if p.startswith("pos")) == 14
    assert sum(1 for p in splits["val"] if p.startswith("pos")) == 3
    assert sum(1 for p in splits["test"] if p.startswith("pos")) == 3
    # Determinism: a second call returns identical assignments.
    splits2 = _split_patients(by_patient, negative_ratio=None)  # type: ignore[arg-type]
    assert splits == splits2


def test_split_patients_negative_ratio_modes() -> None:
    """Verify the three canonical negative_ratio modes shape negatives correctly."""
    # 10 pos + 30 neg → natural 3:1 imbalance per-split.
    by_patient = {
        **{f"pos_{i:03d}": [object()] for i in range(10)},
        **{f"neg_{i:03d}": [] for i in range(30)},
    }
    # keep_all → all 30 negatives end up in splits, total 40 patients
    keep = _split_patients(by_patient, negative_ratio=None)  # type: ignore[arg-type]
    total_neg = sum(1 for split in keep.values() for p in split if p.startswith("neg"))
    assert total_neg == 30
    # drop → zero negatives
    drop = _split_patients(by_patient, negative_ratio=0.0)  # type: ignore[arg-type]
    assert all(not p.startswith("neg") for split in drop.values() for p in split)
    # balanced → per-split negatives == positives in that split
    bal = _split_patients(by_patient, negative_ratio=1.0)  # type: ignore[arg-type]
    for split in ("train", "val", "test"):
        n_pos = sum(1 for p in bal[split] if p.startswith("pos"))  # type: ignore[index]
        n_neg = sum(1 for p in bal[split] if p.startswith("neg"))  # type: ignore[index]
        assert n_pos == n_neg, f"{split}: pos={n_pos} != neg={n_neg}"
    # Determinism within a mode.
    bal2 = _split_patients(by_patient, negative_ratio=1.0)  # type: ignore[arg-type]
    assert bal == bal2


# --- end-to-end prepare -------------------------------------------------


def test_prepare_writes_yolo_layout_and_normalises_bboxes(tmp_path: Path) -> None:
    """Full prepare run: data.yaml + symlinks + label files in the expected shape."""
    raw = make_fake_rsna_root(
        tmp_path / "raw", n_normal=10, n_pneumonia=10, bbox_per_positive=2
    )

    splits = prepare_rsna_pneumonia_yolo(raw_root=raw)
    prepared = raw / "yolo_neg1"  # default negative_ratio=1.0 → balanced

    # --- data.yaml shape
    data = yaml.safe_load(splits.data_yaml_path.read_text())
    assert data["path"] == str(prepared.resolve())
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    assert data["test"] == "images/test"
    assert data["names"] == {0: "pneumonia"}

    # --- splits non-empty (assert_non_empty would also raise)
    splits.assert_non_empty()
    assert splits.train_count + splits.val_count + splits.test_count == 20

    # --- normals get empty label files
    for split in ("train", "val", "test"):
        for label_file in (prepared / "labels" / split).glob("normal_*.txt"):
            assert label_file.stat().st_size == 0, label_file

    # --- positives get one line per bbox row, normalised
    one_pos_label = next((prepared / "labels" / "train").glob("pneumonia_*.txt"), None)
    if one_pos_label is None:
        one_pos_label = next((prepared / "labels" / "val").glob("pneumonia_*.txt"))
    lines = [ln for ln in one_pos_label.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2  # bbox_per_positive=2
    # First bbox: x=32, y=64, w=64, h=64 at 256×256
    # → cx=(32+32)/256=0.25, cy=(64+32)/256=0.375, w=64/256=0.25, h=64/256=0.25
    parts0 = lines[0].split()
    assert int(parts0[0]) == PNEUMONIA_CLASS_ID
    assert float(parts0[1]) == 0.25
    assert float(parts0[2]) == 0.375
    assert float(parts0[3]) == 0.25
    assert float(parts0[4]) == 0.25


def test_prepare_creates_image_symlinks_into_png_cache(tmp_path: Path) -> None:
    """Images directory holds symlinks pointing at the shared png_cache PNGs."""
    raw = make_fake_rsna_root(tmp_path / "raw", n_normal=4, n_pneumonia=4)
    prepare_rsna_pneumonia_yolo(raw_root=raw)
    prepared = raw / "yolo_neg1"  # default negative_ratio=1.0 → balanced

    images = (
        list((prepared / "images" / "train").glob("*.png"))
        + list((prepared / "images" / "val").glob("*.png"))
        + list((prepared / "images" / "test").glob("*.png"))
    )
    assert images, "no prepared images found"
    for img in images:
        # Symlink (or copy on filesystems that can't symlink) — either way,
        # resolves to a file under png_cache/.
        resolved = img.resolve()
        assert resolved.is_file()
        assert resolved.parent.name == "png_cache"


def test_prepare_is_idempotent(tmp_path: Path) -> None:
    """Second prepare run on the same root must not blow up or duplicate."""
    raw = make_fake_rsna_root(tmp_path / "raw", n_normal=3, n_pneumonia=3)
    s1 = prepare_rsna_pneumonia_yolo(raw_root=raw)
    s2 = prepare_rsna_pneumonia_yolo(raw_root=raw)
    assert s1 == s2


def test_prepare_label_dim_uses_actual_image_size(tmp_path: Path) -> None:
    """Normalisation uses real PNG dimensions, not a hardcoded size."""
    raw = make_fake_rsna_root(tmp_path / "raw", n_normal=2, n_pneumonia=2)
    prepare_rsna_pneumonia_yolo(raw_root=raw)
    prepared = raw / "yolo_neg1"  # default negative_ratio=1.0 → balanced

    # Find any positive label file and verify the normalisation matches
    # the synthetic image dimensions (FIXTURE_COLS × FIXTURE_ROWS).
    pos_label = None
    for split in ("train", "val", "test"):
        for f in (prepared / "labels" / split).glob("pneumonia_*.txt"):
            if f.stat().st_size > 0:
                pos_label = f
                break
        if pos_label:
            break
    assert pos_label is not None
    line = pos_label.read_text().splitlines()[0]
    cx = float(line.split()[1])
    # Fixture bbox cx == (32 + 32) / FIXTURE_COLS — verifies divisor matches PNG.
    assert abs(cx - (32 + 32) / FIXTURE_COLS) < 1e-6
    assert FIXTURE_COLS == FIXTURE_ROWS  # sanity for the constant
