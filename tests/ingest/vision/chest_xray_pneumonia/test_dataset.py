"""Chest X-Ray Pneumonia (Kermany 2018) dataset wrapper — smoke tests.

Heavy ``torch`` import in :func:`build_dataset` is exercised only when
the extra is installed; the smoke tests here cover the pure-Python
``discover`` + ``stratified_split`` paths so CI gets coverage without
``vision-server`` extras.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from claritymed.ingest.vision.chest_xray_pneumonia.dataset import (
    CHEST_XRAY_PNEUMONIA_LABELS,
    ChestXrayPneumoniaSample,
    _canonicalize_folder_name,
    discover,
    stratified_split,
)


# --- folder-name normalisation -------------------------------------------


@pytest.mark.parametrize(
    "folder,expected",
    [
        ("NORMAL", "normal"),
        ("PNEUMONIA", "pneumonia"),
        ("normal", "normal"),
        ("pneumonia", "pneumonia"),
        # Trailing whitespace from filesystem oddities.
        ("PNEUMONIA ", "pneumonia"),
    ],
)
def test_canonicalize_folder_name(folder: str, expected: str | None) -> None:
    assert _canonicalize_folder_name(folder) == expected


def test_canonicalize_unknown_returns_none() -> None:
    """Folders we can't map to a canonical label return ``None``."""
    assert _canonicalize_folder_name("benign") is None
    assert _canonicalize_folder_name("") is None
    assert _canonicalize_folder_name("PNEUMONIA_VIRAL") is None


# --- discovery -----------------------------------------------------------


def _make_fake_kermany(
    root: Path,
    *,
    n_normal: int,
    n_pneumonia: int,
) -> Path:
    """Mirror the upstream archive's on-disk layout under
    ``chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/``.
    """
    base = root / "chest_xray"
    base.mkdir()
    for subsplit in ("train", "val", "test"):
        for label, count in (("NORMAL", n_normal), ("PNEUMONIA", n_pneumonia)):
            class_dir = base / subsplit / label
            class_dir.mkdir(parents=True)
            for i in range(count):
                # Real Kermany ships .jpeg; we accept .png/.jpg/.jpeg.
                (class_dir / f"{subsplit}_{label}_{i:03d}.jpeg").write_bytes(
                    b"\xff\xd8\xff\xe0fake"
                )
    return base


def test_discover_finds_all_samples_across_three_splits(tmp_path: Path) -> None:
    """Pools across train + val + test upstream subsplits — we treat
    Kermany's tiny upstream val as just more data and re-stratify.
    """
    root = _make_fake_kermany(tmp_path, n_normal=2, n_pneumonia=3)
    samples = discover(root)
    # 3 subsplits × (2 normal + 3 pneumonia) = 15.
    assert len(samples) == 15
    assert all(0 <= s.label < len(CHEST_XRAY_PNEUMONIA_LABELS) for s in samples)


def test_discover_skips_unknown_folder_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An unrecognised class folder warns and skips — guards against an
    upstream variant adding e.g. ``PNEUMONIA_VIRAL`` without crashing.
    """
    root = _make_fake_kermany(tmp_path, n_normal=2, n_pneumonia=2)
    (root / "train" / "PNEUMONIA_BACTERIAL").mkdir()
    (root / "train" / "PNEUMONIA_BACTERIAL" / "img.jpeg").write_bytes(b"x")

    import logging

    with caplog.at_level(
        logging.WARNING,
        logger="claritymed.ingest.vision.chest_xray_pneumonia.dataset",
    ):
        samples = discover(root)
    # 2 normal + 2 pneumonia × 3 subsplits = 12; PNEUMONIA_BACTERIAL skipped.
    assert len(samples) == 12
    assert any("PNEUMONIA_BACTERIAL" in rec.message for rec in caplog.records)


def test_discover_raises_when_no_upstream_subdirs(tmp_path: Path) -> None:
    root = tmp_path / "chest_xray"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="no upstream subdirs"):
        discover(root)


def test_discover_raises_when_no_images(tmp_path: Path) -> None:
    """An empty ``train/`` with no class folders → no samples → raise."""
    root = tmp_path / "chest_xray"
    (root / "train").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no chest_xray_pneumonia samples"):
        discover(root)


# --- stratified split ---------------------------------------------------


def test_stratified_split_proportions_hold_per_class() -> None:
    """Each class gets the expected ~70/15/15 split."""
    samples = [
        ChestXrayPneumoniaSample(image_path=Path(f"normal_{i}.jpeg"), label=0)
        for i in range(20)
    ] + [
        ChestXrayPneumoniaSample(image_path=Path(f"pneumonia_{i}.jpeg"), label=1)
        for i in range(20)
    ]
    splits = stratified_split(samples)
    assert len(splits["train"]) == 14 + 14
    assert len(splits["val"]) == 3 + 3
    assert len(splits["test"]) == 3 + 3


def test_stratified_split_is_deterministic() -> None:
    """Same seed → same split. The seed is pinned; any change must bump
    the dataset version so a model retrain doesn't quietly score against
    a different held-out chunk.
    """
    samples = [
        ChestXrayPneumoniaSample(image_path=Path(f"img_{i:03d}.jpeg"), label=i % 2)
        for i in range(40)
    ]
    a = stratified_split(samples, seed="deterministic-v1")
    b = stratified_split(samples, seed="deterministic-v1")
    assert [s.image_path for s in a["train"]] == [s.image_path for s in b["train"]]
    assert [s.image_path for s in a["val"]] == [s.image_path for s in b["val"]]
    assert [s.image_path for s in a["test"]] == [s.image_path for s in b["test"]]


def test_stratified_split_rejects_invalid_fractions() -> None:
    with pytest.raises(ValueError, match="invalid split fractions"):
        stratified_split([], train_frac=0.9, val_frac=0.2)
