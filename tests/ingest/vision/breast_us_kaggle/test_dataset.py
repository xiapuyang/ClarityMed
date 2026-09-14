"""Breast-US Kaggle (Vuppalaadithyasairam) dataset wrapper — smoke tests.

Heavy ``torch`` import in :func:`build_dataset` is exercised only when
the extra is installed; the smoke tests here cover the pure-Python
``discover`` + ``stratified_split`` paths so CI gets coverage without
``vision-server`` extras.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from claritymed.ingest.vision.breast_us_kaggle.dataset import (
    BREAST_US_KAGGLE_LABELS,
    BreastUsKaggleSample,
    _canonicalize_folder_name,
    discover,
    stratified_split,
)


# --- folder-name normalisation --------------------------------------------


@pytest.mark.parametrize(
    "folder,expected",
    [
        ("benign", "benign"),
        ("malignant", "malignant"),
        ("Benign", "benign"),
        ("MALIGNANT", "malignant"),
        # Trailing whitespace from filesystem oddities.
        ("benign ", "benign"),
    ],
)
def test_canonicalize_folder_name(folder: str, expected: str | None) -> None:
    assert _canonicalize_folder_name(folder) == expected


def test_canonicalize_unknown_returns_none() -> None:
    """Folders we can't map to a canonical label return ``None``."""
    # No ``normal`` class in this dataset — explicit exclusion.
    assert _canonicalize_folder_name("normal") is None
    assert _canonicalize_folder_name("") is None
    assert _canonicalize_folder_name("benign_subset_v2") is None


# --- discovery ------------------------------------------------------------


def _make_fake_breast_us_kaggle(
    root: Path,
    *,
    n_benign: int,
    n_malignant: int,
) -> Path:
    """Mirror the upstream archive's on-disk layout."""
    base = root / "ultrasound breast classification"
    base.mkdir()
    for subsplit in ("train", "val"):
        for label, count in (("benign", n_benign), ("malignant", n_malignant)):
            class_dir = base / subsplit / label
            class_dir.mkdir(parents=True)
            for i in range(count):
                (class_dir / f"{subsplit}_{label}_{i:03d}.png").write_bytes(
                    b"\x89PNG\r\n\x1a\nfake"
                )
    return base


def test_discover_finds_all_samples(tmp_path: Path) -> None:
    """Discover pools across the train + val upstream subsplits."""
    root = _make_fake_breast_us_kaggle(tmp_path, n_benign=3, n_malignant=4)
    samples = discover(root)
    # Both train/ and val/ get walked — total is double the per-subsplit count.
    assert len(samples) == (3 + 4) * 2
    assert all(0 <= s.label < len(BREAST_US_KAGGLE_LABELS) for s in samples)


def test_discover_skips_unknown_folder_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An unrecognised class folder (e.g. ``normal/``) warns and skips."""
    root = _make_fake_breast_us_kaggle(tmp_path, n_benign=2, n_malignant=2)
    (root / "train" / "normal").mkdir()
    (root / "train" / "normal" / "img.png").write_bytes(b"x")

    import logging

    with caplog.at_level(
        logging.WARNING, logger="claritymed.ingest.vision.breast_us_kaggle.dataset"
    ):
        samples = discover(root)
    # 2 benign + 2 malignant × 2 subsplits = 8; the rogue normal folder is skipped.
    assert len(samples) == 8
    assert any("normal" in rec.message for rec in caplog.records)


def test_discover_raises_when_no_upstream_subdirs(tmp_path: Path) -> None:
    root = tmp_path / "ultrasound breast classification"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="no upstream subdirs"):
        discover(root)


def test_discover_raises_when_no_images(tmp_path: Path) -> None:
    """An empty ``train/`` with no class folders → no samples → raise."""
    root = tmp_path / "ultrasound breast classification"
    (root / "train").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no breast_us_kaggle samples"):
        discover(root)


# --- stratified split -----------------------------------------------------


def test_stratified_split_proportions_hold_per_class() -> None:
    """Each class gets the expected ~70/15/15 split."""
    samples = [
        BreastUsKaggleSample(image_path=Path(f"ben_{i}.png"), label=0)
        for i in range(20)
    ] + [
        BreastUsKaggleSample(image_path=Path(f"mal_{i}.png"), label=1)
        for i in range(20)
    ]
    splits = stratified_split(samples)
    assert len(splits["train"]) == 14 + 14
    assert len(splits["val"]) == 3 + 3
    assert len(splits["test"]) == 3 + 3


def test_stratified_split_is_deterministic() -> None:
    """Same seed → same split."""
    samples = [
        BreastUsKaggleSample(image_path=Path(f"img_{i:03d}.png"), label=i % 2)
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
