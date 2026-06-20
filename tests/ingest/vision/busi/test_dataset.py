"""BUSI dataset wrapper — discovery + stratified split.

Heavy ``torch`` import in :func:`build_dataset` is exercised only when
the extra is installed; the smoke tests here cover the pure-Python
``discover`` + ``stratified_split`` paths so CI gets coverage without
``vision-server`` extras.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from claritymed.ingest.vision.busi.dataset import (
    BUSI_LABELS,
    BUSISample,
    discover,
    stratified_split,
)


def _make_fake_busi(
    root: Path,
    *,
    n_benign: int,
    n_malignant: int,
    n_normal: int,
    aux_masks: dict[str, int] | None = None,
) -> Path:
    """Mirror BUSI's on-disk layout: per-class folders with image + mask pairs.

    ``aux_masks`` maps an image stem (e.g. ``"benign (1)"``) to the
    number of auxiliary ``_mask_<n>.png`` files to create alongside the
    primary mask, mirroring BUSI's multi-lesion samples.
    """
    base = root / "Dataset_BUSI_with_GT"
    base.mkdir()
    counts = {
        "benign": n_benign,
        "malignant": n_malignant,
        "normal": n_normal,
    }
    aux = aux_masks or {}
    for label, n in counts.items():
        class_dir = base / label
        class_dir.mkdir()
        for i in range(n):
            stem = f"{label} ({i + 1})"
            (class_dir / f"{stem}.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
            (class_dir / f"{stem}_mask.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
            for k in range(aux.get(stem, 0)):
                (class_dir / f"{stem}_mask_{k + 1}.png").write_bytes(
                    b"\x89PNG\r\n\x1a\nfake"
                )
    return base


def test_discover_picks_up_every_pair(tmp_path):
    root = _make_fake_busi(tmp_path, n_benign=3, n_malignant=2, n_normal=1)
    samples = discover(root)
    assert len(samples) == 6
    labels = sorted({BUSI_LABELS[s.label] for s in samples})
    assert labels == sorted(BUSI_LABELS)


def test_discover_drops_orphan_images_without_masks(tmp_path):
    root = _make_fake_busi(tmp_path, n_benign=2, n_malignant=2, n_normal=2)
    # Delete one mask — the image should be dropped silently.
    (root / "benign" / "benign (1)_mask.png").unlink()
    samples = discover(root)
    assert len(samples) == 5


def test_discover_attaches_aux_masks_to_primary_sample(tmp_path, caplog):
    """Auxiliary ``_mask_<n>.png`` files attach to the parent sample
    instead of being mistaken for images and dropped. Real BUSI carries
    a few multi-lesion samples — without this they'd emit a spurious
    "mask missing" warning and the extra lesion's pixels would never
    reach the loss.
    """
    root = _make_fake_busi(
        tmp_path,
        n_benign=2,
        n_malignant=1,
        n_normal=1,
        aux_masks={"benign (1)": 2},  # 1 primary + 2 aux → 3 masks total
    )
    with caplog.at_level(logging.WARNING):
        samples = discover(root)
    # Exactly 4 samples; aux masks did NOT spawn fake samples of their own.
    assert len(samples) == 4
    multi = next(s for s in samples if s.image_path.name == "benign (1).png")
    assert len(multi.mask_paths) == 3
    assert multi.mask_paths[0].name == "benign (1)_mask.png"
    assert {p.name for p in multi.mask_paths[1:]} == {
        "benign (1)_mask_1.png",
        "benign (1)_mask_2.png",
    }
    assert not any("mask missing" in r.message for r in caplog.records)


def test_discover_raises_when_class_dir_missing(tmp_path):
    base = tmp_path / "Dataset_BUSI_with_GT"
    base.mkdir()
    (base / "benign").mkdir()
    with pytest.raises(FileNotFoundError):
        discover(base)


def _sample(stem_prefix: str, i: int, label: int) -> BUSISample:
    return BUSISample(
        Path(f"{stem_prefix} ({i}).png"),
        (Path(f"{stem_prefix} ({i})_mask.png"),),
        label,
    )


def test_stratified_split_keeps_every_label_in_every_split():
    samples = (
        [_sample("benign", i, 0) for i in range(20)]
        + [_sample("malignant", i, 1) for i in range(20)]
        + [_sample("normal", i, 2) for i in range(20)]
    )
    splits = stratified_split(samples, train_frac=0.7, val_frac=0.15)
    for split_name in ("train", "val", "test"):
        labels = {s.label for s in splits[split_name]}
        assert labels == {0, 1, 2}, f"missing label in {split_name}"


def test_stratified_split_is_deterministic():
    samples = [_sample("benign", i, 0) for i in range(10)] + [
        _sample("malignant", i, 1) for i in range(10)
    ]
    a = stratified_split(samples, seed="busi-v1")
    b = stratified_split(samples, seed="busi-v1")
    assert [s.image_path for s in a["train"]] == [s.image_path for s in b["train"]]


def test_stratified_split_rejects_invalid_fractions():
    with pytest.raises(ValueError):
        stratified_split([], train_frac=0.9, val_frac=0.2)


def test_busi_labels_are_three_class():
    assert BUSI_LABELS == ("benign", "malignant", "normal")
