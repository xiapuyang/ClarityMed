"""Colon histopathology dataset wrapper — discovery + split smoke tests.

Mirrors lung_histopath's test file. The same split-by-organ invariant
applies in reverse: the colon discover must not pull lung samples even
when a co-extracted lung subdir is on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from claritymed.ingest.vision.colon_histopath.dataset import (
    COLON_HISTOPATH_LABELS,
    ColonHistopathSample,
    _canonicalize_folder_name,
    discover,
    stratified_split,
)


# --- folder-name normalisation --------------------------------------------


@pytest.mark.parametrize(
    "folder,expected",
    [
        ("colon_aca", "adenocarcinoma"),
        ("colon_n", "normal"),
        ("Colon_ACA", "adenocarcinoma"),
        ("COLON-N", "normal"),
        ("colon.aca", "adenocarcinoma"),
        ("adenocarcinoma", "adenocarcinoma"),
        ("normal", "normal"),
    ],
)
def test_canonicalize_folder_name(folder: str, expected: str | None) -> None:
    assert _canonicalize_folder_name(folder) == expected


def test_canonicalize_unknown_returns_none() -> None:
    """Lung short forms must NOT resolve through the colon canonicaliser."""
    assert _canonicalize_folder_name("lung_aca") is None
    assert _canonicalize_folder_name("lung_n") is None
    assert _canonicalize_folder_name("lung_scc") is None
    assert _canonicalize_folder_name("squamous_cell_carcinoma") is None
    assert _canonicalize_folder_name("rare_subtype") is None
    assert _canonicalize_folder_name("") is None


# --- discovery ------------------------------------------------------------


def _make_fake_colon_archive(
    root: Path,
    *,
    counts: dict[str, int],
    folder_variant: str = "short",
) -> Path:
    """Mirror the upstream LC25000 archive's on-disk layout (colon subset)."""
    base = root / "lung_colon_image_set"
    base.mkdir()
    colon_dir = base / "colon_image_sets"
    colon_dir.mkdir()
    short_map = {
        "adenocarcinoma": "colon_aca",
        "normal": "colon_n",
    }
    for canonical, n in counts.items():
        folder_name = short_map[canonical] if folder_variant == "short" else canonical
        class_dir = colon_dir / folder_name
        class_dir.mkdir()
        for i in range(n):
            (class_dir / f"img_{i:03d}.jpeg").write_bytes(b"\xff\xd8\xff\xe0fake")
    return base


def test_discover_finds_all_samples(tmp_path: Path) -> None:
    counts = {"adenocarcinoma": 5, "normal": 3}
    root = _make_fake_colon_archive(tmp_path, counts=counts)
    samples = discover(root)
    assert len(samples) == sum(counts.values())
    assert all(0 <= s.label < len(COLON_HISTOPATH_LABELS) for s in samples)


def test_discover_normalises_folder_names(tmp_path: Path) -> None:
    counts = {label: 1 for label in COLON_HISTOPATH_LABELS}
    for variant in ("short", "canonical"):
        sub = tmp_path / variant
        sub.mkdir()
        root = _make_fake_colon_archive(sub, counts=counts, folder_variant=variant)
        samples = discover(root)
        assert len(samples) == len(COLON_HISTOPATH_LABELS)
        labels_seen = {COLON_HISTOPATH_LABELS[s.label] for s in samples}
        assert labels_seen == set(COLON_HISTOPATH_LABELS)


def test_discover_ignores_lung_subdir(tmp_path: Path) -> None:
    """A LC25000 archive with both organs present yields only colon samples."""
    counts = {label: 1 for label in COLON_HISTOPATH_LABELS}
    root = _make_fake_colon_archive(tmp_path, counts=counts)
    # Co-extract the lung subdir — must be invisible to the colon discover.
    lung_dir = root / "lung_image_sets"
    lung_dir.mkdir()
    (lung_dir / "lung_aca").mkdir()
    (lung_dir / "lung_aca" / "img_000.jpeg").write_bytes(b"\xff\xd8")

    samples = discover(root)
    assert len(samples) == len(COLON_HISTOPATH_LABELS)
    # See the matching lung-side test for why a substring check is unsafe.
    for sample in samples:
        assert sample.image_path.parent.parent.name == "colon_image_sets"


def test_discover_skips_unknown_folder_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    counts = {label: 1 for label in COLON_HISTOPATH_LABELS}
    root = _make_fake_colon_archive(tmp_path, counts=counts)
    (root / "colon_image_sets" / "colon_rare_subtype").mkdir()
    (root / "colon_image_sets" / "colon_rare_subtype" / "img.jpeg").write_bytes(b"x")

    import logging

    with caplog.at_level(
        logging.WARNING, logger="claritymed.ingest.vision.colon_histopath.dataset"
    ):
        samples = discover(root)
    assert len(samples) == len(COLON_HISTOPATH_LABELS)
    assert any("colon_rare_subtype" in rec.message for rec in caplog.records)


def test_discover_raises_when_organ_subdir_missing(tmp_path: Path) -> None:
    root = tmp_path / "lung_colon_image_set"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="colon_histopath organ subdir"):
        discover(root)


def test_discover_raises_when_no_images(tmp_path: Path) -> None:
    root = tmp_path / "lung_colon_image_set"
    (root / "colon_image_sets").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no colon_histopath samples"):
        discover(root)


# --- stratified split -----------------------------------------------------


def test_stratified_split_proportions_hold_per_class() -> None:
    samples = [
        ColonHistopathSample(image_path=Path(f"aca_{i}.jpeg"), label=0)
        for i in range(20)
    ] + [
        ColonHistopathSample(image_path=Path(f"norm_{i}.jpeg"), label=1)
        for i in range(20)
    ]
    splits = stratified_split(samples)
    assert len(splits["train"]) == 14 + 14
    assert len(splits["val"]) == 3 + 3
    assert len(splits["test"]) == 3 + 3


def test_stratified_split_is_deterministic() -> None:
    samples = [
        ColonHistopathSample(image_path=Path(f"img_{i:03d}.jpeg"), label=i % 2)
        for i in range(20)
    ]
    a = stratified_split(samples, seed="deterministic-v1")
    b = stratified_split(samples, seed="deterministic-v1")
    assert [s.image_path for s in a["train"]] == [s.image_path for s in b["train"]]


def test_stratified_split_rejects_invalid_fractions() -> None:
    with pytest.raises(ValueError, match="invalid split fractions"):
        stratified_split([], train_frac=0.9, val_frac=0.2)
