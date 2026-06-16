"""Lung histopathology dataset wrapper — discovery + split smoke tests.

Heavy ``torch`` import in :func:`build_dataset` is exercised only when
the extra is installed; the smoke tests here cover the pure-Python
``discover`` + ``stratified_split`` paths so CI gets coverage without
``vision-server`` extras.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from claritymed.ingest.vision.lung_histopath.dataset import (
    LUNG_HISTOPATH_LABELS,
    LungHistopathSample,
    _canonicalize_folder_name,
    discover,
    stratified_split,
)


# --- folder-name normalisation --------------------------------------------


@pytest.mark.parametrize(
    "folder,expected",
    [
        # Upstream short forms.
        ("lung_aca", "adenocarcinoma"),
        ("lung_n", "normal"),
        ("lung_scc", "squamous_cell_carcinoma"),
        # Mixed case + alternative separators in the short form.
        ("Lung_ACA", "adenocarcinoma"),
        ("LUNG-N", "normal"),
        ("lung.scc", "squamous_cell_carcinoma"),
        # Canonical labels accepted directly.
        ("adenocarcinoma", "adenocarcinoma"),
        ("normal", "normal"),
        ("squamous_cell_carcinoma", "squamous_cell_carcinoma"),
        # Display-cased canonical labels.
        ("Squamous Cell Carcinoma", "squamous_cell_carcinoma"),
        # Runs of separators squash to a single underscore.
        ("squamous__cell__carcinoma", "squamous_cell_carcinoma"),
    ],
)
def test_canonicalize_folder_name(folder: str, expected: str | None) -> None:
    assert _canonicalize_folder_name(folder) == expected


def test_canonicalize_unknown_returns_none() -> None:
    """Folders we can't map to a canonical label return ``None``.

    Importantly, the colon classes (``colon_aca`` / ``colon_n``) must NOT
    resolve through the lung canonicaliser — they belong to the sibling
    colon_histopath module. The split-by-organ-module design depends on
    each module rejecting the other's folder names so a mis-extracted
    archive fails loud instead of silently mixing organs.
    """
    assert _canonicalize_folder_name("colon_aca") is None
    assert _canonicalize_folder_name("colon_n") is None
    assert _canonicalize_folder_name("rare_subtype") is None
    assert _canonicalize_folder_name("") is None


# --- discovery ------------------------------------------------------------


def _make_fake_lung_archive(
    root: Path,
    *,
    counts: dict[str, int],
    folder_variant: str = "short",
) -> Path:
    """Mirror the upstream LC25000 archive's on-disk layout (lung subset).

    ``folder_variant``:

    * ``"short"`` — use the upstream's short folder names
      (``lung_aca`` / ``lung_n`` / ``lung_scc``).
    * ``"canonical"`` — use the canonical expanded labels.

    Returns the path of the ``lung_colon_image_set/`` top-level dir
    that :func:`discover` expects to walk.
    """
    base = root / "lung_colon_image_set"
    base.mkdir()
    lung_dir = base / "lung_image_sets"
    lung_dir.mkdir()
    short_map = {
        "adenocarcinoma": "lung_aca",
        "normal": "lung_n",
        "squamous_cell_carcinoma": "lung_scc",
    }
    for canonical, n in counts.items():
        folder_name = short_map[canonical] if folder_variant == "short" else canonical
        class_dir = lung_dir / folder_name
        class_dir.mkdir()
        for i in range(n):
            (class_dir / f"img_{i:03d}.jpeg").write_bytes(b"\xff\xd8\xff\xe0fake")
    return base


def test_discover_finds_all_samples(tmp_path: Path) -> None:
    counts = {
        "adenocarcinoma": 4,
        "normal": 3,
        "squamous_cell_carcinoma": 2,
    }
    root = _make_fake_lung_archive(tmp_path, counts=counts)
    samples = discover(root)
    assert len(samples) == sum(counts.values())
    assert all(0 <= s.label < len(LUNG_HISTOPATH_LABELS) for s in samples)


def test_discover_normalises_folder_names(tmp_path: Path) -> None:
    """Both ``short`` (upstream) and ``canonical`` folder names resolve."""
    counts = {label: 1 for label in LUNG_HISTOPATH_LABELS}
    for variant in ("short", "canonical"):
        sub = tmp_path / variant
        sub.mkdir()
        root = _make_fake_lung_archive(sub, counts=counts, folder_variant=variant)
        samples = discover(root)
        assert len(samples) == len(LUNG_HISTOPATH_LABELS)
        labels_seen = {LUNG_HISTOPATH_LABELS[s.label] for s in samples}
        assert labels_seen == set(LUNG_HISTOPATH_LABELS)


def test_discover_ignores_colon_subdir(tmp_path: Path) -> None:
    """A LC25000 archive with both organs present yields only lung samples.

    This is the core invariant of the split-by-organ design: lung_histopath
    walks ``lung_image_sets/`` only — even when a co-extracted colon subdir
    sits next to it on disk.
    """
    counts = {label: 1 for label in LUNG_HISTOPATH_LABELS}
    root = _make_fake_lung_archive(tmp_path, counts=counts)
    # Co-extract the colon subdir with a colon class folder — must be
    # invisible to the lung discover.
    colon_dir = root / "colon_image_sets"
    colon_dir.mkdir()
    (colon_dir / "colon_aca").mkdir()
    (colon_dir / "colon_aca" / "img_000.jpeg").write_bytes(b"\xff\xd8")

    samples = discover(root)
    # Three labels x 1 image each = 3 samples; the colon image must not appear.
    assert len(samples) == len(LUNG_HISTOPATH_LABELS)
    # Each sample lives at .../lung_image_sets/<class>/<img>; the immediate
    # grand-parent must be the lung organ subdir, never colon's. Substring
    # checks against the full path are unsafe because the LC25000 top-level
    # dir itself ("lung_colon_image_set") contains both organ words.
    for sample in samples:
        assert sample.image_path.parent.parent.name == "lung_image_sets"


def test_discover_skips_unknown_folder_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An unrecognised class folder warns and skips — doesn't crash."""
    counts = {label: 1 for label in LUNG_HISTOPATH_LABELS}
    root = _make_fake_lung_archive(tmp_path, counts=counts)
    (root / "lung_image_sets" / "lung_rare_subtype").mkdir()
    (root / "lung_image_sets" / "lung_rare_subtype" / "img.jpeg").write_bytes(b"x")

    import logging

    with caplog.at_level(
        logging.WARNING, logger="claritymed.ingest.vision.lung_histopath.dataset"
    ):
        samples = discover(root)
    assert len(samples) == len(LUNG_HISTOPATH_LABELS)
    assert any("lung_rare_subtype" in rec.message for rec in caplog.records)


def test_discover_raises_when_organ_subdir_missing(tmp_path: Path) -> None:
    """Missing ``lung_image_sets`` fails loud — caught before training."""
    root = tmp_path / "lung_colon_image_set"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="lung_histopath organ subdir"):
        discover(root)


def test_discover_raises_when_no_images(tmp_path: Path) -> None:
    """An empty ``lung_image_sets/`` with no class folders → no samples → raise."""
    root = tmp_path / "lung_colon_image_set"
    (root / "lung_image_sets").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no lung_histopath samples"):
        discover(root)


# --- stratified split -----------------------------------------------------


def test_stratified_split_proportions_hold_per_class() -> None:
    """Each class gets the expected ~70/15/15 split."""
    samples = [
        LungHistopathSample(image_path=Path(f"aca_{i}.jpeg"), label=0)
        for i in range(20)
    ] + [
        LungHistopathSample(image_path=Path(f"norm_{i}.jpeg"), label=1)
        for i in range(20)
    ]
    splits = stratified_split(samples)
    assert len(splits["train"]) == 14 + 14
    assert len(splits["val"]) == 3 + 3
    assert len(splits["test"]) == 3 + 3


def test_stratified_split_is_deterministic() -> None:
    """Same seed → same split."""
    samples = [
        LungHistopathSample(image_path=Path(f"img_{i:03d}.jpeg"), label=i % 3)
        for i in range(30)
    ]
    a = stratified_split(samples, seed="deterministic-v1")
    b = stratified_split(samples, seed="deterministic-v1")
    assert [s.image_path for s in a["train"]] == [s.image_path for s in b["train"]]
    assert [s.image_path for s in a["val"]] == [s.image_path for s in b["val"]]
    assert [s.image_path for s in a["test"]] == [s.image_path for s in b["test"]]


def test_stratified_split_rejects_invalid_fractions() -> None:
    with pytest.raises(ValueError, match="invalid split fractions"):
        stratified_split([], train_frac=0.9, val_frac=0.2)
