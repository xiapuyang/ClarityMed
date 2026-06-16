"""Chest CT dataset wrapper — discovery, label normalisation, stratified split.

Heavy ``torch`` import in :func:`build_dataset` is exercised only when
the extra is installed; the smoke tests here cover the pure-Python
``discover`` + ``stratified_split`` paths so CI gets coverage without
``vision-server`` extras.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from claritymed.ingest.vision.chest_ct.dataset import (
    CHEST_CT_LABELS,
    ChestCTSample,
    _canonicalize_folder_name,
    discover,
    stratified_split,
)


# --- folder-name normalisation --------------------------------------------


@pytest.mark.parametrize(
    "folder,expected",
    [
        # Bare class names (the upstream ``normal/`` folder is bare).
        ("normal", "normal"),
        ("adenocarcinoma", "adenocarcinoma"),
        ("large_cell_carcinoma", "large_cell_carcinoma"),
        ("squamous_cell_carcinoma", "squamous_cell_carcinoma"),
        # Upstream variants with staging metadata appended.
        ("adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib", "adenocarcinoma"),
        ("large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa", "large_cell_carcinoma"),
        ("squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa", "squamous_cell_carcinoma"),
        # Case-insensitive.
        ("Adenocarcinoma_Left", "adenocarcinoma"),
        # Mixed dots/underscores collapse.
        ("large..cell..carcinoma_x", "large_cell_carcinoma"),
        # Hyphens normalise to underscores — anything prefixed with a
        # canonical label token + separator still matches that label
        # (mirrors the staging-metadata path above).
        ("normal-subset_v2", "normal"),
    ],
)
def test_canonicalize_folder_name(folder: str, expected: str | None) -> None:
    assert _canonicalize_folder_name(folder) == expected


def test_canonicalize_unknown_returns_none() -> None:
    """Folders we can't map to a canonical label return ``None``.

    ``discover`` turns this into a logged warning + skip so a new
    upstream variant is visible without crashing existing pipelines.
    """
    assert _canonicalize_folder_name("metastasis_unknown") is None
    assert _canonicalize_folder_name("") is None


# --- discovery ------------------------------------------------------------


def _make_fake_chest_ct(
    root: Path,
    *,
    n_adeno: int,
    n_large: int,
    n_normal: int,
    n_squamous: int,
    folder_variant: str = "staged",
) -> Path:
    """Mirror the upstream archive's on-disk layout.

    ``folder_variant``:

    * ``"staged"`` — use the upstream's staging-info folder names.
    * ``"bare"`` — use the canonical bare label as the folder name
      (matches the ``normal/`` folder shape).
    """
    base = root / "Data"
    base.mkdir()

    if folder_variant == "staged":
        folders = {
            "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib": ("adenocarcinoma", n_adeno),
            "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa": (
                "large_cell_carcinoma",
                n_large,
            ),
            "normal": ("normal", n_normal),
            "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa": (
                "squamous_cell_carcinoma",
                n_squamous,
            ),
        }
    else:
        folders = {
            "adenocarcinoma": ("adenocarcinoma", n_adeno),
            "large_cell_carcinoma": ("large_cell_carcinoma", n_large),
            "normal": ("normal", n_normal),
            "squamous_cell_carcinoma": ("squamous_cell_carcinoma", n_squamous),
        }

    # Upstream ships train/test/valid; we put samples in train/ so a
    # smoke test of discover() picks them up the same way it would in
    # production.
    train_dir = base / "train"
    train_dir.mkdir()
    for folder_name, (_canonical, n) in folders.items():
        class_dir = train_dir / folder_name
        class_dir.mkdir()
        for i in range(n):
            (class_dir / f"img_{i:03d}.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    return base


def test_discover_finds_all_samples(tmp_path: Path) -> None:
    root = _make_fake_chest_ct(tmp_path, n_adeno=3, n_large=2, n_normal=4, n_squamous=5)
    samples = discover(root)
    assert len(samples) == 3 + 2 + 4 + 5
    # Every sample maps to a real label index.
    assert all(0 <= s.label < len(CHEST_CT_LABELS) for s in samples)


def test_discover_normalises_folder_names(tmp_path: Path) -> None:
    """Both ``staged`` (with metadata) and ``bare`` folder names resolve."""
    for variant in ("staged", "bare"):
        sub = tmp_path / variant
        sub.mkdir()
        root = _make_fake_chest_ct(
            sub, n_adeno=1, n_large=1, n_normal=1, n_squamous=1, folder_variant=variant
        )
        samples = discover(root)
        assert len(samples) == 4
        labels_seen = {CHEST_CT_LABELS[s.label] for s in samples}
        assert labels_seen == set(CHEST_CT_LABELS)


def test_discover_skips_unknown_folder_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An unrecognised class folder warns and skips — doesn't crash."""
    root = _make_fake_chest_ct(tmp_path, n_adeno=1, n_large=1, n_normal=1, n_squamous=1)
    # Inject a folder the canonicaliser can't map.
    (root / "train" / "metastasis_unknown_variant").mkdir()
    (root / "train" / "metastasis_unknown_variant" / "img.png").write_bytes(b"x")

    import logging

    with caplog.at_level(
        logging.WARNING, logger="claritymed.ingest.vision.chest_ct.dataset"
    ):
        samples = discover(root)
    # The four real classes survive; the unknown folder is dropped.
    assert len(samples) == 4
    assert any("metastasis_unknown_variant" in rec.message for rec in caplog.records)


def test_discover_raises_when_no_upstream_subdirs(tmp_path: Path) -> None:
    """Missing the ``train`` / ``test`` / ``valid`` subdirs fails loud."""
    root = tmp_path / "Data"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="no upstream subdirs"):
        discover(root)


def test_discover_raises_when_no_images(tmp_path: Path) -> None:
    """An empty ``train/`` with no class folders → no samples → raise."""
    root = tmp_path / "Data"
    (root / "train").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no chest CT samples"):
        discover(root)


# --- stratified split -----------------------------------------------------


def test_stratified_split_proportions_hold_per_class() -> None:
    """Each class gets the expected ~70/15/15 split."""
    samples = [
        ChestCTSample(image_path=Path(f"adeno_{i}.png"), label=0) for i in range(20)
    ] + [ChestCTSample(image_path=Path(f"normal_{i}.png"), label=2) for i in range(20)]
    splits = stratified_split(samples)
    assert len(splits["train"]) == 14 + 14  # 70% of each class
    assert len(splits["val"]) == 3 + 3  # 15% of each class
    assert len(splits["test"]) == 3 + 3  # remainder


def test_stratified_split_is_deterministic() -> None:
    """Same seed → same split."""
    samples = [
        ChestCTSample(image_path=Path(f"img_{i:03d}.png"), label=i % 4)
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
