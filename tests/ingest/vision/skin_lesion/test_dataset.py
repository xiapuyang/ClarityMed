"""Skin lesion (ISIC 9-class) dataset wrapper — discovery + split smoke tests.

Heavy ``torch`` import in :func:`build_dataset` is exercised only when
the extra is installed; the smoke tests here cover the pure-Python
``discover`` + ``stratified_split`` paths so CI gets coverage without
``vision-server`` extras.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from claritymed.ingest.vision.skin_lesion.dataset import (
    SKIN_LESION_LABELS,
    SkinLesionSample,
    _canonicalize_folder_name,
    discover,
    stratified_split,
)


# --- folder-name normalisation --------------------------------------------


@pytest.mark.parametrize(
    "folder,expected",
    [
        # Bare canonical labels.
        ("melanoma", "melanoma"),
        ("nevus", "nevus"),
        ("actinic_keratosis", "actinic_keratosis"),
        # Upstream uses display-cased folder names with spaces.
        ("basal cell carcinoma", "basal_cell_carcinoma"),
        ("pigmented benign keratosis", "pigmented_benign_keratosis"),
        ("Vascular Lesion", "vascular_lesion"),
        # Mixed dots/hyphens collapse to underscores.
        ("seborrheic-keratosis", "seborrheic_keratosis"),
        ("squamous.cell.carcinoma", "squamous_cell_carcinoma"),
        # Runs of separators squash to a single underscore (the
        # canonical labels themselves never carry a double underscore).
        ("vascular  lesion", "vascular_lesion"),
        ("basal..cell..carcinoma", "basal_cell_carcinoma"),
    ],
)
def test_canonicalize_folder_name(folder: str, expected: str | None) -> None:
    assert _canonicalize_folder_name(folder) == expected


def test_canonicalize_unknown_returns_none() -> None:
    """Folders we can't map to a canonical label return ``None``."""
    assert _canonicalize_folder_name("metastasis_unknown") is None
    assert _canonicalize_folder_name("") is None
    # Prefix-only — exact match required since ISIC display names
    # are unambiguous; a stray ``basal_cell`` shouldn't match
    # ``basal_cell_carcinoma`` to avoid false positives.
    assert _canonicalize_folder_name("basal_cell") is None


# --- discovery ------------------------------------------------------------


def _make_fake_skin_lesion(
    root: Path,
    *,
    counts: dict[str, int],
    folder_variant: str = "display",
) -> Path:
    """Mirror the upstream archive's on-disk layout.

    ``folder_variant``:

    * ``"display"`` — use the upstream's display-cased folder names
      with spaces (matches the Kaggle archive).
    * ``"canonical"`` — use the canonical underscore-separated labels.
    """
    base = root / "Skin cancer ISIC The International Skin Imaging Collaboration"
    base.mkdir()
    train_dir = base / "Train"
    train_dir.mkdir()
    display_map = {
        "actinic_keratosis": "actinic keratosis",
        "basal_cell_carcinoma": "basal cell carcinoma",
        "dermatofibroma": "dermatofibroma",
        "melanoma": "melanoma",
        "nevus": "nevus",
        "pigmented_benign_keratosis": "pigmented benign keratosis",
        "seborrheic_keratosis": "seborrheic keratosis",
        "squamous_cell_carcinoma": "squamous cell carcinoma",
        "vascular_lesion": "vascular lesion",
    }
    for canonical, n in counts.items():
        folder_name = (
            display_map[canonical] if folder_variant == "display" else canonical
        )
        class_dir = train_dir / folder_name
        class_dir.mkdir()
        for i in range(n):
            (class_dir / f"img_{i:03d}.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    return base


def test_discover_finds_all_samples(tmp_path: Path) -> None:
    counts = {
        "actinic_keratosis": 2,
        "basal_cell_carcinoma": 3,
        "dermatofibroma": 1,
        "melanoma": 4,
        "nevus": 5,
        "pigmented_benign_keratosis": 2,
        "seborrheic_keratosis": 2,
        "squamous_cell_carcinoma": 3,
        "vascular_lesion": 1,
    }
    root = _make_fake_skin_lesion(tmp_path, counts=counts)
    samples = discover(root)
    assert len(samples) == sum(counts.values())
    assert all(0 <= s.label < len(SKIN_LESION_LABELS) for s in samples)


def test_discover_normalises_folder_names(tmp_path: Path) -> None:
    """Both ``display`` (with spaces) and ``canonical`` folder names resolve."""
    counts = {label: 1 for label in SKIN_LESION_LABELS}
    for variant in ("display", "canonical"):
        sub = tmp_path / variant
        sub.mkdir()
        root = _make_fake_skin_lesion(sub, counts=counts, folder_variant=variant)
        samples = discover(root)
        assert len(samples) == len(SKIN_LESION_LABELS)
        labels_seen = {SKIN_LESION_LABELS[s.label] for s in samples}
        assert labels_seen == set(SKIN_LESION_LABELS)


def test_discover_skips_unknown_folder_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An unrecognised class folder warns and skips — doesn't crash."""
    counts = {label: 1 for label in SKIN_LESION_LABELS}
    root = _make_fake_skin_lesion(tmp_path, counts=counts)
    (root / "Train" / "rare_subtype").mkdir()
    (root / "Train" / "rare_subtype" / "img.png").write_bytes(b"x")

    import logging

    with caplog.at_level(
        logging.WARNING, logger="claritymed.ingest.vision.skin_lesion.dataset"
    ):
        samples = discover(root)
    assert len(samples) == len(SKIN_LESION_LABELS)
    assert any("rare_subtype" in rec.message for rec in caplog.records)


def test_discover_raises_when_no_upstream_subdirs(tmp_path: Path) -> None:
    """Missing ``Train`` / ``Test`` subdirs fails loud."""
    root = tmp_path / "Skin cancer ISIC The International Skin Imaging Collaboration"
    root.mkdir()
    with pytest.raises(FileNotFoundError, match="no upstream subdirs"):
        discover(root)


def test_discover_raises_when_no_images(tmp_path: Path) -> None:
    """An empty ``Train/`` with no class folders → no samples → raise."""
    root = tmp_path / "Skin cancer ISIC The International Skin Imaging Collaboration"
    (root / "Train").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="no skin lesion samples"):
        discover(root)


# --- stratified split -----------------------------------------------------


def test_stratified_split_proportions_hold_per_class() -> None:
    """Each class gets the expected ~70/15/15 split."""
    samples = [
        SkinLesionSample(image_path=Path(f"mel_{i}.png"), label=3) for i in range(20)
    ] + [SkinLesionSample(image_path=Path(f"nev_{i}.png"), label=4) for i in range(20)]
    splits = stratified_split(samples)
    assert len(splits["train"]) == 14 + 14
    assert len(splits["val"]) == 3 + 3
    assert len(splits["test"]) == 3 + 3


def test_stratified_split_is_deterministic() -> None:
    """Same seed → same split."""
    samples = [
        SkinLesionSample(image_path=Path(f"img_{i:03d}.png"), label=i % 9)
        for i in range(45)
    ]
    a = stratified_split(samples, seed="deterministic-v1")
    b = stratified_split(samples, seed="deterministic-v1")
    assert [s.image_path for s in a["train"]] == [s.image_path for s in b["train"]]
    assert [s.image_path for s in a["val"]] == [s.image_path for s in b["val"]]
    assert [s.image_path for s in a["test"]] == [s.image_path for s in b["test"]]


def test_stratified_split_rejects_invalid_fractions() -> None:
    with pytest.raises(ValueError, match="invalid split fractions"):
        stratified_split([], train_frac=0.9, val_frac=0.2)
