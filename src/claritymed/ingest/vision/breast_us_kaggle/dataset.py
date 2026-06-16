"""Breast-US Kaggle dataset wrapper for PyTorch training.

The upstream Kaggle archive
(``vuppalaadithyasairam/ultrasound-breast-images-for-breast-cancer``)
ships images grouped by class under
``ultrasound breast classification/{train,val}/<label>/`` where ``<label>``
is one of two classes (no ``normal`` class):

* ``benign``
* ``malignant``

The dataset has been augmented (rotation + sharpening) upstream — total
~9000 images, much larger than BUSI's ~780 — but the augmentation is
already baked into the file set, so we treat each image as an
independent sample.

We pool images across the upstream ``train/`` and ``val/`` subdirs so
our own deterministic, hash-stratified split picks them. Same rationale
as BUSI / chest_ct: the upstream split is small and tilted, and a
hash-stratified split keeps each class proportionally represented in
val + test.

Output shape per item: ``(image: Tensor[C=3, H, W], label: int)``. No
mask channel — the Kaggle archive ships no lesion masks. The ``label``
indexes into :data:`BREAST_US_KAGGLE_LABELS`.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from torch import Tensor
    from torch.utils.data import Dataset
else:  # pragma: no cover — import-time stubs
    Tensor = "Tensor"  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]

# Canonical labels. Order is the model's class-index order; do NOT
# reorder without rewriting every persisted manifest's ``labels`` array.
# The 2-class subset of BUSI's labels — the upstream archive doesn't
# carry a ``normal`` class.
BREAST_US_KAGGLE_LABELS: tuple[str, ...] = ("benign", "malignant")
Split = Literal["train", "val", "test"]
DEFAULT_INPUT_SIZE = 256

# Upstream archive subdirs that hold the class folders.
_UPSTREAM_SUBSPLITS: tuple[str, ...] = ("train", "val")

# Image extensions the dataset ships.
_IMAGE_EXTS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})


@dataclass(frozen=True)
class BreastUsKaggleSample:
    """One on-disk image with its class label."""

    image_path: Path
    label: int  # index into BREAST_US_KAGGLE_LABELS


def _canonicalize_folder_name(name: str) -> str | None:
    """Map an upstream folder name to one of :data:`BREAST_US_KAGGLE_LABELS`."""
    normalised = name.lower().strip()
    if normalised in BREAST_US_KAGGLE_LABELS:
        return normalised
    return None


def discover(root: Path) -> list[BreastUsKaggleSample]:
    """Walk the dataset root and return every classifiable image.

    Pools across ``train/`` and ``val/`` subdirs. Raises
    ``FileNotFoundError`` when none of the upstream subdirs exist —
    surfaces a clear error before training would silently produce a
    zero-sample epoch.

    Unrecognised class folders log a warning and are skipped; this lets
    a new upstream variant ship without crashing existing pipelines,
    while still being visible in logs.
    """
    candidate_subsplits = [d for s in _UPSTREAM_SUBSPLITS if (d := root / s).is_dir()]
    if not candidate_subsplits:
        raise FileNotFoundError(
            f"no upstream subdirs found under {root}; expected at least one of "
            f"{list(_UPSTREAM_SUBSPLITS)!r}. Did `download` complete?"
        )

    samples: list[BreastUsKaggleSample] = []
    skipped: dict[str, int] = {}
    label_to_idx = {label: idx for idx, label in enumerate(BREAST_US_KAGGLE_LABELS)}

    for sub_root in candidate_subsplits:
        for class_dir in sorted(sub_root.iterdir()):
            if not class_dir.is_dir():
                continue
            canonical = _canonicalize_folder_name(class_dir.name)
            if canonical is None:
                skipped[class_dir.name] = skipped.get(class_dir.name, 0) + 1
                continue
            label_idx = label_to_idx[canonical]
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() not in _IMAGE_EXTS:
                    continue
                samples.append(BreastUsKaggleSample(image_path, label_idx))

    if skipped:
        for folder, count in sorted(skipped.items()):
            logger.warning(
                "breast_us_kaggle: skipping unrecognised class folder %r "
                "(seen %d times)",
                folder,
                count,
            )

    if not samples:
        raise RuntimeError(f"no breast_us_kaggle samples found under {root}")
    return samples


def stratified_split(
    samples: list[BreastUsKaggleSample],
    *,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: str = "breast-us-kaggle-v1",
) -> dict[Split, list[BreastUsKaggleSample]]:
    """Deterministic stratified split.

    Hash each ``image_path.stem`` with a salt; route into ``train`` /
    ``val`` / ``test`` per class so the proportions hold within every
    label. Same shape as BUSI / chest_ct so the eval scripts stay
    dataset-agnostic.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError(f"invalid split fractions: train={train_frac} val={val_frac}")
    by_label: dict[int, list[BreastUsKaggleSample]] = {}
    for s in samples:
        by_label.setdefault(s.label, []).append(s)

    out: dict[Split, list[BreastUsKaggleSample]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for label_samples in by_label.values():
        ranked = sorted(
            label_samples,
            key=lambda s: hashlib.sha256(
                f"{seed}|{s.image_path.stem}".encode()
            ).hexdigest(),
        )
        n = len(ranked)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        out["train"].extend(ranked[:n_train])
        out["val"].extend(ranked[n_train : n_train + n_val])
        out["test"].extend(ranked[n_train + n_val :])
    return out


try:
    import numpy as _np
    import torch as _torch
    from PIL import Image as _Image
    from torch.utils.data import Dataset as _DatasetBase

    class BreastUsKaggleDataset(_DatasetBase):  # type: ignore[misc, valid-type]
        # Module-level so DataLoader workers using macOS `spawn` can pickle it.
        def __init__(self, items: list[BreastUsKaggleSample], size: int) -> None:
            self._items = items
            self._size = size

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, idx: int):
            sample = self._items[idx]
            image = (
                _Image.open(sample.image_path)
                .convert("RGB")
                .resize((self._size, self._size), _Image.BILINEAR)
            )
            img_arr = _np.asarray(image, dtype=_np.float32) / 255.0  # HWC
            img_t = _torch.from_numpy(img_arr).permute(2, 0, 1)  # C H W
            return img_t, sample.label

except ImportError:  # pragma: no cover — torch-less envs hit discover/split only
    BreastUsKaggleDataset = None  # type: ignore[assignment, misc]


def build_dataset(
    samples: list[BreastUsKaggleSample],
    *,
    input_size: int = DEFAULT_INPUT_SIZE,
):
    """Construct a ``BreastUsKaggleDataset`` for these samples.

    Tests that import the module without ``torch`` installed can still
    exercise :func:`discover` + :func:`stratified_split`; only this
    constructor requires the heavy deps.
    """
    if BreastUsKaggleDataset is None:
        raise SystemExit("torch not installed — run `uv sync --extra vision-server`.")
    return BreastUsKaggleDataset(samples, input_size)
