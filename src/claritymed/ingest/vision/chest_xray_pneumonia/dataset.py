"""Chest X-Ray Pneumonia (Kermany et al. 2018) dataset wrapper.

The upstream Kaggle archive
(``paultimothymooney/chest-xray-pneumonia``) ships pediatric chest
radiographs from the Guangzhou Women and Children's Medical Center,
grouped by class under
``chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/`` (see
``download.py``'s docstring for the on-disk layout).

We pool images across the upstream ``train`` / ``val`` / ``test``
subdirs and apply our own deterministic hash-stratified split for two
reasons:

1. Kermany's published ``val`` split is famously tiny (16 images, 8 per
   class) — too small to be a usable validation set.
2. The bench / cross-eval want representative held-out chunks per
   class, which a hash-stratified split delivers identically to BUSI
   and breast_us_kaggle.

Output shape per item: ``(image: Tensor[C=3, H, W], label: int)``. No
mask channel — the upstream archive ships no segmentations. The
``label`` indexes into :data:`CHEST_XRAY_PNEUMONIA_LABELS`.
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
# ``normal`` first follows the convention BUSI established (negative class
# at index 0) — matters when the bench remaps gt indices by name.
CHEST_XRAY_PNEUMONIA_LABELS: tuple[str, ...] = ("normal", "pneumonia")
Split = Literal["train", "val", "test"]
DEFAULT_INPUT_SIZE = 256

# Upstream archive subdirs that hold the class folders.
_UPSTREAM_SUBSPLITS: tuple[str, ...] = ("train", "val", "test")

# Image extensions the dataset ships. Kermany is ``.jpeg`` throughout
# upstream, but ``.png`` / ``.jpg`` are accepted for future ingest
# variants that re-encode.
_IMAGE_EXTS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})

# Upstream class folder names → canonical label. Upstream uses
# ``NORMAL`` / ``PNEUMONIA`` in uppercase; we lowercase for the public
# label tuple.
_FOLDER_TO_LABEL: dict[str, str] = {
    "normal": "normal",
    "pneumonia": "pneumonia",
}


@dataclass(frozen=True)
class ChestXrayPneumoniaSample:
    """One on-disk image with its class label."""

    image_path: Path
    label: int  # index into CHEST_XRAY_PNEUMONIA_LABELS


def _canonicalize_folder_name(name: str) -> str | None:
    """Map an upstream folder name to one of :data:`CHEST_XRAY_PNEUMONIA_LABELS`."""
    return _FOLDER_TO_LABEL.get(name.lower().strip())


def discover(root: Path) -> list[ChestXrayPneumoniaSample]:
    """Walk the dataset root and return every classifiable image.

    Pools across ``train`` / ``val`` / ``test`` subdirs. Raises
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

    samples: list[ChestXrayPneumoniaSample] = []
    skipped: dict[str, int] = {}
    label_to_idx = {label: idx for idx, label in enumerate(CHEST_XRAY_PNEUMONIA_LABELS)}

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
                samples.append(ChestXrayPneumoniaSample(image_path, label_idx))

    if skipped:
        for folder, count in sorted(skipped.items()):
            logger.warning(
                "chest_xray_pneumonia: skipping unrecognised class folder %r "
                "(seen %d times)",
                folder,
                count,
            )

    if not samples:
        raise RuntimeError(f"no chest_xray_pneumonia samples found under {root}")
    return samples


def stratified_split(
    samples: list[ChestXrayPneumoniaSample],
    *,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: str = "chest-xray-pneumonia-v1",
) -> dict[Split, list[ChestXrayPneumoniaSample]]:
    """Deterministic stratified split.

    Hash each ``image_path.stem`` with a salt; route into ``train`` /
    ``val`` / ``test`` per class so the proportions hold within every
    label. Same shape as BUSI / breast_us_kaggle so eval scripts stay
    dataset-agnostic.

    A single ``normal`` sample being moved into a different bucket by a
    seed change should not be a silent variance source: the seed is
    pinned and any change must bump the dataset version.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError(f"invalid split fractions: train={train_frac} val={val_frac}")
    by_label: dict[int, list[ChestXrayPneumoniaSample]] = {}
    for s in samples:
        by_label.setdefault(s.label, []).append(s)

    out: dict[Split, list[ChestXrayPneumoniaSample]] = {
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

    class ChestXrayPneumoniaDataset(_DatasetBase):  # type: ignore[misc, valid-type]
        # Module-level so DataLoader workers using macOS `spawn` can pickle it.
        def __init__(self, items: list[ChestXrayPneumoniaSample], size: int) -> None:
            self._items = items
            self._size = size

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, idx: int):
            sample = self._items[idx]
            # Kermany ships single-channel grayscale JPEGs; convert to
            # RGB so ImageNet-pretrained backbones see the 3-channel
            # input they expect. The same channel-replication happens in
            # the runtime adapter's preprocess().
            image = (
                _Image.open(sample.image_path)
                .convert("RGB")
                .resize((self._size, self._size), _Image.BILINEAR)
            )
            img_arr = _np.asarray(image, dtype=_np.float32) / 255.0  # HWC
            img_t = _torch.from_numpy(img_arr).permute(2, 0, 1)  # C H W
            return img_t, sample.label

except ImportError:  # pragma: no cover — torch-less envs hit discover/split only
    ChestXrayPneumoniaDataset = None  # type: ignore[assignment, misc]


def build_dataset(
    samples: list[ChestXrayPneumoniaSample],
    *,
    input_size: int = DEFAULT_INPUT_SIZE,
):
    """Construct a ``ChestXrayPneumoniaDataset`` for these samples.

    Tests that import the module without ``torch`` installed can still
    exercise :func:`discover` + :func:`stratified_split`; only this
    constructor requires the heavy deps.
    """
    if ChestXrayPneumoniaDataset is None:
        raise SystemExit("torch not installed — run `uv sync --extra vision-server`.")
    return ChestXrayPneumoniaDataset(samples, input_size)
