"""BUSI dataset wrapper for PyTorch training.

BUSI (Dataset_BUSI_with_GT) ships ~780 ultrasound images in three
class subdirectories: ``benign/``, ``malignant/``, ``normal/``. Each
class folder pairs ``<name>.png`` with ``<name>_mask.png`` (binary
lesion mask; ``normal`` images carry an all-zero mask). A handful of
multi-lesion samples carry additional ``<name>_mask_<n>.png`` files —
those get OR-fused into the GT mask at load time so the second/third
lesion still contributes to dice supervision and eval.

Splitting is deterministic and stratified by class so the rare
``normal`` class (~10% of samples) shows up in every split — a naive
filename-hash split could otherwise leave a split with zero normals
and bias eval.

Output shape per item: ``(image: Tensor[C=3, H, W], mask: Tensor[1, H, W], label: int)``.
The ``label`` indexes into :data:`BUSI_LABELS`.
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
else:  # pragma: no cover - import-time stubs
    Tensor = "Tensor"  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]

BUSI_LABELS: tuple[str, ...] = ("benign", "malignant", "normal")
Split = Literal["train", "val", "test"]
DEFAULT_INPUT_SIZE = 256


@dataclass(frozen=True)
class BUSISample:
    """One on-disk image with its GT mask(s) and class label.

    ``mask_paths`` holds the primary ``<name>_mask.png`` at index 0; any
    additional ``<name>_mask_<n>.png`` files (multi-lesion samples)
    follow in glob order. The loader OR-fuses them into a single binary
    mask at ``__getitem__`` time.
    """

    image_path: Path
    mask_paths: tuple[Path, ...]
    label: int  # index into BUSI_LABELS


def discover(root: Path) -> list[BUSISample]:
    """Walk ``Dataset_BUSI_with_GT/`` and return every image with its mask(s).

    Raises FileNotFoundError when the root doesn't contain the three
    expected class subdirectories — surfaces a clear error before
    training would silently produce a zero-sample epoch.
    """
    samples: list[BUSISample] = []
    for class_idx, label in enumerate(BUSI_LABELS):
        class_dir = root / label
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"missing BUSI class directory: {class_dir}. Did `download` complete?"
            )
        for image_path in sorted(class_dir.glob("*.png")):
            stem = image_path.stem
            # Skip the primary mask ("<stem>_mask.png") and auxiliary masks
            # for multi-lesion samples ("<stem>_mask_1.png", etc). Without
            # the "_mask_" clause aux masks slipped through as fake images
            # and got dropped with a spurious "mask missing" warning.
            if stem.endswith("_mask") or "_mask_" in stem:
                continue
            primary_mask = class_dir / f"{stem}_mask.png"
            if not primary_mask.exists():
                # BUSI's normal class still ships an (all-zero) mask. Drop
                # the sample on miss so a corrupt download doesn't poison
                # training.
                logger.warning("dropping %s — mask missing", image_path)
                continue
            # Glob pattern "_mask_*.png" requires an underscore after "mask",
            # so the primary "_mask.png" is not re-matched here.
            aux_masks = sorted(class_dir.glob(f"{stem}_mask_*.png"))
            samples.append(
                BUSISample(image_path, (primary_mask, *aux_masks), class_idx)
            )
    if not samples:
        raise RuntimeError(f"no BUSI samples found under {root}")
    return samples


def stratified_split(
    samples: list[BUSISample],
    *,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: str = "busi-v1",
) -> dict[Split, list[BUSISample]]:
    """Deterministic stratified split.

    Hash each ``image_path.stem`` with a salt; route into ``train`` /
    ``val`` / ``test`` per class so the proportions hold within every
    label. The salt makes the assignment reproducible across machines
    without depending on filesystem ordering.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError(f"invalid split fractions: train={train_frac} val={val_frac}")
    by_label: dict[int, list[BUSISample]] = {}
    for s in samples:
        by_label.setdefault(s.label, []).append(s)

    out: dict[Split, list[BUSISample]] = {"train": [], "val": [], "test": []}
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

    class BUSIDataset(_DatasetBase):  # type: ignore[misc, valid-type]
        # Module-level so DataLoader workers using macOS `spawn` can pickle it.
        def __init__(self, items: list[BUSISample], size: int) -> None:
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
            # OR-fuse every mask file. Multi-lesion samples carry one
            # mask per lesion; the model only needs a single "lesion vs
            # background" target, so union them at load time.
            mask_arr: _np.ndarray | None = None
            for mp in sample.mask_paths:
                m = (
                    _Image.open(mp)
                    .convert("L")
                    .resize((self._size, self._size), _Image.NEAREST)
                )
                m_arr = (_np.asarray(m, dtype=_np.float32) > 127).astype(_np.float32)
                mask_arr = m_arr if mask_arr is None else _np.maximum(mask_arr, m_arr)
            assert mask_arr is not None  # discover() guarantees ≥1 mask
            img_t = _torch.from_numpy(img_arr).permute(2, 0, 1)  # C H W
            mask_t = _torch.from_numpy(mask_arr).unsqueeze(0)  # 1 H W
            return img_t, mask_t, sample.label

except ImportError:  # pragma: no cover — torch-less envs hit discover/split only
    BUSIDataset = None  # type: ignore[assignment, misc]


def build_dataset(
    samples: list[BUSISample],
    *,
    input_size: int = DEFAULT_INPUT_SIZE,
):
    """Construct a ``BUSIDataset`` for these samples.

    Tests that import the module without ``torch`` installed can still
    exercise :func:`discover` + :func:`stratified_split`; only this
    constructor requires the heavy deps.
    """
    if BUSIDataset is None:
        raise SystemExit("torch not installed — run `uv sync --extra vision-server`.")
    return BUSIDataset(samples, input_size)
