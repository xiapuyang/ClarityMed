"""BUSI dataset wrapper for PyTorch training.

BUSI (Dataset_BUSI_with_GT) ships ~780 ultrasound images in three
class subdirectories: ``benign/``, ``malignant/``, ``normal/``. Each
class folder pairs ``<name>.png`` with ``<name>_mask.png`` (binary
lesion mask; ``normal`` images carry an all-zero mask).

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
    """One on-disk pair (image + mask) with its class label."""

    image_path: Path
    mask_path: Path
    label: int  # index into BUSI_LABELS


def discover(root: Path) -> list[BUSISample]:
    """Walk ``Dataset_BUSI_with_GT/`` and return every (image, mask) pair.

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
            if image_path.name.endswith("_mask.png"):
                continue
            mask_path = class_dir / f"{image_path.stem}_mask.png"
            if not mask_path.exists():
                # BUSI's normal class still ships a mask (all-zero). Drop the
                # sample on miss so a corrupt download doesn't poison training.
                logger.warning("dropping %s — mask missing", image_path)
                continue
            samples.append(BUSISample(image_path, mask_path, class_idx))
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


def _torch_import():
    """Lazy import keeps the module importable without the heavy extra."""
    try:
        import torch  # noqa: F401
        from torch.utils.data import Dataset as _Dataset

        return _Dataset
    except ImportError as exc:
        raise SystemExit(
            "torch not installed — run `uv sync --extra vision-server`."
        ) from exc


def build_dataset(
    samples: list[BUSISample],
    *,
    input_size: int = DEFAULT_INPUT_SIZE,
):
    """Construct a ``BUSIDataset`` for these samples.

    The class itself is defined inside the closure so the ``Dataset``
    base class is only imported when this constructor is called. Tests
    that import the module without ``torch`` installed can still
    exercise :func:`discover` + :func:`stratified_split`.
    """
    _DatasetBase = _torch_import()
    from PIL import Image
    import numpy as np
    import torch

    class BUSIDataset(_DatasetBase):
        def __init__(self, items: list[BUSISample], size: int) -> None:
            self._items = items
            self._size = size

        def __len__(self) -> int:
            return len(self._items)

        def __getitem__(self, idx: int):
            sample = self._items[idx]
            image = (
                Image.open(sample.image_path)
                .convert("RGB")
                .resize((self._size, self._size), Image.BILINEAR)
            )
            mask = (
                Image.open(sample.mask_path)
                .convert("L")
                .resize((self._size, self._size), Image.NEAREST)
            )
            img_arr = np.asarray(image, dtype=np.float32) / 255.0  # HWC
            mask_arr = (np.asarray(mask, dtype=np.float32) > 127).astype(
                np.float32
            )  # HW
            img_t = torch.from_numpy(img_arr).permute(2, 0, 1)  # C H W
            mask_t = torch.from_numpy(mask_arr).unsqueeze(0)  # 1 H W
            return img_t, mask_t, sample.label

    return BUSIDataset(samples, input_size)
