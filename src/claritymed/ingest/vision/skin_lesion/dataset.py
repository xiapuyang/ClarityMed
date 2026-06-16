"""Skin lesion (ISIC 9-class) dataset wrapper for PyTorch training.

The upstream Kaggle archive (``nodoubttome/skin-cancer9-classesisic``)
ships images grouped by class under
``Skin cancer ISIC The International Skin Imaging Collaboration/{Train,Test}/<folder>/``
where ``<folder>`` is one of the nine ISIC labels in display-cased,
space-separated form, e.g.::

    Train/actinic keratosis/
    Train/basal cell carcinoma/
    Train/dermatofibroma/
    Train/melanoma/
    Train/nevus/
    Train/pigmented benign keratosis/
    Train/seborrheic keratosis/
    Train/squamous cell carcinoma/
    Train/vascular lesion/

We normalise every folder name (lowercase, collapse spaces / hyphens /
dots into underscores) to one of the nine labels in
:data:`SKIN_LESION_LABELS`, then pool images across the upstream
``Train/`` and ``Test/`` subdirs so our own deterministic, hash-stratified
split picks them. Same rationale as ``chest_ct``: the upstream split is
small and tilted, and a hash-stratified split keeps each class
proportionally represented in val and test.

Output shape per item: ``(image: Tensor[C=3, H, W], label: int)``. No
mask channel — ISIC ships no lesion masks. The ``label`` indexes into
:data:`SKIN_LESION_LABELS`.
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
# The upstream ISIC 9-class taxonomy splits "actinic keratosis" (premalignant),
# four explicitly malignant entries (basal_cell_carcinoma, melanoma,
# squamous_cell_carcinoma — large_cell_carcinoma is lung-only),
# and four benign entries.
SKIN_LESION_LABELS: tuple[str, ...] = (
    "actinic_keratosis",
    "basal_cell_carcinoma",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "pigmented_benign_keratosis",
    "seborrheic_keratosis",
    "squamous_cell_carcinoma",
    "vascular_lesion",
)
Split = Literal["train", "val", "test"]
DEFAULT_INPUT_SIZE = 256

# Upstream archive subdirs that hold the class folders. ``Train`` is
# the bulk; ``Test`` is much smaller — pool both before our own split.
_UPSTREAM_SUBSPLITS: tuple[str, ...] = ("Train", "Test")

# Image extensions the dataset ships. Listed lowercase; ``discover``
# matches case-insensitively.
_IMAGE_EXTS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})


@dataclass(frozen=True)
class SkinLesionSample:
    """One on-disk image with its class label."""

    image_path: Path
    label: int  # index into SKIN_LESION_LABELS


# --- label normalisation --------------------------------------------------


def _canonicalize_folder_name(name: str) -> str | None:
    """Map an upstream folder name to one of :data:`SKIN_LESION_LABELS`.

    Rules:

    * Lowercase, collapse spaces, hyphens and dots into underscores,
      squash runs of underscores.
    * Exact match against a canonical label returns that label.
    * Anything that doesn't match returns ``None`` — surfaces in
      :func:`discover` as a warning so a renamed upstream folder is
      caught loud.

    The upstream uses display-cased folder names with spaces (e.g.
    ``"basal cell carcinoma"``); a strict ``startswith`` test would
    over-match ``"basal_cell"`` and similar prefixes, so this
    implementation requires an exact normalised match.
    """
    normalised = name.lower()
    for ch in (" ", "-", "."):
        normalised = normalised.replace(ch, "_")
    while "__" in normalised:
        normalised = normalised.replace("__", "_")
    if normalised in SKIN_LESION_LABELS:
        return normalised
    return None


# --- discovery ------------------------------------------------------------


def discover(root: Path) -> list[SkinLesionSample]:
    """Walk the dataset root and return every classifiable image.

    Pools across ``Train/`` and ``Test/`` subdirs (see module docstring).
    Raises ``FileNotFoundError`` when none of the upstream subdirs
    exist — surfaces a clear error before training would silently
    produce a zero-sample epoch.

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

    samples: list[SkinLesionSample] = []
    skipped: dict[str, int] = {}
    label_to_idx = {label: idx for idx, label in enumerate(SKIN_LESION_LABELS)}

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
                samples.append(SkinLesionSample(image_path, label_idx))

    if skipped:
        for folder, count in sorted(skipped.items()):
            logger.warning(
                "skin_lesion: skipping unrecognised class folder %r (seen %d times)",
                folder,
                count,
            )

    if not samples:
        raise RuntimeError(f"no skin lesion samples found under {root}")
    return samples


def stratified_split(
    samples: list[SkinLesionSample],
    *,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: str = "skin-lesion-v1",
) -> dict[Split, list[SkinLesionSample]]:
    """Deterministic stratified split.

    Hash each ``image_path.stem`` with a salt; route into ``train`` /
    ``val`` / ``test`` per class so the proportions hold within every
    label. The salt makes the assignment reproducible across machines
    without depending on filesystem ordering. Same shape as BUSI and
    chest_ct so the eval scripts stay dataset-agnostic.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError(f"invalid split fractions: train={train_frac} val={val_frac}")
    by_label: dict[int, list[SkinLesionSample]] = {}
    for s in samples:
        by_label.setdefault(s.label, []).append(s)

    out: dict[Split, list[SkinLesionSample]] = {"train": [], "val": [], "test": []}
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


# --- torch dataset --------------------------------------------------------


try:
    import numpy as _np
    import torch as _torch
    from PIL import Image as _Image
    from torch.utils.data import Dataset as _DatasetBase

    class SkinLesionDataset(_DatasetBase):  # type: ignore[misc, valid-type]
        # Module-level so DataLoader workers using macOS `spawn` can pickle it.
        def __init__(self, items: list[SkinLesionSample], size: int) -> None:
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
    SkinLesionDataset = None  # type: ignore[assignment, misc]


def build_dataset(
    samples: list[SkinLesionSample],
    *,
    input_size: int = DEFAULT_INPUT_SIZE,
):
    """Construct a ``SkinLesionDataset`` for these samples.

    Tests that import the module without ``torch`` installed can still
    exercise :func:`discover` + :func:`stratified_split`; only this
    constructor requires the heavy deps.
    """
    if SkinLesionDataset is None:
        raise SystemExit("torch not installed — run `uv sync --extra vision-server`.")
    return SkinLesionDataset(samples, input_size)
