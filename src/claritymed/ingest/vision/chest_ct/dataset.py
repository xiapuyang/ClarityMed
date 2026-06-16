"""Chest CT dataset wrapper for PyTorch training.

The upstream Kaggle archive (``mohamedhanyyy/chest-ctscan-images``)
ships images grouped by class under ``Data/{train,test,valid}/<folder>/``
where ``<folder>`` is one of four cancer classes — but with extra
staging info embedded in the folder name, e.g.::

    adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib/
    large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa/
    normal/
    squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa/

We normalise every folder's name (lowercase, dots and underscores
collapsed) to one of the four labels in :data:`CHEST_CT_LABELS`, then
pool images across the upstream train/test/valid split so our own
deterministic, hash-stratified split picks them. Why ignore the
upstream split: it's small and tilted (the ``test/`` folder is shaped
for demo, not statistically representative), and a hash-stratified
split keeps each class proportionally represented in val and test.

Output shape per item: ``(image: Tensor[C=3, H, W], label: int)``.
No mask channel — chest CT classification is a 4-class problem only.
The ``label`` indexes into :data:`CHEST_CT_LABELS`.
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
CHEST_CT_LABELS: tuple[str, ...] = (
    "adenocarcinoma",
    "large_cell_carcinoma",
    "normal",
    "squamous_cell_carcinoma",
)
Split = Literal["train", "val", "test"]
DEFAULT_INPUT_SIZE = 256

# Upstream archive subdirs that hold the class folders. We pool images
# across all three before doing our own split (see module docstring).
_UPSTREAM_SUBSPLITS: tuple[str, ...] = ("train", "test", "valid")

# Image extensions the dataset ships. Listed lowercase; ``discover``
# matches case-insensitively.
_IMAGE_EXTS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})


@dataclass(frozen=True)
class ChestCTSample:
    """One on-disk image with its class label."""

    image_path: Path
    label: int  # index into CHEST_CT_LABELS


# --- label normalisation --------------------------------------------------


def _canonicalize_folder_name(name: str) -> str | None:
    """Map an upstream folder name to one of :data:`CHEST_CT_LABELS`.

    Rules:

    * Lowercase, collapse dots and hyphens into underscores.
    * If the result *starts with* a canonical label token (with the
      upstream's ``.`` separators normalised), return that label.
      Matches both ``normal`` (bare) and
      ``adenocarcinoma_left_lower_lobe_t2_n0_m0_ib``.
    * Returns ``None`` for anything that doesn't match — surfaces in
      :func:`discover` as a warning so a renamed upstream folder is
      caught loud.
    """
    normalised = name.lower().replace(".", "_").replace("-", "_")
    while "__" in normalised:
        normalised = normalised.replace("__", "_")
    for label in CHEST_CT_LABELS:
        # Compare token-by-token so ``adenocarcinoma_xxx`` matches but a
        # stray ``adenocarcinoma_variant_we_dont_track`` doesn't match
        # ``normal`` by accident.
        if normalised == label or normalised.startswith(label + "_"):
            return label
    return None


# --- discovery ------------------------------------------------------------


def discover(root: Path) -> list[ChestCTSample]:
    """Walk ``Data/`` and return every classifiable image.

    Pools across the upstream ``train/`` ``test/`` ``valid/`` subdirs
    (see module docstring). Raises ``FileNotFoundError`` when none of
    the upstream subdirs exist — surfaces a clear error before training
    would silently produce a zero-sample epoch.

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

    samples: list[ChestCTSample] = []
    skipped: dict[str, int] = {}
    label_to_idx = {label: idx for idx, label in enumerate(CHEST_CT_LABELS)}

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
                samples.append(ChestCTSample(image_path, label_idx))

    if skipped:
        # One log line per unrecognised folder so the operator can tell
        # whether a rename happened upstream or a class was added.
        for folder, count in sorted(skipped.items()):
            logger.warning(
                "chest_ct: skipping unrecognised class folder %r (seen %d times)",
                folder,
                count,
            )

    if not samples:
        raise RuntimeError(f"no chest CT samples found under {root}")
    return samples


def stratified_split(
    samples: list[ChestCTSample],
    *,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: str = "chest-ct-v1",
) -> dict[Split, list[ChestCTSample]]:
    """Deterministic stratified split.

    Hash each ``image_path.stem`` with a salt; route into ``train`` /
    ``val`` / ``test`` per class so the proportions hold within every
    label. The salt makes the assignment reproducible across machines
    without depending on filesystem ordering. Same shape as BUSI's
    splitter — keeping the contract identical lets the eval scripts be
    dataset-agnostic.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError(f"invalid split fractions: train={train_frac} val={val_frac}")
    by_label: dict[int, list[ChestCTSample]] = {}
    for s in samples:
        by_label.setdefault(s.label, []).append(s)

    out: dict[Split, list[ChestCTSample]] = {"train": [], "val": [], "test": []}
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

    class ChestCTDataset(_DatasetBase):  # type: ignore[misc, valid-type]
        # Module-level so DataLoader workers using macOS `spawn` can pickle it.
        def __init__(self, items: list[ChestCTSample], size: int) -> None:
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
    ChestCTDataset = None  # type: ignore[assignment, misc]


def build_dataset(
    samples: list[ChestCTSample],
    *,
    input_size: int = DEFAULT_INPUT_SIZE,
):
    """Construct a ``ChestCTDataset`` for these samples.

    Tests that import the module without ``torch`` installed can still
    exercise :func:`discover` + :func:`stratified_split`; only this
    constructor requires the heavy deps.
    """
    if ChestCTDataset is None:
        raise SystemExit("torch not installed — run `uv sync --extra vision-server`.")
    return ChestCTDataset(samples, input_size)
