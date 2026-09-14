"""Colon histopathology (LC25000 colon subset) dataset wrapper for PyTorch.

The upstream Kaggle archive
(``andrewmvd/lung-and-colon-cancer-histopathological-images``) ships
colon-tissue images under
``lung_colon_image_set/colon_image_sets/<short_folder>/`` where
``<short_folder>`` is one of ``colon_aca`` / ``colon_n``. We normalise
to the canonical labels in :data:`COLON_HISTOPATH_LABELS` so manifests
and i18n bundles drop the organ prefix — the organ is encoded in the
disease_id (``colon_cancer_histopathology``), not in the labels.

Output shape per item: ``(image: Tensor[C=3, H, W], label: int)``. No
mask channel — the upstream archive ships no segmentation masks.
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
COLON_HISTOPATH_LABELS: tuple[str, ...] = (
    "adenocarcinoma",
    "normal",
)
Split = Literal["train", "val", "test"]
DEFAULT_INPUT_SIZE = 256

# The organ-level subdir under LC25000's top-level ``lung_colon_image_set``
# directory. Walked by :func:`discover`; the sibling ``lung_image_sets/``
# is invisible to this module by design.
_ORGAN_SUBDIR = "colon_image_sets"

# Map from the upstream short folder names to the canonical labels.
# Accepts either short or canonical form so a re-uploaded archive with
# expanded names still resolves.
_UPSTREAM_FOLDER_ALIASES: dict[str, str] = {
    "colon_aca": "adenocarcinoma",
    "colon_n": "normal",
}

# Image extensions the dataset ships (jpeg upstream; png allowed for
# operator-augmented variants).
_IMAGE_EXTS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})


@dataclass(frozen=True)
class ColonHistopathSample:
    """One on-disk image with its class label."""

    image_path: Path
    label: int  # index into COLON_HISTOPATH_LABELS


# --- label normalisation --------------------------------------------------


def _canonicalize_folder_name(name: str) -> str | None:
    """Map an upstream folder name to one of :data:`COLON_HISTOPATH_LABELS`.

    Rules:

    * Lowercase, collapse spaces / hyphens / dots into underscores,
      squash runs of underscores.
    * If the normalised name is a known short alias (e.g. ``colon_aca``),
      expand it to the canonical label.
    * Exact match against a canonical label returns that label.
    * Anything that doesn't match returns ``None`` — surfaces in
      :func:`discover` as a warning so a renamed upstream folder is
      caught loud.
    """
    normalised = name.lower()
    for ch in (" ", "-", "."):
        normalised = normalised.replace(ch, "_")
    while "__" in normalised:
        normalised = normalised.replace("__", "_")
    if normalised in _UPSTREAM_FOLDER_ALIASES:
        return _UPSTREAM_FOLDER_ALIASES[normalised]
    if normalised in COLON_HISTOPATH_LABELS:
        return normalised
    return None


# --- discovery ------------------------------------------------------------


def discover(root: Path) -> list[ColonHistopathSample]:
    """Walk the LC25000 top-level dir and return every colon image.

    Walks ``root/colon_image_sets/<class>/``. ``root`` is the
    ``lung_colon_image_set/`` directory inside CLARITYMED_HOME.

    Raises ``FileNotFoundError`` when the organ subdir is missing
    (download not run, or upstream archive shape changed). Unrecognised
    class folders log a warning and are skipped — same conservative
    posture as the other vision modules.
    """
    organ_root = root / _ORGAN_SUBDIR
    if not organ_root.is_dir():
        raise FileNotFoundError(
            f"colon_histopath organ subdir not found at {organ_root}; "
            f"expected {_ORGAN_SUBDIR!r} under {root}. Did `download` complete?"
        )

    samples: list[ColonHistopathSample] = []
    skipped: dict[str, int] = {}
    label_to_idx = {label: idx for idx, label in enumerate(COLON_HISTOPATH_LABELS)}

    for class_dir in sorted(organ_root.iterdir()):
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
            samples.append(ColonHistopathSample(image_path, label_idx))

    if skipped:
        for folder, count in sorted(skipped.items()):
            logger.warning(
                "colon_histopath: skipping unrecognised class folder %r "
                "(seen %d times)",
                folder,
                count,
            )

    if not samples:
        raise RuntimeError(f"no colon_histopath samples found under {organ_root}")
    return samples


def stratified_split(
    samples: list[ColonHistopathSample],
    *,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: str = "colon-histopath-v1",
) -> dict[Split, list[ColonHistopathSample]]:
    """Deterministic stratified split.

    Hash each ``image_path.stem`` with a salt; route into ``train`` /
    ``val`` / ``test`` per class so the proportions hold within every
    label. Same shape as the other vision modules so eval scripts stay
    dataset-agnostic.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError(f"invalid split fractions: train={train_frac} val={val_frac}")
    by_label: dict[int, list[ColonHistopathSample]] = {}
    for s in samples:
        by_label.setdefault(s.label, []).append(s)

    out: dict[Split, list[ColonHistopathSample]] = {
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


# --- torch dataset --------------------------------------------------------


try:
    import numpy as _np
    import torch as _torch
    from PIL import Image as _Image
    from torch.utils.data import Dataset as _DatasetBase

    class ColonHistopathDataset(_DatasetBase):  # type: ignore[misc, valid-type]
        # Module-level so DataLoader workers using macOS `spawn` can pickle it.
        def __init__(self, items: list[ColonHistopathSample], size: int) -> None:
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
    ColonHistopathDataset = None  # type: ignore[assignment, misc]


def build_dataset(
    samples: list[ColonHistopathSample],
    *,
    input_size: int = DEFAULT_INPUT_SIZE,
):
    """Construct a ``ColonHistopathDataset`` for these samples.

    Tests that import the module without ``torch`` installed can still
    exercise :func:`discover` + :func:`stratified_split`; only this
    constructor requires the heavy deps.
    """
    if ColonHistopathDataset is None:
        raise SystemExit("torch not installed — run `uv sync --extra vision-server`.")
    return ColonHistopathDataset(samples, input_size)
