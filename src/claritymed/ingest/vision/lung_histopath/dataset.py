"""Lung histopathology (LC25000 lung subset) dataset wrapper for PyTorch.

The upstream Kaggle archive
(``andrewmvd/lung-and-colon-cancer-histopathological-images``) ships
lung-tissue images under
``lung_colon_image_set/lung_image_sets/<short_folder>/`` where
``<short_folder>`` is one of ``lung_aca`` / ``lung_n`` / ``lung_scc``.
We normalise to the canonical labels in :data:`LUNG_HISTOPATH_LABELS`
so manifests and i18n bundles drop the organ prefix — same naming
convention as the chest-CT lung-cancer disease's labels
(``adenocarcinoma``, ``squamous_cell_carcinoma``, ``normal``).

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
# Matches the chest_ct convention (bare disease labels, no organ
# prefix) — the organ is encoded in the disease_id, not the label.
LUNG_HISTOPATH_LABELS: tuple[str, ...] = (
    "adenocarcinoma",
    "normal",
    "squamous_cell_carcinoma",
)
Split = Literal["train", "val", "test"]
DEFAULT_INPUT_SIZE = 256

# The organ-level subdir under LC25000's top-level ``lung_colon_image_set``
# directory. Walked by :func:`discover`; the sibling
# ``colon_image_sets/`` is invisible to this module by design.
_ORGAN_SUBDIR = "lung_image_sets"

# Map from the upstream short folder names (``lung_aca`` etc.) to the
# canonical labels above. Accepts either short or canonical form so a
# re-uploaded archive with expanded names still resolves.
_UPSTREAM_FOLDER_ALIASES: dict[str, str] = {
    "lung_aca": "adenocarcinoma",
    "lung_n": "normal",
    "lung_scc": "squamous_cell_carcinoma",
}

# Image extensions the dataset ships (jpeg upstream; png allowed for
# operator-augmented variants).
_IMAGE_EXTS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg"})


@dataclass(frozen=True)
class LungHistopathSample:
    """One on-disk image with its class label."""

    image_path: Path
    label: int  # index into LUNG_HISTOPATH_LABELS


# --- label normalisation --------------------------------------------------


def _canonicalize_folder_name(name: str) -> str | None:
    """Map an upstream folder name to one of :data:`LUNG_HISTOPATH_LABELS`.

    Rules:

    * Lowercase, collapse spaces / hyphens / dots into underscores,
      squash runs of underscores.
    * If the normalised name is a known short alias (e.g. ``lung_aca``),
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
    if normalised in LUNG_HISTOPATH_LABELS:
        return normalised
    return None


# --- discovery ------------------------------------------------------------


def discover(root: Path) -> list[LungHistopathSample]:
    """Walk the LC25000 top-level dir and return every lung image.

    Walks ``root/lung_image_sets/<class>/``. ``root`` is the
    ``lung_colon_image_set/`` directory inside CLARITYMED_HOME.

    Raises ``FileNotFoundError`` when the organ subdir is missing
    (download not run, or upstream archive shape changed). Unrecognised
    class folders log a warning and are skipped — same conservative
    posture as the other vision modules.
    """
    organ_root = root / _ORGAN_SUBDIR
    if not organ_root.is_dir():
        raise FileNotFoundError(
            f"lung_histopath organ subdir not found at {organ_root}; "
            f"expected {_ORGAN_SUBDIR!r} under {root}. Did `download` complete?"
        )

    samples: list[LungHistopathSample] = []
    skipped: dict[str, int] = {}
    label_to_idx = {label: idx for idx, label in enumerate(LUNG_HISTOPATH_LABELS)}

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
            samples.append(LungHistopathSample(image_path, label_idx))

    if skipped:
        for folder, count in sorted(skipped.items()):
            logger.warning(
                "lung_histopath: skipping unrecognised class folder %r (seen %d times)",
                folder,
                count,
            )

    if not samples:
        raise RuntimeError(f"no lung_histopath samples found under {organ_root}")
    return samples


def stratified_split(
    samples: list[LungHistopathSample],
    *,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: str = "lung-histopath-v1",
) -> dict[Split, list[LungHistopathSample]]:
    """Deterministic stratified split.

    Hash each ``image_path.stem`` with a salt; route into ``train`` /
    ``val`` / ``test`` per class so the proportions hold within every
    label. Same shape as BUSI / chest_ct / skin_lesion so the eval
    scripts stay dataset-agnostic.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError(f"invalid split fractions: train={train_frac} val={val_frac}")
    by_label: dict[int, list[LungHistopathSample]] = {}
    for s in samples:
        by_label.setdefault(s.label, []).append(s)

    out: dict[Split, list[LungHistopathSample]] = {
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

    class LungHistopathDataset(_DatasetBase):  # type: ignore[misc, valid-type]
        # Module-level so DataLoader workers using macOS `spawn` can pickle it.
        def __init__(self, items: list[LungHistopathSample], size: int) -> None:
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
    LungHistopathDataset = None  # type: ignore[assignment, misc]


def build_dataset(
    samples: list[LungHistopathSample],
    *,
    input_size: int = DEFAULT_INPUT_SIZE,
):
    """Construct a ``LungHistopathDataset`` for these samples.

    Tests that import the module without ``torch`` installed can still
    exercise :func:`discover` + :func:`stratified_split`; only this
    constructor requires the heavy deps.
    """
    if LungHistopathDataset is None:
        raise SystemExit("torch not installed — run `uv sync --extra vision-server`.")
    return LungHistopathDataset(samples, input_size)
