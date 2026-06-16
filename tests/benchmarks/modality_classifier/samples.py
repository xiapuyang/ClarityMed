"""Sample images from the downloaded vision datasets with modality labels.

Each ``DatasetSource`` declares a top-level dir under
``~/.claritymed/data/vision/<bucket>`` and the canonical
:class:`~claritymed.core.medical_clip.schemas.Modality` it represents.
:func:`build_sample_pool` walks the source, filters to recognized image
extensions, drops obvious non-data files (segmentation masks, README
artifacts), and returns at most ``per_dataset`` paths per dataset.

Sampling is seeded so a re-run on the same data emits the same image
set — the per-modality recall numbers stay comparable across runs.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path

from claritymed.config import DATA_DIR

logger = logging.getLogger(__name__)

# Recognized image extensions. JPEG / JPG / PNG cover the four shipped
# training datasets; TIFF is included because the histopath ZIP (once
# extracted) sometimes uses .tiff for slide tiles.
_IMG_EXTS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff"})

# Filename substrings that flag non-image artifacts (segmentation masks
# in BUSI, file-watcher caches). Substring match — keeps the list
# short. ``_mask`` (no trailing dot) is intentional: BUSI ships
# multi-mask variants named ``_mask_1.png`` / ``_mask_2.png`` that the
# dotted form misses.
_EXCLUDE_SUBSTRINGS = frozenset({"_mask", ".DS_Store"})


@dataclass(frozen=True)
class DatasetSource:
    """One dataset on disk + the modality it represents.

    Args:
        name: Stable id (also used in the trials CSV ``dataset`` column).
        modality: Ground-truth label for every image found under
            ``relative_path``. Must be one of the values in
            :data:`claritymed.core.medical_clip.schemas.Modality`.
        relative_path: Path under ``DATA_DIR / "vision"``. Datasets that
            extract into a deeply nested folder name the path verbatim
            here so the benchmark doesn't need fuzzy matching.
    """

    name: str
    modality: str
    relative_path: str


# Order matters for human-readable summary output; keep the per-modality
# pairs adjacent so the table reads as "ultrasound x 2, ct x 1, …".
_DATASET_SOURCES: tuple[DatasetSource, ...] = (
    DatasetSource(
        name="busi",
        modality="ultrasound",
        relative_path="busi/Dataset_BUSI_with_GT",
    ),
    DatasetSource(
        name="breast_us_kaggle",
        modality="ultrasound",
        relative_path="breast_us_kaggle/ultrasound breast classification",
    ),
    DatasetSource(
        name="chest_ct",
        modality="ct",
        relative_path="chest_ct/Data",
    ),
    DatasetSource(
        name="skin_lesion",
        modality="dermoscopy",
        relative_path=(
            "skin_lesion/Skin cancer ISIC The International Skin Imaging Collaboration"
        ),
    ),
    DatasetSource(
        name="lung_colon_histopath",
        modality="histopathology",
        # Filled in once the operator extracts the ZIP. ``_walk`` returns
        # an empty list if the directory doesn't exist, and the benchmark
        # logs a skip line — no exception.
        relative_path="lung_colon_histopath",
    ),
)


@dataclass(frozen=True)
class SamplePath:
    """One image + its ground-truth label.

    The dataset id is carried alongside the modality so the per-trial
    CSV can attribute false positives to a specific source (e.g.
    "chest_ct's mediastinal slices look like skin to BiomedCLIP").
    """

    dataset: str
    modality: str
    path: Path


def _walk(root: Path) -> list[Path]:
    """Return every recognized image under ``root``, mask files removed.

    Empty list when the root doesn't exist (dataset not yet extracted) —
    callers log a single skip, not an exception. This keeps the
    benchmark runnable in a partial environment.
    """
    if not root.exists():
        return []
    found: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in _IMG_EXTS:
            continue
        if any(token in p.name for token in _EXCLUDE_SUBSTRINGS):
            continue
        found.append(p)
    return found


def build_sample_pool(
    *,
    per_dataset: int = 20,
    seed: int = 0,
    datasets: list[str] | None = None,
) -> list[SamplePath]:
    """Sample up to ``per_dataset`` images from each available dataset.

    Args:
        per_dataset: Maximum samples drawn from each :class:`DatasetSource`.
            20 keeps a full A/B run under ~5 minutes on a CPU-bound LLM
            classifier; raise for stronger statistical power.
        seed: ``random.Random`` seed for reproducibility. Same seed +
            same data on disk → same sampled paths.
        datasets: Optional allow-list of dataset names (``["busi",
            "chest_ct"]``). ``None`` keeps every source. Use to scope a
            run to one or two modalities while iterating on a classifier.

    Returns:
        Flat list of :class:`SamplePath` across all selected sources.
        Sources that resolve to zero images on disk are logged once and
        omitted from the result — they do *not* raise.
    """
    rng = random.Random(seed)
    allow = set(datasets) if datasets is not None else None
    out: list[SamplePath] = []
    for source in _DATASET_SOURCES:
        if allow is not None and source.name not in allow:
            continue
        root = Path(str(DATA_DIR)) / "vision" / source.relative_path
        candidates = _walk(root)
        if not candidates:
            logger.warning(
                "dataset %r resolved to zero images under %s — skipped. "
                "Extract the dataset there and re-run if you want this "
                "modality included.",
                source.name,
                root,
            )
            continue
        if len(candidates) > per_dataset:
            candidates = rng.sample(candidates, per_dataset)
        out.extend(
            SamplePath(dataset=source.name, modality=source.modality, path=p)
            for p in candidates
        )
    return out


def list_known_datasets() -> list[str]:
    """Return the configured dataset names — useful for ``--datasets`` help text."""
    return [s.name for s in _DATASET_SOURCES]
