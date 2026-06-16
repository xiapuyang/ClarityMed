"""``CHEST_CT_DATASET`` — :class:`DatasetSpec` for 4-class lung-cancer classification.

Three malignant subtypes (``adenocarcinoma`` / ``large_cell_carcinoma`` /
``squamous_cell_carcinoma``) plus ``normal``. No segmentation masks
ship with the upstream Kaggle dataset, so every chest CT model variant
uses :class:`ClassificationTask`.
"""

from __future__ import annotations

from claritymed.core.vision.schemas import LabelMeta
from claritymed.ingest.vision.chest_ct.dataset import (
    CHEST_CT_LABELS,
    build_dataset,
    discover,
    stratified_split,
)
from claritymed.ingest.vision.chest_ct.download import (
    DATASET_SUBDIR,
    KAGGLE_SLUG,
    chest_ct_data_root,
)
from claritymed.ingest.vision.forge.spec import DatasetSpec, Splits


# All three malignant labels carry the same routing: cancer + urgent
# specialist review. Subtype distinction matters to a radiologist, but
# from a clinical-action perspective they're identical.
_URGENT_CANCER = {
    "cancer_status": "malignant",
    "clinical_action": "urgent_specialist",
}

CHEST_CT_LABELS_META: dict[str, LabelMeta] = {
    "adenocarcinoma": LabelMeta(
        description="Adenocarcinoma of the lung. Specialist review required.",
        cancer_status=_URGENT_CANCER["cancer_status"],
        clinical_action=_URGENT_CANCER["clinical_action"],
    ),
    "large_cell_carcinoma": LabelMeta(
        description="Large cell carcinoma of the lung. Specialist review required.",
        cancer_status=_URGENT_CANCER["cancer_status"],
        clinical_action=_URGENT_CANCER["clinical_action"],
    ),
    "normal": LabelMeta(
        description="No lung-cancer findings identified. No action required from this image alone.",
        cancer_status="normal",
        clinical_action="no_action",
    ),
    "squamous_cell_carcinoma": LabelMeta(
        description="Squamous cell carcinoma of the lung. Specialist review required.",
        cancer_status=_URGENT_CANCER["cancer_status"],
        clinical_action=_URGENT_CANCER["clinical_action"],
    ),
}


def _build_splits() -> Splits:
    """Run discover + stratified split + build the three Torch Datasets.

    Pools across the upstream archive's ``train/`` ``test/`` ``valid/``
    subdirs first (see chest_ct/dataset.py) because the upstream split
    is tilted; a hash-stratified split keeps each class proportionally
    represented in val + test.
    """
    root = chest_ct_data_root() / DATASET_SUBDIR
    if not root.is_dir():
        raise SystemExit(
            f"chest CT not present at {root}. Run "
            "`uv run python -m claritymed.ingest.vision.chest_ct.download` first."
        )
    samples = discover(root)
    raw = stratified_split(samples)
    return Splits(
        train=build_dataset(raw["train"]),
        val=build_dataset(raw["val"]),
        test=build_dataset(raw["test"]),
    )


CHEST_CT_DATASET = DatasetSpec(
    disease_id="lung_cancer_chest_ct",
    accepted_modality="ct",
    labels=CHEST_CT_LABELS,
    labels_meta=CHEST_CT_LABELS_META,
    cancer_class=True,
    download_slug=KAGGLE_SLUG,
    dataset_subdir=DATASET_SUBDIR,
    build_splits=_build_splits,
)


__all__ = ["CHEST_CT_DATASET", "CHEST_CT_LABELS_META"]
