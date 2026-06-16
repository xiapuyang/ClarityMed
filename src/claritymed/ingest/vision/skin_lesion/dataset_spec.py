"""``SKIN_LESION_DATASET`` — :class:`DatasetSpec` for 9-class ISIC skin-lesion classification.

Per-label cancer_status mapping follows standard dermatology — four
cancer / pre-cancer entries (``actinic_keratosis``,
``basal_cell_carcinoma``, ``melanoma``, ``squamous_cell_carcinoma``)
all carry ``malignant`` + ``urgent_specialist`` routing, and the five
benign entries (``dermatofibroma``, ``nevus``,
``pigmented_benign_keratosis``, ``seborrheic_keratosis``,
``vascular_lesion``) carry ``benign`` + ``routine_followup``. The
medical bar for skin cancer is "don't miss the cancer" — clinical
action is identical across malignant subtypes from a routing
perspective.

No segmentation masks ship with the upstream ISIC archive, so every
skin lesion model variant uses :class:`ClassificationTask`.
"""

from __future__ import annotations

from claritymed.core.vision.schemas import LabelMeta
from claritymed.ingest.vision.forge.spec import DatasetSpec, Splits
from claritymed.ingest.vision.skin_lesion.dataset import (
    SKIN_LESION_LABELS,
    build_dataset,
    discover,
    stratified_split,
)
from claritymed.ingest.vision.skin_lesion.download import (
    DATASET_SUBDIR,
    KAGGLE_SLUG,
    skin_lesion_data_root,
)


# Malignant routing is shared across the four cancer / pre-cancer
# entries: a clinical-action perspective doesn't distinguish subtypes;
# the routing tells the patient "see a specialist soon" regardless.
_URGENT_CANCER = {
    "cancer_status": "malignant",
    "clinical_action": "urgent_specialist",
}
_BENIGN_ROUTINE = {
    "cancer_status": "benign",
    "clinical_action": "routine_followup",
}

SKIN_LESION_LABELS_META: dict[str, LabelMeta] = {
    "actinic_keratosis": LabelMeta(
        description=(
            "Pre-malignant sun-damage lesion. Dermatology review is "
            "recommended to assess progression risk."
        ),
        cancer_status=_URGENT_CANCER["cancer_status"],
        clinical_action=_URGENT_CANCER["clinical_action"],
    ),
    "basal_cell_carcinoma": LabelMeta(
        description=(
            "Basal cell carcinoma — the most common skin cancer. "
            "Dermatology review is recommended."
        ),
        cancer_status=_URGENT_CANCER["cancer_status"],
        clinical_action=_URGENT_CANCER["clinical_action"],
    ),
    "dermatofibroma": LabelMeta(
        description="Benign fibrous nodule. Routine follow-up is usually appropriate.",
        cancer_status=_BENIGN_ROUTINE["cancer_status"],
        clinical_action=_BENIGN_ROUTINE["clinical_action"],
    ),
    "melanoma": LabelMeta(
        description=(
            "Suspicious for melanoma — the most serious skin cancer. "
            "A dermatologist should review urgently."
        ),
        cancer_status=_URGENT_CANCER["cancer_status"],
        clinical_action=_URGENT_CANCER["clinical_action"],
    ),
    "nevus": LabelMeta(
        description="Common mole (benign). Routine self-monitoring is appropriate.",
        cancer_status=_BENIGN_ROUTINE["cancer_status"],
        clinical_action=_BENIGN_ROUTINE["clinical_action"],
    ),
    "pigmented_benign_keratosis": LabelMeta(
        description=(
            "Benign pigmented keratosis. Routine follow-up is usually appropriate."
        ),
        cancer_status=_BENIGN_ROUTINE["cancer_status"],
        clinical_action=_BENIGN_ROUTINE["clinical_action"],
    ),
    "seborrheic_keratosis": LabelMeta(
        description="Benign seborrheic keratosis. Routine follow-up is appropriate.",
        cancer_status=_BENIGN_ROUTINE["cancer_status"],
        clinical_action=_BENIGN_ROUTINE["clinical_action"],
    ),
    "squamous_cell_carcinoma": LabelMeta(
        description=("Squamous cell carcinoma. Dermatology review is recommended."),
        cancer_status=_URGENT_CANCER["cancer_status"],
        clinical_action=_URGENT_CANCER["clinical_action"],
    ),
    "vascular_lesion": LabelMeta(
        description=(
            "Benign vascular lesion (e.g. cherry angioma). Routine "
            "follow-up is appropriate."
        ),
        cancer_status=_BENIGN_ROUTINE["cancer_status"],
        clinical_action=_BENIGN_ROUTINE["clinical_action"],
    ),
}


def _build_splits() -> Splits:
    """Run discover + stratified split + build the three Torch Datasets.

    Pools across the upstream ``Train/`` and ``Test/`` subdirs first
    (see dataset.py) because the upstream split is small and tilted; a
    hash-stratified split keeps each class proportionally represented
    in val + test.
    """
    root = skin_lesion_data_root() / DATASET_SUBDIR
    if not root.is_dir():
        raise SystemExit(
            f"skin lesion data not present at {root}. Run "
            "`uv run python -m claritymed.ingest.vision.skin_lesion.download` first."
        )
    samples = discover(root)
    raw = stratified_split(samples)
    return Splits(
        train=build_dataset(raw["train"]),
        val=build_dataset(raw["val"]),
        test=build_dataset(raw["test"]),
    )


SKIN_LESION_DATASET = DatasetSpec(
    disease_id="skin_cancer_dermoscopy",
    accepted_modality="dermoscopy",
    labels=SKIN_LESION_LABELS,
    labels_meta=SKIN_LESION_LABELS_META,
    cancer_class=True,
    download_slug=KAGGLE_SLUG,
    dataset_subdir=DATASET_SUBDIR,
    build_splits=_build_splits,
)


__all__ = ["SKIN_LESION_DATASET", "SKIN_LESION_LABELS_META"]
