"""``BUSI_DATASET`` — :class:`DatasetSpec` for breast-ultrasound classification + segmentation.

Holds the per-dataset identity + IO that every BUSI model variant
shares. Specific model architectures (currently only the U-Net) pick
this dataset up via :class:`ModelSpec` in ``busi/models/<name>.py``.
"""

from __future__ import annotations

from claritymed.core.vision.schemas import LabelMeta
from claritymed.ingest.vision.busi.dataset import (
    BUSI_LABELS,
    build_dataset,
    discover,
    stratified_split,
)
from claritymed.ingest.vision.busi.download import (
    DATASET_SUBDIR,
    KAGGLE_SLUG,
    busi_data_root,
)
from claritymed.ingest.vision.forge.spec import DatasetSpec, Splits


# Per-label metadata. Kept here (not in ``models/<x>.py``) because
# clinical_action + cancer_status are dataset-level facts: any model
# trained on BUSI surfaces the same routing instructions per label.
BUSI_LABELS_META: dict[str, LabelMeta] = {
    "benign": LabelMeta(
        description="Non-cancerous lesion. Routine follow-up is usually appropriate.",
        cancer_status="benign",
        clinical_action="routine_followup",
    ),
    "malignant": LabelMeta(
        description="Suspicious for cancer. A breast specialist should review the image.",
        cancer_status="malignant",
        clinical_action="urgent_specialist",
    ),
    "normal": LabelMeta(
        description="No lesion identified. No action required from this image alone.",
        cancer_status="normal",
        clinical_action="no_action",
    ),
}


def _build_splits() -> Splits:
    """Run discover + stratified split + build the three Torch Datasets."""
    root = busi_data_root() / DATASET_SUBDIR
    if not root.is_dir():
        raise SystemExit(
            f"BUSI not present at {root}. Run "
            "`uv run python -m claritymed.ingest.vision.busi.download` first."
        )
    samples = discover(root)
    raw = stratified_split(samples)
    return Splits(
        train=build_dataset(raw["train"]),
        val=build_dataset(raw["val"]),
        test=build_dataset(raw["test"]),
    )


BUSI_DATASET = DatasetSpec(
    disease_id="breast_cancer_ultrasound",
    accepted_modality="ultrasound",
    labels=BUSI_LABELS,
    labels_meta=BUSI_LABELS_META,
    cancer_class=True,
    download_slug=KAGGLE_SLUG,
    dataset_subdir=DATASET_SUBDIR,
    build_splits=_build_splits,
    # BUSI is tiny (~780 images, ~40 train batches at bs=16). The (8, 4)
    # default is sized for RSNA-scale (~20k images) where worker spawn
    # cost amortises across many batches per epoch. Here it dominated.
    search_num_workers=(2, 2),
)


__all__ = ["BUSI_DATASET", "BUSI_LABELS_META"]
