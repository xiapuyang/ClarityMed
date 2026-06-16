"""``BREAST_US_KAGGLE_DATASET`` — alternative 2-class :class:`DatasetSpec`.

Shares ``disease_id="breast_cancer_ultrasound"`` with
:data:`~claritymed.ingest.vision.busi.dataset_spec.BUSI_DATASET`. The
two specs are alternate training data for the same vision tool — BUSI
is the 3-class primary; this entry is a much larger 2-class (augmented)
backup that an operator can promote into the disease's flow when they
want to trade the ``normal`` class for more positive-class recall.

No segmentation masks ship with the upstream archive, so every
breast_us_kaggle model variant uses :class:`ClassificationTask`.
"""

from __future__ import annotations

from claritymed.core.vision.schemas import LabelMeta
from claritymed.ingest.vision.breast_us_kaggle.dataset import (
    BREAST_US_KAGGLE_LABELS,
    build_dataset,
    discover,
    stratified_split,
)
from claritymed.ingest.vision.breast_us_kaggle.download import (
    DATASET_SUBDIR,
    KAGGLE_SLUG,
    breast_us_kaggle_data_root,
)
from claritymed.ingest.vision.forge.spec import DatasetSpec, Splits


# Per-label metadata. Mirrors BUSI's benign + malignant entries —
# operators routing between the two models see identical clinical-action
# semantics for these labels.
BREAST_US_KAGGLE_LABELS_META: dict[str, LabelMeta] = {
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
}


def _build_splits() -> Splits:
    """Run discover + stratified split + build the three Torch Datasets."""
    root = breast_us_kaggle_data_root() / DATASET_SUBDIR
    if not root.is_dir():
        raise SystemExit(
            f"breast_us_kaggle not present at {root}. Run "
            "`uv run python -m claritymed.ingest.vision.breast_us_kaggle.download` first."
        )
    samples = discover(root)
    raw = stratified_split(samples)
    return Splits(
        train=build_dataset(raw["train"]),
        val=build_dataset(raw["val"]),
        test=build_dataset(raw["test"]),
    )


BREAST_US_KAGGLE_DATASET = DatasetSpec(
    disease_id="breast_cancer_ultrasound",
    accepted_modality="ultrasound",
    labels=BREAST_US_KAGGLE_LABELS,
    labels_meta=BREAST_US_KAGGLE_LABELS_META,
    cancer_class=True,
    download_slug=KAGGLE_SLUG,
    dataset_subdir=DATASET_SUBDIR,
    build_splits=_build_splits,
)


__all__ = ["BREAST_US_KAGGLE_DATASET", "BREAST_US_KAGGLE_LABELS_META"]
