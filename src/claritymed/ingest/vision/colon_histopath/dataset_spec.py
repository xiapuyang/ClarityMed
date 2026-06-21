"""``COLON_HISTOPATH_DATASET`` — :class:`DatasetSpec` for LC25000 colon subset.

Sister module to :mod:`claritymed.ingest.vision.lung_histopath`. Same
upstream archive, different organ subset, independent disease_id
(``colon_cancer_histopathology``). Splitting the LC25000 dataset into
two per-organ modules reflects the clinical reality: lung and colon
biopsies are routed to different specialists with different follow-up
pathways.

Per-label cancer_status mapping: ``adenocarcinoma`` → ``malignant`` +
``urgent_specialist``; ``normal`` → ``normal`` + ``no_action`` because
LC25000's "colon_n" class is healthy baseline tissue, not a benign
lesion.

No segmentation masks ship with the upstream archive, so this dataset
uses :class:`ClassificationTask`.
"""

from __future__ import annotations

from claritymed.core.vision.schemas import LabelMeta
from claritymed.ingest.vision.colon_histopath.dataset import (
    COLON_HISTOPATH_LABELS,
    build_dataset,
    discover,
    stratified_split,
)
from claritymed.ingest.vision.forge.spec import DatasetSpec, Splits, standard_splits
from claritymed.ingest.vision.lung_colon_histopath.download import (
    DATASET_SUBDIR,
    KAGGLE_SLUG,
    lung_colon_histopath_data_root,
)


COLON_HISTOPATH_LABELS_META: dict[str, LabelMeta] = {
    "adenocarcinoma": LabelMeta(
        description=(
            "Colon adenocarcinoma on histology. Specialist review is recommended."
        ),
        cancer_status="malignant",
        clinical_action="urgent_specialist",
    ),
    "normal": LabelMeta(
        description=(
            "Healthy colon tissue on histology. No malignant features identified."
        ),
        cancer_status="normal",
        clinical_action="no_action",
    ),
}


def _build_splits() -> Splits:
    """Run discover + stratified split + build the three Torch Datasets.

    The on-disk root is shared with lung_histopath via the
    :mod:`lung_colon_histopath` download wrapper.
    """
    root = lung_colon_histopath_data_root() / DATASET_SUBDIR
    return standard_splits(
        root=root,
        missing_message=(
            f"LC25000 archive not present at {root}. Run "
            "`uv run python -m claritymed.ingest.vision.lung_colon_histopath.download` first."
        ),
        discover=discover,
        stratified_split=stratified_split,
        build_dataset=build_dataset,
    )


COLON_HISTOPATH_DATASET = DatasetSpec(
    disease_id="colon_cancer_histopathology",
    accepted_modality="histopathology",
    labels=COLON_HISTOPATH_LABELS,
    labels_meta=COLON_HISTOPATH_LABELS_META,
    cancer_class=True,
    download_slug=KAGGLE_SLUG,
    dataset_subdir=DATASET_SUBDIR,
    build_splits=_build_splits,
)


__all__ = ["COLON_HISTOPATH_DATASET", "COLON_HISTOPATH_LABELS_META"]
