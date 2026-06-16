"""``LUNG_HISTOPATH_DATASET`` — :class:`DatasetSpec` for LC25000 lung subset.

Same disease as :data:`~claritymed.ingest.vision.chest_ct.dataset_spec.CHEST_CT_DATASET`
(``lung_cancer_*``) but a different modality: the lung-cancer condition
is the same, the imaging is histopathology rather than chest CT. The
disease_id is therefore ``lung_cancer_histopathology`` — sister entry
to ``lung_cancer_chest_ct`` in ``configs/vision.yaml``.

Per-label cancer_status mapping follows standard pathology: the two
malignant entries (``adenocarcinoma``, ``squamous_cell_carcinoma``)
carry ``malignant`` + ``urgent_specialist`` routing; ``normal`` carries
``normal`` + ``no_action`` because LC25000's "lung_n" class is healthy
baseline tissue, not a benign lesion.

No segmentation masks ship with the upstream archive, so this dataset
uses :class:`ClassificationTask`.
"""

from __future__ import annotations

from claritymed.core.vision.schemas import LabelMeta
from claritymed.ingest.vision.forge.spec import DatasetSpec, Splits
from claritymed.ingest.vision.lung_colon_histopath.download import (
    DATASET_SUBDIR,
    KAGGLE_SLUG,
    lung_colon_histopath_data_root,
)
from claritymed.ingest.vision.lung_histopath.dataset import (
    LUNG_HISTOPATH_LABELS,
    build_dataset,
    discover,
    stratified_split,
)


LUNG_HISTOPATH_LABELS_META: dict[str, LabelMeta] = {
    "adenocarcinoma": LabelMeta(
        description=(
            "Lung adenocarcinoma on histology. Specialist review is recommended."
        ),
        cancer_status="malignant",
        clinical_action="urgent_specialist",
    ),
    "normal": LabelMeta(
        description=(
            "Healthy lung tissue on histology. No malignant features identified."
        ),
        cancer_status="normal",
        clinical_action="no_action",
    ),
    "squamous_cell_carcinoma": LabelMeta(
        description=(
            "Lung squamous cell carcinoma on histology. Specialist "
            "review is recommended."
        ),
        cancer_status="malignant",
        clinical_action="urgent_specialist",
    ),
}


def _build_splits() -> Splits:
    """Run discover + stratified split + build the three Torch Datasets.

    The on-disk root is shared with colon_histopath via the
    :mod:`lung_colon_histopath` download wrapper — both per-organ
    modules walk different organ subdirs of the same extracted
    archive, so downloading once feeds both training runs.
    """
    root = lung_colon_histopath_data_root() / DATASET_SUBDIR
    if not root.is_dir():
        raise SystemExit(
            f"LC25000 archive not present at {root}. Run "
            "`uv run python -m claritymed.ingest.vision.lung_colon_histopath.download` first."
        )
    samples = discover(root)
    raw = stratified_split(samples)
    return Splits(
        train=build_dataset(raw["train"]),
        val=build_dataset(raw["val"]),
        test=build_dataset(raw["test"]),
    )


LUNG_HISTOPATH_DATASET = DatasetSpec(
    disease_id="lung_cancer_histopathology",
    accepted_modality="histopathology",
    labels=LUNG_HISTOPATH_LABELS,
    labels_meta=LUNG_HISTOPATH_LABELS_META,
    cancer_class=True,
    download_slug=KAGGLE_SLUG,
    dataset_subdir=DATASET_SUBDIR,
    build_splits=_build_splits,
)


__all__ = ["LUNG_HISTOPATH_DATASET", "LUNG_HISTOPATH_LABELS_META"]
