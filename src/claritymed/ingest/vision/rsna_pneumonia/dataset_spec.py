"""``RSNA_PNEUMONIA_DATASET`` — :class:`DatasetSpec` for RSNA Pneumonia Detection Challenge.

The adult-population counterpart to Kermany. Same modality (frontal
chest X-ray), same binary task (normal vs pneumonia), wildly different
acquisition context (adult vs pediatric, multi-center vs single
hospital, portable AP vs standing PA). Exists purely as an eval target
for the cross-dataset drift bench — no model is trained on RSNA in
this iteration.

``disease_id`` deliberately matches ``CHEST_XRAY_PNEUMONIA_DATASET``
so the bench's binary clinical task collapse stays meaningful when a
Kermany-trained model evaluates against RSNA's test split.

There is no segmentation head (RSNA's bounding boxes are ignored in
the bench), so the classification-only ``ClassificationTask`` is the
fit if RSNA ever becomes a training target.

See ``docs/plans/2026-06-17-001-feat-cross-dataset-drift-bench-plan.md``.
"""

from __future__ import annotations

from claritymed.core.vision.schemas import LabelMeta
from claritymed.ingest.vision.forge.spec import DatasetSpec, Splits
from claritymed.ingest.vision.rsna_pneumonia.dataset import (
    RSNA_PNEUMONIA_LABELS,
    build_dataset,
    discover,
    stratified_split,
)
from claritymed.ingest.vision.rsna_pneumonia.download import (
    DATASET_SUBDIR,
    KAGGLE_SLUG,
    rsna_pneumonia_data_root,
)


# Per-label metadata. Same clinical actions as Kermany — pneumonia is
# urgent regardless of patient age, normal needs no action.
RSNA_PNEUMONIA_LABELS_META: dict[str, LabelMeta] = {
    "normal": LabelMeta(
        description=(
            "No pneumonia identified. No action required from this image alone."
        ),
        cancer_status=None,
        clinical_action="no_action",
    ),
    "pneumonia": LabelMeta(
        description=(
            "Pneumonia identified. A clinician should review the image promptly."
        ),
        cancer_status=None,
        clinical_action="urgent_specialist",
    ),
}


def _build_splits() -> Splits:
    """Run discover + stratified split + build the three Torch Datasets."""
    root = rsna_pneumonia_data_root() / DATASET_SUBDIR
    # DATASET_SUBDIR is "." for RSNA (no top-level wrapper); resolve so
    # the fail-loud check below points at the actual root path.
    root = root.resolve()
    if not (root / "stage_2_train_labels.csv").is_file():
        raise SystemExit(
            f"rsna_pneumonia not present at {root}. Run "
            "`uv run python -m claritymed.ingest.vision.rsna_pneumonia.download` "
            "first (requires accepting competition rules on Kaggle)."
        )
    samples = discover(root)
    raw = stratified_split(samples)
    return Splits(
        train=build_dataset(raw["train"]),
        val=build_dataset(raw["val"]),
        test=build_dataset(raw["test"]),
    )


RSNA_PNEUMONIA_DATASET = DatasetSpec(
    disease_id="chest_xray_pneumonia",  # same disease, different source
    accepted_modality="xray",
    labels=RSNA_PNEUMONIA_LABELS,
    labels_meta=RSNA_PNEUMONIA_LABELS_META,
    cancer_class=False,
    download_slug=KAGGLE_SLUG,
    dataset_subdir=DATASET_SUBDIR,
    build_splits=_build_splits,
)


__all__ = ["RSNA_PNEUMONIA_DATASET", "RSNA_PNEUMONIA_LABELS_META"]
