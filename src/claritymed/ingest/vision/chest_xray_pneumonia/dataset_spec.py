"""``CHEST_XRAY_PNEUMONIA_DATASET`` — :class:`DatasetSpec` for Kermany 2018.

The dataset is pediatric (Guangzhou Women and Children's Medical
Center) NORMAL / PNEUMONIA binary classification. ``cancer_class=False``
because pneumonia isn't cancer — the per-label ``clinical_action``
metadata sets pneumonia's routing to ``urgent_specialist`` directly,
without going through the cancer_status_mapping shortcut.

This module's primary near-term consumer is the **cross-dataset drift
bench**: a Kermany-trained classifier evaluated against an adult chest
X-ray dataset (RSNA Pneumonia Detection Challenge) measures how much a
pediatric model degrades on adult anatomy. See
``docs/plans/2026-06-17-001-feat-cross-dataset-drift-bench-plan.md``.

There is no segmentation head (Kermany has no masks), so every Kermany
model variant uses :class:`ClassificationTask`.
"""

from __future__ import annotations

from claritymed.core.vision.schemas import LabelMeta
from claritymed.ingest.vision.chest_xray_pneumonia.dataset import (
    CHEST_XRAY_PNEUMONIA_LABELS,
    build_dataset,
    discover,
    stratified_split,
)
from claritymed.ingest.vision.chest_xray_pneumonia.download import (
    DATASET_SUBDIR,
    KAGGLE_SLUG,
    chest_xray_pneumonia_data_root,
)
from claritymed.ingest.vision.forge.spec import DatasetSpec, Splits

# Per-label metadata. ``cancer_status`` left as ``None`` everywhere
# because pneumonia is not on the cancer-status enum; downstream code
# that branches on cancer_status treats ``None`` as "not applicable",
# which is the right semantic for an infection class.
CHEST_XRAY_PNEUMONIA_LABELS_META: dict[str, LabelMeta] = {
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
    root = chest_xray_pneumonia_data_root() / DATASET_SUBDIR
    if not root.is_dir():
        raise SystemExit(
            f"chest_xray_pneumonia not present at {root}. Run "
            "`uv run python -m claritymed.ingest.vision.chest_xray_pneumonia.download` "
            "first."
        )
    samples = discover(root)
    raw = stratified_split(samples)
    return Splits(
        train=build_dataset(raw["train"]),
        val=build_dataset(raw["val"]),
        test=build_dataset(raw["test"]),
    )


CHEST_XRAY_PNEUMONIA_DATASET = DatasetSpec(
    disease_id="chest_xray_pneumonia",
    accepted_modality="xray",
    labels=CHEST_XRAY_PNEUMONIA_LABELS,
    labels_meta=CHEST_XRAY_PNEUMONIA_LABELS_META,
    cancer_class=False,
    download_slug=KAGGLE_SLUG,
    dataset_subdir=DATASET_SUBDIR,
    build_splits=_build_splits,
)


__all__ = ["CHEST_XRAY_PNEUMONIA_DATASET", "CHEST_XRAY_PNEUMONIA_LABELS_META"]
