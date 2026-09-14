"""``RSNA_PNEUMONIA_DATASET`` — :class:`DatasetSpec` for RSNA Pneumonia Detection Challenge.

The adult-population counterpart to Kermany. Same modality (frontal
chest X-ray), same binary task (normal vs pneumonia), wildly different
acquisition context (adult vs pediatric, multi-center vs single
hospital, portable AP vs standing PA).

Serves two roles, both pivoting on ``disease_id="chest_xray_pneumonia"``:

* **Training source** — paired with ``models/resnet50_v1.py``
  (``RESNET50_V1``) so the forge pipeline can train an RSNA-native
  classifier alongside Kermany's. Both ModelSpecs share the artifact
  root and ``LATEST.jsonl``; the ``model_id`` discriminates rows.
* **Drift-bench target** — a Kermany-trained model evaluates against
  RSNA's test split (and now symmetrically the other way too) to
  quantify pediatric ↔ adult distribution shift. The shared
  ``disease_id`` is what makes the binary clinical task collapse
  apples-to-apples across the pair.

Bounding boxes from ``stage_2_train_labels.csv`` are collapsed to
image-level positives (any bbox → ``pneumonia``); the bbox-preserving
parse lives in the sibling ``rsna_pneumonia_yolo/`` detection adapter.
No segmentation head — RSNA ships no masks — so
``ClassificationTask`` is the right fit.

See ``docs/plans/2026-06-17-001-feat-cross-dataset-drift-bench-plan.md``.
"""

from __future__ import annotations

from claritymed.core.vision.schemas import LabelMeta
from claritymed.ingest.vision.forge.spec import DatasetSpec, Splits, standard_splits
from claritymed.ingest.vision.rsna_pneumonia.dataset import (
    DEFAULT_INPUT_SIZE,
    RSNA_PNEUMONIA_LABELS,
    build_dataset,
    discover,
    ensure_resized_cache,
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
    """Run discover + stratified split + build the three Torch Datasets.

    DATASET_SUBDIR is ``"."`` for RSNA (no top-level wrapper); resolve
    so the fail-loud message points at the actual root path. Presence
    is keyed on the labels CSV rather than the directory because the
    Kaggle archive is sometimes extracted to a flat tree where the dir
    exists empty before the user has run the download.
    """
    root = (rsna_pneumonia_data_root() / DATASET_SUBDIR).resolve()
    return standard_splits(
        root=root,
        missing_message=(
            f"rsna_pneumonia not present at {root}. Run "
            "`uv run python -m claritymed.ingest.vision.rsna_pneumonia.download` "
            "first (requires accepting competition rules on Kaggle)."
        ),
        discover=discover,
        stratified_split=stratified_split,
        build_dataset=build_dataset,
        is_present=lambda p: (p / "stage_2_train_labels.csv").is_file(),
        post_discover=lambda samples, root: ensure_resized_cache(
            samples, root, input_size=DEFAULT_INPUT_SIZE
        ),
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
