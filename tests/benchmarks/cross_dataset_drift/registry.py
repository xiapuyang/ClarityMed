"""Cross-dataset drift bench registry — what to evaluate against what.

Each :class:`BenchEntry` is one ``(model_artifact, eval_dataset)`` pair
that the bench driver will run. Self-eval entries (model evaluated on
its own dataset's test split) live alongside cross-dataset entries so
the in-distribution baseline sits visibly next to the drift cell in
every Markdown report — the gap is the headline.

Why this is a Python module, not YAML: ``eval_dataset_dotted_path``
references a :class:`~claritymed.ingest.vision.forge.spec.DatasetSpec`
instance via the same ``module.path:ATTR`` shape forge's CLI uses
(``ingest/vision/forge/cli.py::_resolve_model_spec``). Python is the
natural home for that, and the bench driver imports this module
directly — no parsing layer, no schema duplication.

How ``positive_labels`` works: each entry declares the label name(s)
that count as the positive class in the binary clinical task. The
driver passes ``positive_labels`` into
:func:`~claritymed.core.vision.eval_metrics.binary_clinical_metrics`,
which collapses both the model's predictions and the eval dataset's
ground truth to ``positive vs not``. This is how we make BUSI 3-class
vs breast_us_kaggle 2-class comparisons apples-to-apples.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

# --- entry schema --------------------------------------------------------


class BenchEntry(BaseModel):
    """One bench row: load a model from ``model_artifact_dir``, run it
    over the test split of ``eval_dataset_dotted_path``, collapse to the
    binary clinical task defined by ``positive_labels``.

    ``train_dataset_id`` and ``eval_dataset_id`` are human-readable
    short labels used in the output Markdown — they don't have to match
    any internal id, they're for the report table.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair_id: str = Field(min_length=1, max_length=32)
    model_artifact_dir: str = Field(min_length=1)
    train_dataset_id: str = Field(min_length=1, max_length=64)
    eval_dataset_dotted_path: str = Field(min_length=1)
    eval_dataset_id: str = Field(min_length=1, max_length=64)
    positive_labels: frozenset[str] = Field(min_length=1)


# --- breast ultrasound pair (Phase 1) ------------------------------------
#
# Both models share ``disease_id=breast_cancer_ultrasound`` and tag
# ``positive_labels={"malignant"}``. The binary collapse means BUSI's
# ``normal`` ground-truth folds into the negative class when evaluating
# breast_us_kaggle's 2-class model, and the BUSI model's ``normal``
# predictions on the 2-class kaggle test set similarly count as
# negative — both directions stay clinically meaningful.

_MALIGNANT: frozenset[str] = frozenset({"malignant"})

_BREAST_US_ENTRIES: list[BenchEntry] = [
    # --- BUSI U-Net (cls+seg) -------------------------------------------
    BenchEntry(
        pair_id="breast_us",
        model_artifact_dir="breast_cancer_ultrasound/breast_busi_unet_v1",
        train_dataset_id="busi",
        eval_dataset_dotted_path=(
            "claritymed.ingest.vision.busi.dataset_spec:BUSI_DATASET"
        ),
        eval_dataset_id="busi",
        positive_labels=_MALIGNANT,
    ),
    BenchEntry(
        pair_id="breast_us",
        model_artifact_dir="breast_cancer_ultrasound/breast_busi_unet_v1",
        train_dataset_id="busi",
        eval_dataset_dotted_path=(
            "claritymed.ingest.vision.breast_us_kaggle.dataset_spec:"
            "BREAST_US_KAGGLE_DATASET"
        ),
        eval_dataset_id="breast_us_kaggle",
        positive_labels=_MALIGNANT,
    ),
    # --- breast_us_kaggle ResNet-50 (cls only) --------------------------
    BenchEntry(
        pair_id="breast_us",
        model_artifact_dir="breast_cancer_ultrasound/breast_us_kaggle_resnet50_v1",
        train_dataset_id="breast_us_kaggle",
        eval_dataset_dotted_path=(
            "claritymed.ingest.vision.breast_us_kaggle.dataset_spec:"
            "BREAST_US_KAGGLE_DATASET"
        ),
        eval_dataset_id="breast_us_kaggle",
        positive_labels=_MALIGNANT,
    ),
    BenchEntry(
        pair_id="breast_us",
        model_artifact_dir="breast_cancer_ultrasound/breast_us_kaggle_resnet50_v1",
        train_dataset_id="breast_us_kaggle",
        eval_dataset_dotted_path=(
            "claritymed.ingest.vision.busi.dataset_spec:BUSI_DATASET"
        ),
        eval_dataset_id="busi",
        positive_labels=_MALIGNANT,
    ),
]


# --- chest X-ray pair (Phase 2) ------------------------------------------
#
# Asymmetric — only Kermany has a trained model in this iteration. The
# matrix has two cells: Kermany self-eval (the in-distribution baseline)
# and Kermany on RSNA (the drift cell). No RSNA-trained model exists so
# the reverse direction is absent; training one would be a separate
# ingest module + forge pipeline run.
#
# Prereqs for ``make bench-drift PAIR=chest_xray``:
#   1. ``uv run python -m claritymed.ingest.vision.chest_xray_pneumonia.download``
#   2. ``uv run python -m claritymed.ingest.vision.rsna_pneumonia.download``
#      (Kaggle competition rules must be accepted first).
#   3. ``uv run claritymed-vision-forge pipeline --model
#      claritymed.ingest.vision.chest_xray_pneumonia.models.resnet50_v1:RESNET50_V1``
#      to train the Kermany classifier (~20-30 min on M1).

_PNEUMONIA: frozenset[str] = frozenset({"pneumonia"})

_CHEST_XRAY_ENTRIES: list[BenchEntry] = [
    # --- Kermany ResNet (cls only) — in-distribution baseline ------------
    BenchEntry(
        pair_id="chest_xray",
        model_artifact_dir="chest_xray_pneumonia/chest_xray_pneumonia_resnet50_v1",
        train_dataset_id="kermany",
        eval_dataset_dotted_path=(
            "claritymed.ingest.vision.chest_xray_pneumonia.dataset_spec:"
            "CHEST_XRAY_PNEUMONIA_DATASET"
        ),
        eval_dataset_id="kermany",
        positive_labels=_PNEUMONIA,
    ),
    # --- Kermany ResNet on RSNA — the drift cell --------------------------
    BenchEntry(
        pair_id="chest_xray",
        model_artifact_dir="chest_xray_pneumonia/chest_xray_pneumonia_resnet50_v1",
        train_dataset_id="kermany",
        eval_dataset_dotted_path=(
            "claritymed.ingest.vision.rsna_pneumonia.dataset_spec:"
            "RSNA_PNEUMONIA_DATASET"
        ),
        eval_dataset_id="rsna",
        positive_labels=_PNEUMONIA,
    ),
]


# --- consolidated registry ------------------------------------------------

BENCH_ENTRIES: list[BenchEntry] = [
    *_BREAST_US_ENTRIES,
    # *_CHEST_XRAY_ENTRIES,
]


def entries_for_pair(pair_id: str) -> list[BenchEntry]:
    """Filter the registry to one pair. Fail-loud on unknown pair.

    Unknown pair ids almost always mean a typo at the CLI (``--pair
    breast-us`` vs ``--pair breast_us``); surfacing the known ids in
    the error keeps the operator from guessing.
    """
    matches = [e for e in BENCH_ENTRIES if e.pair_id == pair_id]
    if not matches:
        known = sorted({e.pair_id for e in BENCH_ENTRIES})
        raise ValueError(f"no bench entries for pair_id={pair_id!r}; known: {known!r}")
    return matches


__all__ = ["BENCH_ENTRIES", "BenchEntry", "entries_for_pair"]
