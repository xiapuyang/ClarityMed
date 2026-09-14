"""Validate ``CHEST_XRAY_PNEUMONIA_DATASET`` imports + has the expected shape."""

from __future__ import annotations


def test_dataset_spec_imports_cleanly() -> None:
    """The spec module must import without side effects (no data root scan)."""
    from claritymed.ingest.vision.chest_xray_pneumonia.dataset_spec import (
        CHEST_XRAY_PNEUMONIA_DATASET,
        CHEST_XRAY_PNEUMONIA_LABELS_META,
    )

    assert CHEST_XRAY_PNEUMONIA_DATASET.disease_id == "chest_xray_pneumonia"
    assert CHEST_XRAY_PNEUMONIA_DATASET.accepted_modality == "xray"
    assert CHEST_XRAY_PNEUMONIA_DATASET.labels == ("normal", "pneumonia")
    assert CHEST_XRAY_PNEUMONIA_DATASET.cancer_class is False
    # Meta must cover every label, otherwise downstream rendering will
    # crash on a KeyError at runtime.
    for label in CHEST_XRAY_PNEUMONIA_DATASET.labels:
        assert label in CHEST_XRAY_PNEUMONIA_LABELS_META


def test_clinical_actions_match_clinical_intent() -> None:
    """Pneumonia routes to urgent specialist; normal needs no action.

    Sanity check — these strings are translator-facing and drive the
    runtime's clinical-action branching; getting them wrong silently
    flips a triage decision.
    """
    from claritymed.ingest.vision.chest_xray_pneumonia.dataset_spec import (
        CHEST_XRAY_PNEUMONIA_LABELS_META,
    )

    assert (
        CHEST_XRAY_PNEUMONIA_LABELS_META["pneumonia"].clinical_action
        == "urgent_specialist"
    )
    assert CHEST_XRAY_PNEUMONIA_LABELS_META["normal"].clinical_action == "no_action"


def test_build_splits_fails_loud_when_data_root_missing(tmp_path, monkeypatch) -> None:
    """A missing data root surfaces a clear instruction, not a stack trace."""
    import pytest

    from claritymed.ingest.vision.chest_xray_pneumonia import dataset_spec

    # Redirect the data root to an empty tmp_path so the check fires.
    monkeypatch.setattr(
        dataset_spec, "chest_xray_pneumonia_data_root", lambda: tmp_path
    )
    with pytest.raises(SystemExit, match="not present at"):
        dataset_spec._build_splits()
