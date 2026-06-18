"""Validate ``RSNA_PNEUMONIA_DATASET`` imports + has the expected shape."""

from __future__ import annotations


def test_dataset_spec_imports_cleanly() -> None:
    """The spec module must import without side effects (no data root scan)."""
    from claritymed.ingest.vision.rsna_pneumonia.dataset_spec import (
        RSNA_PNEUMONIA_DATASET,
        RSNA_PNEUMONIA_LABELS_META,
    )

    # disease_id intentionally matches Kermany's so the bench's binary
    # collapse stays apples-to-apples across the cross-eval pair.
    assert RSNA_PNEUMONIA_DATASET.disease_id == "chest_xray_pneumonia"
    assert RSNA_PNEUMONIA_DATASET.accepted_modality == "xray"
    assert RSNA_PNEUMONIA_DATASET.labels == ("normal", "pneumonia")
    assert RSNA_PNEUMONIA_DATASET.cancer_class is False
    for label in RSNA_PNEUMONIA_DATASET.labels:
        assert label in RSNA_PNEUMONIA_LABELS_META


def test_build_splits_fails_loud_when_data_root_missing(tmp_path, monkeypatch) -> None:
    """Missing data root surfaces an instruction to run download first."""
    import pytest

    from claritymed.ingest.vision.rsna_pneumonia import dataset_spec

    monkeypatch.setattr(
        dataset_spec, "rsna_pneumonia_data_root", lambda: tmp_path / "missing"
    )
    with pytest.raises(SystemExit, match="not present at"):
        dataset_spec._build_splits()
