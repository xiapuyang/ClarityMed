"""Tests for the per-condition curated catalog loader + startup validator."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from claritymed.core.symptoms.conditions_catalog import (
    ConditionEntry,
    SymptomsConditionsCatalog,
    enumerate_dataset_condition_ids,
    validate_conditions_catalog,
)
from claritymed.core.symptoms.schemas import DatasetSpec
from claritymed.errors import SymptomsCatalogValidationError


def _write_catalog(dir_path: Path, lang: str, payload: dict) -> Path:
    lang_dir = dir_path / lang
    lang_dir.mkdir(parents=True, exist_ok=True)
    path = lang_dir / "symptoms_conditions.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _minimal_entry() -> dict:
    return {
        "display_name": "Pneumonia",
        "report": "Infection of the lung tissue.",
        "suggestion": "Seek same-day medical evaluation.",
    }


def test_entry_rejects_blank_fields():
    from pydantic import ValidationError

    payload = _minimal_entry()
    payload["display_name"] = ""
    with pytest.raises(ValidationError):
        ConditionEntry.model_validate(payload)


def test_entry_rejects_extra_fields():
    """Legacy ``citations:`` on a payload must be rejected — the field was
    removed with the card-simplification pass; carrying it silently would
    hide stale data."""
    from pydantic import ValidationError

    payload = _minimal_entry()
    payload["citations"] = [{"source_id": "x", "title": "y", "authority_tier": 1}]
    with pytest.raises(ValidationError):
        ConditionEntry.model_validate(payload)


def test_loader_returns_entry_when_present(tmp_path):
    _write_catalog(tmp_path, "en", {"conditions": {"pneumonia": _minimal_entry()}})
    catalog = SymptomsConditionsCatalog(base_dir=tmp_path)
    entry = catalog.get("pneumonia", "en")
    assert entry is not None
    assert entry.display_name == "Pneumonia"
    assert entry.suggestion.startswith("Seek same-day")


def test_loader_returns_none_for_uncovered(tmp_path):
    _write_catalog(tmp_path, "en", {"conditions": {"pneumonia": _minimal_entry()}})
    catalog = SymptomsConditionsCatalog(base_dir=tmp_path)
    assert catalog.get("influenza", "en") is None


def test_loader_missing_file_returns_empty(tmp_path):
    catalog = SymptomsConditionsCatalog(base_dir=tmp_path)
    assert catalog.language_entries("en") == {}


def test_loader_rejects_wrong_top_level(tmp_path):
    _write_catalog(tmp_path, "en", {"conditions": ["a", "b"]})
    catalog = SymptomsConditionsCatalog(base_dir=tmp_path)
    with pytest.raises(SymptomsCatalogValidationError):
        catalog.language_entries("en")


def test_loader_reports_validation_error_with_id(tmp_path):
    bad = _minimal_entry()
    bad["suggestion"] = ""  # blank suggestion violates min_length=1
    _write_catalog(tmp_path, "en", {"conditions": {"pneumonia": bad}})
    catalog = SymptomsConditionsCatalog(base_dir=tmp_path)
    with pytest.raises(SymptomsCatalogValidationError) as exc:
        catalog.language_entries("en")
    assert "pneumonia" in str(exc.value)


def test_validator_passes_on_full_coverage(tmp_path):
    _write_catalog(tmp_path, "en", {"conditions": {"pneumonia": _minimal_entry()}})
    _write_catalog(tmp_path, "zh", {"conditions": {"pneumonia": _minimal_entry()}})
    catalog = SymptomsConditionsCatalog(base_dir=tmp_path)
    validate_conditions_catalog(catalog, ["pneumonia"])


def test_validator_reports_missing_language(tmp_path):
    _write_catalog(tmp_path, "en", {"conditions": {"pneumonia": _minimal_entry()}})
    catalog = SymptomsConditionsCatalog(base_dir=tmp_path)
    with pytest.raises(SymptomsCatalogValidationError) as exc:
        validate_conditions_catalog(catalog, ["pneumonia"])
    assert "zh:pneumonia" in str(exc.value)


def test_validator_reports_missing_slug(tmp_path):
    _write_catalog(tmp_path, "en", {"conditions": {"pneumonia": _minimal_entry()}})
    _write_catalog(tmp_path, "zh", {"conditions": {"pneumonia": _minimal_entry()}})
    catalog = SymptomsConditionsCatalog(base_dir=tmp_path)
    with pytest.raises(SymptomsCatalogValidationError) as exc:
        validate_conditions_catalog(catalog, ["pneumonia", "influenza"])
    assert "influenza" in str(exc.value)


def test_validator_empty_condition_set_is_noop(tmp_path):
    catalog = SymptomsConditionsCatalog(base_dir=tmp_path)
    # No conditions to check, empty catalog OK.
    validate_conditions_catalog(catalog, [])


def _ddxplus_spec() -> DatasetSpec:
    return DatasetSpec(
        id="ddxplus",
        enabled=True,
        model_ids=["typed_basd_v2"],
    )


def test_enumerate_dataset_condition_ids_reads_i18n_yaml(tmp_path):
    ds = _ddxplus_spec()
    # Write a minimal symptoms_ddxplus.yaml at both langs matching the
    # DatasetSpec's resolved_i18n_prefix() convention.
    payload = {
        "symptoms": {
            "ddxplus": {
                "conditions": {
                    "pneumonia": {"name": "Pneumonia"},
                    "influenza": {"name": "Influenza"},
                }
            }
        }
    }
    for lang in ("en", "zh"):
        (tmp_path / lang).mkdir()
        (tmp_path / lang / "symptoms_ddxplus.yaml").write_text(
            yaml.safe_dump(payload), encoding="utf-8"
        )
    ids = enumerate_dataset_condition_ids(ds, i18n_dir=tmp_path)
    assert ids == {"pneumonia", "influenza"}


def test_enumerate_dataset_condition_ids_bilingual_union(tmp_path):
    """Slug present in only one language still enumerates."""
    ds = _ddxplus_spec()
    en_payload = {
        "symptoms": {"ddxplus": {"conditions": {"pneumonia": {"name": "Pneumonia"}}}}
    }
    zh_payload = {
        "symptoms": {
            "ddxplus": {
                "conditions": {
                    "pneumonia": {"name": "肺炎"},
                    "influenza": {"name": "流感"},
                }
            }
        }
    }
    (tmp_path / "en").mkdir()
    (tmp_path / "zh").mkdir()
    (tmp_path / "en" / "symptoms_ddxplus.yaml").write_text(
        yaml.safe_dump(en_payload), encoding="utf-8"
    )
    (tmp_path / "zh" / "symptoms_ddxplus.yaml").write_text(
        yaml.safe_dump(zh_payload), encoding="utf-8"
    )
    ids = enumerate_dataset_condition_ids(ds, i18n_dir=tmp_path)
    assert ids == {"pneumonia", "influenza"}


def test_repo_ddxplus_catalog_full_coverage():
    """Regression guard: the shipped catalog covers every enabled dataset's
    condition_id in both en and zh. The plugin construct depends on this."""
    from claritymed.config import load_symptoms_config

    cfg = load_symptoms_config()
    condition_ids: set[str] = set()
    for ds in cfg.datasets:
        if ds.enabled:
            condition_ids.update(enumerate_dataset_condition_ids(ds))
    assert condition_ids, "no enabled dataset condition_ids found"
    validate_conditions_catalog(SymptomsConditionsCatalog(), condition_ids)
