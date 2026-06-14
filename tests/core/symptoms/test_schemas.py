"""Validate ``configs/symptoms.yaml`` parses + every fail-loud path fires."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from claritymed.config import load_symptoms_config
from claritymed.core.symptoms.schemas import (
    DatasetSpec,
    EligibilityCatalogConfig,
    ModelSpec,
    SymptomsConfig,
    TranslationEligibilityEntry,
)
from claritymed.errors import (
    EligibilityStrategyConfigError,
    EligibilityStrategyUnavailableError,
    SymptomsServerUnreachableError,
    UnknownDatasetError,
    UnknownEligibilityStrategyError,
)

_DUMMY_SHA = "a" * 64


def _cfg(**overrides) -> SymptomsConfig:
    payload = {
        "datasets": [
            {
                "id": "ddxplus",
                "enabled": True,
                "model_ids": ["typed_basd_v1"],
                "maxstep": 8,
            }
        ],
        "models": [
            {
                "id": "typed_basd_v1",
                "algorithm_module": "typed_basd",
                "weights_subpath": "ddxplus/typed_basd_v1",
                "manifest_sha256": _DUMMY_SHA,
            }
        ],
        "eligibility": {
            "active": "direct",
            "catalog": [{"id": "direct", "kind": "direct"}],
        },
    }
    payload.update(overrides)
    return SymptomsConfig.model_validate(payload)


# --- happy paths -----------------------------------------------------------


def test_shipped_configs_symptoms_yaml_loads() -> None:
    """The repo's configs/symptoms.yaml is parseable + valid.

    Catches drift between the YAML and the schema during code review.
    """
    cfg = load_symptoms_config()
    assert cfg.datasets[0].id == "ddxplus"
    assert cfg.eligibility.resolved().id == "direct"


def test_resolved_eligibility_returns_active_entry() -> None:
    cfg = _cfg(
        eligibility={
            "active": "translation",
            "catalog": [
                {"id": "direct", "kind": "direct"},
                {
                    "id": "translation",
                    "kind": "translation",
                    "provider_id": "omlx",
                    "prompt_name": "translate_complaint",
                },
            ],
        }
    )
    entry = cfg.eligibility.resolved()
    assert isinstance(entry, TranslationEligibilityEntry)
    assert entry.provider_id == "omlx"


# --- fail-loud paths -------------------------------------------------------


def test_eligibility_typo_raises_unknown_strategy() -> None:
    """KTD-4: catalog + active resolution mirrors TermServiceConfig."""
    with pytest.raises(UnknownEligibilityStrategyError):
        EligibilityCatalogConfig.model_validate(
            {
                "active": "trasnaltion",  # noqa intentional typo
                "catalog": [{"id": "direct", "kind": "direct"}],
            }
        )


def test_dataset_references_unknown_model_id_rejected() -> None:
    """Cross-reference: every entry in dataset.model_ids must resolve."""
    with pytest.raises(ValidationError) as excinfo:
        _cfg(
            datasets=[
                {
                    "id": "ddxplus",
                    "model_ids": ["missing_model"],
                    "maxstep": 8,
                }
            ],
        )
    assert "missing_model" in str(excinfo.value)


def test_dataset_multi_model_ids_all_must_resolve() -> None:
    with pytest.raises(ValidationError) as excinfo:
        _cfg(
            datasets=[
                {
                    "id": "ddxplus",
                    "model_ids": ["typed_basd_v1", "absent"],
                    "maxstep": 8,
                }
            ],
        )
    assert "absent" in str(excinfo.value)


def test_dataset_model_ids_must_be_unique() -> None:
    with pytest.raises(ValidationError) as excinfo:
        _cfg(
            datasets=[
                {
                    "id": "ddxplus",
                    "model_ids": ["typed_basd_v1", "typed_basd_v1"],
                    "maxstep": 8,
                }
            ],
        )
    assert "unique" in str(excinfo.value).lower()


def test_dataset_default_model_selection_is_first() -> None:
    cfg = _cfg()
    assert cfg.datasets[0].model_selection == "first"
    assert cfg.datasets[0].primary_model_id() == "typed_basd_v1"


def test_partial_min_confidence_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        DatasetSpec.model_validate(
            {
                "id": "ddxplus",
                "model_ids": ["typed_basd_v1"],
                "maxstep": 8,
                "partial_min_confidence": 1.5,
            }
        )


def test_maxstep_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        DatasetSpec.model_validate(
            {"id": "ddxplus", "model_ids": ["typed_basd_v1"], "maxstep": 0}
        )


def test_weights_subpath_absolute_path_rejected() -> None:
    """KTD-6: weights live under CLARITYMED_HOME/models/symptoms/."""
    with pytest.raises(ValidationError) as excinfo:
        ModelSpec.model_validate(
            {
                "id": "typed_basd_v1",
                "algorithm_module": "typed_basd",
                "weights_subpath": "/tmp/evil",
                "manifest_sha256": _DUMMY_SHA,
            }
        )
    assert "relative" in str(excinfo.value).lower()


def test_weights_subpath_traversal_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        ModelSpec.model_validate(
            {
                "id": "typed_basd_v1",
                "algorithm_module": "typed_basd",
                "weights_subpath": "../etc/passwd",
                "manifest_sha256": _DUMMY_SHA,
            }
        )
    assert ".." in str(excinfo.value)


def test_manifest_sha256_wrong_length_rejected() -> None:
    with pytest.raises(ValidationError):
        ModelSpec.model_validate(
            {
                "id": "typed_basd_v1",
                "algorithm_module": "typed_basd",
                "weights_subpath": "ddxplus/v1",
                "manifest_sha256": "abc",
            }
        )


def test_duplicate_dataset_ids_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        _cfg(
            datasets=[
                {"id": "ddxplus", "model_ids": ["typed_basd_v1"], "maxstep": 8},
                {"id": "ddxplus", "model_ids": ["typed_basd_v1"], "maxstep": 8},
            ],
        )
    assert "unique" in str(excinfo.value).lower()


def test_unknown_eligibility_kind_rejected_by_discriminator() -> None:
    with pytest.raises(ValidationError):
        EligibilityCatalogConfig.model_validate(
            {
                "active": "bogus",
                "catalog": [{"id": "bogus", "kind": "graph_neural_net"}],
            }
        )


# --- error class smoke -----------------------------------------------------


def test_all_symptoms_error_types_importable() -> None:
    """The 5 typed errors documented by the plan must exist + inherit right."""
    assert issubclass(UnknownEligibilityStrategyError, KeyError)
    assert issubclass(UnknownDatasetError, KeyError)
    assert issubclass(SymptomsServerUnreachableError, RuntimeError)
    assert issubclass(EligibilityStrategyConfigError, RuntimeError)
    assert issubclass(EligibilityStrategyUnavailableError, RuntimeError)


# --- i18n key helpers ------------------------------------------------------


def test_dataset_spec_default_i18n_prefix_derives_from_id() -> None:
    cfg = _cfg()
    ds = cfg.datasets[0]
    assert ds.resolved_i18n_prefix() == "symptoms.ddxplus"


def test_dataset_spec_explicit_i18n_prefix_wins() -> None:
    cfg = _cfg(
        datasets=[
            {
                "id": "ddxplus",
                "model_ids": ["typed_basd_v1"],
                "maxstep": 8,
                "i18n_key_prefix": "custom.bundle",
            }
        ]
    )
    ds = cfg.datasets[0]
    assert ds.resolved_i18n_prefix() == "custom.bundle"


def test_dataset_spec_question_key_pattern() -> None:
    cfg = _cfg()
    ds = cfg.datasets[0]
    assert ds.question_key("E_91") == "symptoms.ddxplus.E_91.question"


def test_dataset_spec_value_key_pattern() -> None:
    cfg = _cfg()
    ds = cfg.datasets[0]
    assert ds.value_key("E_55", "V_14") == "symptoms.ddxplus.E_55.values.V_14"


def test_dataset_spec_condition_name_key_pattern() -> None:
    cfg = _cfg()
    ds = cfg.datasets[0]
    assert (
        ds.condition_name_key("spontaneous_pneumothorax")
        == "symptoms.ddxplus.conditions.spontaneous_pneumothorax.name"
    )


def test_dataset_spec_default_binary_keys_are_globals() -> None:
    """Default Yes/No keys point at the global symptoms.binary namespace.

    A second dataset doesn't have to ship its own Yes/No translations.
    """
    cfg = _cfg()
    ds = cfg.datasets[0]
    assert ds.binary_yes_key == "symptoms.binary.yes"
    assert ds.binary_no_key == "symptoms.binary.no"
