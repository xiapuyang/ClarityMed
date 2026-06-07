"""Tests for ``ModelsConfig`` / ``ProviderConfig``."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from claritymed.core.schemas import ModelsConfig, ProviderConfig


def _provider(**over) -> dict:
    base = {
        "id": "ollama-openai",
        "kind": "local",
        "api": "openai",
        "base_url": "http://127.0.0.1:11434/v1",
        "model": "qwen3:14b",
    }
    base.update(over)
    return base


def test_local_provider_omits_api_key_env():
    p = ProviderConfig.model_validate(_provider())
    assert p.api_key_env is None


def test_cloud_provider_with_key_env():
    p = ProviderConfig.model_validate(
        _provider(
            id="claude",
            kind="cloud",
            api="anthropic",
            base_url="https://api.anthropic.com",
            model="claude-sonnet-4-5",
            api_key_env="ANTHROPIC_API_KEY",
        )
    )
    assert p.kind == "cloud"
    assert p.api == "anthropic"
    assert p.api_key_env == "ANTHROPIC_API_KEY"


def test_unknown_kind_rejected():
    with pytest.raises(ValidationError):
        ProviderConfig.model_validate(_provider(kind="hybrid"))


def test_unknown_api_rejected():
    with pytest.raises(ValidationError):
        ProviderConfig.model_validate(_provider(api="cohere"))


def test_provider_id_must_be_path_safe():
    with pytest.raises(ValidationError):
        ProviderConfig.model_validate(_provider(id="bad id with spaces"))


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        ProviderConfig.model_validate(_provider(temperature=0.7))


def test_models_config_happy_path():
    cfg = ModelsConfig.model_validate(
        {
            "providers": [
                _provider(),
                _provider(id="claude", kind="cloud", api="anthropic"),
            ],
            "default_provider": "ollama-openai",
        }
    )
    assert cfg.default_provider == "ollama-openai"
    assert {p.id for p in cfg.providers} == {"ollama-openai", "claude"}


def test_duplicate_provider_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate provider ids"):
        ModelsConfig.model_validate(
            {
                "providers": [_provider(), _provider()],
                "default_provider": "ollama-openai",
            }
        )


def test_default_must_exist_in_providers():
    with pytest.raises(ValidationError, match="not in providers"):
        ModelsConfig.model_validate(
            {
                "providers": [_provider()],
                "default_provider": "claude",
            }
        )


def test_providers_cannot_be_empty():
    with pytest.raises(ValidationError):
        ModelsConfig.model_validate(
            {"providers": [], "default_provider": "ollama-openai"}
        )


def test_shipped_models_yaml_parses():
    """Sanity check: the committed configs/models.yaml is valid."""
    from claritymed.stores.models import load_models

    cfg = load_models()
    ids = {p.id for p in cfg.providers}
    # Every cloud vendor the user asked for.
    assert {
        "openai",
        "deepseek",
        "gemini",
        "qwen",
        "claude",
        "kimi",
        "openrouter",
    } <= ids
    # Both local wire formats.
    assert {"ollama-openai", "ollama-anthropic"} <= ids
