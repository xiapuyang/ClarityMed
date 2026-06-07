"""Tests for ``ModelsConfig`` / ``ProviderConfig``."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from claritymed.core.schemas import ModelsConfig, ProviderConfig


def _provider(**over) -> dict:
    base = {
        "id": "ollama",
        "kind": "local",
        "model": "qwen3:14b",
        "base_url": "http://127.0.0.1:11434/v1",
    }
    base.update(over)
    return base


# ---------- happy paths ----------


def test_local_with_base_url_accepts_bare_model_name():
    p = ProviderConfig.model_validate(_provider())
    assert p.model == "qwen3:14b"
    assert p.api_key_env is None


def test_cloud_stock_requires_prefix():
    """No base_url → pydantic-ai needs '<prefix>:<model>'."""
    p = ProviderConfig.model_validate(
        {"id": "claude", "kind": "cloud", "model": "anthropic:claude-sonnet-4-5"}
    )
    assert p.kind == "cloud"
    assert p.base_url is None


def test_api_key_env_accepted_with_base_url():
    p = ProviderConfig.model_validate(_provider(api_key_env="MY_LOCAL_KEY"))
    assert p.api_key_env == "MY_LOCAL_KEY"


# ---------- model-shape validator ----------


def test_bare_model_without_base_url_rejected():
    """Stock path needs '<prefix>:<model>' so infer_model can dispatch."""
    with pytest.raises(ValidationError, match="<provider>:<model>"):
        ProviderConfig.model_validate({"id": "x", "kind": "cloud", "model": "gpt-4o"})


def test_prefix_model_without_base_url_accepted():
    p = ProviderConfig.model_validate(
        {"id": "x", "kind": "cloud", "model": "openai:gpt-4o"}
    )
    assert p.model == "openai:gpt-4o"


# ---------- api_key_env x base_url validator ----------


def test_api_key_env_without_base_url_rejected():
    """Stock providers read their own env vars; a second layer would just
    confuse where the key comes from."""
    with pytest.raises(ValidationError, match="api_key_env"):
        ProviderConfig.model_validate(
            {
                "id": "x",
                "kind": "cloud",
                "model": "openai:gpt-4o",
                "api_key_env": "MY_KEY",
            }
        )


# ---------- thinking field ----------


def test_thinking_defaults_to_none():
    """Omitting ``thinking`` means 'use vendor default' — no settings emitted."""
    p = ProviderConfig.model_validate(_provider())
    assert p.thinking is None


def test_thinking_accepts_bool():
    """``True`` = enable with vendor default effort; ``False`` = explicitly disable."""
    p_on = ProviderConfig.model_validate(_provider(thinking=True))
    p_off = ProviderConfig.model_validate(_provider(thinking=False))
    assert p_on.thinking is True
    assert p_off.thinking is False


@pytest.mark.parametrize("level", ["minimal", "low", "medium", "high", "xhigh"])
def test_thinking_accepts_effort_levels(level):
    """The five canonical effort labels pydantic-ai accepts."""
    p = ProviderConfig.model_validate(_provider(thinking=level))
    assert p.thinking == level


def test_thinking_rejects_unknown_level():
    """Typos like 'extreme' must fail at YAML load, not at first request."""
    with pytest.raises(ValidationError):
        ProviderConfig.model_validate(_provider(thinking="extreme"))


def test_thinking_rejects_integer():
    """Raw token budgets aren't supported — use the effort levels instead.
    A future escape hatch would go through extra_body, not this field."""
    with pytest.raises(ValidationError):
        ProviderConfig.model_validate(_provider(thinking=8000))


# ---------- generic field validation ----------


def test_unknown_kind_rejected():
    with pytest.raises(ValidationError):
        ProviderConfig.model_validate(_provider(kind="hybrid"))


def test_provider_id_must_be_path_safe():
    with pytest.raises(ValidationError):
        ProviderConfig.model_validate(_provider(id="bad id with spaces"))


def test_extra_field_rejected():
    """``api`` was dropped — stale YAML keys should fail loudly."""
    with pytest.raises(ValidationError):
        ProviderConfig.model_validate(_provider(api="openai"))


# ---------- ModelsConfig ----------


def test_models_config_happy_path():
    cfg = ModelsConfig.model_validate(
        {
            "providers": [
                _provider(),
                {
                    "id": "claude",
                    "kind": "cloud",
                    "model": "anthropic:claude-sonnet-4-5",
                },
            ],
            "default_provider": "ollama",
        }
    )
    assert cfg.default_provider == "ollama"
    assert {p.id for p in cfg.providers} == {"ollama", "claude"}


def test_duplicate_provider_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate provider ids"):
        ModelsConfig.model_validate(
            {
                "providers": [_provider(), _provider()],
                "default_provider": "ollama",
            }
        )


def test_default_must_exist_in_providers():
    with pytest.raises(ValidationError, match="not in providers"):
        ModelsConfig.model_validate(
            {"providers": [_provider()], "default_provider": "claude"}
        )


def test_providers_cannot_be_empty():
    with pytest.raises(ValidationError):
        ModelsConfig.model_validate({"providers": [], "default_provider": "ollama"})


def test_shipped_models_yaml_parses():
    """Sanity check: the committed configs/models.yaml is valid."""
    from claritymed.stores.models import load_models

    cfg = load_models()
    ids = {p.id for p in cfg.providers}
    assert {
        "openai",
        "deepseek",
        "gemini",
        "qwen",
        "claude",
        "kimi",
        "openrouter",
        "ollama",
        "omlx",
    } <= ids
