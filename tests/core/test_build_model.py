"""Tests for ``claritymed.core.llm.model.build_model``.

Two paths in ``build_model``, mirrored by two test groups:

1. **No ``base_url``** — pure delegation to pydantic-ai's ``infer_model``.
   We never re-implement the prefix → Model + Provider + env-var lookup.
2. **``base_url`` set** — always ``OllamaProvider`` (the only pydantic-ai
   provider that does not require an API key). ``api_key_env`` is read
   here and only here; declared-but-unset raises ``MissingApiKeyError``.
"""

from __future__ import annotations

import pytest

from claritymed.core.llm import build_model
from claritymed.core.schemas import ProviderConfig
from claritymed.errors import MissingApiKeyError


def _provider(**over) -> ProviderConfig:
    base = {
        "id": "test",
        "kind": "local",
        "model": "qwen3:14b",
        "base_url": "http://127.0.0.1:11434/v1",
    }
    base.update(over)
    return ProviderConfig.model_validate(base)


# ---------- delegation to infer_model (no base_url) ----------


def test_no_base_url_uses_infer_model_for_anthropic(monkeypatch):
    """Stock anthropic entry: infer_model picks AnthropicModel and reads
    ANTHROPIC_API_KEY itself — we do nothing."""
    from pydantic_ai.models.anthropic import AnthropicModel

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    m = build_model(
        _provider(kind="cloud", model="anthropic:claude-sonnet-4-5", base_url=None)
    )
    assert isinstance(m, AnthropicModel)


def test_no_base_url_uses_infer_model_for_openai(monkeypatch):
    from pydantic_ai.models.openai import OpenAIChatModel

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    m = build_model(_provider(kind="cloud", model="openai:gpt-4o", base_url=None))
    assert isinstance(m, OpenAIChatModel)


def test_missing_cloud_key_surfaces_pydantic_ai_error(monkeypatch):
    """We never re-implement the env check for stock providers. pydantic-ai
    raises its own UserError when the env var is unset; that's the contract."""
    from pydantic_ai.exceptions import UserError

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(UserError, match="DEEPSEEK_API_KEY"):
        build_model(
            _provider(kind="cloud", model="deepseek:deepseek-chat", base_url=None)
        )


# ---------- base_url path: always OllamaProvider ----------


def test_local_with_base_url_uses_ollama_provider():
    """Bare model name + base_url → OllamaProvider, no env var needed."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.ollama import OllamaProvider

    m = build_model(_provider())
    assert isinstance(m, OpenAIChatModel)
    assert isinstance(m._provider, OllamaProvider)
    assert "127.0.0.1:11434" in str(m._provider.base_url)


def test_mlx_style_endpoint_uses_ollama_provider():
    """MLX server is just another OpenAI-compatible local endpoint."""
    from pydantic_ai.providers.ollama import OllamaProvider

    m = build_model(
        _provider(
            id="mlx",
            model="mlx-community/Llama-3.2-3B-Instruct-4bit",
            base_url="http://127.0.0.1:8080/v1",
        )
    )
    assert isinstance(m._provider, OllamaProvider)
    assert "8080" in str(m._provider.base_url)


def test_api_key_env_passes_key_through(monkeypatch):
    """When api_key_env is declared and set, the value is handed to OllamaProvider."""
    from pydantic_ai.providers.ollama import OllamaProvider

    monkeypatch.setenv("MY_LOCAL_KEY", "secret-123")
    m = build_model(_provider(api_key_env="MY_LOCAL_KEY"))
    assert isinstance(m._provider, OllamaProvider)
    # OllamaProvider stores api_key on the underlying AsyncOpenAI client.
    assert m._provider.client.api_key == "secret-123"


def test_api_key_env_declared_but_unset_raises(monkeypatch):
    """Fail loud — declaring api_key_env means 'this endpoint needs auth',
    silent fallback to OllamaProvider's placeholder would hide the misconfig."""
    monkeypatch.delenv("MY_LOCAL_KEY", raising=False)
    with pytest.raises(MissingApiKeyError, match="MY_LOCAL_KEY"):
        build_model(_provider(api_key_env="MY_LOCAL_KEY"))


def test_api_key_env_blank_treated_as_missing(monkeypatch):
    """Empty string is not a valid key."""
    monkeypatch.setenv("MY_LOCAL_KEY", "")
    with pytest.raises(MissingApiKeyError):
        build_model(_provider(api_key_env="MY_LOCAL_KEY"))


def test_no_api_key_env_means_no_auth(monkeypatch):
    """When api_key_env is None, we hand OllamaProvider no key — it uses
    its built-in placeholder. Works for local servers that ignore auth."""
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    m = build_model(_provider(api_key_env=None))
    # Whatever OllamaProvider's placeholder is, it should be non-empty and
    # not look like a real API key.
    key = m._provider.client.api_key
    assert key and "not-set" in key.lower()
