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

from claritymed.core.llm import build_model, build_model_settings
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
    ANTHROPIC_API_KEY itself — we do nothing.

    Unit 5 wraps cloud-kind providers in ``PhiAssertionModel(LoggingModel(…))``;
    unwrap two layers before checking the base type.
    """
    from pydantic_ai.models.anthropic import AnthropicModel

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    m = build_model(
        _provider(kind="cloud", model="anthropic:claude-sonnet-4-5", base_url=None)
    )
    assert isinstance(_unwrap_phi_logging(m), AnthropicModel)


def test_no_base_url_uses_infer_model_for_openai(monkeypatch):
    from pydantic_ai.models.openai import OpenAIChatModel

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    m = build_model(_provider(kind="cloud", model="openai:gpt-4o", base_url=None))
    assert isinstance(_unwrap_phi_logging(m), OpenAIChatModel)


def _unwrap_phi_logging(model):
    """Cloud providers wrap base → LoggingModel → PhiAssertionModel; peel."""
    from claritymed.core.observability.llm_logger import LoggingModel
    from claritymed.core.phi.assertion_model import PhiAssertionModel

    if isinstance(model, PhiAssertionModel):
        model = model._inner  # noqa: SLF001
    if isinstance(model, LoggingModel):
        model = model._inner  # noqa: SLF001
    return model


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


# ---------- build_model_settings ----------


def test_settings_none_when_thinking_unset():
    """No thinking → no settings → caller omits model_settings from Agent."""
    assert build_model_settings(_provider()) is None


def test_settings_pass_thinking_true_through():
    """``thinking: true`` → ModelSettings(thinking=True). pydantic-ai's
    Model layer translates it to each vendor's native field."""
    s = build_model_settings(_provider(thinking=True))
    assert s == {"thinking": True}


def test_settings_pass_thinking_false_through():
    """``thinking: false`` is *explicit* disable — distinct from omitted."""
    s = build_model_settings(_provider(thinking=False))
    assert s == {"thinking": False}


@pytest.mark.parametrize("level", ["minimal", "low", "medium", "high", "xhigh"])
def test_settings_pass_effort_level_through(level):
    """Effort levels go through verbatim; pydantic-ai owns the vendor mapping."""
    s = build_model_settings(_provider(thinking=level))
    assert s == {"thinking": level}


def test_settings_work_for_stock_cloud_provider(monkeypatch):
    """Stock cloud entry (no base_url) — same translation, no special-casing."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    p = _provider(
        id="claude",
        kind="cloud",
        model="anthropic:claude-sonnet-4-5",
        base_url=None,
        thinking="high",
    )
    assert build_model_settings(p) == {"thinking": "high"}


def test_cloud_models_share_phi_guard_singleton(monkeypatch):
    """Two ``build_model`` calls must return models that share the same
    ``PhiGuard`` instance.  Without this, each call creates a fresh
    ``ScrubService`` which lazy-loads the 809 MB ONNX pipeline again —
    memory grows by ~809 MB per call in benchmark / eval loops.
    """
    from claritymed.core.phi.assertion_model import PhiAssertionModel
    from claritymed.core.phi.guard import invalidate_guard_cache

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    invalidate_guard_cache()

    p = _provider(kind="cloud", model="anthropic:claude-sonnet-4-5", base_url=None)
    m1 = build_model(p)
    m2 = build_model(p)

    assert isinstance(m1, PhiAssertionModel)
    assert isinstance(m2, PhiAssertionModel)
    assert m1._guard is m2._guard, (  # noqa: SLF001
        "build_model() returned two different PhiGuard instances — "
        "ScrubService / ONNX pipeline would be duplicated per call"
    )
