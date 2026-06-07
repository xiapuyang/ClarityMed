"""Tests for ``claritymed.core.llm.client.LLMClient``.

The pydantic-ai ``Model`` is constructed for real (cheap, no network) so the
provider dispatch / api_key plumbing is exercised. ``model_request`` is
monkeypatched so no actual HTTP traffic leaves the test box.
"""

from __future__ import annotations

import pytest
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from claritymed.core.llm import ChatMessage, ChatRequest, LLMClient, MissingApiKeyError
from claritymed.core.schemas import ProviderConfig


def _local_openai() -> ProviderConfig:
    return ProviderConfig(
        id="ollama-openai",
        kind="local",
        api="openai",
        base_url="http://127.0.0.1:11434/v1",
        model="qwen3:14b",
    )


def _local_anthropic() -> ProviderConfig:
    return ProviderConfig(
        id="ollama-anthropic",
        kind="local",
        api="anthropic",
        base_url="http://127.0.0.1:11434",
        model="qwen3:14b",
    )


def _cloud_openai() -> ProviderConfig:
    return ProviderConfig(
        id="deepseek",
        kind="cloud",
        api="openai",
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        api_key_env="DEEPSEEK_API_KEY",
    )


def _cloud_anthropic() -> ProviderConfig:
    return ProviderConfig(
        id="claude",
        kind="cloud",
        api="anthropic",
        base_url="https://api.anthropic.com",
        model="claude-sonnet-4-5",
        api_key_env="ANTHROPIC_API_KEY",
    )


# ---------- construction / key resolution ----------


def test_local_provider_constructs_without_env(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    c = LLMClient(_local_openai())
    assert c.provider.id == "ollama-openai"


def test_cloud_provider_missing_key_fails_fast(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(MissingApiKeyError, match="DEEPSEEK_API_KEY"):
        LLMClient(_cloud_openai())


def test_cloud_provider_blank_key_treated_as_missing(monkeypatch):
    """Empty string is not a valid key — surface it instead of silently sending blanks."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    with pytest.raises(MissingApiKeyError):
        LLMClient(_cloud_openai())


def test_cloud_provider_with_key_constructs(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    c = LLMClient(_cloud_openai())
    assert c.provider.id == "deepseek"


def test_dispatch_picks_openai_chat_model(monkeypatch):
    from pydantic_ai.models.openai import OpenAIChatModel

    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    c = LLMClient(_cloud_openai())
    assert isinstance(c._model, OpenAIChatModel)


def test_dispatch_picks_anthropic_model(monkeypatch):
    from pydantic_ai.models.anthropic import AnthropicModel

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    c = LLMClient(_cloud_anthropic())
    assert isinstance(c._model, AnthropicModel)


def test_anthropic_local_uses_custom_base_url(monkeypatch):
    """Local Anthropic-emulator must hit our base_url, not api.anthropic.com."""
    c = LLMClient(_local_anthropic())
    assert "127.0.0.1" in str(c._model.client.base_url)


# ---------- ChatRequest -> ModelMessage translation ----------


def test_to_model_messages_packs_system_and_user_into_one_request():
    req = ChatRequest(
        messages=[
            ChatMessage(role="system", content="be brief"),
            ChatMessage(role="user", content="hi"),
        ]
    )
    msgs = LLMClient._to_model_messages(req)
    assert len(msgs) == 1
    assert isinstance(msgs[0], ModelRequest)
    assert isinstance(msgs[0].parts[0], SystemPromptPart)
    assert isinstance(msgs[0].parts[1], UserPromptPart)


def test_to_model_messages_splits_on_assistant_turn():
    req = ChatRequest(
        messages=[
            ChatMessage(role="system", content="be brief"),
            ChatMessage(role="user", content="q1"),
            ChatMessage(role="assistant", content="a1"),
            ChatMessage(role="user", content="q2"),
        ]
    )
    msgs = LLMClient._to_model_messages(req)
    assert len(msgs) == 3
    assert isinstance(msgs[0], ModelRequest)  # system + q1
    assert isinstance(msgs[1], ModelResponse)  # a1
    assert isinstance(msgs[2], ModelRequest)  # q2
    assert isinstance(msgs[1].parts[0], TextPart)
    assert msgs[1].parts[0].content == "a1"


def test_to_settings_includes_max_tokens_always():
    req = ChatRequest(messages=[ChatMessage(role="user", content="hi")], max_tokens=512)
    settings = LLMClient._to_settings(req)
    assert settings["max_tokens"] == 512
    assert "temperature" not in settings


def test_to_settings_includes_temperature_when_set():
    req = ChatRequest(
        messages=[ChatMessage(role="user", content="hi")], temperature=0.2
    )
    settings = LLMClient._to_settings(req)
    assert settings["temperature"] == 0.2


# ---------- ModelResponse -> ChatResponse translation ----------


def test_to_chat_response_concatenates_text_parts(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    c = LLMClient(_cloud_openai())
    resp = ModelResponse(
        parts=[TextPart("hello "), TextPart("world")],
        usage=RequestUsage(input_tokens=10, output_tokens=2),
        model_name="deepseek-chat",
        finish_reason="stop",
    )
    out = c._to_chat_response(resp)
    assert out.text == "hello world"
    assert out.model == "deepseek-chat"
    assert out.provider_id == "deepseek"
    assert out.stop_reason == "stop"
    assert out.usage.input_tokens == 10
    assert out.usage.output_tokens == 2


def test_to_chat_response_omits_usage_when_zero(monkeypatch):
    """Some local servers return 0/0 — that's noise, not data."""
    c = LLMClient(_local_openai())
    resp = ModelResponse(
        parts=[TextPart("ok")],
        usage=RequestUsage(input_tokens=0, output_tokens=0),
        model_name="qwen3:14b",
    )
    out = c._to_chat_response(resp)
    assert out.usage is None


def test_to_chat_response_falls_back_to_configured_model_name(monkeypatch):
    """If the server omits model_name, use whatever we configured."""
    c = LLMClient(_local_openai())
    resp = ModelResponse(parts=[TextPart("ok")], usage=RequestUsage())
    out = c._to_chat_response(resp)
    assert out.model == "qwen3:14b"


# ---------- end-to-end chat() with mocked model_request ----------


async def test_chat_round_trip(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    captured: dict = {}

    async def fake_request(model, messages, *, model_settings=None, **_):
        captured["model"] = model
        captured["messages"] = messages
        captured["settings"] = model_settings
        return ModelResponse(
            parts=[TextPart("hi back")],
            usage=RequestUsage(input_tokens=3, output_tokens=2),
            model_name="deepseek-chat",
            finish_reason="stop",
        )

    monkeypatch.setattr("claritymed.core.llm.client.model_request", fake_request)

    c = LLMClient(_cloud_openai())
    out = await c.chat(
        ChatRequest(
            messages=[
                ChatMessage(role="system", content="be brief"),
                ChatMessage(role="user", content="hi"),
            ],
            max_tokens=64,
            temperature=0.0,
        )
    )
    assert out.text == "hi back"
    assert captured["settings"]["max_tokens"] == 64
    assert captured["settings"]["temperature"] == 0.0
    assert len(captured["messages"]) == 1
    assert isinstance(captured["messages"][0], ModelRequest)
