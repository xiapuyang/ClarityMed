"""TranslationEligibility — local LLM translation then EN delegate.

All collaborators are mocked in-process: the provider resolver returns a
synthetic ``ProviderConfig``, the agent factory returns a strict stub
that records calls (or refuses to be called for EN short-circuit), and
the prompt registry is stubbed to return a fixed system prompt without
touching disk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from claritymed.core.schemas import ProviderConfig
from claritymed.core.schemas.patient import Profile
from claritymed.core.symptoms.eligibility import (
    DirectEligibility,
    EligibilityResult,
    TranslationEligibility,
)
from claritymed.core.symptoms.schemas import DatasetSpec
from claritymed.errors import (
    EligibilityStrategyConfigError,
    EligibilityStrategyUnavailableError,
)


def _dataset() -> DatasetSpec:
    return DatasetSpec(
        id="ddxplus",
        enabled=True,
        model_ids=["typed_basd_v1"],
        maxstep=8,
    )


def _profile() -> Profile:
    return Profile()


def _vocabs() -> dict[str, dict[str, frozenset[str]]]:
    return {
        "ddxplus": {
            "E_1": frozenset({"chest pain"}),
            "E_2": frozenset({"nausea"}),
            "E_3": frozenset({"shortness of breath"}),
        }
    }


def _local_provider(provider_id: str = "omlx") -> ProviderConfig:
    """Synthesise a local provider that ``is_provider_available`` will accept."""
    return ProviderConfig(
        id=provider_id,
        kind="local",
        model="qwen3:14b",
        base_url="http://127.0.0.1:8000",
    )


def _cloud_provider() -> ProviderConfig:
    return ProviderConfig(
        id="anthropic",
        kind="cloud",
        model="anthropic:claude-sonnet-4-5",
    )


class _StubRegistry:
    """Minimal PromptRegistry stand-in returning a fixed template."""

    def __init__(self, template: str = "Translate to English:") -> None:
        self._template = template

    def get(self, name: str, version: str = "latest", language: Any = None) -> str:
        return self._template


@dataclass
class _AgentRunResult:
    """Minimal pydantic-ai ``AgentRunResult`` lookalike."""

    output: str


class _StubAgent:
    """Records ``run`` calls and returns the configured translation."""

    def __init__(self, translation: str = "chest pain") -> None:
        self.translation = translation
        self.calls: list[str] = []

    async def run(self, prompt: str) -> _AgentRunResult:
        self.calls.append(prompt)
        return _AgentRunResult(output=self.translation)


class _StrictAgentNotInvoked:
    """Fails if ``run`` is invoked — used to verify EN short-circuit."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def run(self, prompt: str) -> _AgentRunResult:  # pragma: no cover
        self.calls.append(prompt)
        raise AssertionError(
            "Translation agent was invoked but EN short-circuit should have prevented it"
        )


def _make_strategy(
    *,
    agent: Any,
    provider: ProviderConfig | None = None,
    available: bool = True,
    direct: DirectEligibility | None = None,
) -> TranslationEligibility:
    """Construct the strategy with all collaborators stubbed."""
    return TranslationEligibility(
        provider_id="omlx",
        prompt_name="translate_complaint_to_en",
        direct_strategy=direct or DirectEligibility(vocabs=_vocabs()),
        agent_factory=lambda sys_prompt, prov: agent,
        provider_resolver=lambda pid: provider or _local_provider(pid),
        availability_check=lambda prov: available,
        registry=_StubRegistry(),
    )


async def test_en_short_circuits_to_direct() -> None:
    """An EN complaint must skip the LLM entirely — translating EN→EN is
    wasted latency and a possible source of drift."""
    strict_agent = _StrictAgentNotInvoked()
    strategy = _make_strategy(agent=strict_agent)
    result = await strategy.check(
        complaint="I have chest pain and nausea",
        language="en",
        profile=_profile(),
        dataset=_dataset(),
    )
    assert isinstance(result, EligibilityResult)
    assert result.eligible is True
    assert strict_agent.calls == []


async def test_zh_translation_then_direct_match() -> None:
    agent = _StubAgent(translation="chest pain and nausea")
    strategy = _make_strategy(agent=agent)
    result = await strategy.check(
        complaint="胸口疼，恶心",
        language="zh",
        profile=_profile(),
        dataset=_dataset(),
    )
    assert agent.calls == ["胸口疼，恶心"]
    assert result.eligible is True
    assert result.reason == "in_scope"


async def test_zh_translation_blank_falls_through_out_of_scope() -> None:
    """A model that returns an empty translation isn't a hard failure —
    it's just an unmatched complaint. We surface out_of_scope so the
    caller can fall back to free-text answering."""
    agent = _StubAgent(translation="   ")
    strategy = _make_strategy(agent=agent)
    result = await strategy.check(
        complaint="...",
        language="zh",
        profile=_profile(),
        dataset=_dataset(),
    )
    assert result.eligible is False
    assert result.reason == "out_of_scope"


async def test_zh_translation_gibberish_yields_out_of_scope() -> None:
    """Translation succeeded but produced nonsense that doesn't match
    any evidence vocab. Same path as a real out-of-scope complaint."""
    agent = _StubAgent(translation="banana flugelhorn")
    strategy = _make_strategy(agent=agent)
    result = await strategy.check(
        complaint="some unintelligible zh complaint",
        language="zh",
        profile=_profile(),
        dataset=_dataset(),
    )
    assert result.eligible is False
    assert result.reason == "out_of_scope"


async def test_zh_agent_raises_yields_strategy_unavailable() -> None:
    """Transient backing-LLM failure (timeout, 500) → strategy_unavailable
    so the caller knows the result is a no-op, not a real negative."""

    class _BoomAgent:
        async def run(self, prompt: str) -> _AgentRunResult:
            raise RuntimeError("upstream down")

    strategy = _make_strategy(agent=_BoomAgent())
    result = await strategy.check(
        complaint="胸口疼",
        language="zh",
        profile=_profile(),
        dataset=_dataset(),
    )
    assert result.eligible is False
    assert result.reason == "strategy_unavailable"


def test_cloud_provider_rejected_at_construct() -> None:
    """PHI must not cross the network for translation — the construct
    check is the structural defense, not a runtime branch."""
    with pytest.raises(EligibilityStrategyConfigError) as info:
        TranslationEligibility(
            provider_id="anthropic",
            prompt_name="translate_complaint_to_en",
            direct_strategy=DirectEligibility(vocabs=_vocabs()),
            agent_factory=lambda s, p: _StubAgent(),
            provider_resolver=lambda pid: _cloud_provider(),
            availability_check=lambda prov: True,
            registry=_StubRegistry(),
        )
    assert "cloud" in str(info.value).lower()


def test_local_provider_unreachable_at_construct() -> None:
    """An unreachable local provider (env var unset, server down at
    startup) fails loud rather than at first request."""
    with pytest.raises(EligibilityStrategyUnavailableError) as info:
        TranslationEligibility(
            provider_id="omlx",
            prompt_name="translate_complaint_to_en",
            direct_strategy=DirectEligibility(vocabs=_vocabs()),
            agent_factory=lambda s, p: _StubAgent(),
            provider_resolver=lambda pid: _local_provider(pid),
            availability_check=lambda prov: False,
            registry=_StubRegistry(),
        )
    assert "reachable" in str(info.value).lower()


async def test_agent_factory_receives_system_prompt_from_registry() -> None:
    """The translation prompt must be pulled from the registry (no hard-
    coded strings) and passed into the agent factory unchanged."""
    captured: dict[str, Any] = {}

    def _factory(system_prompt: str, provider: ProviderConfig) -> _StubAgent:
        captured["system_prompt"] = system_prompt
        captured["provider"] = provider
        return _StubAgent(translation="chest pain and nausea")

    strategy = TranslationEligibility(
        provider_id="omlx",
        prompt_name="translate_complaint_to_en",
        direct_strategy=DirectEligibility(vocabs=_vocabs()),
        agent_factory=_factory,
        provider_resolver=lambda pid: _local_provider(pid),
        availability_check=lambda prov: True,
        registry=_StubRegistry(template="SYS_PROMPT_TOKEN"),
    )
    await strategy.check(
        complaint="胸口疼", language="zh", profile=_profile(), dataset=_dataset()
    )
    assert captured["system_prompt"] == "SYS_PROMPT_TOKEN"
    assert captured["provider"].id == "omlx"


async def test_factory_dispatches_translation_kind() -> None:
    """End-to-end factory wiring: ``eligibility.active = "translation"``
    builds a working strategy when overrides are injected for tests."""
    from claritymed.core.symptoms.eligibility import build_eligibility_strategy
    from claritymed.core.symptoms.schemas import (
        EligibilityCatalogConfig,
        TranslationEligibilityEntry,
    )

    cfg = EligibilityCatalogConfig(
        active="translation",
        catalog=[
            TranslationEligibilityEntry(
                id="translation",
                kind="translation",
                provider_id="omlx",
                prompt_name="translate_complaint_to_en",
            )
        ],
    )
    agent = _StubAgent(translation="chest pain and nausea")
    strategy = build_eligibility_strategy(
        cfg,
        vocabs=_vocabs(),
        translation_overrides={
            "agent_factory": lambda s, p: agent,
            "provider_resolver": lambda pid: _local_provider(pid),
            "availability_check": lambda prov: True,
            "registry": _StubRegistry(),
        },
    )
    assert isinstance(strategy, TranslationEligibility)
    result = await strategy.check(
        complaint="胸口疼，恶心",
        language="zh",
        profile=_profile(),
        dataset=_dataset(),
    )
    assert result.eligible is True
