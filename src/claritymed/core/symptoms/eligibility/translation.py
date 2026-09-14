"""LLM-translation eligibility strategy.

When the user's complaint is in a different language than the
dataset's evidence vocab (``DatasetSpec.native_language``), translate
the complaint into the dataset's native language via a ``kind: local``
pydantic-ai ``Agent`` and then run the translated complaint through an
injected :class:`DirectEligibility`. The matching logic lives in one
place — this module is glue.

PHI hygiene: the provider's ``kind`` must be ``"local"`` (see
``ProviderConfig.kind``). A cloud provider would send PHI off-device,
so construct raises :class:`~claritymed.errors.EligibilityStrategyConfigError`
when the resolved provider is cloud-bound. This is a deliberate
*construct-time* failure — postponing the check until the first call
would let a sub-second provider swap silently degrade hygiene.

Runtime PHI hygiene: an unreachable local provider raises
:class:`~claritymed.errors.EligibilityStrategyUnavailableError` at
construct so the plugin can fall through to another strategy.
``pydantic_ai`` transport errors during ``run`` are caught and
re-raised as ``EligibilityStrategyUnavailableError`` — the strategy is
not a hard requirement; the caller's contract is "fall through to
free-text answer", not "crash the request".
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol, runtime_checkable

from claritymed.core.llm.model import build_model, build_model_settings
from claritymed.core.prompts.registry import PromptRegistry
from claritymed.core.rag.terms.base import ConceptLanguage
from claritymed.core.schemas import ProviderConfig
from claritymed.core.schemas.patient import Profile
from claritymed.core.symptoms.eligibility.base import (
    EligibilityResult,
    EligibilityStrategy,
)
from claritymed.core.symptoms.eligibility.direct import DirectEligibility
from claritymed.core.symptoms.schemas import DatasetSpec
from claritymed.errors import (
    EligibilityStrategyConfigError,
    EligibilityStrategyUnavailableError,
)
from claritymed.stores.models import is_provider_available, resolve_provider

logger = logging.getLogger(__name__)

# Default max_tokens for the translation Agent. Catalog entry can lower
# this; raising past 1024 risks the model hallucinating extra prose
# instead of returning a tight translation.
_DEFAULT_MAX_TOKENS = 256


@runtime_checkable
class _AgentLike(Protocol):
    """Minimal pydantic-ai Agent surface used by this strategy.

    Declared as a Protocol so tests can inject a stub without importing
    the heavy ``pydantic_ai.Agent`` class graph.
    """

    async def run(self, prompt: str) -> Any: ...


# Callable that turns a system-prompt template + resolved provider into
# an Agent. Default uses ``pydantic_ai.Agent`` directly; tests pass a
# stub. Keeps this module free of an explicit ``pydantic_ai`` import
# inside the hot path.
AgentFactory = Callable[[str, ProviderConfig], _AgentLike]


def _default_agent_factory(system_prompt: str, provider: ProviderConfig) -> _AgentLike:
    """Build a pydantic-ai ``Agent`` for the resolved provider."""
    # Local import: ``pydantic_ai`` is a heavy dependency; importing at
    # module load would slow down every place that touches the
    # eligibility package, even when the active strategy isn't
    # translation.
    from pydantic_ai import Agent

    model = build_model(provider)
    settings = build_model_settings(provider)
    if settings is None:
        return Agent(model=model, system_prompt=system_prompt, output_type=str)
    return Agent(
        model=model,
        system_prompt=system_prompt,
        output_type=str,
        model_settings=settings,
    )


class TranslationEligibility(EligibilityStrategy):
    """Translate the complaint to EN via a local LLM, then delegate matching.

    The injected :class:`DirectEligibility` is the single source of truth
    for the matching rule — this module only adds the translation step.
    Tests verify EN short-circuit by injecting a strict mock agent that
    fails if invoked, then asserting it wasn't called.
    """

    def __init__(
        self,
        *,
        provider_id: str,
        prompt_name: str,
        direct_strategy: DirectEligibility,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        agent_factory: AgentFactory = _default_agent_factory,
        provider_resolver: Callable[
            [str], ProviderConfig
        ] = lambda pid: resolve_provider(override=pid),
        availability_check: Callable[[ProviderConfig], bool] = is_provider_available,
        registry: PromptRegistry | None = None,
    ) -> None:
        provider = provider_resolver(provider_id)
        if provider.kind != "local":
            raise EligibilityStrategyConfigError(
                f"TranslationEligibility requires a local provider; "
                f"{provider_id!r} has kind={provider.kind!r}. "
                f"PHI must not cross the network for symptom translation."
            )
        if not availability_check(provider):
            raise EligibilityStrategyUnavailableError(
                f"TranslationEligibility provider {provider_id!r} is not "
                f"reachable (env var / endpoint not configured). Set the "
                f"backing env var or pick a different eligibility strategy."
            )
        self._provider = provider
        self._prompt_name = prompt_name
        self._direct = direct_strategy
        self._max_tokens = max_tokens
        self._agent_factory = agent_factory
        self._registry = registry or PromptRegistry()

    async def check(
        self,
        complaint: str,
        language: ConceptLanguage,
        profile: Profile,
        dataset: DatasetSpec,
    ) -> EligibilityResult:
        """Translate the complaint to the dataset's native language then run direct matching."""
        # Same-language short-circuit — skip the LLM entirely when the
        # complaint already speaks the dataset's vocab language. The
        # injected direct strategy's check is already async.
        target = dataset.native_language
        if language == target:
            return await self._direct.check(complaint, language, profile, dataset)

        try:
            translated = await self._translate(complaint, target_language=target)
        except EligibilityStrategyUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 — propagate as typed error
            logger.warning(
                "translation eligibility: provider %s failed (%s); "
                "degrading to strategy_unavailable",
                self._provider.id,
                exc,
            )
            return EligibilityResult(eligible=False, reason="strategy_unavailable")

        if not translated.strip():
            return EligibilityResult(eligible=False, reason="out_of_scope")
        return await self._direct.check(translated, target, profile, dataset)

    async def _translate(self, complaint: str, *, target_language: str) -> str:
        """Run the Agent and return the model's text output.

        The prompt YAML indexes its ``languages`` map by **target**
        language (the dataset's vocab language), so we fetch with
        ``language=target_language``. EN target → English body
        instructing "translate to English"; ZH target → Chinese body
        instructing "translate to Chinese". The user's chat language
        does not enter this lookup.
        """
        from pydantic_ai.settings import ModelSettings

        system_prompt = self._registry.get(
            self._prompt_name,
            language=target_language,  # type: ignore[arg-type]
        )
        agent = self._agent_factory(system_prompt, self._provider)
        result = await agent.run(
            complaint, model_settings=ModelSettings(max_tokens=self._max_tokens)
        )
        return getattr(result, "output", str(result))
