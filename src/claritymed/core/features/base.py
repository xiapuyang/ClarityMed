"""``FeaturePlugin`` protocol + the turn-time context handed to plugins.

A plugin is a small adapter: it declares its mode and exposes whichever
hooks that mode needs. ``AskService`` composes the turn by gathering
deterministic pre-invoke text and registering tool-mode callables;
neither the plugin nor the service knows about the others.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pydantic_ai.toolsets import AbstractToolset

    from claritymed.core.turn_state import TurnState


FeatureMode = Literal["deterministic", "tool", "agentic"]


@runtime_checkable
class PostProcessHook(Protocol):
    """Optional protocol — implement on a plugin to run logic after the
    LLM stream completes.

    ``AskService`` walks the active plugins after each turn's stream
    finishes; for any plugin whose tool was the last-called in the turn
    AND implements this protocol, it invokes ``post_process`` and uses
    the returned text as the user-visible reply.

    Today only :class:`~claritymed.orchestrator.features.symptoms_plugin.SymptomsFeature`
    implements this — the contract is intentionally generic so future
    plugins can run their own final-reply checks without an
    AskService change.

    Implementations should:

    * Be idempotent (the same input must yield the same output —
      AskService calls this once per turn).
    * Never raise on bad input; log a warning and return ``text``
      unchanged. A buggy post_process must not break the reply.
    * Treat the call as audit-only when possible: returning ``text``
      unchanged is the safe default. Mutation is reserved for cases
      where the prompt-level safety net is provably insufficient.
    """

    async def post_process(self, text: str, tool_result: dict) -> str: ...


@dataclass
class TurnContext:
    """Inputs visible to plugins at turn time.

    ``scrubbed`` is the post-PHI-scrub user input; ``deps`` carries the
    runtime state the agent will see (strategy, user_id, language,
    event queue, accumulated chunks) — typed as ``TurnState`` so core
    code never reaches into the concrete ``AskDeps`` orchestrator owns.
    Plugins write back to ``deps`` when they have side effects (e.g.
    accumulating RAG chunks for the Sources block).
    """

    scrubbed: str
    deps: "TurnState"


class FeaturePlugin(Protocol):
    """One pluggable per-turn capability.

    Implementations declare ``mode`` and implement the matching hook:

    * ``mode == "deterministic"`` → ``pre_invoke`` returns prompt text
      (empty string for no-op).
    * ``mode == "tool"`` → ``as_tool`` returns a pydantic-ai tool
      callable OR ``as_toolset`` returns a pre-built
      ``AbstractToolset`` (used when the plugin needs framework
      machinery like ``ApprovalRequiredToolset``); ``pre_invoke``
      returns ``""``.
    * ``mode == "agentic"`` → reserved; v1 raises at
      ``build_features`` time.

    A plugin can expose tools via ``as_tool`` (single callable),
    ``as_toolset`` (full toolset, e.g. ingest tools behind an approval
    gate), or both. ``AskService`` collects each side independently
    and passes them through to ``Agent`` as ``tools=`` and
    ``toolsets=`` respectively.
    """

    name: str
    mode: FeatureMode

    async def pre_invoke(self, ctx: TurnContext) -> str:
        """Return text to splice into the user prompt. ``""`` means no-op."""
        ...

    def as_tool(self) -> Callable | None:
        """Return the pydantic-ai tool callable, or ``None`` if not in tool mode."""
        ...

    def as_toolset(self) -> "AbstractToolset[Any] | None":
        """Return a pre-built toolset, or ``None`` when the plugin
        exposes no toolset (the common case — RAG, attachments etc.
        register a single callable via ``as_tool``).

        Used by the ingest-tools plugin to hand back an
        ``ApprovalRequiredToolset`` wrapping all seven write tools so
        the framework's own approval flow runs uniformly across them.
        """
        ...
