"""``FeaturePlugin`` protocol + the turn-time context handed to plugins.

A plugin is a small adapter: it declares its mode and exposes whichever
hooks that mode needs. ``AskService`` composes the turn by gathering
deterministic pre-invoke text and registering tool-mode callables;
neither the plugin nor the service knows about the others.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Literal, Protocol

if TYPE_CHECKING:
    from claritymed.core.turn_state import TurnState


FeatureMode = Literal["deterministic", "tool", "agentic"]


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
      callable; ``pre_invoke`` returns ``""``.
    * ``mode == "agentic"`` → reserved; v1 raises at
      ``build_features`` time.
    """

    name: str
    mode: FeatureMode

    async def pre_invoke(self, ctx: TurnContext) -> str:
        """Return text to splice into the user prompt. ``""`` means no-op."""
        ...

    def as_tool(self) -> Callable | None:
        """Return the pydantic-ai tool callable, or ``None`` if not in tool mode."""
        ...
