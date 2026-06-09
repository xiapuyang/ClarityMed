"""Pluggable per-turn capabilities (RAG, future: vision, symptoms).

A ``FeaturePlugin`` declares one of three interaction shapes via its
``mode`` attribute and exposes the corresponding hook(s):

* ``deterministic`` — ``pre_invoke`` runs before the LLM; its text is
  spliced into the user prompt.
* ``tool`` — ``as_tool`` returns a pydantic-ai tool callable the agent
  can invoke 1..N times during streaming.
* ``agentic`` — reserved for the future state-graph workflow; v1 raises
  ``NotImplementedError`` if any active feature picks it.

Each feature owns its own config namespace (e.g. ``rag``); the global
factory ``build_features(retrieval_cfg)`` builds the list of active
plugins for an ask turn. Adding ``vision`` or ``symptoms`` later is one
new ``FeaturePlugin`` class + one config branch.
"""

from claritymed.core.features.base import FeaturePlugin, TurnContext
from claritymed.core.features.factory import build_features

__all__ = [
    "FeaturePlugin",
    "TurnContext",
    "build_features",
]
