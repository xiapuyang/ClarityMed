"""Shared :class:`AskService` constructor used by TUI and web hosts.

Both hosts need the same toolset stack (vision, symptoms, translation,
profile_context, RAG mode) but differ in two things they own themselves:

* **Channels** — TUI uses Textual modals; web uses an in-memory
  rendezvous future bridge. Each host builds and passes its own
  :class:`PromptChannel` / :class:`ToolApprovalChannel`.
* **RAG strategy** — TUI caches one strategy per session (qdrant clients
  outlive a turn); web has no session-cache layer yet. The factory takes
  an optional pre-built ``strategy``; ``None`` disables RAG retrieval
  for this :class:`AskService` (the agent can still answer from chat
  context + tools).

Why this lives under ``orchestrator/services/`` and not ``core/``: it
imports ``orchestrator.features.vision_plugin`` and
``orchestrator.features.symptoms_plugin`` — orchestrator may depend on
core but not vice versa (see CLAUDE.md "core/ 不许反向依赖
core/orchestrator/").
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from claritymed.orchestrator.services.ask_service import AskService

if TYPE_CHECKING:
    from pydantic_ai.models import Model

    from claritymed.core.rag.strategies.base import RagStrategy
    from claritymed.core.schemas import ProviderConfig
    from claritymed.orchestrator.interaction.channel import PromptChannel
    from claritymed.orchestrator.interaction.tool_approval_channel import (
        ToolApprovalChannel,
    )
    from claritymed.orchestrator.services.chat_session import ChatSession

logger = logging.getLogger(__name__)


def build_ask_service(
    *,
    model: "Model",
    language: str,
    chat_session: "ChatSession | None",
    provider: "ProviderConfig",
    prompt_channel: "PromptChannel | None" = None,
    tool_approval_channel: "ToolApprovalChannel | None" = None,
    strategy: "RagStrategy | None" = None,
) -> AskService:
    """Build an ``AskService`` with the full host-shared toolset stack.

    Args:
        model: pydantic-ai ``Model`` for the main LLM calls.
        language: per-turn language (``"en"`` / ``"zh"``); host resolves
            this from its user-facing source (StatusBar in TUI,
            ``account.language`` in web).
        chat_session: persistence target. ``None`` for headless one-shot
            usage; both TUI and web always pass a real session.
        provider: resolved :class:`ProviderConfig`; supplies ``id`` /
            ``model`` / ``kind`` for the audit row and PHI gate.
        prompt_channel: host's ``ask_user_question`` rendezvous channel.
            ``None`` means the host is non-interactive — the tool body
            falls back to a plain hint instead of blocking.
        tool_approval_channel: host's per-tool PHI write-approval channel.
            ``None`` omits the ingest toolset entirely (no UI to host the
            modal), mirroring ``prompt_channel``'s gate.
        strategy: pre-built RAG retriever. ``None`` disables RAG for this
            service instance; pass a cached strategy from the host when
            qdrant client lifetime should span multiple turns.

    Reads ``configs/app.yaml`` (``profile_context.mode``) and
    ``configs/retrieval.yaml`` (``rag.mode``) so the choice of
    deterministic-vs-tool context lives in config, not code.
    """
    from claritymed.config import load_yaml
    from claritymed.core.rag import load_retrieval_config
    from claritymed.core.translation import make_translation_provider
    from claritymed.orchestrator.features.symptoms_plugin import (
        make_symptoms_factory,
    )
    from claritymed.orchestrator.features.vision_plugin import make_vision_factory

    rag_mode = load_retrieval_config().rag.mode
    profile_context_mode = (
        load_yaml("app.yaml").get("profile_context", {}).get("mode", "deterministic")
    )

    def _session_id() -> str | None:
        return chat_session.session_id if chat_session is not None else None

    return AskService(
        model=model,
        language=language,
        chat_session=chat_session,
        provider_id=provider.id,
        model_name=provider.model,
        strategy=strategy,
        provider_config=provider,
        translation_service=make_translation_provider(model, phi_kind=provider.kind),
        rag_mode=rag_mode,
        profile_context_mode=profile_context_mode,
        prompt_channel=prompt_channel,
        tool_approval_channel=tool_approval_channel,
        symptoms_factory=make_symptoms_factory(),
        vision_factory=make_vision_factory(get_session_id=_session_id),
    )


def build_rag_strategy(model: "Model | None" = None) -> "RagStrategy | None":
    """Build a process-wide :class:`RagStrategy` from ``retrieval.yaml``.

    Returns ``None`` when ``rag.enabled=false`` so callers can fast-path
    without RAG. Otherwise constructs:

    * one :class:`HybridRetriever` (embedder / reranker / system stores
      shared by all users; per-user qdrant clients cached *inside* the
      retriever's factory closures);
    * one :class:`RagStrategy` over that retriever, bound to ``model``
      for HyDE-style strategies (other strategies ignore it).

    Sync because:

    * the retriever construction itself is fast (no network I/O — just
      object wiring);
    * the qdrant *local-mode* client only acquires its file lock when a
      collection is first touched at query time, not at retriever
      construction, so we don't need ``asyncio.to_thread`` here;
    * TUI's existing ``_strategy_for_session`` is sync, and matching its
      shape keeps the cache-and-reuse pattern unchanged.

    Process-wide singleton is correct: the retriever's per-user qdrant
    client cache (see ``_make_user_store_factory``) handles per-user
    isolation; the strategy itself holds no user state.
    """
    from claritymed.core.rag import build_hybrid_retriever, load_retrieval_config
    from claritymed.core.rag.strategies import build_strategy

    cfg = load_retrieval_config()
    if not cfg.rag.enabled:
        return None
    retriever = build_hybrid_retriever(cfg)
    return build_strategy(
        retriever,
        config=cfg.strategies,
        max_evidence=cfg.rag.max_evidence,
        model=model,
    )


__all__ = ["build_ask_service", "build_rag_strategy"]
