"""Resolve the active ``RagStrategy`` from ``configs/retrieval.yaml``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from claritymed.core.rag.retriever import HybridRetriever
from claritymed.core.rag.schemas import (
    AgenticStrategyConfig,
    HydeStrategyConfig,
    NaiveHybridStrategyConfig,
    StrategiesConfig,
    load_retrieval_config,
)
from claritymed.core.rag.strategies.base import RagStrategy
from claritymed.core.rag.strategies.hyde import HydeStrategy
from claritymed.core.rag.strategies.naive_hybrid import NaiveHybridStrategy
from claritymed.errors import UnknownStrategyError

if TYPE_CHECKING:
    from pydantic_ai.models import Model


def build_strategy(
    retriever: HybridRetriever,
    config: StrategiesConfig | None = None,
    *,
    max_evidence: int = 5,
    model: "Model | None" = None,
) -> RagStrategy:
    """Instantiate the active strategy.

    Args:
        retriever: Pre-built HybridRetriever instance the strategy will
            delegate to.
        config: Optional StrategiesConfig override.
        max_evidence: Final cap on chunks returned to the caller.
        model: pydantic-ai Model used by strategies that issue LLM calls
            during retrieval (HyDE). Optional for naive_hybrid / agentic
            because they do not call the LLM at the retrieval layer.

    Raises:
        UnknownStrategyError: Active id has no factory branch yet (e.g.
            "graph", "raptor" reserved future plugs).
    """
    cfg = config or load_retrieval_config().strategies
    entry = cfg.resolved()
    if isinstance(entry, NaiveHybridStrategyConfig):
        return NaiveHybridStrategy(
            retriever=retriever,
            grader=entry.grader,
            max_evidence=max_evidence,
        )
    if isinstance(entry, HydeStrategyConfig):
        if model is None:
            raise ValueError(
                "hyde strategy requires a Model — pass build_strategy(..., "
                "model=...) from the CLI/TUI bootstrap"
            )
        return HydeStrategy(
            retriever=retriever,
            config=entry,
            model=model,
            max_evidence=max_evidence,
        )
    if isinstance(entry, AgenticStrategyConfig):
        # Agentic mode shares the retrieval layer with naive_hybrid; the
        # behavioural switch (skip pre-retrieval, let the tool loop drive)
        # lives in AskService. Attach ``is_agentic`` so the service can
        # detect it without reaching back into the catalog.
        strategy = NaiveHybridStrategy(
            retriever=retriever,
            grader=entry.grader,
            max_evidence=max_evidence,
        )
        strategy.is_agentic = True  # type: ignore[attr-defined]
        return strategy
    raise UnknownStrategyError(
        f"build_strategy has no factory branch for id={entry.id!r} "
        f"(see docs/plans/2026-06-07-002-feat-rag-module-plan.md "
        f"for the RagStrategy plug-in roadmap)"
    )
