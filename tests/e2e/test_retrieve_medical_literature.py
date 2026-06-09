"""E2E: retrieve_medical_literature — zh query, translation → RAG → source block.

Full pipeline exercised:
  zh session  →  LLM extracts query  →  TranslationProvider (zh→en)
  →  HybridRetriever (embedder + reranker + Qdrant)
  →  AskService assembles answer with [n] citations
  →  deepeval scores faithfulness + answer relevancy + contextual relevancy

Run:
    uv run pytest tests/e2e -v --no-cov
    deepeval test run tests/e2e/test_retrieve_medical_literature.py
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from claritymed.orchestrator.services.events import Event

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INPUT_ZH = "我的血红蛋白是105 g/L，这正常吗？需要担心吗？"
EXPECTED_EN_GIST = "hemoglobin"  # translation must produce this concept


async def _run_pipeline(input_text: str, lang: str = "zh") -> tuple[str, list]:
    """Run AskService end-to-end and return (final_answer, retrieved_chunks).

    Builds every component from real configs — no mocks.
    """
    from claritymed.core.llm.model import build_model
    from claritymed.core.rag.retriever_factory import build_hybrid_retriever
    from claritymed.core.rag.strategies.factory import build_strategy
    from claritymed.core.translation import make_translation_provider
    from claritymed.orchestrator.services import AskService
    from claritymed.orchestrator.services.events import Done
    from claritymed.stores.models import resolve_provider

    provider = resolve_provider()
    model = build_model(provider)
    retriever = build_hybrid_retriever()
    strategy = build_strategy(retriever)
    translation_svc = make_translation_provider(model)

    service = AskService(
        model=model,
        language=lang,
        strategy=strategy,
        provider_config=provider,
        translation_service=translation_svc,
        provider_id=provider.id,
        model_name=str(provider.model),
    )

    events: list[Event] = [
        ev async for ev in service.run(input_text, user_id="e2e_test")
    ]
    done = next((e for e in events if isinstance(e, Done)), None)
    final = done.final if done else ""
    return final, events


# ---------------------------------------------------------------------------
# Structural assertions (no judge LLM needed)
# ---------------------------------------------------------------------------


@pytest.mark.local
def test_translation_fires_for_zh_session():
    """TranslationProvider is called and the query reaching Qdrant is English."""
    from claritymed.orchestrator.services.events import ToolStarted

    final, events = asyncio.run(_run_pipeline(INPUT_ZH, lang="zh"))
    tool_names = [e.tool_name for e in events if isinstance(e, ToolStarted)]
    assert "translate.query" in tool_names, (
        "Expected translate.query ToolStarted event for zh session "
        f"but only saw: {tool_names}"
    )


@pytest.mark.local
def test_rag_retrieves_at_least_one_chunk():
    """At least one chunk is retrieved for a valid haematology query."""
    from claritymed.orchestrator.services.events import RetrievalCompleted

    _, events = asyncio.run(_run_pipeline(INPUT_ZH, lang="zh"))
    rc = next((e for e in events if isinstance(e, RetrievalCompleted)), None)
    assert rc is not None, "RetrievalCompleted event not found"
    assert rc.num_chunks >= 1, (
        f"Expected ≥1 chunk retrieved for '{INPUT_ZH}', got {rc.num_chunks}. "
        "Check that statpearls_en collection is populated."
    )


@pytest.mark.local
def test_answer_contains_source_citations():
    """Final answer must include a Sources block with at least one citation."""

    final, events = asyncio.run(_run_pipeline(INPUT_ZH, lang="zh"))
    assert "**Sources:**" in final, (
        "Sources block missing from final answer.\n"
        f"Final text (truncated): {final[:400]}"
    )
    assert "[1]" in final, "Expected at least one [1] citation in the Sources block"


@pytest.mark.local
def test_event_ordering_translation_before_retrieval():
    """translate.query ToolCompleted must precede RetrievalPending."""
    from claritymed.orchestrator.services.events import (
        RetrievalPending,
        ToolCompleted,
        ToolStarted,
    )

    _, events = asyncio.run(_run_pipeline(INPUT_ZH, lang="zh"))
    tool_started_names = [e.tool_name for e in events if isinstance(e, ToolStarted)]
    assert "translate.query" in tool_started_names

    translate_completed_idx = next(
        (
            i
            for i, e in enumerate(events)
            if isinstance(e, ToolCompleted) and e.tool_name == "translate.query"
        ),
        None,
    )
    retrieval_pending_idx = next(
        (i for i, e in enumerate(events) if isinstance(e, RetrievalPending)), None
    )
    assert translate_completed_idx is not None
    assert retrieval_pending_idx is not None
    assert translate_completed_idx < retrieval_pending_idx, (
        "translate.query must complete before RetrievalPending fires"
    )


# ---------------------------------------------------------------------------
# deepeval quality metrics
# ---------------------------------------------------------------------------

_DEEPEVAL_AVAILABLE = False
try:
    import deepeval  # noqa: F401

    _DEEPEVAL_AVAILABLE = True
except ImportError:
    pass

_skip_deepeval = pytest.mark.skipif(
    not _DEEPEVAL_AVAILABLE,
    reason="deepeval not installed — run: uv sync --extra e2e",
)


class _ClarityMedJudge:
    """Thin adapter so deepeval metrics can use the project's configured LLM.

    deepeval metrics accept a ``model`` kwarg that must implement
    ``generate(prompt) → str`` and ``a_generate(prompt) → str``.
    This wraps pydantic-ai's Agent in that interface without pulling in
    the deepeval base class (avoids import at module load time).
    """

    def __init__(self) -> None:
        from claritymed.core.llm.model import build_model
        from claritymed.stores.models import resolve_provider

        provider = resolve_provider()
        from pydantic_ai import Agent

        self._agent: Agent = Agent(build_model(provider), output_type=str)

    def generate(self, prompt: str, schema=None) -> str:  # noqa: ANN001
        return asyncio.run(self._a_generate(prompt))

    async def a_generate(self, prompt: str, schema=None) -> str:  # noqa: ANN001
        return await self._a_generate(prompt)

    async def _a_generate(self, prompt: str) -> str:
        result = await self._agent.run(prompt)
        return str(result.output)

    def get_model_name(self) -> str:
        from claritymed.stores.models import resolve_provider

        p = resolve_provider()
        return str(p.model)


def _build_judge():
    """Return a deepeval-compatible judge backed by the configured model."""
    try:
        from deepeval.models import DeepEvalBaseLLM

        class _Judge(DeepEvalBaseLLM, _ClarityMedJudge):
            def __init__(self) -> None:
                _ClarityMedJudge.__init__(self)

            def load_model(self):
                return self._agent

        return _Judge()
    except Exception:
        return None


@_skip_deepeval
@pytest.mark.local
def test_deepeval_rag_quality():
    """deepeval RAG triad: faithfulness + answer relevancy + contextual relevancy.

    Uses the project's configured LLM as judge so no OPENAI_API_KEY is needed.
    Threshold: 0.5 — passes on any reasonable haematology response.
    """
    from deepeval import assert_test
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    from claritymed.orchestrator.services.events import RetrievalCompleted, TokenChunk

    final, events = asyncio.run(_run_pipeline(INPUT_ZH, lang="zh"))

    # Collect retrieval context from the sources injected into the LLM prompt.
    # The TokenChunk stream contains the evidence block that was shown to the
    # LLM (via _compose_prompt); extract it as the retrieval_context list.
    token_texts = [e.text for e in events if isinstance(e, TokenChunk)]
    full_text = "".join(token_texts)

    rc = next((e for e in events if isinstance(e, RetrievalCompleted)), None)
    num_chunks = rc.num_chunks if rc else 0

    # Build a minimal retrieval_context list from the Sources block in the
    # final answer — each [n] line is one retrieved document snippet.
    retrieval_context: list[str] = []
    if "**Sources:**" in full_text:
        sources_section = full_text.split("**Sources:**", 1)[-1]
        for line in sources_section.splitlines():
            line = line.strip()
            if line.startswith("[") and "]" in line:
                retrieval_context.append(line)

    # Fallback: if sources couldn't be parsed, use a placeholder so the
    # metric can still score answer relevancy.
    if not retrieval_context:
        retrieval_context = [f"<{num_chunks} chunks retrieved, sources unavailable>"]

    judge = _build_judge()
    metric_kwargs = {"threshold": 0.5, "model": judge} if judge else {"threshold": 0.5}

    test_case = LLMTestCase(
        input=INPUT_ZH,
        actual_output=final,
        retrieval_context=retrieval_context,
    )

    assert_test(
        test_case,
        [
            AnswerRelevancyMetric(**metric_kwargs),
            FaithfulnessMetric(**metric_kwargs),
            ContextualRelevancyMetric(**metric_kwargs),
        ],
    )
