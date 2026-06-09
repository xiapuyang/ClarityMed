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
    from claritymed.core.events import Event

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INPUT_ZH = "我的血红蛋白是105 g/L，这正常吗？需要担心吗？"
EXPECTED_EN_GIST = "hemoglobin"  # translation must produce this concept


async def _run_pipeline(
    input_text: str, lang: str = "zh", provider_id: str | None = None
) -> tuple[str, list, list]:
    """Run AskService end-to-end and return (final_answer, events, chunks).

    Builds every component from real configs — no mocks.
    Pass provider_id to override the default from models.yaml.
    Returns (Done.final, all events, retrieved RetrievedChunk objects).
    service.last_chunks holds the RetrievedChunk list after the run.
    """
    from claritymed.core.llm.model import build_model
    from claritymed.core.rag.retriever_factory import build_hybrid_retriever
    from claritymed.core.rag.strategies.factory import build_strategy
    from claritymed.core.translation import make_translation_provider
    from claritymed.orchestrator.services import AskService
    from claritymed.core.events import Done
    from claritymed.stores.models import resolve_provider

    provider = resolve_provider(override=provider_id)
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
    return final, events, service.last_chunks


# ---------------------------------------------------------------------------
# Structural assertions (no judge LLM needed)
# ---------------------------------------------------------------------------


@pytest.mark.local
@pytest.mark.flaky(reruns=2, reruns_delay=5)
def test_translation_fires_for_zh_session(e2e_provider_id):
    """TranslationProvider is called and the query reaching Qdrant is English."""
    from claritymed.core.events import ToolStarted

    final, events, _ = asyncio.run(
        _run_pipeline(INPUT_ZH, lang="zh", provider_id=e2e_provider_id)
    )
    tool_names = [e.tool_name for e in events if isinstance(e, ToolStarted)]
    assert "translate.query" in tool_names, (
        "Expected translate.query ToolStarted event for zh session "
        f"but only saw: {tool_names}"
    )


@pytest.mark.local
@pytest.mark.flaky(reruns=2, reruns_delay=5)
def test_rag_retrieves_at_least_one_chunk(e2e_provider_id):
    """At least one chunk is retrieved for a valid haematology query."""
    from claritymed.core.events import RetrievalCompleted

    _, events, _ = asyncio.run(
        _run_pipeline(INPUT_ZH, lang="zh", provider_id=e2e_provider_id)
    )
    rc = next((e for e in events if isinstance(e, RetrievalCompleted)), None)
    assert rc is not None, "RetrievalCompleted event not found"
    assert rc.num_chunks >= 1, (
        f"Expected ≥1 chunk retrieved for '{INPUT_ZH}', got {rc.num_chunks}. "
        "Check that statpearls_en collection is populated."
    )


@pytest.mark.local
@pytest.mark.flaky(reruns=2, reruns_delay=5)
def test_answer_contains_source_citations(e2e_provider_id):
    """Full streamed output must include a Sources block with at least one citation.

    The Sources block is emitted as a TokenChunk just before Done — it is not
    included in Done.final (which holds only the raw LLM response).
    """
    from claritymed.core.events import TokenChunk

    _, events, _ = asyncio.run(
        _run_pipeline(INPUT_ZH, lang="zh", provider_id=e2e_provider_id)
    )
    full_text = "".join(e.text for e in events if isinstance(e, TokenChunk))
    assert "**Sources:**" in full_text, (
        "Sources block missing from streamed output.\n"
        f"Full text (truncated): {full_text[:400]}"
    )
    assert "[1]" in full_text, "Expected at least one [1] citation in the Sources block"


@pytest.mark.local
@pytest.mark.flaky(reruns=2, reruns_delay=5)
def test_event_ordering_translation_before_retrieval(e2e_provider_id):
    """translate.query ToolCompleted must precede RetrievalPending."""
    from claritymed.core.events import (
        RetrievalPending,
        ToolCompleted,
        ToolStarted,
    )

    _, events, _ = asyncio.run(
        _run_pipeline(INPUT_ZH, lang="zh", provider_id=e2e_provider_id)
    )
    tool_started_names = [e.tool_name for e in events if isinstance(e, ToolStarted)]
    if "retrieve_medical_literature" not in tool_started_names:
        pytest.skip(
            "LLM answered without calling retrieve_medical_literature — "
            "ordering test not applicable for this run"
        )
    assert "translate.query" in tool_started_names, (
        "retrieve_medical_literature was called but translate.query was not. "
        f"Tool calls: {tool_started_names}"
    )

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

    def _init_agent(self, provider_id: str | None = None) -> None:
        from claritymed.core.llm.model import build_model
        from claritymed.stores.models import resolve_provider
        from pydantic_ai import Agent

        provider = resolve_provider(override=provider_id)
        self._agent: Agent = Agent(build_model(provider), output_type=str)
        self._provider_model = str(provider.model)

    def generate(self, prompt: str, schema=None) -> str:  # noqa: ANN001
        return asyncio.run(self._a_generate(prompt))

    async def a_generate(self, prompt: str, schema=None) -> str:  # noqa: ANN001
        return await self._a_generate(prompt)

    async def _a_generate(self, prompt: str) -> str:
        result = await self._agent.run(prompt)
        return str(result.output)

    def get_model_name(self) -> str:
        return self._provider_model


def _build_judge(provider_id: str | None = None):
    """Return a deepeval-compatible judge backed by the configured model."""
    try:
        from deepeval.models import DeepEvalBaseLLM

        # _ClarityMedJudge must come first in MRO so its concrete generate/
        # a_generate shadow DeepEvalBaseLLM's @abstractmethod declarations.
        class _Judge(_ClarityMedJudge, DeepEvalBaseLLM):
            def __init__(self, pid: str | None) -> None:
                # Set up the pydantic-ai agent first so load_model() can return it.
                _ClarityMedJudge._init_agent(self, provider_id=pid)
                # DeepEvalBaseLLM.__init__ calls load_model() and sets self.name.
                DeepEvalBaseLLM.__init__(self)

            def load_model(self):
                return self._agent

            def get_model_name(self) -> str:
                return _ClarityMedJudge.get_model_name(self)

        return _Judge(provider_id)
    except Exception as exc:
        import warnings

        warnings.warn(f"Could not build deepeval judge: {exc}", stacklevel=2)
        return None


@_skip_deepeval
@pytest.mark.local
@pytest.mark.xfail(
    strict=False,
    reason="Local LLM judge may not follow deepeval JSON schema reliably",
)
def test_deepeval_rag_quality(e2e_provider_id):
    """deepeval RAG triad: faithfulness + answer relevancy + contextual relevancy.

    Uses the project's configured LLM as judge so no OPENAI_API_KEY is needed.
    Threshold: 0.3 — set low because local models have limited capability.
    """
    from deepeval import assert_test
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
    )
    from deepeval.test_case import LLMTestCase

    final, events, chunks = asyncio.run(
        _run_pipeline(INPUT_ZH, lang="zh", provider_id=e2e_provider_id)
    )

    # Use actual retrieved chunk text as retrieval_context so deepeval's
    # faithfulness and contextual-relevancy metrics have real evidence to judge.
    retrieval_context: list[str] = [
        c.parent_text or c.text for c in chunks if (c.parent_text or c.text)
    ]
    if not retrieval_context:
        retrieval_context = ["<no chunks retrieved>"]

    judge = _build_judge(e2e_provider_id)
    metric_kwargs = {"threshold": 0.3, "model": judge} if judge else {"threshold": 0.3}

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
