"""Unit 8: AskService + RagStrategy integration tests.

Stub strategy/provider so we never call a real embedder/reranker/qdrant.
Verifies the event order, PHI-filter behavior at the cloud boundary, and
that evidence is spliced into the agent prompt (via TestModel echoing).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic_ai.models.test import TestModel

from claritymed.core.rag.schemas import EvidenceBundle, RetrievalTrace
from claritymed.core.rag.strategies.base import RagStrategy, RetrievalContext
from claritymed.core.schemas.models import ProviderConfig
from claritymed.core.schemas.retrieval import RetrievedChunk
from claritymed.orchestrator.services import AskService
from claritymed.orchestrator.services.events import (
    Done,
    LlmCallStarted,
    LlmFirstToken,
    RetrievalCompleted,
    RetrievalFiltered,
    RetrievalStarted,
    TokenChunk,
    ToolCompleted,
    ToolStarted,
)

if TYPE_CHECKING:
    from claritymed.core.translation import TranslationProvider


def _chunk(
    *,
    text: str,
    is_phi: bool = False,
    can_cloud: bool = True,
    source: str = "system_rag",
    parent_text: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        source=source,  # type: ignore[arg-type]
        score=0.9,
        doc_id="d1",
        chunk_index=0,
        is_phi=is_phi,
        can_cloud=can_cloud,
        collection_name="statpearls_en" if source == "system_rag" else "user_rag_alice",
        parent_id="d1#p0",
        parent_text=parent_text,
        rerank_score=0.9,
    )


class StubStrategy(RagStrategy):
    def __init__(self, bundle: EvidenceBundle) -> None:
        self._bundle = bundle
        self.calls: list[RetrievalContext] = []

    async def retrieve(self, ctx: RetrievalContext) -> EvidenceBundle:
        self.calls.append(ctx)
        return self._bundle


def _provider(kind: str = "local") -> ProviderConfig:
    return ProviderConfig(id="test_provider", kind=kind, model="openai:gpt-4o")


# --- event ordering --------------------------------------------------


async def test_run_with_strategy_emits_retrieval_then_tokens_then_done():
    bundle = EvidenceBundle(
        chunks=[_chunk(text="aspirin treats pain", parent_text="aspirin parent text")],
        trace=RetrievalTrace(
            strategy="naive_hybrid",
            active_collections=["statpearls_en"],
            embed_ms=5,
            search_ms=10,
            rerank_ms=8,
            parent_expand_ms=2,
        ),
    )
    strategy = StubStrategy(bundle)
    service = AskService(
        model=TestModel(custom_output_text="answer"),
        strategy=strategy,
        provider_config=_provider("local"),
    )
    events = [ev async for ev in service.run("what is aspirin", user_id="alice")]
    types = [type(e).__name__ for e in events]
    # Retrieval events come before TokenChunk and before Done.
    assert types.index("RetrievalStarted") < types.index("TokenChunk")
    assert types.index("RetrievalCompleted") < types.index("TokenChunk")
    assert types.index("TokenChunk") < types.index("Done")
    # Pending fires first — that's the whole point of having a separate
    # event: the UI needs a signal *before* the slow embed+search+rerank
    # await, not after it returns with bundle metadata.
    assert types.index("RetrievalPending") < types.index("RetrievalStarted")
    # LlmCallStarted fires before the tool call — the LLM decides to call
    # the retrieval tool, so retrieval is always nested inside the LLM run.
    assert types.index("LlmCallStarted") < types.index("RetrievalPending")
    assert types.index("LlmCallStarted") < types.index("LlmFirstToken")
    assert types.index("LlmFirstToken") <= types.index("TokenChunk")


async def test_retrieval_started_carries_strategy_and_collections():
    bundle = EvidenceBundle(
        chunks=[_chunk(text="x")],
        trace=RetrievalTrace(
            strategy="naive_hybrid",
            active_collections=["statpearls_en", "user_rag_alice"],
        ),
    )
    service = AskService(
        model=TestModel(custom_output_text="answer"),
        strategy=StubStrategy(bundle),
        provider_config=_provider("local"),
    )
    events = [ev async for ev in service.run("q", user_id="alice")]
    started = next(e for e in events if isinstance(e, RetrievalStarted))
    assert started.strategy == "naive_hybrid"
    assert started.active_collections == ["statpearls_en", "user_rag_alice"]


# --- cloud PHI filter -----------------------------------------------


async def test_cloud_provider_filters_phi_chunks():
    bundle = EvidenceBundle(
        chunks=[
            _chunk(text="public reference", is_phi=False, can_cloud=True),
            _chunk(
                text="private user upload",
                is_phi=True,
                can_cloud=False,
                source="user_rag",
            ),
        ],
        trace=RetrievalTrace(strategy="naive_hybrid"),
    )
    service = AskService(
        model=TestModel(custom_output_text="answer"),
        strategy=StubStrategy(bundle),
        provider_config=_provider("cloud"),
    )
    events = [ev async for ev in service.run("q", user_id="alice")]
    filtered = next(e for e in events if isinstance(e, RetrievalFiltered))
    assert filtered.filtered_phi == 1
    assert filtered.kept == 1
    completed = next(e for e in events if isinstance(e, RetrievalCompleted))
    assert completed.num_chunks == 1


async def test_local_provider_skips_phi_filter():
    bundle = EvidenceBundle(
        chunks=[
            _chunk(text="public", is_phi=False, can_cloud=True),
            _chunk(text="private", is_phi=True, can_cloud=False, source="user_rag"),
        ],
        trace=RetrievalTrace(strategy="naive_hybrid"),
    )
    service = AskService(
        model=TestModel(custom_output_text="answer"),
        strategy=StubStrategy(bundle),
        provider_config=_provider("local"),
    )
    events = [ev async for ev in service.run("q", user_id="alice")]
    # No RetrievalFiltered event when nothing is filtered.
    assert not any(isinstance(e, RetrievalFiltered) for e in events)
    completed = next(e for e in events if isinstance(e, RetrievalCompleted))
    assert completed.num_chunks == 2


# --- no strategy backward compatible ---------------------------------


async def test_no_strategy_skips_retrieval_path():
    """AskService.strategy=None preserves legacy behavior — no retrieval
    events (pending/started/completed/filtered), no evidence in the prompt.
    LlmCallStarted/LlmFirstToken still fire because they're not RAG-gated."""
    from claritymed.orchestrator.services import RetrievalPending

    service = AskService(model=TestModel(custom_output_text="answer"))
    events = [ev async for ev in service.run("q", user_id="alice")]
    assert not any(isinstance(e, RetrievalPending) for e in events)
    assert not any(isinstance(e, RetrievalStarted) for e in events)
    assert not any(isinstance(e, RetrievalCompleted) for e in events)
    # LLM lifecycle events fire regardless — they're about the model
    # call, not retrieval.
    assert any(isinstance(e, LlmCallStarted) for e in events)
    assert any(isinstance(e, LlmFirstToken) for e in events)
    assert next(e for e in events if isinstance(e, Done))


async def test_llm_call_started_carries_model_and_provider():
    """LlmCallStarted payload feeds the TUI panel — must include the
    model name + provider id so the user sees *which* model they're
    waiting on, not just that *a* model is generating."""
    bundle = EvidenceBundle(
        chunks=[_chunk(text="x")],
        trace=RetrievalTrace(strategy="naive_hybrid"),
    )
    service = AskService(
        model=TestModel(custom_output_text="answer"),
        strategy=StubStrategy(bundle),
        provider_config=_provider("local"),
        provider_id="omlx",
        model_name="Qwen3.6-35B-A3B-oQ4-mtp",
    )
    events = [ev async for ev in service.run("q", user_id="alice")]
    started = next(e for e in events if isinstance(e, LlmCallStarted))
    assert started.model_name == "Qwen3.6-35B-A3B-oQ4-mtp"
    assert started.provider_id == "omlx"


# --- evidence content -----------------------------------------------


def test_evidence_format_prefers_parent_text_when_present():
    """Prompt assembler uses parent_text when available (more context),
    falling back to the child chunk text otherwise."""
    formatted = AskService._format_evidence(
        [
            _chunk(text="child A", parent_text="PARENT_A_FULL"),
            _chunk(text="child B with no parent"),
        ]
    )
    assert "PARENT_A_FULL" in formatted
    assert "child A" not in formatted  # parent preferred over child
    assert "child B with no parent" in formatted
    assert "[1]" in formatted
    assert "[2]" in formatted


def test_evidence_format_empty_returns_empty_string():
    assert AskService._format_evidence([]) == ""


def test_format_debug_collections_lists_collection_and_score():
    """_format_debug_collections builds a markdown table the LLM cannot produce
    itself — it uses the raw chunk metadata, not LLM-generated source text."""
    chunk = _chunk(text="body")
    block = AskService._format_debug_collections([chunk])
    assert "statpearls_en" in block
    assert "[1]" in block
    assert "score" in block


def test_compose_prompt_appends_evidence_with_question_label():
    out = AskService._compose_prompt("what is X?", "EVIDENCE_BLOCK")
    assert "EVIDENCE_BLOCK" in out
    assert "Question: what is X?" in out


def test_compose_prompt_no_evidence_returns_scrubbed_unchanged():
    assert AskService._compose_prompt("plain query", "") == "plain query"


def test_format_sources_uses_source_uri_when_available():
    chunk = _chunk(text="body", parent_text=None)
    chunk2 = RetrievedChunk(
        text="body",
        source="system_rag",
        score=0.9,
        doc_id="d2",
        chunk_index=0,
        is_phi=False,
        can_cloud=True,
        collection_name="statpearls_en",
        parent_id=None,
        parent_text=None,
        rerank_score=0.9,
        source_uri="https://ncbi.nlm.nih.gov/books/NBK123",
    )
    block = AskService._format_sources([chunk, chunk2])
    assert "**Sources:**" in block
    assert "[1]" in block
    assert "[2]" in block
    # source_uri takes priority over collection_name
    assert "https://ncbi.nlm.nih.gov/books/NBK123" in block


def test_format_sources_empty_returns_empty():
    assert AskService._format_sources([]) == ""


async def test_sources_block_always_injected_before_done():
    """Sources section is emitted as a TokenChunk before Done when chunks exist."""
    bundle = EvidenceBundle(
        chunks=[_chunk(text="x")], trace=RetrievalTrace(strategy="naive_hybrid")
    )
    service = AskService(
        model=TestModel(custom_output_text="answer"),
        strategy=StubStrategy(bundle),
        provider_config=_provider("local"),
    )
    events = [ev async for ev in service.run("q", user_id="alice")]
    types = [type(e).__name__ for e in events]
    # Sources token appears before Done
    token_texts = [e.text for e in events if isinstance(e, TokenChunk)]
    assert any("**Sources:**" in t for t in token_texts)
    last_token_idx = max(i for i, t in enumerate(types) if t == "TokenChunk")
    done_idx = types.index("Done")
    assert last_token_idx < done_idx


async def test_debug_mode_emits_collections_token_before_done(monkeypatch):
    """CLARITYMED_DEBUG=1 injects a TokenChunk with collection names just
    before Done so the TUI renders it in the final markdown response."""
    monkeypatch.setenv("CLARITYMED_DEBUG", "1")
    bundle = EvidenceBundle(
        chunks=[_chunk(text="x")],
        trace=RetrievalTrace(strategy="naive_hybrid"),
    )
    service = AskService(
        model=TestModel(custom_output_text="answer"),
        strategy=StubStrategy(bundle),
        provider_config=_provider("local"),
    )
    events = [ev async for ev in service.run("q", user_id="alice")]
    types = [type(e).__name__ for e in events]
    # A debug TokenChunk must appear between the last regular TokenChunk and Done.
    assert "TokenChunk" in types
    assert types.index("TokenChunk") < types.index("Done")
    # The debug block comes right before Done — find the last TokenChunk.
    last_token_idx = max(i for i, t in enumerate(types) if t == "TokenChunk")
    done_idx = types.index("Done")
    assert last_token_idx < done_idx
    last_token = events[last_token_idx]
    assert "statpearls_en" in last_token.text


async def test_debug_mode_off_no_collections_block(monkeypatch):
    """Without CLARITYMED_DEBUG the debug collections block is not emitted.

    Note: the Sources section IS emitted unconditionally (it may contain the
    collection name). The debug block is distinguished by its 'Debug — RAG
    Collections:' header and score annotations.
    """
    monkeypatch.delenv("CLARITYMED_DEBUG", raising=False)
    bundle = EvidenceBundle(
        chunks=[_chunk(text="x")],
        trace=RetrievalTrace(strategy="naive_hybrid"),
    )
    service = AskService(
        model=TestModel(custom_output_text="answer"),
        strategy=StubStrategy(bundle),
        provider_config=_provider("local"),
    )
    events = [ev async for ev in service.run("q", user_id="alice")]
    token_texts = [e.text for e in events if isinstance(e, TokenChunk)]
    assert not any("Debug" in t for t in token_texts)


# --- retrieval failure ----------------------------------------------


async def test_retrieval_failure_surfaces_error_event():
    class _BoomStrategy(RagStrategy):
        async def retrieve(self, ctx: RetrievalContext) -> EvidenceBundle:
            raise RuntimeError("simulated retrieval crash")

    service = AskService(
        model=TestModel(custom_output_text="never seen"),
        strategy=_BoomStrategy(),
        provider_config=_provider("local"),
    )
    events = [ev async for ev in service.run("q", user_id="alice")]
    from claritymed.orchestrator.services.events import Error

    err = next(e for e in events if isinstance(e, Error))
    assert err.error_type == "retrieval_failed"


# --- whitelist passthrough ------------------------------------------


@pytest.mark.parametrize("whitelist", [None, [], ["statpearls_en"]])
async def test_user_whitelist_threads_through_to_strategy(whitelist):
    bundle = EvidenceBundle(
        chunks=[_chunk(text="x")], trace=RetrievalTrace(strategy="naive_hybrid")
    )
    strategy = StubStrategy(bundle)
    service = AskService(
        model=TestModel(custom_output_text="answer"),
        strategy=strategy,
        provider_config=_provider("local"),
        user_whitelist=whitelist,
    )
    [ev async for ev in service.run("q", user_id="alice")]
    assert strategy.calls[0].user_whitelist == whitelist


# --- query translation via TranslationProvider -----------------------


def _translation_svc(output: str = "hemoglobin 105") -> "TranslationProvider":
    from claritymed.core.translation import LLMTranslationProvider

    return LLMTranslationProvider(TestModel(custom_output_text=output))


async def test_translate_queries_auto_fires_for_cross_lingual():
    """zh session + en collection triggers translation automatically — no env var."""
    bundle = EvidenceBundle(
        chunks=[_chunk(text="x")], trace=RetrievalTrace(strategy="naive_hybrid")
    )
    strategy = StubStrategy(bundle)
    service = AskService(
        model=TestModel(custom_output_text="answer"),
        strategy=strategy,
        provider_config=_provider("local"),
        language="zh",
        translation_service=_translation_svc("hemoglobin 105"),
    )
    [ev async for ev in service.run("我血红蛋白105", user_id="alice")]
    ctx = strategy.calls[0]
    assert ctx.query == "hemoglobin 105"
    assert ctx.language == "zh"  # routing language stays zh


async def test_translate_queries_skipped_when_no_service():
    """Without a TranslationProvider the tool-extracted query reaches the strategy."""
    bundle = EvidenceBundle(
        chunks=[_chunk(text="x")], trace=RetrievalTrace(strategy="naive_hybrid")
    )
    strategy = StubStrategy(bundle)
    service = AskService(
        model=TestModel(custom_output_text="answer"),
        strategy=strategy,
        provider_config=_provider("local"),
        language="zh",
        # no translation_service
    )
    [ev async for ev in service.run("我血红蛋白105", user_id="alice")]
    # TestModel generates query='a' for the tool — no translation, 'a' reaches strategy.
    assert strategy.calls[0].query == "a"


async def test_translate_queries_only_fires_for_collection_mismatch():
    """en session (collection lang == session lang) → no translation even with service."""
    bundle = EvidenceBundle(
        chunks=[_chunk(text="x")], trace=RetrievalTrace(strategy="naive_hybrid")
    )
    strategy = StubStrategy(bundle)
    service = AskService(
        model=TestModel(custom_output_text="answer"),
        strategy=strategy,
        provider_config=_provider("local"),
        language="en",
        translation_service=_translation_svc(),
    )
    [ev async for ev in service.run("hemoglobin 105", user_id="alice")]
    # No language mismatch → no translation → TestModel's tool arg 'a' reaches strategy.
    assert strategy.calls[0].query == "a"


async def test_translate_queries_fallback_on_failure():
    """When translate_query raises, the original query is used and retrieval proceeds."""
    bundle = EvidenceBundle(
        chunks=[_chunk(text="x")], trace=RetrievalTrace(strategy="naive_hybrid")
    )
    strategy = StubStrategy(bundle)
    svc = _translation_svc()

    async def _boom(text, *, target_lang, context="general"):  # noqa: ANN001
        raise RuntimeError("simulated translation failure")

    svc._call = _boom  # type: ignore[method-assign]

    service = AskService(
        model=TestModel(custom_output_text="answer"),
        strategy=strategy,
        provider_config=_provider("local"),
        language="zh",
        translation_service=svc,
    )
    events = [ev async for ev in service.run("我血红蛋白105", user_id="alice")]
    # Translation failed → fallback to tool-extracted query 'a' (TestModel arg).
    assert strategy.calls[0].query == "a"
    assert any(isinstance(e, Done) for e in events)


async def test_translate_queries_emits_tool_events():
    """ToolStarted/ToolCompleted appear for the translation step, before retrieval."""
    bundle = EvidenceBundle(
        chunks=[_chunk(text="x")], trace=RetrievalTrace(strategy="naive_hybrid")
    )
    strategy = StubStrategy(bundle)
    service = AskService(
        model=TestModel(custom_output_text="answer"),
        strategy=strategy,
        provider_config=_provider("local"),
        language="zh",
        translation_service=_translation_svc("hemoglobin 105"),
    )
    events = [ev async for ev in service.run("我血红蛋白105", user_id="alice")]
    types = [type(e).__name__ for e in events]

    assert "ToolStarted" in types
    assert "ToolCompleted" in types
    started = next(e for e in events if isinstance(e, ToolStarted))
    completed = next(e for e in events if isinstance(e, ToolCompleted))
    assert started.tool_name == "translate.query"
    assert completed.tool_name == "translate.query"
    assert completed.summary == "done"
    assert types.index("ToolStarted") < types.index("RetrievalPending")


# --- auto output language -------------------------------------------


async def test_auto_language_off_uses_session_lang(monkeypatch):
    """Without CLARITYMED_AUTO_LANGUAGE the output language follows --lang."""
    monkeypatch.delenv("CLARITYMED_AUTO_LANGUAGE", raising=False)
    bundle = EvidenceBundle(
        chunks=[_chunk(text="x")], trace=RetrievalTrace(strategy="naive_hybrid")
    )
    service = AskService(
        model=TestModel(custom_output_text="answer"),
        strategy=StubStrategy(bundle),
        provider_config=_provider("local"),
        language="en",
    )
    events = [ev async for ev in service.run("我血红蛋白105", user_id="alice")]
    assert any(isinstance(e, Done) for e in events)
