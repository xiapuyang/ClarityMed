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
from claritymed.core.events import (
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
    doc_id: str = "d1",
) -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        source=source,  # type: ignore[arg-type]
        score=0.9,
        doc_id=doc_id,
        chunk_index=0,
        is_phi=is_phi,
        can_cloud=can_cloud,
        collection_name="statpearls_en" if source == "system_rag" else "user_rag_alice",
        parent_id=f"{doc_id}#p0",
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
            _chunk(text="child A", parent_text="PARENT_A_FULL", doc_id="d1"),
            _chunk(text="child B with no parent", doc_id="d2"),
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


def test_format_debug_collections_renders_turn_context_preamble():
    """When turn-context kwargs are supplied the debug block prepends
    a ``key: `value``` preamble above the collection list. Pasting a
    UI screenshot into a bug report should carry the same request_id
    that audit.log / access.log use."""
    chunk = _chunk(text="body")
    block = AskService._format_debug_collections(
        [chunk],
        request_id="20260706185349E47C5447",
        user_id="alice",
        provider_id="omlx",
        model_name="mlx-community/Llama-3.2-3B",
        language="en",
        sensitivity="off",
    )
    assert "request_id" in block and "20260706185349E47C5447" in block
    assert "user_id" in block and "alice" in block
    assert "omlx" in block and "Llama-3.2-3B" in block
    assert "language" in block and "en" in block
    assert "sensitivity" in block and "off" in block
    # Collections list still rendered below the preamble.
    assert "statpearls_en" in block
    assert "[1]" in block


def test_format_debug_collections_omits_missing_context_lines():
    """Historical two-arg call (no kwargs) still works — no preamble
    section renders when nothing was passed."""
    chunk = _chunk(text="body")
    block = AskService._format_debug_collections([chunk])
    assert "request_id" not in block
    assert "provider" not in block
    assert "sensitivity" not in block
    assert "statpearls_en" in block


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


def test_format_sources_never_leaks_local_filesystem_paths():
    """Regression: ``source_uri`` set to a local path (typical for
    disk-ingested PDFs) must not surface the developer's home
    directory to end users. When ``doc_title`` is available we prefer
    it; otherwise we fall back to the basename."""
    with_title = RetrievedChunk(
        text="body",
        source="system_rag",
        score=0.9,
        doc_id="d1",
        chunk_index=0,
        is_phi=False,
        can_cloud=True,
        collection_name="ats_idsa_pneumonia_guidelines_en",
        parent_id=None,
        parent_text=None,
        rerank_score=0.9,
        source_uri="/Users/sharp/projects/ClarityMed/data/download/ATS-IDSA CAP Guidelines.pdf",
        doc_title="ATS IDSA CAP Guidelines",
    )
    without_title = RetrievedChunk(
        text="body",
        source="system_rag",
        score=0.9,
        doc_id="d2",
        chunk_index=0,
        is_phi=False,
        can_cloud=True,
        collection_name="ats_idsa_pneumonia_guidelines_en",
        parent_id=None,
        parent_text=None,
        rerank_score=0.9,
        source_uri="/Users/sharp/projects/ClarityMed/data/download/Some Study.pdf",
        doc_title=None,
    )
    block = AskService._format_sources([with_title, without_title])
    assert "/Users/sharp" not in block, (
        f"local path leaked into Sources block:\n{block}"
    )
    assert "/data/download/" not in block
    assert "ATS IDSA CAP Guidelines" in block
    # Basename fallback for the no-title chunk.
    assert "Some Study.pdf" in block


def test_display_source_uri_keeps_real_urls():
    assert (
        AskService._display_source_uri(
            "https://www.cdc.gov/pneumonia/", "CDC Pneumonia"
        )
        == "https://www.cdc.gov/pneumonia/"
    )
    assert (
        AskService._display_source_uri("s3://bucket/paper.pdf", "Paper")
        == "s3://bucket/paper.pdf"
    )


def test_display_source_uri_prefers_doc_title_over_absolute_path():
    assert (
        AskService._display_source_uri("/Users/sharp/foo/bar.pdf", "Nice Title")
        == "Nice Title"
    )


def test_display_source_uri_falls_back_to_basename_when_no_title():
    assert AskService._display_source_uri("/Users/sharp/foo/bar.pdf", None) == "bar.pdf"


def test_display_source_uri_none_returns_doc_title():
    assert AskService._display_source_uri(None, "Only Title") == "Only Title"
    assert AskService._display_source_uri("", "Only Title") == "Only Title"
    assert AskService._display_source_uri(None, None) is None


def test_format_sources_uses_i18n_display_label_for_system_rag():
    """Sources line format: ``[N] <title> · <i18n display label>``.

    The label is looked up from ``configs/i18n/<lang>.yaml`` under
    ``rag.collection.<collection_name>``. statpearls_en → "MedRAG-StatPearls",
    textbooks_en → "MedRAG-Textbooks" — adding a new corpus only needs an i18n
    entry, no code change.
    """
    chunk = RetrievedChunk(
        text="body",
        source="system_rag",
        score=0.9,
        doc_id="d1",
        chunk_index=0,
        is_phi=False,
        can_cloud=True,
        collection_name="statpearls_en",
        parent_id=None,
        parent_text=None,
        rerank_score=0.9,
        source_uri=None,
        doc_title="StatPearls: Iron Deficiency Anemia",
    )
    block_en = AskService._format_sources([chunk], lang="en")
    assert " · MedRAG-StatPearls" in block_en
    # Brand names match across languages, so zh resolves to the same string.
    block_zh = AskService._format_sources([chunk], lang="zh")
    assert " · MedRAG-StatPearls" in block_zh


def test_format_sources_zh_translates_user_library_label():
    """user_rag chunks render as ``我的资料库`` in zh, ``My Library`` in en.

    Plus: never leak the per-user collection id (``user_rag_alice``).
    """
    chunk = RetrievedChunk(
        text="body",
        source="user_rag",
        score=0.9,
        doc_id="upload-42",
        chunk_index=0,
        is_phi=False,
        can_cloud=False,
        collection_name="user_rag_alice",
        parent_id=None,
        parent_text=None,
        rerank_score=0.9,
        source_uri=None,
        doc_title="my-lab-report.pdf",
    )
    block_en = AskService._format_sources([chunk], lang="en")
    assert " · My Library" in block_en
    assert "user_rag_alice" not in block_en
    assert "alice" not in block_en

    block_zh = AskService._format_sources([chunk], lang="zh")
    assert " · 我的资料库" in block_zh
    assert "user_rag_alice" not in block_zh


def test_format_sources_unknown_collection_falls_back_to_raw_id():
    """A new corpus without an i18n entry surfaces its raw id, not the key.

    Without this guard, adding a new collection but forgetting to add
    an i18n key would render ``rag.collection.foo_en`` in the UI —
    visibly broken. Falling back to ``foo_en`` is uglier than the brand
    label but still informative.
    """
    chunk = RetrievedChunk(
        text="body",
        source="system_rag",
        score=0.9,
        doc_id="d1",
        chunk_index=0,
        is_phi=False,
        can_cloud=True,
        collection_name="brand_new_corpus_en",
        parent_id=None,
        parent_text=None,
        rerank_score=0.9,
        source_uri=None,
        doc_title="A Document Title",
    )
    block = AskService._format_sources([chunk], lang="en")
    assert " · brand_new_corpus_en" in block
    assert "rag.collection." not in block


def test_format_sources_no_title_shows_only_corpus_label():
    """No URI + no doc_title → just the corpus label (no stray separator).

    Older chunks ingested before source_uri / doc_title backfill must
    still render usefully — `· ` with nothing on its left would be
    visual noise.
    """
    chunk = _chunk(text="body", parent_text=None)
    assert chunk.source_uri is None
    assert chunk.doc_title is None
    block = AskService._format_sources([chunk], lang="en")
    assert "- [1] MedRAG-StatPearls" in block
    assert " · " not in block


def test_ask_prompt_v5_is_latest_and_allows_graceful_fallback():
    """v5 must win ``latest`` resolution and contain the per-claim mode rule.

    Regression for the NSAID-pharmacology screenshot: v4 trained the
    model to refuse outright when retrieval mismatched the question.
    v5 splits answer modes — general knowledge can fall back to model
    training (without [N]) while specific numbers stay strictly cited.
    """
    from claritymed.core.prompts.registry import PromptRegistry

    registry = PromptRegistry()
    en = registry.get("ask", language="en")
    zh = registry.get("ask", language="zh")
    # English contract markers.
    assert "Answer mode — pick PER CLAIM" in en
    assert "Beyond the retrieved sources" in en
    assert "Each turn judges its topic independently" in en
    # Chinese contract markers.
    assert "回答模式" in zh
    assert "超出检索资料范围" in zh
    assert "每一轮独立判定主题" in zh


def test_format_evidence_never_leaks_collection_name():
    """LLM-facing evidence must not contain the collection id either.

    The LLM mimics whatever string it sees as a "source", so leaking the
    collection name into evidence trains it to write fake "[1] statpearls_en"
    footnotes in later turns (the original failure mode).
    """
    chunk = _chunk(text="aspirin treats pain", parent_text="aspirin parent text")
    formatted = AskService._format_evidence([chunk])
    assert "statpearls_en" not in formatted
    # No source identifier available → unlabeled [N] line (still cite-able).
    assert "[1]" in formatted
    assert "aspirin parent text" in formatted


def test_format_evidence_uses_source_uri_label_when_available():
    chunk = RetrievedChunk(
        text="aspirin treats pain",
        source="system_rag",
        score=0.9,
        doc_id="d1",
        chunk_index=0,
        is_phi=False,
        can_cloud=True,
        collection_name="statpearls_en",
        parent_id=None,
        parent_text="aspirin parent text",
        rerank_score=0.9,
        source_uri="https://ncbi.nlm.nih.gov/books/NBK123",
    )
    formatted = AskService._format_evidence([chunk])
    assert "(https://ncbi.nlm.nih.gov/books/NBK123)" in formatted
    assert "statpearls_en" not in formatted


def test_clamp_citations_strips_out_of_range_markers():
    """``[N]`` with ``N > max_n`` is hallucinated; strip it.

    Direct regression for the screenshot showing ``[11]`` in an answer
    whose current turn only surfaced 2 Sources entries.
    """
    text = "首先 [1]，然后 [11]，最后 [2] 收尾。"
    cleaned, offending = AskService._clamp_citations(text, max_n=2)
    assert "[11]" not in cleaned
    assert "[1]" in cleaned
    assert "[2]" in cleaned
    assert offending == [11]


def test_clamp_citations_max_n_zero_strips_all_markers():
    """When no chunks were retrieved this turn, NO [N] should survive.

    Belt-and-suspenders for the v4 prompt rule "no Evidence → no citations".
    """
    text = "Body claim [1] more claim [2]."
    cleaned, offending = AskService._clamp_citations(text, max_n=0)
    assert "[1]" not in cleaned
    assert "[2]" not in cleaned
    assert offending == [1, 2]


def test_clamp_citations_in_range_unchanged():
    text = "claim [1] claim [2] claim [3]"
    cleaned, offending = AskService._clamp_citations(text, max_n=3)
    assert cleaned == text
    assert offending == []


def test_clamp_citations_dedupes_offending_list():
    text = "claim [11] claim [11] claim [12] claim [11]"
    cleaned, offending = AskService._clamp_citations(text, max_n=2)
    assert "[11]" not in cleaned and "[12]" not in cleaned
    assert offending == [11, 12]  # sorted, unique


def test_strip_evidence_block_removes_inline_splice():
    """The exact splice ``_compose_prompt`` produces is recognized + removed."""
    from claritymed.orchestrator.services.ask_service import _strip_evidence_block

    spliced = (
        "\nEvidence (cite by [n]):\n"
        "[1] (uri-1) body one\n"
        "[2] (uri-2) body two\n\n"
        "Question: what is my hemoglobin?"
    )
    cleaned = _strip_evidence_block(spliced)
    assert "Evidence (cite by [n]):" not in cleaned
    assert "uri-1" not in cleaned and "uri-2" not in cleaned
    assert cleaned.endswith("what is my hemoglobin?")


def test_strip_evidence_block_passes_through_plain_text():
    """Plain user messages (no splice) must round-trip untouched."""
    from claritymed.orchestrator.services.ask_service import _strip_evidence_block

    plain = "我贫血了吗？血红蛋白 105 g/L。"
    assert _strip_evidence_block(plain) == plain


def test_sanitize_history_strips_evidence_from_user_prompts():
    """Drives the helper through a realistic two-turn history."""
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        TextPart,
        UserPromptPart,
    )

    from claritymed.orchestrator.services.ask_service import (
        _sanitize_history_for_llm,
    )

    spliced = "\nEvidence (cite by [n]):\n[1] (x) body\n\nQuestion: prior question"
    history = [
        ModelRequest(parts=[UserPromptPart(content=spliced)]),
        ModelResponse(parts=[TextPart(content="prior answer with [1]")]),
        ModelRequest(parts=[UserPromptPart(content="follow-up question")]),
    ]
    sanitized = _sanitize_history_for_llm(history)
    # User-prompt evidence is gone; assistant response (which references
    # [1]) is preserved — that's fine, the index is now opaque to the LLM.
    assert "Evidence (cite by [n]):" not in sanitized[0].parts[0].content
    assert "prior question" in sanitized[0].parts[0].content
    assert sanitized[1].parts[0].content == "prior answer with [1]"
    assert sanitized[2].parts[0].content == "follow-up question"


def test_format_debug_collections_dedupes_by_doc_id():
    """Debug indices must match Sources [N], not surface raw chunk duplicates.

    Regression for the second screenshot: two chunks from
    ``article-132882`` formerly produced Debug ``[1]`` + ``[2]`` while
    Sources merged them into ``[1]`` — so Sources ``[2]`` mapped to
    Debug ``[3]`` and the reader couldn't cross-reference.
    """
    c1 = RetrievedChunk(
        text="x",
        source="system_rag",
        score=0.7,
        doc_id="article-132882",
        chunk_index=0,
        is_phi=False,
        can_cloud=True,
        collection_name="statpearls_en",
        parent_id=None,
        parent_text=None,
        rerank_score=0.678,
    )
    c2 = RetrievedChunk(
        text="y",
        source="system_rag",
        score=0.6,
        doc_id="article-132882",  # SAME doc_id as c1
        chunk_index=1,
        is_phi=False,
        can_cloud=True,
        collection_name="statpearls_en",
        parent_id=None,
        parent_text=None,
        rerank_score=0.594,
    )
    c3 = RetrievedChunk(
        text="z",
        source="system_rag",
        score=0.4,
        doc_id="article-56333",
        chunk_index=0,
        is_phi=False,
        can_cloud=True,
        collection_name="statpearls_en",
        parent_id=None,
        parent_text=None,
        rerank_score=0.441,
    )
    block = AskService._format_debug_collections([c1, c2, c3])
    # Two rows, matching what Sources would show.
    assert block.count("\n- [") == 2
    assert "[1]" in block and "[2]" in block
    assert "[3]" not in block
    # Best score reported for the duplicated doc + chunk count annotation.
    assert "0.678" in block
    assert "2 chunks" in block
    # Single-chunk row has no count annotation.
    assert "1 chunks" not in block


def test_latest_ask_prompt_forbids_phantom_citations():
    """The active ask prompt must always carry the no-evidence-no-cite rule.

    Originally added in v4 to stop fake ``[1] statpearls_en`` footnotes
    when retrieval returned nothing. v5 inherits the rule; future
    versions must too — that's why this asserts on ``latest`` rather
    than pinning a version.
    """
    from claritymed.core.prompts.registry import PromptRegistry

    registry = PromptRegistry()
    en = registry.get("ask", language="en")
    zh = registry.get("ask", language="zh")
    assert "Citation honesty" in en
    assert "Evidence (cite by [n]):" in en
    assert "引用诚实性" in zh
    assert "Evidence (cite by [n]):" in zh


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


async def test_sources_block_persisted_to_jsonl_transcript():
    """Sources markdown must be in the persisted assistant text.

    The live stream shows a Sources block as trailing TokenChunks, but the
    persisted assistant event feeds the SPA on refresh (via load_turns).
    If the block is only streamed and never merged into ``final_text``
    before finalize, refresh drops the citations — the bug the screenshot
    at CLM-<n> reported.
    """
    from claritymed.orchestrator.services import ChatSession

    session = ChatSession.new("alice")
    bundle = EvidenceBundle(
        chunks=[_chunk(text="x")], trace=RetrievalTrace(strategy="naive_hybrid")
    )
    service = AskService(
        model=TestModel(custom_output_text="answer"),
        strategy=StubStrategy(bundle),
        provider_config=_provider("local"),
        chat_session=session,
    )
    async for _ in service.run("q", user_id="alice"):
        pass

    turns = ChatSession.resume("alice", session.session_id).load_turns()
    assistant_turns = [t for t in turns if t.role == "assistant"]
    assert assistant_turns, "assistant turn missing from persisted transcript"
    assert "**Sources:**" in assistant_turns[-1].text, (
        "Sources block was streamed live but never persisted — refresh will "
        "render an assistant bubble without citations."
    )


async def test_debug_block_persisted_when_env_flag_set(monkeypatch):
    """CLARITYMED_DEBUG=1 must persist the Debug preamble to the transcript.

    Same reason as Sources: the on-refresh render reads persisted text, so
    a Debug block emitted only as a TokenChunk vanishes on refresh even
    though the developer explicitly opted in via the env var.
    """
    from claritymed.orchestrator.services import ChatSession

    monkeypatch.setenv("CLARITYMED_DEBUG", "1")
    session = ChatSession.new("alice")
    bundle = EvidenceBundle(
        chunks=[_chunk(text="x")], trace=RetrievalTrace(strategy="naive_hybrid")
    )
    service = AskService(
        model=TestModel(custom_output_text="answer"),
        strategy=StubStrategy(bundle),
        provider_config=_provider("local"),
        chat_session=session,
    )
    async for _ in service.run("q", user_id="alice"):
        pass

    turns = ChatSession.resume("alice", session.session_id).load_turns()
    assistant_turns = [t for t in turns if t.role == "assistant"]
    assert assistant_turns, "assistant turn missing from persisted transcript"
    assert "**Debug**" in assistant_turns[-1].text, (
        "Debug block was streamed live but never persisted — refresh drops "
        "the request_id / provider preamble the developer needs to correlate."
    )


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
    from claritymed.core.events import Error

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
    # retrieve_medical_literature emits its own ToolStarted first; verify the
    # translate.query step is also present somewhere in the event stream.
    tool_names_started = [e.tool_name for e in events if isinstance(e, ToolStarted)]
    tool_names_completed = [e.tool_name for e in events if isinstance(e, ToolCompleted)]
    assert "translate.query" in tool_names_started
    assert "translate.query" in tool_names_completed
    translate_completed = next(
        e
        for e in events
        if isinstance(e, ToolCompleted) and e.tool_name == "translate.query"
    )
    assert translate_completed.summary == "done"
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
