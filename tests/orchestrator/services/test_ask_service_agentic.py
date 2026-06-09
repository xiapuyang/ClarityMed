"""Feature-plugin tests for AskService.

The "three modes" are now per-feature plugin attributes (not a single
turn-level dispatch). These tests verify:

* default ``rag_mode='tool'`` → tool gets registered, LLM calls it,
  Sources accumulate;
* ``rag_mode='deterministic'`` → retrieval runs *before* the LLM call,
  no tool registered;
* ``rag_mode='agentic'`` → ``build_features`` fails loud with
  NotImplementedError (state-graph workflow reserved for v2).
"""

from __future__ import annotations

import pytest
from pydantic_ai.models.test import TestModel

from claritymed.core.events import Done, TokenChunk
from claritymed.core.rag.schemas import EvidenceBundle, RetrievalTrace
from claritymed.core.rag.strategies.base import RagStrategy, RetrievalContext
from claritymed.core.schemas.models import ProviderConfig
from claritymed.core.schemas.retrieval import RetrievedChunk
from claritymed.orchestrator.services import AskService


def _chunk(*, text: str = "evidence") -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        source="system_rag",
        score=0.9,
        doc_id="d1",
        chunk_index=0,
        is_phi=False,
        can_cloud=True,
        collection_name="statpearls_en",
        parent_id="d1#p0",
        rerank_score=0.9,
    )


class _RecordingStrategy(RagStrategy):
    def __init__(self, *, bundle: EvidenceBundle) -> None:
        self._bundle = bundle
        self.calls: list[RetrievalContext] = []

    async def retrieve(self, ctx: RetrievalContext) -> EvidenceBundle:
        self.calls.append(ctx)
        return self._bundle


def _provider() -> ProviderConfig:
    return ProviderConfig(id="test", kind="local", model="openai:gpt-4o")


def _bundle() -> EvidenceBundle:
    return EvidenceBundle(
        chunks=[_chunk()],
        trace=RetrievalTrace(strategy="naive_hybrid"),
    )


async def test_default_tool_mode_routes_through_retrieve_tool():
    """``rag_mode='tool'`` registers the tool; TestModel calls every
    registered tool once → strategy.retrieve runs."""
    strategy = _RecordingStrategy(bundle=_bundle())
    service = AskService(
        model=TestModel(custom_output_text="answer"),
        strategy=strategy,
        provider_config=_provider(),
    )
    events = [ev async for ev in service.run("what is aspirin", user_id="alice")]
    assert any(isinstance(e, Done) for e in events)
    assert len(strategy.calls) >= 1


async def test_tool_mode_accumulates_sources_block():
    strategy = _RecordingStrategy(bundle=_bundle())
    service = AskService(
        model=TestModel(custom_output_text="answer"),
        strategy=strategy,
        provider_config=_provider(),
    )
    captured: list[str] = []
    async for event in service.run("aspirin and ibuprofen", user_id="alice"):
        if isinstance(event, TokenChunk):
            captured.append(event.text)
    joined = "".join(captured)
    assert "Sources:" in joined


async def test_deterministic_mode_retrieves_before_llm():
    """Deterministic ``RagFeature.pre_invoke`` runs the retrieval
    pipeline *before* agent.run_stream. A TestModel that ignores tools
    proves it: the strategy is still hit."""
    strategy = _RecordingStrategy(bundle=_bundle())
    service = AskService(
        model=TestModel(custom_output_text="aspirin is a salicylate."),
        strategy=strategy,
        provider_config=_provider(),
        rag_mode="deterministic",
    )
    captured: list[str] = []
    async for event in service.run("what is aspirin", user_id="alice"):
        if isinstance(event, TokenChunk):
            captured.append(event.text)
    assert len(strategy.calls) == 1
    assert "Sources:" in "".join(captured)


async def test_agentic_mode_raises_at_construction():
    """v1 has no state-graph runtime; opting any feature into agentic
    fails at AskService() rather than mid-stream so operators see the
    misconfig immediately."""
    strategy = _RecordingStrategy(bundle=_bundle())
    with pytest.raises(NotImplementedError):
        AskService(
            model=TestModel(custom_output_text="answer"),
            strategy=strategy,
            provider_config=_provider(),
            rag_mode="agentic",
        )


async def test_announced_but_skipped_emits_audit_event(monkeypatch):
    """LLM in tool mode that writes "I will search..." but never fires
    the tool call should produce a ``mode.ask.tool_announced_but_skipped``
    audit row so we can quantify per-provider compliance.
    """
    from claritymed.orchestrator.services import ask_service as ask_mod

    captured: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        ask_mod, "audit_event", lambda event, payload: captured.append((event, payload))
    )

    strategy = _RecordingStrategy(bundle=_bundle())
    # TestModel that doesn't call any tool and emits an announce phrase.
    service = AskService(
        model=TestModel(
            call_tools=[],  # no tool calls at all
            custom_output_text="我将首先检索相关指南，请稍候。",
        ),
        strategy=strategy,
        provider_config=_provider(),
    )
    async for _ in service.run("我血红蛋白 105", user_id="alice"):
        pass

    events = [name for name, _ in captured]
    assert "mode.ask.tool_announced_but_skipped" in events
    payload = next(
        p for name, p in captured if name == "mode.ask.tool_announced_but_skipped"
    )
    assert payload["tool"] == "retrieve_medical_literature"
    assert "检索" in payload["snippet"]


async def test_no_announce_no_audit(monkeypatch):
    """Plain answer (no announce phrase) does NOT trigger the audit row
    even when the tool wasn't called — false positives would drown out
    real compliance signals."""
    from claritymed.orchestrator.services import ask_service as ask_mod

    captured: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        ask_mod, "audit_event", lambda event, payload: captured.append((event, payload))
    )

    strategy = _RecordingStrategy(bundle=_bundle())
    service = AskService(
        model=TestModel(
            call_tools=[],
            custom_output_text="Your hemoglobin is in the mild anemia range.",
        ),
        strategy=strategy,
        provider_config=_provider(),
    )
    async for _ in service.run("hb 105", user_id="alice"):
        pass

    events = [name for name, _ in captured]
    assert "mode.ask.tool_announced_but_skipped" not in events


async def test_tool_call_counted_when_invoked(monkeypatch):
    """When the tool fires, ``deps.tool_calls['retrieve_medical_literature']``
    reaches the ``mode.ask`` audit payload — operators can grep
    per-turn tool usage from one audit row."""
    from claritymed.orchestrator.services import ask_service as ask_mod

    captured: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        ask_mod, "audit_event", lambda event, payload: captured.append((event, payload))
    )

    strategy = _RecordingStrategy(bundle=_bundle())
    service = AskService(
        model=TestModel(custom_output_text="answer"),
        strategy=strategy,
        provider_config=_provider(),
    )
    async for _ in service.run("aspirin", user_id="alice"):
        pass

    mode_ask = next(p for name, p in captured if name == "mode.ask")
    assert mode_ask.get("tool_calls", {}).get("retrieve_medical_literature", 0) >= 1
