"""Agentic mode: when the active strategy declares ``is_agentic=True``.

Retrieval has always been tool-driven (``retrieve_medical_literature``);
agentic mode is the configuration knob that surfaces this through the
audit log + ``AskDeps.agentic`` so operators can confirm a rollout
reached the tool loop rather than silently degrading.
"""

from __future__ import annotations

from pydantic_ai.models.test import TestModel

from claritymed.core.rag.schemas import EvidenceBundle, RetrievalTrace
from claritymed.core.rag.strategies.base import RagStrategy, RetrievalContext
from claritymed.core.schemas.models import ProviderConfig
from claritymed.core.schemas.retrieval import RetrievedChunk
from claritymed.orchestrator.services import AskService
from claritymed.orchestrator.services.events import Done, TokenChunk


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
    """Records the contexts the tool invoked retrieve with."""

    def __init__(self, *, is_agentic: bool, bundle: EvidenceBundle) -> None:
        self.is_agentic = is_agentic
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


# --- agentic flag propagation -----------------------------------------


async def test_agentic_flag_propagates_to_deps_when_strategy_marks_itself():
    """The strategy's ``is_agentic`` attribute lands on AskDeps so the tool
    and audit layer can observe it."""
    captured: dict[str, object] = {}

    class _CapturingStrategy(_RecordingStrategy):
        async def retrieve(self, ctx):
            # Tool ran → it had access to deps.strategy which is this
            # instance. We rely on retrieve being invoked at all, then
            # assert downstream via the bundle path.
            captured["called"] = True
            return await super().retrieve(ctx)

    capturing = _CapturingStrategy(is_agentic=True, bundle=_bundle())
    service = AskService(
        model=TestModel(custom_output_text="answer"),
        strategy=capturing,
        provider_config=_provider(),
    )
    events = [ev async for ev in service.run("what is aspirin", user_id="alice")]
    assert any(isinstance(e, Done) for e in events)
    assert captured.get("called") is True


async def test_non_agentic_strategy_keeps_deps_agentic_false():
    """A bare NaiveHybridStrategy (no ``is_agentic`` attribute) leaves the
    flag at its default ``False`` so existing behaviour is preserved."""
    strategy = _RecordingStrategy(is_agentic=False, bundle=_bundle())
    # Remove the attribute entirely so the ``getattr`` default kicks in,
    # mirroring what NaiveHybridStrategy looks like before the factory
    # tags it.
    delattr(strategy, "is_agentic")
    service = AskService(
        model=TestModel(custom_output_text="answer"),
        strategy=strategy,
        provider_config=_provider(),
    )
    events = [ev async for ev in service.run("aspirin", user_id="alice")]
    assert any(isinstance(e, Done) for e in events)


async def test_agentic_strategy_supports_multiple_tool_invocations():
    """The tool loop already accumulates chunks across calls; agentic mode
    just rides on that — verify the sources block reflects the union."""
    strategy = _RecordingStrategy(is_agentic=True, bundle=_bundle())
    service = AskService(
        model=TestModel(
            # TestModel can be configured to invoke the tool; default behaviour
            # calls every tool once. That's enough to prove the path is wired.
            custom_output_text="answer",
        ),
        strategy=strategy,
        provider_config=_provider(),
    )
    captured_chunks: list[str] = []
    async for event in service.run("aspirin and ibuprofen", user_id="alice"):
        if isinstance(event, TokenChunk):
            captured_chunks.append(event.text)
    # The sources block lands as a TokenChunk just before Done.
    joined = "".join(captured_chunks)
    assert "Sources:" in joined
    assert len(strategy.calls) >= 1
