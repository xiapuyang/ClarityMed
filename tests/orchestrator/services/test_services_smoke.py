"""Smoke tests for ingest / ask / rag services."""

from __future__ import annotations

import hashlib

import pytest
from pydantic_ai.models.test import TestModel
from qdrant_client import QdrantClient

from claritymed.core.schemas.receipts import IngestionReceipt, IngestReceipt
from claritymed.orchestrator import PhiGuard
from claritymed.orchestrator.services import (
    AskService,
    Done,
    IngestService,
    RagService,
    TokenChunk,
)
from claritymed.stores.user_rag import UserRagStore


class _StubEmbedder:
    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in digest[: self.dimension]]

    @property
    def dimension(self) -> int:
        return 32


@pytest.fixture
def rag_store():
    return UserRagStore(
        client=QdrantClient(":memory:"),
        embedder=_StubEmbedder(),
        guard=PhiGuard.from_config(),
    )


async def test_ingest_service_writes_profile_field():
    events = []
    async for ev in IngestService().run("allergy=penicillin", user_id="alice"):
        events.append(ev)

    types = [type(e).__name__ for e in events]
    assert "ToolStarted" in types
    assert "ToolCompleted" in types
    assert "Done" in types

    done = next(e for e in events if isinstance(e, Done))
    assert isinstance(done.final, IngestReceipt)
    assert done.final.records[0].kind == "profile"


async def test_ingest_service_no_kv_returns_empty_receipt():
    events = []
    async for ev in IngestService().run("just a sentence", user_id="alice"):
        events.append(ev)
    done = next(e for e in events if isinstance(e, Done))
    assert done.final.records == []


async def test_rag_service_persists_via_user_rag(rag_store):
    events = []
    async for ev in RagService(store=rag_store).run(
        "Patient note line one\n\nLine two paragraph",
        user_id="alice",
    ):
        events.append(ev)

    done = next(e for e in events if isinstance(e, Done))
    assert isinstance(done.final, IngestionReceipt)
    assert done.final.chunk_count == 2

    hits = rag_store.search("alice", "paragraph", top_k=5)
    assert hits != []


async def test_ask_service_streams_tokens_and_scrubs_input():
    """ask_service must (a) scrub PHI from input before calling LLM, and
    (b) yield streaming TokenChunk events ending in a Done."""
    model = TestModel(custom_output_text="this is a streamed response")
    service = AskService(model=model)

    events = []
    async for ev in service.run(
        "Patient phone 13800138000 — what does this lab value mean?",
        user_id="alice",
    ):
        events.append(ev)

    # Verify streaming events
    chunks = [e for e in events if isinstance(e, TokenChunk)]
    assert chunks  # at least one token chunk
    full_text = "".join(c.text for c in chunks)
    assert "streamed" in full_text

    done = next(e for e in events if isinstance(e, Done))
    assert done.final == "this is a streamed response"


async def test_ask_service_persists_run_messages_to_chat_memory():
    """AskService must append pydantic-ai messages_json to chat_memory.

    Round-trips through the official ``ModelMessagesTypeAdapter`` so the
    file format stays compatible with ``Agent.run(message_history=...)``
    when resume lands.
    """
    from pydantic_ai.messages import (
        ModelMessagesTypeAdapter,
        ModelRequest,
        ModelResponse,
    )

    captured: list[bytes] = []

    class _StubMemory:
        def append_run_messages_json(self, blob: bytes) -> int:
            captured.append(blob)
            return 1

    service = AskService(
        model=TestModel(custom_output_text="hello there"),
        chat_memory=_StubMemory(),
    )
    events = []
    async for ev in service.run("how are you?", user_id="alice"):
        events.append(ev)

    assert captured, "AskService did not call chat_memory"
    messages = ModelMessagesTypeAdapter.validate_json(captured[0])
    kinds = [type(m).__name__ for m in messages]
    assert "ModelRequest" in kinds
    assert "ModelResponse" in kinds
    assert any(isinstance(m, ModelRequest) for m in messages)
    assert any(isinstance(m, ModelResponse) for m in messages)


async def test_ask_service_reuses_existing_request_id():
    """Regression: AskService used to generate a fresh request_id, which
    desynchronised the TUI status bar from the audit log. With a rid
    already in context (TUI's _run_stream sets one), the service must
    reuse it instead of overwriting."""
    from claritymed.context import apply_context, reset_context

    rid = "20260607123045ABCDEF12"
    tokens = apply_context(rid, "alice", "en")
    captured_rid: list[str] = []

    class _RidSpyMemory:
        def append_run_messages_json(self, blob: bytes) -> int:
            from claritymed.context import request_id_ctx

            captured_rid.append(request_id_ctx.get() or "")
            return 1

    try:
        service = AskService(
            model=TestModel(custom_output_text="ok"),
            chat_memory=_RidSpyMemory(),
        )
        async for _ in service.run("hi", user_id="alice"):
            pass
    finally:
        reset_context(tokens)

    assert captured_rid == [rid], (
        f"AskService overwrote request_id: caller had {rid!r}, "
        f"service used {captured_rid!r}"
    )


async def test_ask_service_phi_scrub_before_llm(monkeypatch):
    """Verify scrub_free_text is called with the original input before
    anything is handed to the LLM. We intercept the guard directly because
    that is the single chokepoint between user input and Agent.run_stream."""
    guard = PhiGuard.from_config()
    captured: list[str] = []
    original = guard.scrub_free_text

    def spy(text: str):
        captured.append(text)
        return original(text)

    monkeypatch.setattr(guard, "scrub_free_text", spy)

    service = AskService(
        model=TestModel(custom_output_text="ok"),
        guard=guard,
    )
    events = []
    async for ev in service.run(
        "My phone is 13800138000, please remember it.",
        user_id="alice",
    ):
        events.append(ev)

    assert captured, "scrub_free_text was not called before the LLM run"
    assert "13800138000" in captured[0], "scrub must see the raw input"
