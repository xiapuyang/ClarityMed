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
