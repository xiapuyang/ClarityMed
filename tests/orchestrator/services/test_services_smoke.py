"""Smoke tests for ingest / ask / rag services."""

from __future__ import annotations

import hashlib

import pytest
from pydantic_ai.models.test import TestModel
from qdrant_client import AsyncQdrantClient

from claritymed.core.rag.chunking.base import (
    ChildChunk,
    ChunkedDocument,
    ParentChunk,
    RawDocument,
)
from claritymed.core.rag.embedding.base import Embedder, SparseVector
from claritymed.core.schemas.receipts import IngestionReceipt
from claritymed.core.phi.guard import PhiGuard
from claritymed.orchestrator.services import (
    AskService,
    Done,
    RagService,
    TokenChunk,
)
from claritymed.stores.user_rag import UserRagStore


class _StubEmbedder(Embedder):
    @property
    def dimension(self) -> int:
        return 32

    async def embed_dense(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            out.append([b / 255.0 for b in digest[: self.dimension]])
        return out

    async def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        return [{abs(hash(t)) % 100: 0.5} for t in texts]


class _StubChunker:
    def chunk(self, doc: RawDocument) -> ChunkedDocument:
        if not doc.text.strip():
            return ChunkedDocument(parents=[], children=[])
        import uuid

        parent_id = f"{doc.doc_id}#p0"
        parent = ParentChunk(
            parent_id=parent_id,
            text=doc.text,
            doc_id=doc.doc_id,
            parent_index=0,
        )
        child = ChildChunk(
            child_id=str(uuid.uuid5(uuid.NAMESPACE_URL, doc.doc_id)),
            text=doc.text,
            parent_id=parent_id,
            doc_id=doc.doc_id,
            chunk_index=0,
        )
        return ChunkedDocument(parents=[parent], children=[child])


@pytest.fixture(scope="module")
def _phi_guard() -> PhiGuard:
    return PhiGuard.from_config()


@pytest.fixture(autouse=True)
def _disable_cosine_dedupe(monkeypatch):
    """Disable per-chunk cosine-sim dedupe for these smoke tests.

    The StubEmbedder is deterministic — running these tests with the
    production default threshold would silently drop identical-text
    re-ingests, masking the assertions. Tests that need to exercise
    the dedupe path live in tests/stores/test_user_rag.py.
    """
    monkeypatch.setattr("claritymed.config.upload_dedupe_cosine_threshold", lambda: 0.0)


@pytest.fixture
def rag_store(_phi_guard: PhiGuard):
    return UserRagStore(
        aclient=AsyncQdrantClient(":memory:"),
        embedder=_StubEmbedder(),
        chunker=_StubChunker(),
        guard=_phi_guard,
    )


async def test_rag_service_persists_via_user_rag(rag_store):
    events = []
    async for ev in RagService(store=rag_store).run(
        "Patient note line one\n\nLine two paragraph",
        user_id="alice",
    ):
        events.append(ev)

    done = next(e for e in events if isinstance(e, Done))
    assert isinstance(done.final, IngestionReceipt)
    # _StubChunker emits one child per doc.
    assert done.final.chunk_count == 1

    hits = await rag_store.search("alice", "paragraph", top_k=5)
    assert hits != []


async def test_rag_service_public_flag_passes_through(rag_store):
    events = []
    async for ev in RagService(store=rag_store).run(
        "public paper content",
        user_id="alice",
        public=True,
    ):
        events.append(ev)

    done = next(e for e in events if isinstance(e, Done))
    assert done.final.public is True
    hits = await rag_store.search("alice", "public", top_k=5)
    assert all(h.can_cloud for h in hits)


async def test_rag_service_empty_text_yields_stub_status(rag_store):
    """Empty input → chunker returns no children → embedding_status == 'stub'.

    Guards the conditional in RagService that distinguishes a real write
    (``"ok"``) from a no-op (``"stub"``).
    """
    events = []
    async for ev in RagService(store=rag_store).run("", user_id="alice"):
        events.append(ev)

    done = next(e for e in events if isinstance(e, Done))
    assert done.final.chunk_count == 0
    assert done.final.embedding_status == "stub"


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


async def test_ask_service_persists_run_messages_to_chat_session():
    """AskService must append the user+assistant turns to the chat session.

    The persisted assistant event embeds the pydantic-ai messages array;
    we round-trip it through ``ModelMessagesTypeAdapter`` to prove it can
    feed back into ``Agent.run(message_history=...)`` next turn.
    """
    import json

    from pydantic_ai.messages import (
        ModelMessagesTypeAdapter,
        ModelRequest,
        ModelResponse,
    )

    from claritymed.orchestrator.services import ChatSession

    session = ChatSession.new("alice")
    service = AskService(
        model=TestModel(custom_output_text="hello there"),
        chat_session=session,
        provider_id="test_provider",
        model_name="test:model",
    )
    events = []
    async for ev in service.run("how are you?", user_id="alice"):
        events.append(ev)

    lines = session.path.read_text("utf-8").splitlines()
    kinds = [json.loads(line)["type"] for line in lines]
    assert "user" in kinds and "assistant" in kinds

    assistant = json.loads(
        next(line for line in lines if json.loads(line)["type"] == "assistant")
    )
    assert assistant["model"] == "test:model"
    assert assistant["providerId"] == "test_provider"
    assert assistant["latency"]["totalMs"] >= 0
    assert "steps" in assistant and len(assistant["steps"]) >= 1
    assert "messages" in assistant

    messages = ModelMessagesTypeAdapter.validate_python(assistant["messages"])
    assert any(isinstance(m, ModelRequest) for m in messages)
    assert any(isinstance(m, ModelResponse) for m in messages)


async def test_ask_service_reuses_existing_request_id():
    """Regression: AskService used to generate a fresh request_id, which
    desynchronised the TUI status bar from the audit log. With a rid
    already in context (TUI's _run_stream sets one), the service must
    reuse it instead of overwriting."""
    from claritymed.context import apply_context, request_id_ctx, reset_context
    from claritymed.orchestrator.services import ChatSession

    rid = "20260607123045ABCDEF12"
    tokens = apply_context(rid, "alice", "en")
    captured_rid: list[str] = []

    class _RidSpySession(ChatSession):
        def append_assistant(self, **kwargs) -> str:
            captured_rid.append(request_id_ctx.get() or "")
            return super().append_assistant(**kwargs)

    try:
        session = _RidSpySession.new("alice")
        service = AskService(
            model=TestModel(custom_output_text="ok"),
            chat_session=session,
        )
        async for _ in service.run("hi", user_id="alice"):
            pass
    finally:
        reset_context(tokens)

    assert captured_rid == [rid], (
        f"AskService overwrote request_id: caller had {rid!r}, "
        f"service used {captured_rid!r}"
    )


async def test_ask_service_passes_message_history_for_multi_turn():
    """Multi-turn proof: turn 2 sees turn 1's messages as message_history.

    Without ChatSession plumbing this regresses to the original bug where
    the agent had no memory of prior turns.
    """
    from pydantic_ai.messages import ModelRequest, UserPromptPart

    from claritymed.orchestrator.services import ChatSession

    session = ChatSession.new("alice")
    service = AskService(
        model=TestModel(custom_output_text="first answer"),
        chat_session=session,
    )
    async for _ in service.run("question one", user_id="alice"):
        pass

    history = session.message_history()
    assert history, "first turn left no in-memory history"
    user_prompts = [
        part.content
        for msg in history
        if isinstance(msg, ModelRequest)
        for part in msg.parts
        if isinstance(part, UserPromptPart)
    ]
    assert "question one" in user_prompts


async def test_ask_service_emits_token_and_latency_audit():
    """``mode.ask`` audit payload must include token + latency telemetry."""
    import json
    import logging

    from claritymed.orchestrator.services import ChatSession

    session = ChatSession.new("alice")
    service = AskService(
        model=TestModel(custom_output_text="ok"),
        chat_session=session,
        provider_id="test_provider",
        model_name="test:model",
    )

    audit_records: list[dict] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
            audit_records.append(json.loads(record.getMessage()))

    handler = _Capture()
    audit_logger = logging.getLogger("claritymed.audit")
    audit_logger.addHandler(handler)
    try:
        async for _ in service.run("hi", user_id="alice"):
            pass
    finally:
        audit_logger.removeHandler(handler)

    asks = [r for r in audit_records if r["kind"] == "mode.ask"]
    assert asks, "no mode.ask audit event emitted"
    payload = asks[-1]["payload"]
    assert payload["model"] == "test:model"
    assert payload["provider_id"] == "test_provider"
    assert payload["session_id"] == session.session_id
    assert "latency_ms" in payload
    assert "ttft_ms" in payload
    assert "completion_ms" in payload
    assert "input_tokens" in payload
    assert "output_tokens" in payload
    assert "total_tokens" in payload
    assert "steps" in payload and len(payload["steps"]) >= 1


async def test_ask_service_audit_picks_up_trace_id_when_tracing_active(monkeypatch):
    """Regression: the `mode.ask` audit line used to land with `trace_id=null`
    because we wrote it after the pydantic-ai span had already closed. With
    the request-scoped `ask.request` span in place, both the scrub audit
    and the final mode.ask audit must share the same non-null trace_id.
    """
    import json
    import logging

    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from opentelemetry.sdk.trace import TracerProvider

    from claritymed.core.phi.guard import PhiGuard
    from claritymed.core.scrub.service import ScrubReport
    from claritymed.orchestrator.services import ChatSession

    # ONNX model is not available in CI; patch scrub_free_text so the cloud
    # scrub step succeeds and both mode.ask.scrub and mode.ask audit events fire.
    def _noop_scrub(self, text: str) -> tuple[str, ScrubReport]:
        return text, ScrubReport(text_len_before=len(text), text_len_after=len(text))

    monkeypatch.setattr(PhiGuard, "scrub_free_text", _noop_scrub)

    # Local provider — we patch get_tracer instead of mutating the global
    # so concurrent tests in the session don't inherit our tracer state.
    # ask_service does ``from opentelemetry import trace as otel_trace``
    # inside _run_inner, so the patch target is opentelemetry.trace.
    provider = TracerProvider(resource=Resource.create({SERVICE_NAME: "test"}))
    monkeypatch.setattr(
        otel_trace,
        "get_tracer",
        lambda name: provider.get_tracer(name),
    )

    from claritymed.core.schemas.models import ProviderConfig

    session = ChatSession.new("alice")
    service = AskService(
        model=TestModel(custom_output_text="ok"),
        chat_session=session,
        provider_id="test_provider",
        model_name="test:model",
        # Cloud provider so PHI scrub (mode.ask.scrub) is emitted — the test
        # verifies that scrub and ask share the same trace_id.
        provider_config=ProviderConfig(
            id="test_provider", kind="cloud", model="openai:gpt-4o"
        ),
    )

    audit_records: list[dict] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:  # noqa: D401
            audit_records.append(json.loads(record.getMessage()))

    handler = _Capture()
    audit_logger = logging.getLogger("claritymed.audit")
    audit_logger.addHandler(handler)
    try:
        async for _ in service.run("hi", user_id="alice"):
            pass
    finally:
        audit_logger.removeHandler(handler)
        provider.shutdown()

    asks = [r for r in audit_records if r["kind"] == "mode.ask"]
    scrubs = [r for r in audit_records if r["kind"] == "mode.ask.scrub"]
    assert asks and scrubs, "expected mode.ask and mode.ask.scrub events"
    assert asks[-1]["trace_id"] is not None, (
        "mode.ask trace_id is null — the request-scoped span did not wrap the audit"
    )
    assert scrubs[-1]["trace_id"] == asks[-1]["trace_id"], (
        "scrub and final audit must share trace_id under the same ask.request span"
    )


async def test_ask_service_phi_scrub_before_llm(monkeypatch, _phi_guard: PhiGuard):
    """Verify scrub_free_text is called for cloud providers before the LLM.

    Local providers skip the scrub (data stays on device); cloud providers
    must scrub PHI before any data leaves the machine.
    """
    from claritymed.core.schemas.models import ProviderConfig

    guard = _phi_guard
    captured: list[str] = []
    original = guard.scrub_free_text

    def spy(text: str):
        captured.append(text)
        return original(text)

    monkeypatch.setattr(guard, "scrub_free_text", spy)

    service = AskService(
        model=TestModel(custom_output_text="ok"),
        guard=guard,
        provider_config=ProviderConfig(
            id="cloud_test", kind="cloud", model="openai:gpt-4o"
        ),
    )
    events = []
    async for ev in service.run(
        "My phone is 13800138000, please remember it.",
        user_id="alice",
    ):
        events.append(ev)

    assert captured, "scrub_free_text was not called for cloud provider"
    assert "13800138000" in captured[0], "scrub must see the raw input"


async def test_ask_service_local_skips_phi_scrub(monkeypatch, _phi_guard: PhiGuard):
    """Local providers must NOT call scrub_free_text — data never leaves device."""
    guard = _phi_guard
    captured: list[str] = []
    original = guard.scrub_free_text

    def spy(text: str):
        captured.append(text)
        return original(text)

    monkeypatch.setattr(guard, "scrub_free_text", spy)

    service = AskService(
        model=TestModel(custom_output_text="ok"),
        guard=guard,
        # No provider_config → kind defaults to None → treated as local
    )
    async for _ in service.run("My phone is 13800138000.", user_id="alice"):
        pass

    assert not captured, "scrub_free_text must not be called for local provider"
