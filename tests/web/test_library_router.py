"""``/api/v1/library`` integration tests.

Three endpoints under test:

* GET /library — collection listing. RAG-off path returns empty system
  list, ``rag_enabled=False``, and the user_collection name still
  reflects the requester so the SPA can render the chip even before any
  ingest has happened.
* POST /library/search — 503 when the strategy cache resolves to
  ``None`` (``rag.enabled=false``); 200 + chunks/trace when a fake
  strategy is injected.
* POST /library/ingest — validates the upload bundle. Empty / sub-floor
  text lands as 422 with the bundle reasons embedded; a long-enough
  inline-text payload is fed to a fake ``RagService`` so the route runs
  end-to-end without touching qdrant or the embedder.

RAG strategy injection: the chat router caches the strategy on
``app.state.rag_strategy``. Tests pre-fill that slot with either
``None`` (off) or a fake to bypass the lazy build.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from claritymed.core.events import Done, ToolCompleted, ToolStarted
from claritymed.core.rag.schemas import EvidenceBundle, RetrievalTrace
from claritymed.core.schemas.receipts import IngestionReceipt
from claritymed.core.schemas.retrieval import RetrievedChunk
from claritymed.web.csrf import HEADER_CSRF_TOKEN
from claritymed.web.jwt import create_token
from claritymed.web.middleware import COOKIE_ACCESS_TOKEN


@pytest.fixture
def auth_cookies(test_user):
    """Reuses the conftest test user; mirrors test_attachments_router.py."""
    token = create_token(test_user.user_id, test_user.language)
    return {COOKIE_ACCESS_TOKEN: token, "csrf_token": "csrf-test"}


def _csrf() -> dict[str, str]:
    return {HEADER_CSRF_TOKEN: "csrf-test"}


@pytest.fixture
def rag_disabled(web_app):
    """Pre-fill the strategy cache with ``None`` — rag-off code path.

    Skips the lazy build inside ``_resolve_strategy`` which would try
    to call ``build_rag_strategy`` against a real qdrant. The list
    endpoint returns ``rag_enabled=False`` and search returns 503.
    """
    web_app.state.rag_strategy = None
    yield
    # Restore the sentinel so other tests rebuild from scratch.
    from claritymed.web.routers.chat import RAG_UNSET

    web_app.state.rag_strategy = RAG_UNSET


# --- GET /library -----------------------------------------------------


async def test_list_library_rag_disabled(
    web_client, test_user, auth_cookies, rag_disabled
):  # noqa: ARG001
    """RAG off: list still returns 200; counts are null; flag is False."""
    resp = await web_client.get("/api/v1/library", cookies=auth_cookies)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["rag_enabled"] is False
    # System counts collapse to None when there's no Qdrant server to ask.
    for entry in body["system_collections"]:
        assert entry["chunk_count"] is None
    # User collection always carries the requester's id.
    assert body["user_collection"]["name"] == f"user_rag_{test_user.user_id}"


async def test_list_library_requires_auth(web_client):
    """No cookie → 401 (anonymous downgrade is allowed by middleware)."""
    resp = await web_client.get("/api/v1/library")
    assert resp.status_code == 401


# --- POST /library/search ---------------------------------------------


async def test_search_rag_disabled_returns_503(
    web_client, test_user, auth_cookies, rag_disabled
):  # noqa: ARG001
    resp = await web_client.post(
        "/api/v1/library/search",
        json={"q": "anything"},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert resp.status_code == 503
    assert resp.json()["detail"] == "Retrieval disabled"


@dataclass
class _FakeStrategy:
    """Minimal stand-in for :class:`RagStrategy`.

    Returns one user-rag chunk + one system chunk so the response
    coverage exercises both tag branches.
    """

    async def retrieve(self, ctx):  # noqa: ARG002 — protocol shape
        return EvidenceBundle(
            chunks=[
                RetrievedChunk(
                    text="user chunk content " * 20,
                    source="user_rag",
                    score=0.5,
                    doc_id="doc-user-1",
                    chunk_index=0,
                    is_phi=True,
                    can_cloud=False,
                    collection_name=f"user_rag_{ctx.user_id}",
                    rerank_score=0.82,
                ),
                RetrievedChunk(
                    text="system chunk content " * 20,
                    source="system_rag",
                    score=0.4,
                    doc_id="doc-sys-1",
                    chunk_index=0,
                    is_phi=False,
                    can_cloud=True,
                    collection_name="nccn_guidelines",
                ),
            ],
            trace=RetrievalTrace(
                strategy="naive_hybrid",
                active_collections=["nccn_guidelines", f"user_rag_{ctx.user_id}"],
                expanded_query="anything expanded",
                embed_ms=12,
                search_ms=34,
                rerank_ms=56,
                parent_expand_ms=78,
            ),
        )


async def test_search_with_fake_strategy(web_client, web_app, test_user, auth_cookies):
    """Injected strategy: response carries both USER and SYS tags."""
    web_app.state.rag_strategy = _FakeStrategy()
    try:
        resp = await web_client.post(
            "/api/v1/library/search",
            json={"q": "hello"},
            cookies=auth_cookies,
            headers=_csrf(),
        )
    finally:
        from claritymed.web.routers.chat import RAG_UNSET

        web_app.state.rag_strategy = RAG_UNSET

    assert resp.status_code == 200, resp.text
    body = resp.json()
    tags = [c["tag"] for c in body["chunks"]]
    assert "USER" in tags
    assert "SYS" in tags
    user_chunk = next(c for c in body["chunks"] if c["tag"] == "USER")
    # Rerank score wins when present; pre-rerank score is dropped on the
    # wire so the SPA can sort on one field without disambiguation.
    assert user_chunk["score"] == pytest.approx(0.82)
    assert user_chunk["collection_name"] == f"user_rag_{test_user.user_id}"
    assert body["trace"]["embed_ms"] == 12
    assert body["trace"]["expanded_query"] == "anything expanded"


# --- POST /library/ingest ---------------------------------------------


async def test_ingest_empty_text_returns_422(web_client, test_user, auth_cookies):  # noqa: ARG001
    """Pydantic-level min_length rejects empty payloads before bundle build."""
    resp = await web_client.post(
        "/api/v1/library/ingest",
        json={"text": ""},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    # ``min_length=1`` on LibraryIngestRequest.text means FastAPI's own
    # validator catches this before the bundle builder runs.
    assert resp.status_code == 422


async def test_ingest_sub_floor_text_returns_422_with_reasons(
    web_client,
    test_user,
    auth_cookies,  # noqa: ARG001
):
    """Below ``upload.min_part_chars`` → silently dropped → empty bundle.

    A 9-char segment falls below the per-part floor; the builder's
    ``_append_text_part`` drops it rather than emitting a ``low_content``
    part (matches the TUI's behaviour for connector phrases between
    placeholders). The bundle ends up empty, so the validator reports a
    single ``empty`` reason — distinct from ``total_too_short`` which
    requires *some* non-trivial part to be present.
    """
    resp = await web_client.post(
        "/api/v1/library/ingest",
        json={"text": "too short"},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert resp.status_code == 422
    body = resp.json()
    detail = body["detail"]
    assert detail["detail"] == "Upload bundle invalid"
    assert "empty" in detail["reasons"]


async def test_ingest_total_too_short_returns_422_with_reasons(
    web_client,
    test_user,
    auth_cookies,  # noqa: ARG001
):
    """A part above part-floor but below total-floor surfaces ``total_too_short``."""
    # 50 chars: clears the 30-char per-part floor, but below the 100-char total.
    resp = await web_client.post(
        "/api/v1/library/ingest",
        json={"text": "a" * 50},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert resp.status_code == 422
    body = resp.json()
    detail = body["detail"]
    assert detail["detail"] == "Upload bundle invalid"
    assert any(r.startswith("total_too_short:") for r in detail["reasons"])


class _FakeRagService:
    """Stand-in for :class:`RagService` that yields a synthetic Done.

    Records ``run`` invocations so the test can assert the part text /
    source_uri the router passed down.
    """

    def __init__(self, store):  # noqa: ARG002 — match real signature
        self.calls: list[dict] = []

    async def run(
        self, user_input, user_id, public=False, language="en", source_uri=None
    ):
        self.calls.append(
            {
                "user_input": user_input,
                "user_id": user_id,
                "public": public,
                "language": language,
                "source_uri": source_uri,
            }
        )
        yield ToolStarted(tool_name="embed_and_store", args_preview=user_id)
        yield ToolCompleted(tool_name="embed_and_store", duration_ms=1, summary="ok")
        yield Done(
            final=IngestionReceipt(
                doc_id="doc-fake",
                chunk_count=3,
                skipped_chunk_count=0,
                embedding_status="ok",
                public=public,
            )
        )


async def test_ingest_inline_text_runs_through_fake_service(
    web_client,
    test_user,
    auth_cookies,
    monkeypatch,  # noqa: ARG001
):
    """Long-enough inline text → one part → one RagService.run call."""
    # Replace the RagService class the router imports lazily.
    import claritymed.orchestrator.services as services_mod

    fake_holder: dict = {}

    class _FakeRagServiceSpy(_FakeRagService):
        def __init__(self, store):
            super().__init__(store)
            fake_holder["instance"] = self

    monkeypatch.setattr(services_mod, "RagService", _FakeRagServiceSpy)

    # Replace make_user_rag_store so the real qdrant build is bypassed.
    import claritymed.stores.user_rag as user_rag_mod

    monkeypatch.setattr(user_rag_mod, "make_user_rag_store", lambda _uid: object())

    body_text = "x" * 500  # well above the 100-char floor and 30-char part floor
    resp = await web_client.post(
        "/api/v1/library/ingest",
        json={"text": body_text},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["added_parts"] == 1
    assert body["added_chunks"] == 3
    assert body["skipped_parts"] == 0
    assert body["failed_parts"] == 0
    assert len(body["parts"]) == 1
    assert body["parts"][0]["status"] == "added"

    # The fake recorded exactly one call with the bundle's part content.
    spy = fake_holder["instance"]
    assert len(spy.calls) == 1
    # Default ingest is scrub-on-ingest + can_cloud=False — the SPA
    # must explicitly pass ``public: true`` to opt into the unscrubbed
    # cloud-queryable path (see ``test_ingest_public_opt_in_propagates``).
    assert spy.calls[0]["public"] is False


async def test_ingest_public_opt_in_propagates(
    web_client,
    test_user,  # noqa: ARG001
    auth_cookies,
    monkeypatch,
):
    """``public: true`` in the request body reaches ``RagService.run``."""
    import claritymed.orchestrator.services as services_mod

    fake_holder: dict = {}

    class _FakeRagServiceSpy(_FakeRagService):
        def __init__(self, store):
            super().__init__(store)
            fake_holder["instance"] = self

    monkeypatch.setattr(services_mod, "RagService", _FakeRagServiceSpy)

    import claritymed.stores.user_rag as user_rag_mod

    monkeypatch.setattr(user_rag_mod, "make_user_rag_store", lambda _uid: object())

    resp = await web_client.post(
        "/api/v1/library/ingest",
        json={"text": "x" * 500, "public": True},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert resp.status_code == 200, resp.text
    spy = fake_holder["instance"]
    assert len(spy.calls) == 1
    assert spy.calls[0]["public"] is True


async def test_ingest_requires_csrf(web_client, test_user, auth_cookies):  # noqa: ARG001
    """CSRF middleware rejects POSTs missing the header."""
    resp = await web_client.post(
        "/api/v1/library/ingest",
        json={"text": "x" * 500},
        cookies=auth_cookies,
    )
    assert resp.status_code == 403


async def test_ingest_store_build_failure_returns_503(
    web_client,
    test_user,  # noqa: ARG001
    auth_cookies,
    monkeypatch,
):
    """When make_user_rag_store raises, the endpoint returns 503."""
    import claritymed.stores.user_rag as user_rag_mod

    def _explode(_uid):
        raise RuntimeError("qdrant client init failed")

    monkeypatch.setattr(user_rag_mod, "make_user_rag_store", _explode)

    resp = await web_client.post(
        "/api/v1/library/ingest",
        json={"text": "x" * 500},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert resp.status_code == 503
    assert "unavailable" in resp.json()["detail"].lower()


async def test_ingest_error_event_from_service_marks_part_failed(
    web_client,
    test_user,  # noqa: ARG001
    auth_cookies,
    monkeypatch,
):
    """When RagService.run yields an Error event, the part status is 'failed'."""
    from claritymed.core.events import Error

    import claritymed.orchestrator.services as services_mod
    import claritymed.stores.user_rag as user_rag_mod

    class ErrorService:
        def __init__(self, store):  # noqa: ARG002
            pass

        async def run(self, *_a, **_kw):
            yield Error(
                error_type="retrieval_failed",
                message="index unavailable",
                retryable=False,
            )

    monkeypatch.setattr(services_mod, "RagService", ErrorService)
    monkeypatch.setattr(user_rag_mod, "make_user_rag_store", lambda _uid: object())

    resp = await web_client.post(
        "/api/v1/library/ingest",
        json={"text": "x" * 500},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["failed_parts"] == 1
    assert body["added_parts"] == 0
    assert body["parts"][0]["status"] == "failed"


async def test_ingest_all_chunks_deduped_marks_part_skipped(
    web_client,
    test_user,  # noqa: ARG001
    auth_cookies,
    monkeypatch,
):
    """Done event with chunk_count=0 and skipped_chunk_count>0 → part 'skipped'."""
    import claritymed.orchestrator.services as services_mod
    import claritymed.stores.user_rag as user_rag_mod

    class DedupeService:
        def __init__(self, store):  # noqa: ARG002
            pass

        async def run(self, *_a, **_kw):
            yield Done(
                final=IngestionReceipt(
                    doc_id="doc-dup",
                    chunk_count=0,
                    skipped_chunk_count=5,  # all chunks deduped
                    embedding_status="ok",
                    public=False,
                )
            )

    monkeypatch.setattr(services_mod, "RagService", DedupeService)
    monkeypatch.setattr(user_rag_mod, "make_user_rag_store", lambda _uid: object())

    resp = await web_client.post(
        "/api/v1/library/ingest",
        json={"text": "x" * 500},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["skipped_parts"] == 1
    assert body["added_parts"] == 0
    assert body["parts"][0]["status"] == "skipped"


async def test_ingest_part_exception_marks_part_failed(
    web_client,
    test_user,  # noqa: ARG001
    auth_cookies,
    monkeypatch,
):
    """Exception raised by RagService.run is caught; part status is 'failed'."""
    import claritymed.orchestrator.services as services_mod
    import claritymed.stores.user_rag as user_rag_mod

    class ExceptionService:
        def __init__(self, store):  # noqa: ARG002
            pass

        async def run(self, *_a, **_kw):
            raise RuntimeError("embedding server down")
            yield  # make it an async generator

    monkeypatch.setattr(services_mod, "RagService", ExceptionService)
    monkeypatch.setattr(user_rag_mod, "make_user_rag_store", lambda _uid: object())

    resp = await web_client.post(
        "/api/v1/library/ingest",
        json={"text": "x" * 500},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["failed_parts"] == 1
    assert body["parts"][0]["status"] == "failed"
    assert "embedding server down" in body["parts"][0]["error"]


async def test_ingest_no_terminal_event_marks_part_failed(
    web_client,
    test_user,  # noqa: ARG001
    auth_cookies,
    monkeypatch,
):
    """Generator that yields only ToolCompleted events (no Done/Error) → 'failed'."""
    import claritymed.orchestrator.services as services_mod
    import claritymed.stores.user_rag as user_rag_mod

    class NoTerminalService:
        def __init__(self, store):  # noqa: ARG002
            pass

        async def run(self, *_a, **_kw):
            yield ToolCompleted(
                tool_name="embed_and_store", duration_ms=1, summary="done"
            )
            # generator ends without a Done or Error event

    monkeypatch.setattr(services_mod, "RagService", NoTerminalService)
    monkeypatch.setattr(user_rag_mod, "make_user_rag_store", lambda _uid: object())

    resp = await web_client.post(
        "/api/v1/library/ingest",
        json={"text": "x" * 500},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["failed_parts"] == 1
    assert body["parts"][0]["status"] == "failed"
    assert "no terminal event" in body["parts"][0]["error"]


async def test_ingest_duplicate_document_marks_part_skipped(
    web_client,
    test_user,  # noqa: ARG001
    auth_cookies,
    monkeypatch,
):
    """DuplicateDocumentError raised by RagService → part status 'skipped'."""
    from claritymed.errors import DuplicateDocumentError

    import claritymed.orchestrator.services as services_mod
    import claritymed.stores.user_rag as user_rag_mod

    class DuplicateService:
        def __init__(self, store):  # noqa: ARG002
            pass

        async def run(self, *_a, **_kw):
            raise DuplicateDocumentError("sha:aabbcc", "doc-old-id")
            yield  # make it an async generator

    monkeypatch.setattr(services_mod, "RagService", DuplicateService)
    monkeypatch.setattr(user_rag_mod, "make_user_rag_store", lambda _uid: object())

    resp = await web_client.post(
        "/api/v1/library/ingest",
        json={"text": "x" * 500},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["skipped_parts"] == 1
    assert body["added_parts"] == 0
    assert body["parts"][0]["status"] == "skipped"


async def test_list_library_skips_collection_entries_without_name(
    web_client,
    test_user,  # noqa: ARG001
    auth_cookies,
    rag_disabled,
    monkeypatch,
):
    """Collection entries with no 'name' key are silently skipped."""
    import claritymed.web.routers.library as lib_mod

    monkeypatch.setattr(
        lib_mod,
        "_load_system_rag_collections",
        lambda: [{"language": "en"}, {"name": "valid_col", "language": "en"}],
    )

    resp = await web_client.get("/api/v1/library", cookies=auth_cookies)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    # The nameless entry is skipped; only valid_col appears.
    assert len(body["system_collections"]) == 1
    assert body["system_collections"][0]["name"] == "valid_col"


# --- _one_line helper -------------------------------------------------


def test_one_line_truncation():
    """Text longer than the limit gets '…' appended."""
    from claritymed.web.routers.library import _one_line

    long_text = "word " * 100  # 500 chars
    result = _one_line(long_text, limit=40)
    assert result.endswith("…")
    assert len(result) == 41  # 40 chars + '…'


def test_one_line_short_passes_through():
    from claritymed.web.routers.library import _one_line

    result = _one_line("short text", limit=100)
    assert result == "short text"


def test_one_line_none_returns_empty():
    from claritymed.web.routers.library import _one_line

    assert _one_line(None, limit=100) == ""


def test_one_line_collapses_newlines():
    from claritymed.web.routers.library import _one_line

    result = _one_line("line one\nline two\nline three", limit=100)
    assert "\n" not in result
    assert "line one" in result


# --- _count_user_rag_chunks unit tests -------------------------------------------


@pytest.mark.asyncio
async def test_count_user_rag_chunks_exception_returns_none(monkeypatch):
    """If make_user_rag_store or list_documents raises, returns None."""
    import claritymed.stores.user_rag as user_rag_mod

    monkeypatch.setattr(
        user_rag_mod,
        "make_user_rag_store",
        lambda _uid: (_ for _ in ()).throw(RuntimeError("qdrant down")),
    )
    from claritymed.web.routers.library import _count_user_rag_chunks

    result = await _count_user_rag_chunks("test-user")
    assert result is None


@pytest.mark.asyncio
async def test_count_user_rag_chunks_sums_chunk_counts(monkeypatch):
    """With docs in the store, returns the sum of chunk_count fields."""
    from unittest.mock import AsyncMock, MagicMock

    import claritymed.stores.user_rag as user_rag_mod

    fake_store = MagicMock()
    fake_store.list_documents = AsyncMock(
        return_value=[{"chunk_count": 5}, {"chunk_count": 3}, {"chunk_count": 0}]
    )
    monkeypatch.setattr(user_rag_mod, "make_user_rag_store", lambda _uid: fake_store)

    from claritymed.web.routers.library import _count_user_rag_chunks

    result = await _count_user_rag_chunks("test-user")
    assert result == 8
