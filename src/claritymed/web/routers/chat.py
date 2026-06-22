"""``/api/v1/sessions`` — list, create, history, and SSE-streamed turns.

Why ``POST`` for the stream endpoint, not ``GET`` with ``?q=…``:
``EventSource`` would force ``GET`` with the user's question in the
query string. Uvicorn (and every reverse proxy) writes the full URL to
access logs at INFO level — that would persist raw, pre-PHI-guard user
input to plaintext logs, breaking the PHI invariant. ``POST`` with the
question in the body keeps PHI off the URL. See plan A5 deepening
update.

Per-session concurrency: ``app.state.busy_sessions`` is a plain
``set[str]``. The route handler does a synchronous check-and-add
before returning the StreamingResponse so a second concurrent stream
against the same session_id gets a 409 immediately rather than
queueing — interleaved tokens would be user-visible garbage. Different
sessions for the same user can still stream concurrently. The
``asyncio.Lock`` shape the plan originally specified would acquire
inside the streaming generator, by which time the route handler has
already committed to a 200 response.

AskService construction is per-request via
``app.state.ask_service_factory`` (defaults to
:func:`build_default_ask_service` which uses the catalog + per-user
preference). Tests inject a fake factory before issuing requests.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Callable

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.responses import StreamingResponse

from claritymed.core.events import Cancelled, Error, Event
from claritymed.core.observability.audit import audit_event
from claritymed.core.schemas import Account
from claritymed.errors import UnknownProviderError
from claritymed.orchestrator.services import AskService, ChatSession
from claritymed.stores.paths import list_user_ids, user_sessions_dir
from claritymed.stores.session_attachments import SessionAttachments
from claritymed.web.deps import get_current_user
from claritymed.web.schemas import (
    InteractionResponse,
    NewSessionResponse,
    SessionMetaResponse,
    StreamRequest,
    TurnResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["chat"])

# Cap the rendered error message in stream-side Error events so a
# pydantic-ai exception text can't echo PHI back to the client.
_ERROR_MESSAGE_MAX = 200

# Sentinel marking ``app.state.rag_strategy`` as "not yet built". Used
# instead of ``None`` because ``None`` is a *valid* cached value when
# ``rag.enabled=false`` — caching it stops every request from re-reading
# ``retrieval.yaml`` and re-building the retriever object.
RAG_UNSET = object()


AskServiceFactory = Callable[..., AskService]
"""Build an ``AskService`` for one request.

Signature:

    factory(account, chat_session, *, provider_override=None,
            prompt_channel=None, tool_approval_channel=None) -> AskService

The keyword-only args are optional so production and test factories can
opt in piecemeal. ``provider_override`` is the per-turn override from
:class:`StreamRequest.provider_id`; channels are the web-side rendezvous
shims installed by ``event_gen``.
"""


# --- routes -----------------------------------------------------------


@router.get("/sessions", response_model=list[SessionMetaResponse])
async def list_sessions(
    account: Account = Depends(get_current_user),
) -> list[SessionMetaResponse]:
    """List the current user's sessions, newest first."""
    metas = ChatSession.list_sessions(account.user_id)
    return [
        SessionMetaResponse(
            session_id=m.session_id,
            preview=m.preview,
            modified_at=m.modified_at.isoformat(),
        )
        for m in metas
    ]


@router.post("/sessions", response_model=NewSessionResponse)
async def new_session(
    account: Account = Depends(get_current_user),
) -> NewSessionResponse:
    """Create a fresh ChatSession and persist a ``start`` system event.

    Without the explicit append, ``ChatSession.new`` produces only an
    in-memory object; the file is created lazily on the first turn.
    The plan requires that a freshly-created session appears in the
    next ``GET /api/v1/sessions`` listing, so we write a start event
    here. The event has ``kind="start"`` and is filtered out of
    ``load_turns`` (only ``kind="info"`` system events surface there).
    """
    session = ChatSession.new(account.user_id)
    session.append_system("Session created", kind="start")
    return NewSessionResponse(session_id=session.session_id)


@router.get(
    "/sessions/{session_id}/turns",
    response_model=list[TurnResponse],
)
async def list_turns(
    session_id: str,
    account: Account = Depends(get_current_user),
) -> list[TurnResponse]:
    _validate_session_ownership(account.user_id, session_id)
    session = ChatSession.resume(account.user_id, session_id)
    turns = session.load_turns()
    return [
        TurnResponse(role=t.role, text=t.text, cancelled=t.cancelled) for t in turns
    ]


@router.post("/sessions/{session_id}/stream")
async def stream(
    session_id: str,
    req: StreamRequest,
    request: Request,
    account: Account = Depends(get_current_user),
) -> StreamingResponse:
    """Per-turn SSE-formatted byte stream over POST.

    Body carries ``q`` (the user's question) so PHI never enters the
    URL. CSRF is enforced by middleware. The actual streaming happens
    inside ``event_gen``; the lock is acquired on the first generator
    step so 409-on-busy fires before any bytes are emitted.
    """
    _validate_session_ownership(account.user_id, session_id)

    # ``busy_sessions`` is a plain ``set[str]`` (not an asyncio.Lock
    # dict) so the check-and-add below is synchronous and free of
    # interleaving in the single-threaded asyncio loop. An asyncio.Lock
    # would force us to acquire INSIDE the streaming generator, which
    # means the route handler returns 200 before contention is even
    # checked — exactly the bug this design avoids.
    busy: set[str] = request.app.state.busy_sessions
    if session_id in busy:
        audit_event("web.chat.session_busy", payload={"session_id": session_id})
        raise HTTPException(status_code=409, detail="Session busy")
    busy.add(session_id)

    factory: AskServiceFactory | None = getattr(
        request.app.state, "ask_service_factory", None
    )
    if factory is None:
        busy.discard(session_id)
        logger.error("ask_service_factory is not installed on app.state")
        raise HTTPException(status_code=500, detail="Configuration error")

    chat_session = ChatSession.resume(account.user_id, session_id)

    # Build per-stream web rendezvous channels — defined here so the
    # POST /interactions endpoint can find the pending future on
    # ``app.state.web_interactions``. Headless tests pass ``None`` via
    # the factory shape and skip the rendezvous wiring entirely.
    from claritymed.web.channels import build_web_channels

    interactions: dict = getattr(request.app.state, "web_interactions", None) or {}
    request.app.state.web_interactions = interactions
    interaction_event_queue: asyncio.Queue[Event] = asyncio.Queue()
    prompt_channel, approval_channel = build_web_channels(
        user_id=account.user_id,
        session_id=session_id,
        emit_queue=interaction_event_queue,
        rendezvous=interactions,
    )

    # Resolve the cached RAG strategy *before* invoking the factory so
    # the cache miss serialises across concurrent first requests rather
    # than every parallel turn building its own strategy. The lazy model
    # builder is only called on cache miss — keeps the steady-state path
    # (cache hit) free of pydantic-ai imports + provider lookup.
    def _model_for_strategy():
        from claritymed.core.llm.model import build_model
        from claritymed.stores.models import resolve_provider

        provider = resolve_provider(account=account, override=req.provider_id)
        return build_model(provider)

    try:
        strategy = await _resolve_strategy(request.app.state, _model_for_strategy)
    except UnknownProviderError:
        busy.discard(session_id)
        audit_event(
            "web.chat.unknown_provider",
            payload={
                "provider_id": req.provider_id or account.provider_id or "<default>",
            },
        )
        status_code = 422 if req.provider_id else 500
        raise HTTPException(
            status_code=status_code, detail="Unknown provider"
        ) from None

    try:
        ask_service = factory(
            account,
            chat_session,
            provider_override=req.provider_id,
            language_override=req.language,
            prompt_channel=prompt_channel,
            tool_approval_channel=approval_channel,
            strategy=strategy,
        )
    except UnknownProviderError:
        busy.discard(session_id)
        audit_event(
            "web.chat.unknown_provider",
            payload={
                "provider_id": req.provider_id or account.provider_id or "<default>",
            },
        )
        # Per-turn override is the user's typed/clicked input — surface
        # the typo as 422 so the SPA can flag it inline. The persisted
        # account.provider_id path is operator config; that one stays
        # as 500 because it should have been caught by PATCH /me.
        status_code = 422 if req.provider_id else 500
        raise HTTPException(
            status_code=status_code, detail="Unknown provider"
        ) from None

    user_id = account.user_id
    q = _prepend_attachment_placeholders(user_id, session_id, req.attachment_ids, req.q)

    async def event_gen() -> AsyncIterator[bytes]:
        """Drive ``ask_service.run`` and merge in interaction events.

        Channel events live on ``interaction_event_queue`` rather than
        the AskService event loop because the channels are constructed
        by the web layer, outside the orchestrator. The two streams are
        merged so an ``InteractionRequested`` arrives in user-visible
        order with the surrounding ``ToolStarted`` / ``LlmCallStarted``
        rows.

        ``ask_service.run`` is an async generator that ``apply_context``
        / ``reset_context`` its request-scoped ContextVars across the
        yield boundary. Async generators have no Context of their own —
        each ``__anext__`` runs in the caller's. We therefore drive the
        generator from a single long-lived pump task so the set and
        reset happen in the same asyncio Context (driving it from a
        fresh ``create_task`` per event raises
        ``Token was created in a different Context``).
        """
        ASK, INT, DONE, ERR = "ask", "int", "done", "err"
        merged: asyncio.Queue[tuple[str, object]] = asyncio.Queue()

        async def pump_ask() -> None:
            try:
                async for ev in ask_service.run(q, user_id):
                    await merged.put((ASK, ev))
            except BaseException as exc:  # noqa: BLE001
                await merged.put((ERR, exc))
            finally:
                await merged.put((DONE, None))

        async def pump_interaction() -> None:
            while True:
                ev = await interaction_event_queue.get()
                await merged.put((INT, ev))

        ask_pump = asyncio.create_task(pump_ask())
        int_pump = asyncio.create_task(pump_interaction())
        ask_done = False
        try:
            while True:
                if ask_done and interaction_event_queue.empty() and merged.empty():
                    break
                if ask_done:
                    # AskService finished. Drain any remaining interaction
                    # events so a late-arriving cancellation / completion
                    # signal still flushes, then stop.
                    try:
                        kind, payload = await asyncio.wait_for(
                            merged.get(), timeout=0.05
                        )
                    except asyncio.TimeoutError:
                        break
                else:
                    kind, payload = await merged.get()
                if kind == DONE:
                    ask_done = True
                    continue
                if kind == ERR:
                    raise payload  # type: ignore[misc]
                yield _format_sse(payload)  # type: ignore[arg-type]
        except asyncio.CancelledError:
            try:
                yield _format_sse(Cancelled(reason="client_disconnected"))
            except Exception:  # noqa: BLE001
                pass
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("stream error for user=%s session=%s", user_id, session_id)
            yield _format_sse(
                Error(
                    error_type="llm_error",
                    message=_truncate_error(exc),
                    retryable=True,
                )
            )
        finally:
            if not ask_pump.done():
                ask_pump.cancel()
            if not int_pump.done():
                int_pump.cancel()
            # Reject any pending interactions for this stream — the
            # rendezvous holders are tied to the now-dead generator.
            for iid in [
                iid
                for iid, rec in interactions.items()
                if rec.get("session_id") == session_id
            ]:
                rec = interactions.pop(iid, None)
                fut = rec.get("future") if rec else None
                if fut is not None and not fut.done():
                    fut.cancel()
            busy.discard(session_id)

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


# --- helpers ----------------------------------------------------------


def _format_sse(event: Event) -> bytes:
    """``data: <json>\\n\\n`` framing — the wire format EventSource expects."""
    return f"data: {event.model_dump_json()}\n\n".encode("utf-8")


def _truncate_error(exc: BaseException) -> str:
    msg = str(exc) or type(exc).__name__
    return msg[:_ERROR_MESSAGE_MAX]


@router.post(
    "/sessions/{session_id}/interactions/{interaction_id}",
    status_code=204,
)
async def respond_interaction(
    session_id: str,
    interaction_id: str,
    body: InteractionResponse,
    request: Request,
    account: Account = Depends(get_current_user),
) -> None:
    """Resolve a pending ``ask_user_question`` or ``tool_approval`` interaction.

    The stream task is parked on an ``asyncio.Future``; this endpoint
    finds the record on ``app.state.web_interactions`` (404 if missing),
    asserts ownership (403 if different user/session), validates the
    ``kind`` matches what the channel raised, and sets the future. The
    stream resumes and the next ``ask_service`` event flushes out the
    SSE pipe.
    """
    _validate_session_ownership(account.user_id, session_id)
    interactions: dict = getattr(request.app.state, "web_interactions", {})
    record = interactions.get(interaction_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Interaction not found")
    if (
        record.get("user_id") != account.user_id
        or record.get("session_id") != session_id
    ):
        raise HTTPException(status_code=403, detail="Forbidden")
    if record.get("kind") != body.kind:
        raise HTTPException(
            status_code=409,
            detail=f"Interaction kind mismatch (expected {record['kind']!r})",
        )
    fut: asyncio.Future = record["future"]
    if fut.done():
        # Stream already torn down or duplicate POST. 410 is more
        # informative than 404 here — the resource existed but is
        # spent.
        raise HTTPException(status_code=410, detail="Interaction already resolved")
    fut.set_result(body.payload)
    interactions.pop(interaction_id, None)


def _prepend_attachment_placeholders(
    user_id: str,
    session_id: str,
    attachment_ids: list[str],
    text: str,
) -> str:
    """Inject ``[Image sha:…]`` / ``[File sha:…]`` for each attachment id.

    Returns the user's text with one placeholder per id prepended in the
    order the client sent them. Unknown ids (not in this session's
    attachments tray) are silently skipped — the SPA should never send
    them, but the alternative is failing the whole turn for one stale
    chip, which is worse UX than a missing inline placeholder.

    The placeholder format mirrors the TUI's paste path so
    :class:`AttachmentsFeature` expands them via the same regex.
    """
    if not attachment_ids:
        return text
    try:
        rows = SessionAttachments(user_id, session_id).list()
    except Exception:  # noqa: BLE001
        logger.exception("attachment list failed for %s/%s", user_id, session_id)
        return text
    by_sha = {r.sha256: r for r in rows}
    placeholders: list[str] = []
    for sha in attachment_ids:
        row = by_sha.get(sha)
        if row is None:
            logger.warning(
                "stream request referenced unknown attachment %s for %s/%s",
                sha[:8],
                user_id,
                session_id,
            )
            continue
        kind = "Image" if (row.mime or "").startswith("image/") else "File"
        placeholders.append(f"[{kind} sha:{sha[:8]}]")
    if not placeholders:
        return text
    return " ".join(placeholders) + ("\n\n" + text if text.strip() else "")


def _validate_session_ownership(user_id: str, session_id: str) -> None:
    """Raise 404 if the user has no such session; 403 if another user owns it.

    The 403 path is more informative for the legitimate user (they
    typo'd an id that exists elsewhere) but requires a small cross-user
    scan. Acceptable for the project's user-count.
    """
    own = user_sessions_dir(user_id) / f"{session_id}.jsonl"
    if own.exists():
        return
    for other in list_user_ids():
        if other == user_id:
            continue
        if (user_sessions_dir(other) / f"{session_id}.jsonl").exists():
            raise HTTPException(status_code=403, detail="Forbidden")
    raise HTTPException(status_code=404, detail="Session not found")


# --- RAG strategy cache -----------------------------------------------


async def _resolve_strategy(app_state, model_builder: Callable[[], object]):
    """Return the process-wide cached ``RagStrategy``, building once on miss.

    The web worker is stateless across HTTP requests, but the RAG
    strategy owns expensive state (qdrant clients, BGE embedder/reranker
    HTTP clients, optional HyDE sub-agent) that *must* outlive a single
    turn. Building per request would (a) burn ~tens of ms per call on
    object wiring, (b) re-open qdrant's local-mode file lock and risk
    contention with a still-running prior request, and (c) re-do
    embedder/reranker TCP handshakes the connection pool exists to
    amortise.

    Returns ``None`` when ``rag.enabled=false`` — also cached so
    subsequent requests don't re-read ``retrieval.yaml``.

    ``app.state.rag_strategy_lock`` serialises concurrent first-misses
    so two simultaneous requests don't both pay the build cost; the
    second waiter sees the cached value when it reacquires.

    ``model_builder`` is invoked only on cache miss. Hot path (cache
    hit) never imports pydantic-ai or hits the provider catalog —
    matters because the steady-state cost would otherwise be paid by
    every stream request.

    ``build_rag_strategy`` itself is sync. Local-mode qdrant only
    acquires its file lock at query time (not at construction), so we
    don't need ``asyncio.to_thread``; running inline keeps the lock
    release order obvious.
    """
    cached = app_state.rag_strategy
    if cached is not RAG_UNSET:
        return cached
    async with app_state.rag_strategy_lock:
        cached = app_state.rag_strategy
        if cached is not RAG_UNSET:
            return cached
        # Read ``rag.enabled`` *before* invoking ``model_builder`` so the
        # rag-off path never pays the model-resolve cost (and never
        # raises ``MissingApiKeyError`` for a provider we'd have ignored
        # anyway). ``build_rag_strategy`` would short-circuit on the
        # same flag, but only after Python eagerly evaluates
        # ``model_builder()`` as an argument.
        from claritymed.core.rag import load_retrieval_config
        from claritymed.orchestrator.services import build_rag_strategy

        if not load_retrieval_config().rag.enabled:
            app_state.rag_strategy = None
            return None
        strategy = build_rag_strategy(model=model_builder())
        app_state.rag_strategy = strategy
        return strategy


# --- default factory --------------------------------------------------


def build_default_ask_service(
    account: Account,
    chat_session: ChatSession,
    *,
    provider_override: str | None = None,
    language_override: str | None = None,
    prompt_channel=None,
    tool_approval_channel=None,
    strategy=None,
) -> AskService:
    """Production ``ask_service_factory`` — per-request AskService build.

    Lifespan installs this on ``app.state.ask_service_factory``; tests
    override before issuing chat requests. Per-request construction
    (rather than provider-keyed singletons) keeps each request's
    ``chat_session`` private to that request — singletons would have
    to mutate shared state, which is racy across concurrent streams.

    ``provider_override`` is the per-turn id from ``StreamRequest``;
    unknown ids raise ``UnknownProviderError`` which the caller maps to
    422. ``prompt_channel`` / ``tool_approval_channel`` are the web
    rendezvous shims built in ``event_gen``. ``strategy`` is the
    process-cached ``RagStrategy`` (or ``None`` when RAG is off);
    resolved by :func:`get_or_build_rag_strategy` in the stream handler.

    Delegates to :func:`build_ask_service` so the toolset stack (vision,
    symptoms, translation, profile_context, RAG mode) matches the TUI.
    """
    from claritymed.core.llm.model import build_model
    from claritymed.orchestrator.services import build_ask_service
    from claritymed.stores.models import resolve_provider

    provider = resolve_provider(account=account, override=provider_override)
    model = build_model(provider)
    return build_ask_service(
        model=model,
        language=language_override or account.language,
        chat_session=chat_session,
        provider=provider,
        prompt_channel=prompt_channel,
        tool_approval_channel=tool_approval_channel,
        strategy=strategy,
    )
