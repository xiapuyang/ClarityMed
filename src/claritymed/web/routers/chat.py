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
from claritymed.web.deps import get_current_user
from claritymed.web.schemas import (
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


AskServiceFactory = Callable[[Account, ChatSession], AskService]


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
    try:
        ask_service = factory(account, chat_session)
    except UnknownProviderError:
        busy.discard(session_id)
        audit_event(
            "web.chat.unknown_provider",
            payload={"provider_id": account.provider_id or "<default>"},
        )
        raise HTTPException(status_code=500, detail="Configuration error") from None

    user_id = account.user_id
    q = req.q

    async def event_gen() -> AsyncIterator[bytes]:
        try:
            async for event in ask_service.run(q, user_id):
                yield _format_sse(event)
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


# --- default factory --------------------------------------------------


def build_default_ask_service(
    account: Account, chat_session: ChatSession
) -> AskService:
    """Production ``ask_service_factory`` — per-request AskService build.

    Lifespan installs this on ``app.state.ask_service_factory``; tests
    override before issuing chat requests. Per-request construction
    (rather than provider-keyed singletons) keeps each request's
    ``chat_session`` private to that request — singletons would have
    to mutate shared state, which is racy across concurrent streams.
    """
    from claritymed.core.llm.model import build_model
    from claritymed.stores.models import resolve_provider

    provider = resolve_provider(account=account)
    model = build_model(provider)
    return AskService(
        model=model,
        language=account.language,
        chat_session=chat_session,
        provider_id=provider.id,
        model_name=provider.model,
        provider_config=provider,
    )
