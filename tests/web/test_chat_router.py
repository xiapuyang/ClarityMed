"""Chat SSE router: list/new/turns + streaming + per-session lock."""

from __future__ import annotations

import asyncio
import json
import logging

import pytest

from claritymed.core.events import Done, TokenChunk
from claritymed.orchestrator.services import ChatSession
from claritymed.stores.account import init_user
from claritymed.web.csrf import HEADER_CSRF_TOKEN
from claritymed.web.jwt import create_token
from claritymed.web.middleware import COOKIE_ACCESS_TOKEN

# --- factory / fixtures -----------------------------------------------


class FakeAskService:
    """Yield a scripted event sequence. Used wherever the chat router
    would otherwise build a real AskService against a live provider."""

    def __init__(self, events=None, sleep_s: float = 0.0):
        self._events = events or [
            TokenChunk(text="hello "),
            TokenChunk(text="world"),
            Done(final="hello world"),
        ]
        self._sleep_s = sleep_s

    async def run(self, q: str, user_id: str):  # noqa: ARG002 — match real signature
        for ev in self._events:
            if self._sleep_s:
                await asyncio.sleep(self._sleep_s)
            yield ev


def _install_factory(app, events=None, sleep_s: float = 0.0):
    """Replace the default ask_service_factory with a FakeAskService."""

    def factory(account, chat_session):  # noqa: ARG001
        return FakeAskService(events=events, sleep_s=sleep_s)

    app.state.ask_service_factory = factory


@pytest.fixture
def auth_cookies(test_user):
    token = create_token(test_user.user_id, test_user.language)
    return {COOKIE_ACCESS_TOKEN: token, "csrf_token": "csrf-test"}


def _csrf() -> dict[str, str]:
    return {HEADER_CSRF_TOKEN: "csrf-test"}


# --- session list / new / turns ---------------------------------------


async def test_list_sessions_empty(web_client, test_user, auth_cookies):  # noqa: ARG001
    resp = await web_client.get("/api/v1/sessions", cookies=auth_cookies)
    assert resp.status_code == 200
    assert resp.json() == []


async def test_post_session_then_list_includes_it(web_client, test_user, auth_cookies):  # noqa: ARG001
    resp = await web_client.post(
        "/api/v1/sessions", cookies=auth_cookies, headers=_csrf()
    )
    assert resp.status_code == 200
    sid = resp.json()["session_id"]

    listing = await web_client.get("/api/v1/sessions", cookies=auth_cookies)
    assert any(s["session_id"] == sid for s in listing.json())


async def test_post_session_csrf_required(web_client, test_user, auth_cookies):  # noqa: ARG001
    resp = await web_client.post("/api/v1/sessions", cookies=auth_cookies)
    assert resp.status_code == 403


async def test_get_turns_for_nonexistent_session_returns_404(
    web_client, test_user, auth_cookies
):  # noqa: ARG001
    resp = await web_client.get(
        "/api/v1/sessions/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/turns",
        cookies=auth_cookies,
    )
    assert resp.status_code == 404


async def test_get_turns_for_other_users_session_returns_403(
    web_client, test_user, auth_cookies
):  # noqa: ARG001
    # Create a session as another user, then try to read it.
    other = init_user("other", display_name="Other")
    other_session = ChatSession.new(other.user_id)
    other_session.append_system("init", kind="start")
    resp = await web_client.get(
        f"/api/v1/sessions/{other_session.session_id}/turns",
        cookies=auth_cookies,
    )
    assert resp.status_code == 403


async def test_get_turns_returns_prior_turns(web_client, test_user, auth_cookies):
    session = ChatSession.new(test_user.user_id)
    session.append_user("hello")
    session.append_system("note", kind="info")
    resp = await web_client.get(
        f"/api/v1/sessions/{session.session_id}/turns", cookies=auth_cookies
    )
    assert resp.status_code == 200
    body = resp.json()
    assert [t["role"] for t in body] == ["user", "system"]
    assert body[0]["text"] == "hello"


# --- streaming happy path ---------------------------------------------


async def _new_session_id(web_client, auth_cookies) -> str:
    resp = await web_client.post(
        "/api/v1/sessions", cookies=auth_cookies, headers=_csrf()
    )
    assert resp.status_code == 200
    return resp.json()["session_id"]


def _parse_sse_lines(body: bytes) -> list[dict]:
    events: list[dict] = []
    for chunk in body.decode("utf-8").split("\n\n"):
        chunk = chunk.strip()
        if not chunk.startswith("data: "):
            continue
        events.append(json.loads(chunk[len("data: ") :]))
    return events


async def test_stream_emits_token_chunks_then_done(web_client, test_user, auth_cookies):  # noqa: ARG001
    _install_factory(web_client._transport.app)  # type: ignore[attr-defined]
    sid = await _new_session_id(web_client, auth_cookies)
    resp = await web_client.post(
        f"/api/v1/sessions/{sid}/stream",
        json={"q": "hi"},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["cache-control"] == "no-store"
    assert resp.headers.get("x-accel-buffering") == "no"

    events = _parse_sse_lines(resp.content)
    types = [e["type"] for e in events]
    assert "token_chunk" in types
    assert types[-1] == "done"


async def test_stream_requires_csrf_header(web_client, test_user, auth_cookies):  # noqa: ARG001
    _install_factory(web_client._transport.app)  # type: ignore[attr-defined]
    sid = await _new_session_id(web_client, auth_cookies)
    resp = await web_client.post(
        f"/api/v1/sessions/{sid}/stream",
        json={"q": "hi"},
        cookies=auth_cookies,
    )
    assert resp.status_code == 403


# --- body validation --------------------------------------------------


async def test_stream_empty_q_returns_422(web_client, test_user, auth_cookies):  # noqa: ARG001
    _install_factory(web_client._transport.app)  # type: ignore[attr-defined]
    sid = await _new_session_id(web_client, auth_cookies)
    resp = await web_client.post(
        f"/api/v1/sessions/{sid}/stream",
        json={"q": ""},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert resp.status_code == 422


async def test_stream_oversized_q_returns_422(web_client, test_user, auth_cookies):  # noqa: ARG001
    _install_factory(web_client._transport.app)  # type: ignore[attr-defined]
    sid = await _new_session_id(web_client, auth_cookies)
    resp = await web_client.post(
        f"/api/v1/sessions/{sid}/stream",
        json={"q": "x" * 8001},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert resp.status_code == 422


# --- ownership --------------------------------------------------------


async def test_stream_against_other_users_session_returns_403(
    web_client, test_user, auth_cookies
):  # noqa: ARG001
    other = init_user("other", display_name="Other")
    other_session = ChatSession.new(other.user_id)
    other_session.append_system("init", kind="start")

    _install_factory(web_client._transport.app)  # type: ignore[attr-defined]
    resp = await web_client.post(
        f"/api/v1/sessions/{other_session.session_id}/stream",
        json={"q": "hi"},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert resp.status_code == 403


async def test_stream_nonexistent_session_returns_404(
    web_client, test_user, auth_cookies
):  # noqa: ARG001
    _install_factory(web_client._transport.app)  # type: ignore[attr-defined]
    resp = await web_client.post(
        "/api/v1/sessions/aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/stream",
        json={"q": "hi"},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert resp.status_code == 404


# --- concurrency ------------------------------------------------------


async def test_concurrent_same_session_second_returns_409(
    web_client, test_user, auth_cookies, caplog
):  # noqa: ARG001
    # Make the first stream slow enough that the second arrives mid-flight.
    _install_factory(web_client._transport.app, sleep_s=0.05)  # type: ignore[attr-defined]
    sid = await _new_session_id(web_client, auth_cookies)

    async def hit():
        return await web_client.post(
            f"/api/v1/sessions/{sid}/stream",
            json={"q": "hi"},
            cookies=auth_cookies,
            headers=_csrf(),
        )

    with caplog.at_level(logging.INFO):
        r1, r2 = await asyncio.gather(hit(), hit())
    statuses = sorted([r1.status_code, r2.status_code])
    assert statuses == [200, 409], statuses
    assert any("web.chat.session_busy" in r.message for r in caplog.records)

    # Third stream after first completes should succeed (lock released).
    third = await web_client.post(
        f"/api/v1/sessions/{sid}/stream",
        json={"q": "hi"},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert third.status_code == 200


async def test_concurrent_different_sessions_both_succeed(
    web_client, test_user, auth_cookies
):  # noqa: ARG001
    _install_factory(web_client._transport.app, sleep_s=0.02)  # type: ignore[attr-defined]
    s1 = await _new_session_id(web_client, auth_cookies)
    s2 = await _new_session_id(web_client, auth_cookies)
    assert s1 != s2

    async def hit(sid):
        return await web_client.post(
            f"/api/v1/sessions/{sid}/stream",
            json={"q": "hi"},
            cookies=auth_cookies,
            headers=_csrf(),
        )

    r1, r2 = await asyncio.gather(hit(s1), hit(s2))
    assert r1.status_code == 200
    assert r2.status_code == 200


# --- unknown provider / error path ------------------------------------


async def test_unknown_provider_returns_500_and_audits(
    web_client, test_user, auth_cookies, caplog
):  # noqa: ARG001
    from claritymed.errors import UnknownProviderError

    def boom_factory(account, chat_session):  # noqa: ARG001
        raise UnknownProviderError("test-typo")

    web_client._transport.app.state.ask_service_factory = boom_factory  # type: ignore[attr-defined]
    sid = await _new_session_id(web_client, auth_cookies)

    with caplog.at_level(logging.INFO):
        resp = await web_client.post(
            f"/api/v1/sessions/{sid}/stream",
            json={"q": "hi"},
            cookies=auth_cookies,
            headers=_csrf(),
        )
    assert resp.status_code == 500
    assert any("web.chat.unknown_provider" in r.message for r in caplog.records)


async def test_stream_error_inside_run_yields_error_event(
    web_client, test_user, auth_cookies
):  # noqa: ARG001
    class ExplodingService:
        async def run(self, q, user_id):  # noqa: ARG002
            yield TokenChunk(text="hello")
            raise RuntimeError("provider down")

    web_client._transport.app.state.ask_service_factory = (  # type: ignore[attr-defined]
        lambda a, c: ExplodingService()
    )
    sid = await _new_session_id(web_client, auth_cookies)

    resp = await web_client.post(
        f"/api/v1/sessions/{sid}/stream",
        json={"q": "hi"},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    # The streaming response itself is 200 (errors are surfaced in-band
    # as Error events, not as transport-level failures).
    assert resp.status_code == 200
    events = _parse_sse_lines(resp.content)
    types = [e["type"] for e in events]
    assert "token_chunk" in types
    assert types[-1] == "error"
    assert events[-1]["error_type"] == "llm_error"


# --- auth -------------------------------------------------------------


async def test_list_sessions_unauthenticated_returns_401(web_client):
    resp = await web_client.get("/api/v1/sessions")
    assert resp.status_code == 401


async def test_stream_unauthenticated_returns_401(web_client, test_user):  # noqa: ARG001
    sid = "00000000-0000-0000-0000-000000000000"
    resp = await web_client.post(
        f"/api/v1/sessions/{sid}/stream",
        json={"q": "hi"},
        headers=_csrf() | {"X-CSRF-Token": "csrf-test"},
        cookies={"csrf_token": "csrf-test"},
    )
    assert resp.status_code == 401
