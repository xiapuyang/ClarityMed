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
    """Replace the default ask_service_factory with a FakeAskService.

    The production factory accepts ``provider_override`` /
    ``prompt_channel`` / ``tool_approval_channel`` kwargs (web layer
    plumbing for interactions + per-turn model picker). The fake
    accepts the same shape via ``**_`` so router tests don't need to
    care which kwargs land.
    """

    def factory(account, chat_session, **_):  # noqa: ARG001
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
    # ``differential`` is optional and defaults to null for non-symptoms turns.
    for t in body:
        assert t.get("differential") is None


async def test_get_turns_carries_persisted_differential(
    web_client, test_user, auth_cookies
):
    """Symptoms turn's DifferentialReady payload round-trips through the
    JSONL log and the GET /turns projection, so the SPA can rehydrate
    cards after a page refresh without a rerun."""
    from claritymed.core.events import (
        DifferentialCard,
        DifferentialReady,
        DifferentialSessionMeta,
    )

    payload = DifferentialReady(
        cards=[
            DifferentialCard(
                condition_id="pneumonia",
                condition_name="Pneumonia",
                probability=0.72,
                severity_tier="Urgent",
                confidence_bucket="high",
                confidence_label="Confident",
                headline="Likely Pneumonia",
                report="Infection of the lung tissue.",
            ),
        ],
        session=DifferentialSessionMeta(),
        language="en",
    )
    session = ChatSession.new(test_user.user_id)
    session.append_user("fever + chest pain")
    session.append_assistant(
        text="The follow-up surfaced one finding.",
        messages_json=b"",  # not needed for the turns projection
        model="omlx",
        provider_id="omlx",
        usage=None,
        differential=payload,
    )

    resp = await web_client.get(
        f"/api/v1/sessions/{session.session_id}/turns", cookies=auth_cookies
    )
    assert resp.status_code == 200
    body = resp.json()
    assistant = [t for t in body if t["role"] == "assistant"]
    assert assistant, "no assistant turn returned"
    diff = assistant[0]["differential"]
    assert diff is not None
    assert diff["type"] == "differential_ready"
    assert len(diff["cards"]) == 1
    assert diff["cards"][0]["condition_id"] == "pneumonia"
    assert "citations" not in diff["cards"][0]
    assert "suggestion" not in diff["cards"][0]
    assert diff["session"]["banner_key"] is None
    assert diff["language"] == "en"


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

    def boom_factory(account, chat_session, **_):  # noqa: ARG001
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


async def test_factory_raising_generic_exception_clears_busy_sessions(
    web_client, test_user, auth_cookies
):  # noqa: ARG001
    """Regression: unhandled factory exception must not orphan busy_sessions.

    The route handler adds session_id to busy_sessions synchronously
    before invoking the AskService factory (so 409-on-busy fires
    before any bytes stream). If the factory then raises anything
    other than a specifically-handled error (UnknownProviderError,
    etc.), the discard call must still happen — otherwise the next
    request on the same session receives a permanent 409 with no way
    to recover short of restarting the backend.

    Reproduces the failure mode we saw when a stale in-process
    ``claritymed.errors`` module cache made ``symptoms_plugin`` fail
    to import ``SymptomsCatalogValidationError`` mid-factory.
    """

    def boom_factory(account, chat_session, **_):  # noqa: ARG001
        raise RuntimeError("factory blew up for reasons")

    web_client._transport.app.state.ask_service_factory = boom_factory  # type: ignore[attr-defined]
    sid = await _new_session_id(web_client, auth_cookies)

    # First request: the factory crashes. In production the ASGI stack
    # surfaces this as HTTP 500 via the default exception handler; in
    # this test's ASGITransport the exception is re-raised through the
    # client. Either way, ``busy_sessions`` MUST be cleaned up before
    # control leaves the route handler — otherwise the next request on
    # the same session hits a permanent 409 with no recovery short of
    # a backend restart.
    with pytest.raises(RuntimeError, match="factory blew up"):
        await web_client.post(
            f"/api/v1/sessions/{sid}/stream",
            json={"q": "hi"},
            cookies=auth_cookies,
            headers=_csrf(),
        )
    assert sid not in web_client._transport.app.state.busy_sessions, (  # type: ignore[attr-defined]
        "busy_sessions leaked session_id after factory exception"
    )

    # Second request: SHOULD NOT be 409 — busy_sessions was cleared.
    _install_factory(web_client._transport.app)  # type: ignore[attr-defined]
    r2 = await web_client.post(
        f"/api/v1/sessions/{sid}/stream",
        json={"q": "hi"},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert r2.status_code == 200, (
        f"expected 200 after crashed factory cleaned up busy_sessions, "
        f"got {r2.status_code} (regressed to permanent-409 bug)"
    )


async def test_stream_error_inside_run_yields_error_event(
    web_client, test_user, auth_cookies
):  # noqa: ARG001
    class ExplodingService:
        async def run(self, q, user_id):  # noqa: ARG002
            yield TokenChunk(text="hello")
            raise RuntimeError("provider down")

    web_client._transport.app.state.ask_service_factory = (  # type: ignore[attr-defined]
        lambda a, c, **_: ExplodingService()
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


# --- provider override -------------------------------------------------


async def test_stream_unknown_provider_override_returns_422(
    web_client, test_user, auth_cookies
):  # noqa: ARG001
    """Per-turn override is user input; surface the typo as 422, not 500."""
    from claritymed.errors import UnknownProviderError

    def factory(account, chat_session, *, provider_override=None, **_):  # noqa: ARG001
        if provider_override == "not-a-real-provider":
            raise UnknownProviderError("not-a-real-provider")
        return FakeAskService()

    web_client._transport.app.state.ask_service_factory = factory  # type: ignore[attr-defined]
    sid = await _new_session_id(web_client, auth_cookies)
    resp = await web_client.post(
        f"/api/v1/sessions/{sid}/stream",
        json={"q": "hi", "provider_id": "not-a-real-provider"},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert resp.status_code == 422


async def test_stream_provider_override_passed_to_factory(
    web_client, test_user, auth_cookies
):  # noqa: ARG001
    """StreamRequest.provider_id is plumbed into the factory."""
    seen: dict[str, str | None] = {}

    def factory(account, chat_session, *, provider_override=None, **_):  # noqa: ARG001
        seen["override"] = provider_override
        return FakeAskService()

    web_client._transport.app.state.ask_service_factory = factory  # type: ignore[attr-defined]
    sid = await _new_session_id(web_client, auth_cookies)
    await web_client.post(
        f"/api/v1/sessions/{sid}/stream",
        json={"q": "hi", "provider_id": "omlx"},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert seen["override"] == "omlx"


# --- interactions endpoint --------------------------------------------


async def test_respond_interaction_404_when_missing(
    web_client, test_user, auth_cookies
):  # noqa: ARG001
    sid = await _new_session_id(web_client, auth_cookies)
    resp = await web_client.post(
        f"/api/v1/sessions/{sid}/interactions/no-such-id",
        json={"kind": "ask_user_question", "payload": {}},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert resp.status_code == 404


async def test_respond_interaction_resumes_stream(web_client, test_user, auth_cookies):  # noqa: ARG001
    """End-to-end rendezvous: stream → InteractionRequested → POST → resume."""
    from claritymed.core.events import Done, TokenChunk
    from claritymed.web.channels import WebPromptChannel

    captured: dict = {}

    async def fake_stream_run(channel, q, user_id):  # noqa: ARG001
        yield TokenChunk(text="thinking… ")
        # Real PromptChannel.ask: pause until POST resolves the future.
        from claritymed.core.interaction.schemas import (
            AskUserQuestionInput,
            Question,
            QuestionOption,
        )

        payload = AskUserQuestionInput(
            questions=[
                Question(
                    question="What is your symptom?",
                    header="Symptom",
                    options=[
                        QuestionOption(label="Fever", description="High temperature"),
                        QuestionOption(label="Cough", description="Persistent cough"),
                    ],
                )
            ]
        )
        answer = await channel.ask(payload)
        captured["answer"] = answer.model_dump(mode="json")
        yield TokenChunk(text="ok")
        yield Done(final="ok")

    class WrappedService:
        def __init__(self, channel):
            self._channel = channel

        async def run(self, q, user_id):
            async for ev in fake_stream_run(self._channel, q, user_id):
                yield ev

    def factory(account, chat_session, *, prompt_channel=None, **_):  # noqa: ARG001
        assert isinstance(prompt_channel, WebPromptChannel)
        return WrappedService(prompt_channel)

    app = web_client._transport.app  # type: ignore[attr-defined]
    app.state.ask_service_factory = factory
    sid = await _new_session_id(web_client, auth_cookies)

    # Issue the stream and POST the answer once we see the prompt event.
    async def respond_when_pending():
        # Poll the rendezvous until the channel registers the interaction.
        deadline = asyncio.get_running_loop().time() + 5.0
        while asyncio.get_running_loop().time() < deadline:
            recs = list(app.state.web_interactions.items())
            if recs:
                iid, _ = recs[0]
                return await web_client.post(
                    f"/api/v1/sessions/{sid}/interactions/{iid}",
                    json={
                        "kind": "ask_user_question",
                        "payload": {
                            "answers": {"What is your symptom?": "Fever"},
                        },
                    },
                    cookies=auth_cookies,
                    headers=_csrf(),
                )
            await asyncio.sleep(0.01)
        raise AssertionError("interaction never registered")

    async def issue_stream():
        return await web_client.post(
            f"/api/v1/sessions/{sid}/stream",
            json={"q": "tell me"},
            cookies=auth_cookies,
            headers=_csrf(),
        )

    stream_resp, post_resp = await asyncio.gather(
        issue_stream(), respond_when_pending()
    )
    assert stream_resp.status_code == 200
    assert post_resp.status_code == 204
    events = _parse_sse_lines(stream_resp.content)
    kinds = [e["type"] for e in events]
    assert "interaction_requested" in kinds
    assert kinds[-1] == "done"
    assert captured["answer"]["answers"]["What is your symptom?"] == "Fever"


# --- tokens_used event ------------------------------------------------


async def test_stream_emits_tokens_used_before_done(
    web_client, test_user, auth_cookies
):  # noqa: ARG001
    """AskService yields TokensUsed before Done; router relays it."""
    from claritymed.core.events import Done, TokenChunk, TokensUsed

    def factory(account, chat_session, **_):  # noqa: ARG001
        return FakeAskService(
            events=[
                TokenChunk(text="hi"),
                TokensUsed(
                    model_name="m",
                    provider_id="p",
                    input_tokens=10,
                    output_tokens=5,
                    total_tokens=15,
                    context_window=128_000,
                ),
                Done(final="hi"),
            ]
        )

    web_client._transport.app.state.ask_service_factory = factory  # type: ignore[attr-defined]
    sid = await _new_session_id(web_client, auth_cookies)
    resp = await web_client.post(
        f"/api/v1/sessions/{sid}/stream",
        json={"q": "hi"},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    events = _parse_sse_lines(resp.content)
    types = [e["type"] for e in events]
    assert "tokens_used" in types
    tu = next(e for e in events if e["type"] == "tokens_used")
    assert tu["input_tokens"] == 10
    assert tu["context_window"] == 128_000


# --- attachments inline ------------------------------------------------


async def test_stream_attachment_ids_prepend_placeholder(
    web_client, test_user, auth_cookies
):  # noqa: ARG001
    """attachment_ids → [Image sha:...] prepended to q before AskService."""
    from claritymed.stores.session_attachments import SessionAttachments

    seen: dict[str, str] = {}

    class CaptureService:
        async def run(self, q, user_id):  # noqa: ARG002
            seen["q"] = q
            from claritymed.core.events import Done

            yield Done(final="ok")

    def factory(account, chat_session, **_):  # noqa: ARG001
        return CaptureService()

    web_client._transport.app.state.ask_service_factory = factory  # type: ignore[attr-defined]
    sid = await _new_session_id(web_client, auth_cookies)

    sha = "a" * 64
    SessionAttachments(test_user.user_id, sid).add(
        sha256=sha,
        filename="scan.png",
        mime="image/png",
        size=42,
        source="upload",
    )

    resp = await web_client.post(
        f"/api/v1/sessions/{sid}/stream",
        json={"q": "what is this", "attachment_ids": [sha]},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert resp.status_code == 200
    assert "[Image sha:aaaaaaaa]" in seen["q"]


async def test_stream_unknown_attachment_sha_skipped(
    web_client, test_user, auth_cookies
):  # noqa: ARG001
    """Unknown attachment SHA is silently skipped and q is passed unmodified."""
    seen: dict[str, str] = {}

    class CaptureService:
        async def run(self, q, user_id):  # noqa: ARG002
            seen["q"] = q
            yield Done(final="ok")

    web_client._transport.app.state.ask_service_factory = (  # type: ignore[attr-defined]
        lambda a, c, **_: CaptureService()
    )
    sid = await _new_session_id(web_client, auth_cookies)

    # Send an attachment_id that was never registered in this session.
    resp = await web_client.post(
        f"/api/v1/sessions/{sid}/stream",
        json={"q": "what is this", "attachment_ids": ["b" * 64]},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert resp.status_code == 200
    # No placeholder was prepended; q is unchanged.
    assert seen.get("q") == "what is this"


async def test_stream_attachment_non_image_file_gets_file_tag(
    web_client, test_user, auth_cookies
):
    """Non-image MIME type gets a [File sha:...] placeholder, not [Image sha:...]."""
    from claritymed.stores.session_attachments import SessionAttachments

    seen: dict[str, str] = {}

    class CaptureService:
        async def run(self, q, user_id):  # noqa: ARG002
            seen["q"] = q
            yield Done(final="ok")

    web_client._transport.app.state.ask_service_factory = (  # type: ignore[attr-defined]
        lambda a, c, **_: CaptureService()
    )
    sid = await _new_session_id(web_client, auth_cookies)

    sha = "c" * 64
    SessionAttachments(test_user.user_id, sid).add(
        sha256=sha,
        filename="report.pdf",
        mime="application/pdf",
        size=100,
        source="upload",
    )

    resp = await web_client.post(
        f"/api/v1/sessions/{sid}/stream",
        json={"q": "summarise this", "attachment_ids": [sha]},
        cookies=auth_cookies,
        headers=_csrf(),
    )
    assert resp.status_code == 200
    assert "[File sha:cccccccc]" in seen["q"]


async def test_stream_factory_none_returns_500(web_client, test_user, auth_cookies):  # noqa: ARG001
    """When ask_service_factory is not set on app.state, stream returns 500."""
    app = web_client._transport.app  # type: ignore[attr-defined]
    original = app.state.ask_service_factory
    app.state.ask_service_factory = None
    sid = await _new_session_id(web_client, auth_cookies)
    try:
        resp = await web_client.post(
            f"/api/v1/sessions/{sid}/stream",
            json={"q": "hi"},
            cookies=auth_cookies,
            headers=_csrf(),
        )
    finally:
        app.state.ask_service_factory = original
    assert resp.status_code == 500


async def test_respond_interaction_403_wrong_session(
    web_client, test_user, auth_cookies
):
    """Interaction registered for a different session returns 403."""
    import asyncio as _asyncio

    sid = await _new_session_id(web_client, auth_cookies)
    app = web_client._transport.app  # type: ignore[attr-defined]
    loop = _asyncio.get_event_loop()
    fut = loop.create_future()
    iid = "wrong-session-iid"
    app.state.web_interactions[iid] = {
        "user_id": test_user.user_id,
        "session_id": "completely-different-session",
        "kind": "ask_user_question",
        "future": fut,
    }
    try:
        resp = await web_client.post(
            f"/api/v1/sessions/{sid}/interactions/{iid}",
            json={"kind": "ask_user_question", "payload": {"answers": {}}},
            cookies=auth_cookies,
            headers=_csrf(),
        )
        assert resp.status_code == 403
    finally:
        app.state.web_interactions.pop(iid, None)
        if not fut.done():
            fut.cancel()


async def test_respond_interaction_409_kind_mismatch(
    web_client, test_user, auth_cookies
):
    """Posting a 'tool_approval' payload for an 'ask_user_question' record returns 409."""
    import asyncio as _asyncio

    sid = await _new_session_id(web_client, auth_cookies)
    app = web_client._transport.app  # type: ignore[attr-defined]
    loop = _asyncio.get_event_loop()
    fut = loop.create_future()
    iid = "kind-mismatch-iid"
    app.state.web_interactions[iid] = {
        "user_id": test_user.user_id,
        "session_id": sid,
        "kind": "ask_user_question",
        "future": fut,
    }
    try:
        resp = await web_client.post(
            f"/api/v1/sessions/{sid}/interactions/{iid}",
            json={"kind": "tool_approval", "payload": {"decision": "once"}},
            cookies=auth_cookies,
            headers=_csrf(),
        )
        assert resp.status_code == 409
    finally:
        app.state.web_interactions.pop(iid, None)
        if not fut.done():
            fut.cancel()


async def test_respond_interaction_410_already_resolved(
    web_client, test_user, auth_cookies
):
    """Posting to a future that is already resolved returns 410."""
    import asyncio as _asyncio

    sid = await _new_session_id(web_client, auth_cookies)
    app = web_client._transport.app  # type: ignore[attr-defined]
    loop = _asyncio.get_event_loop()
    fut = loop.create_future()
    fut.set_result({"answers": {}})  # already resolved
    iid = "already-done-iid"
    app.state.web_interactions[iid] = {
        "user_id": test_user.user_id,
        "session_id": sid,
        "kind": "ask_user_question",
        "future": fut,
    }
    try:
        resp = await web_client.post(
            f"/api/v1/sessions/{sid}/interactions/{iid}",
            json={"kind": "ask_user_question", "payload": {"answers": {}}},
            cookies=auth_cookies,
            headers=_csrf(),
        )
        assert resp.status_code == 410
    finally:
        app.state.web_interactions.pop(iid, None)
