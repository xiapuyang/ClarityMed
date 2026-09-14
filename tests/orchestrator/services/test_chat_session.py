"""ChatSession: Claude-Code-style JSONL transcript per chat session.

Covers the contract AskService and the TUI depend on: lazy file creation,
event schema shape, pydantic-ai round-trip via ``messages``, multi-turn
history reconstruction on resume, /clear semantics (rotate session_id,
keep old file), and the listing surface for resume UI.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, ModelResponse
from pydantic_ai.models.test import TestModel

from claritymed.orchestrator.services import (
    ChatSession,
    ChatTurn,
    LatencyTrace,
    SessionMeta,
    build_step_records,
)
from claritymed.stores.paths import user_sessions_dir


def _run_one_turn(prompt: str, history=None) -> tuple[bytes, object, str]:
    """Drive a TestModel agent for one turn; return messages_json, usage, text."""
    agent = Agent(TestModel(custom_output_text=f"reply to: {prompt}"))
    result = agent.run_sync(prompt, message_history=history)
    return result.all_messages_json(), result.usage, result.output


def test_new_does_not_touch_disk():
    session = ChatSession.new("alice")
    assert not session.path.exists()
    assert session.session_id  # uuid4 string


def test_new_session_ids_are_unique():
    a = ChatSession.new("alice")
    b = ChatSession.new("alice")
    assert a.session_id != b.session_id


def test_append_user_creates_file_lazily():
    session = ChatSession.new("alice")
    assert not session.path.exists()
    session.append_user("hello")
    assert session.path.exists()
    assert session.path.parent == user_sessions_dir("alice")
    lines = session.path.read_text("utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["type"] == "user"
    assert event["text"] == "hello"
    assert event["sessionId"] == session.session_id
    assert event["userId"] == "alice"
    assert event["parentUuid"] is None
    assert "uuid" in event
    assert "timestamp" in event


def test_event_chain_uses_parent_uuid():
    session = ChatSession.new("alice")
    u1 = session.append_user("first")
    u2 = session.append_user("second")
    events = [json.loads(line) for line in session.path.read_text("utf-8").splitlines()]
    assert events[0]["uuid"] == u1
    assert events[0]["parentUuid"] is None
    assert events[1]["uuid"] == u2
    assert events[1]["parentUuid"] == u1


def test_append_assistant_records_metrics_and_messages():
    session = ChatSession.new("alice")
    session.append_user("hello")
    messages_json, usage, output = _run_one_turn("hello")
    session.append_assistant(
        text=output,
        messages_json=messages_json,
        model="openai:gpt-4o",
        provider_id="openai_cloud",
        usage=usage,
        latency=LatencyTrace(total_ms=123, ttft_ms=42, completion_ms=81),
        steps=[{"model": "openai:gpt-4o", "usage": {"input_tokens": 10}}],
    )
    events = [json.loads(line) for line in session.path.read_text("utf-8").splitlines()]
    assert events[-1]["type"] == "assistant"
    assert events[-1]["text"] == output
    assert events[-1]["model"] == "openai:gpt-4o"
    assert events[-1]["providerId"] == "openai_cloud"
    assert events[-1]["latency"] == {"totalMs": 123, "ttftMs": 42, "completionMs": 81}
    assert events[-1]["steps"] == [
        {"model": "openai:gpt-4o", "usage": {"input_tokens": 10}}
    ]
    assert events[-1]["cancelled"] is False
    assert "messages" in events[-1]
    assert isinstance(events[-1]["messages"], list)
    # Usage round-trips into the documented schema
    assert set(events[-1]["usage"].keys()) == {
        "input_tokens",
        "output_tokens",
        "total_tokens",
    }
    # pydantic-ai trace ids land for OTel correlation
    assert "conversationId" in events[-1]
    assert "runId" in events[-1]


def test_message_history_updates_after_assistant_turn():
    session = ChatSession.new("alice")
    assert session.message_history() == []
    session.append_user("hi")
    messages_json, usage, output = _run_one_turn("hi")
    session.append_assistant(
        text=output,
        messages_json=messages_json,
        model="?",
        provider_id="?",
        usage=usage,
        latency=LatencyTrace(total_ms=0),
    )
    history = session.message_history()
    assert history, "message_history empty after assistant turn"
    # Shape proves it's feedable to Agent.run(message_history=...) next turn.
    # The follow-up test exercises the runtime path end-to-end.
    assert any(isinstance(m, ModelRequest) for m in history)
    assert any(isinstance(m, ModelResponse) for m in history)


def test_resume_rebuilds_message_history():
    session = ChatSession.new("alice")
    session.append_user("hi")
    messages_json, usage, output = _run_one_turn("hi")
    session.append_assistant(
        text=output,
        messages_json=messages_json,
        model="?",
        provider_id="?",
        usage=usage,
        latency=LatencyTrace(total_ms=0),
    )
    sid = session.session_id

    resumed = ChatSession.resume("alice", sid)
    assert resumed.session_id == sid
    assert resumed.message_history(), "resume did not rebuild history"
    # The next agent run can use it directly without conversion
    agent = Agent(TestModel(custom_output_text="follow-up reply"))
    result = agent.run_sync("and?", message_history=resumed.message_history())
    assert result.output == "follow-up reply"


def test_resume_carries_parent_uuid_so_chain_continues():
    session = ChatSession.new("alice")
    u1 = session.append_user("hi")
    messages_json, usage, _ = _run_one_turn("hi")
    session.append_assistant(
        text="ok",
        messages_json=messages_json,
        model="?",
        provider_id="?",
        usage=usage,
        latency=LatencyTrace(total_ms=0),
    )
    resumed = ChatSession.resume("alice", session.session_id)
    next_uuid = resumed.append_user("next")
    events = [json.loads(line) for line in resumed.path.read_text("utf-8").splitlines()]
    # The new event's parentUuid is the previous last event, not None,
    # not the original first user uuid.
    assert events[-1]["uuid"] == next_uuid
    assert events[-1]["parentUuid"] is not None
    assert events[-1]["parentUuid"] != u1


def test_resume_missing_file_yields_empty_history():
    resumed = ChatSession.resume("alice", "deadbeef-dead-beef-dead-beefdeadbeef")
    assert resumed.message_history() == []


def test_load_turns_skips_bookkeeping_system_events():
    session = ChatSession.new("alice")
    session.append_system("session opened", kind="start")
    session.append_user("hi")
    session.append_system("FYI", kind="info")
    turns = session.load_turns()
    roles = [t.role for t in turns]
    # "start" / "clear" are bookkeeping — only "info" surfaces to users.
    assert roles == ["user", "system"]


def test_clear_semantics_keeps_old_file_starts_new_session():
    # Simulate /clear: create session A, write a turn, then create session B
    # for the same user. The new file is separate; old file is untouched.
    a = ChatSession.new("alice")
    a.append_user("first session message")
    old_path = a.path
    old_contents = old_path.read_text("utf-8")

    b = ChatSession.new("alice")
    assert b.session_id != a.session_id
    b.append_user("second session message")
    assert b.path != old_path
    assert old_path.exists()
    assert old_path.read_text("utf-8") == old_contents


def test_list_sessions_returns_newest_first():
    a = ChatSession.new("alice")
    a.append_user("alpha")
    b = ChatSession.new("alice")
    b.append_user("bravo")
    # Force b's mtime newer than a's
    import os
    import time

    now = time.time()
    os.utime(a.path, (now - 60, now - 60))
    os.utime(b.path, (now, now))

    metas = ChatSession.list_sessions("alice")
    assert isinstance(metas[0], SessionMeta)
    assert [m.session_id for m in metas[:2]] == [b.session_id, a.session_id]
    assert metas[0].preview == "bravo"


def test_list_sessions_empty_when_no_sessions_dir():
    metas = ChatSession.list_sessions("alice")
    assert metas == []


def test_isolation_between_users():
    a = ChatSession.new("alice")
    a.append_user("alice secret")
    b = ChatSession.new("bob")
    b.append_user("bob secret")
    assert a.path.parent != b.path.parent
    assert "alice" in str(a.path)
    assert "bob" in str(b.path)


def test_invalid_session_id_rejected():
    with pytest.raises(ValueError):
        ChatSession(user_id="alice", session_id="../etc").path
    with pytest.raises(ValueError):
        ChatSession(user_id="alice", session_id="").path
    with pytest.raises(ValueError):
        ChatSession(user_id="alice", session_id=".hidden").path


def test_append_assistant_persists_differential_and_load_rehydrates():
    """The multi-card renderer's sidecar payload is stored alongside
    the summary text and round-trips through ``load_turns`` — cards
    survive a page refresh / session resume instead of vanishing with
    the in-memory stream state."""
    from claritymed.core.events import (
        DifferentialCard,
        DifferentialReady,
        DifferentialSessionMeta,
    )

    card = DifferentialCard(
        condition_id="pneumonia",
        condition_name="Pneumonia",
        probability=0.72,
        severity_tier="Urgent",
        confidence_bucket="high",
        confidence_label="Confident",
        headline="Likely Pneumonia",
        report="Infection of the lung tissue.",
    )
    payload = DifferentialReady(
        cards=[card],
        session=DifferentialSessionMeta(),
        language="en",
    )
    session = ChatSession.new("alice")
    session.append_user("fever + chest pain")
    messages_json, usage, _ = _run_one_turn("hi")
    session.append_assistant(
        text="The follow-up surfaced one finding.",
        messages_json=messages_json,
        model="?",
        provider_id="?",
        usage=usage,
        latency=LatencyTrace(total_ms=0),
        differential=payload,
    )
    # Fresh load — simulates the SPA hitting GET /turns after a refresh.
    turns = ChatSession.resume("alice", session.session_id).load_turns()
    assistant = [t for t in turns if t.role == "assistant"]
    assert assistant, "assistant turn missing after resume"
    reloaded = assistant[0].differential
    assert reloaded is not None
    assert len(reloaded.cards) == 1
    assert reloaded.cards[0].condition_id == "pneumonia"
    assert reloaded.cards[0].headline == "Likely Pneumonia"
    assert reloaded.session.banner_key is None
    assert reloaded.language == "en"


def test_load_turns_skips_malformed_persisted_differential():
    """Schema-drifted or corrupt differential payloads must not break
    the whole session load — the turn's summary text still renders,
    just without cards."""
    session = ChatSession.new("alice")
    session.path.parent.mkdir(parents=True, exist_ok=True)
    with session.path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": "user", "text": "hi"}) + "\n")
        # Missing required fields on the differential payload.
        fh.write(
            json.dumps(
                {
                    "type": "assistant",
                    "text": "text still here",
                    "differential": {"type": "differential_ready"},
                }
            )
            + "\n"
        )
    turns = session.load_turns()
    assistant = [t for t in turns if t.role == "assistant"]
    assert assistant and assistant[0].text == "text still here"
    assert assistant[0].differential is None


def test_append_assistant_without_differential_writes_no_field():
    """No differential → the assistant event contains no ``differential``
    key, keeping the JSONL compact for non-symptoms turns."""
    session = ChatSession.new("alice")
    session.append_user("hi")
    messages_json, usage, _ = _run_one_turn("hi")
    session.append_assistant(
        text="ok",
        messages_json=messages_json,
        model="?",
        provider_id="?",
        usage=usage,
        latency=LatencyTrace(total_ms=0),
    )
    lines = session.path.read_text("utf-8").splitlines()
    events = [json.loads(line) for line in lines]
    assistant_event = next(e for e in events if e["type"] == "assistant")
    assert "differential" not in assistant_event


def test_load_turns_recovers_a_cancelled_assistant_event():
    session = ChatSession.new("alice")
    session.append_user("hi")
    messages_json, usage, _ = _run_one_turn("hi")
    session.append_assistant(
        text="streamed partial",
        messages_json=messages_json,
        model="?",
        provider_id="?",
        usage=usage,
        latency=LatencyTrace(total_ms=0),
        cancelled=True,
    )
    turns = session.load_turns()
    assistant = [t for t in turns if t.role == "assistant"]
    assert assistant and assistant[0].cancelled is True


def test_cache_tokens_only_emitted_when_nonzero():
    """Audit + JSONL payloads stay lean when no prompt caching is in use,
    and surface cache_read/write tokens when the provider reports them."""
    from claritymed.orchestrator.services.chat_session import _usage_dict

    class _FakeUsage:
        input_tokens = 100
        output_tokens = 50
        total_tokens = 150
        cache_read_tokens = 0
        cache_write_tokens = 0

    no_cache = _usage_dict(_FakeUsage())
    assert "cache_read_tokens" not in no_cache
    assert "cache_write_tokens" not in no_cache

    class _CachedUsage:
        input_tokens = 100
        output_tokens = 50
        total_tokens = 150
        cache_read_tokens = 80
        cache_write_tokens = 20

    with_cache = _usage_dict(_CachedUsage())
    assert with_cache["cache_read_tokens"] == 80
    assert with_cache["cache_write_tokens"] == 20


def test_build_step_records_emits_one_record_per_model_response():
    """Per-step latency / usage breakdown for multi-hop runs.

    Single-step ask agent today → 1 record. Tools land → N records.
    Same schema both ways so the assistant event log doesn't have to
    change.
    """
    from pydantic_ai import Agent
    from pydantic_ai.models.test import TestModel

    agent = Agent(TestModel(custom_output_text="hi"))
    result = agent.run_sync("ping")
    steps = build_step_records(list(result.new_messages()))
    assert len(steps) >= 1
    assert "model" in steps[0]
    assert "usage" in steps[0]
    assert "input_tokens" in steps[0]["usage"]


def test_chatturn_is_a_frozen_model():
    t = ChatTurn(role="user", text="hi")
    with pytest.raises(Exception):
        t.text = "mutated"  # type: ignore[misc]


def test_corrupt_line_is_skipped_not_fatal():
    session = ChatSession.new("alice")
    session.append_user("hi")
    # Corrupt the file mid-stream
    with session.path.open("a", encoding="utf-8") as fh:
        fh.write("this-is-not-json\n")
    session.append_user("after corruption")
    turns = session.load_turns()
    assert [t.text for t in turns] == ["hi", "after corruption"]
    resumed = ChatSession.resume("alice", session.session_id)
    # No assistant event present, so message_history stays empty
    assert resumed.message_history() == []


def test_session_file_in_expected_location(tmp_path: Path):
    session = ChatSession.new("alice")
    session.append_user("hi")
    expected = user_sessions_dir("alice") / f"{session.session_id}.jsonl"
    assert session.path == expected
    assert expected.exists()


# ---------------------------------------------------------------------------
# Resume / load_turns error-tolerance branches
# ---------------------------------------------------------------------------


def test_resume_swallows_oserror_on_read(monkeypatch):
    """When the on-disk session file can't be opened, resume yields empty history."""
    session = ChatSession.new("alice")
    session.append_user("hi")

    from pathlib import Path as _P

    real_read_text = _P.read_text

    def _boom(self, *args, **kwargs):
        if self == session.path:
            raise OSError("read denied")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(_P, "read_text", _boom)
    resumed = ChatSession.resume("alice", session.session_id)
    assert resumed.message_history() == []


def test_load_turns_swallows_oserror_on_read(monkeypatch):
    """Same OSError tolerance for load_turns."""
    session = ChatSession.new("alice")
    session.append_user("hi")

    from pathlib import Path as _P

    real_read_text = _P.read_text

    def _boom(self, *args, **kwargs):
        if self == session.path:
            raise OSError("read denied")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(_P, "read_text", _boom)
    assert session.load_turns() == []


def test_load_turns_skips_blank_lines():
    session = ChatSession.new("alice")
    session.append_user("hi")
    with session.path.open("a", encoding="utf-8") as fh:
        fh.write("\n\n   \n")
    session.append_user("again")
    turns = session.load_turns()
    assert [t.text for t in turns] == ["hi", "again"]


def test_resume_skips_blank_lines():
    session = ChatSession.new("alice")
    session.append_user("hi")
    with session.path.open("a", encoding="utf-8") as fh:
        fh.write("\n\n")
    resumed = ChatSession.resume("alice", session.session_id)
    # No assistant event → empty message history.
    assert resumed.message_history() == []


def test_resume_logs_when_messages_validate_fails(caplog):
    """Bad message payload in an assistant event must not abort resume."""
    import json

    session = ChatSession.new("alice")
    session.path.parent.mkdir(parents=True, exist_ok=True)
    with session.path.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "type": "assistant",
                    "uuid": "abc",
                    "messages": [{"role": "totally_invalid", "bogus": True}],
                }
            )
            + "\n"
        )

    with caplog.at_level("ERROR"):
        resumed = ChatSession.resume("alice", session.session_id)
    # No history rebuilt — corrupt payload is logged + skipped.
    assert resumed.message_history() == []
    assert any("rehydrate" in r.message for r in caplog.records)


def test_list_sessions_skips_non_jsonl_files():
    """Files whose suffix isn't .jsonl are silently ignored by list_sessions."""
    from claritymed.stores.paths import user_sessions_dir

    session_a = ChatSession.new("alice")
    session_a.append_user("hello A")

    sessions_dir = user_sessions_dir("alice")
    # Drop a random file in there to exercise the suffix-filter continue.
    (sessions_dir / "notes.txt").write_text("ignore me", encoding="utf-8")
    (sessions_dir / "sub").mkdir()  # ignored — not a file

    metas = ChatSession.list_sessions("alice")
    # Only the real session_a remains.
    assert [m.session_id for m in metas] == [session_a.session_id]


# ---------------------------------------------------------------------------
# _decode_messages error branches
# ---------------------------------------------------------------------------


def test_decode_messages_with_empty_bytes_returns_nones():
    from claritymed.orchestrator.services.chat_session import _decode_messages

    assert _decode_messages(b"") == (None, None, None, None)


def test_decode_messages_handles_invalid_json(caplog):
    from claritymed.orchestrator.services.chat_session import _decode_messages

    with caplog.at_level("ERROR"):
        result = _decode_messages(b"{not valid json")
    assert result == (None, None, None, None)
    assert any("invalid messages_json bytes" in r.message for r in caplog.records)


def test_decode_messages_handles_invalid_utf8(caplog):
    from claritymed.orchestrator.services.chat_session import _decode_messages

    with caplog.at_level("ERROR"):
        result = _decode_messages(b"\xff\xfe")
    assert result == (None, None, None, None)


def test_decode_messages_invalid_model_messages_yields_none_objs(caplog):
    """JSON parses fine but doesn't validate as ModelMessage list."""
    from claritymed.orchestrator.services.chat_session import _decode_messages

    with caplog.at_level("ERROR"):
        msgs_obj, msg_objs, _conv, _run = _decode_messages(
            b'[{"role": "unknown_role"}]'
        )
    assert msgs_obj == [{"role": "unknown_role"}]
    assert msg_objs is None


# ---------------------------------------------------------------------------
# _first_user_preview branches
# ---------------------------------------------------------------------------


def test_first_user_preview_returns_empty_on_oserror(monkeypatch, tmp_path):
    from claritymed.orchestrator.services.chat_session import _first_user_preview

    bogus = tmp_path / "missing.jsonl"
    assert _first_user_preview(bogus) == ""


def test_first_user_preview_truncates_long_text():
    from claritymed.orchestrator.services.chat_session import _first_user_preview

    session = ChatSession.new("alice")
    session.append_user("A" * 200)
    preview = _first_user_preview(session.path, max_chars=80)
    assert len(preview) == 80
    assert preview.endswith("…")


def test_first_user_preview_skips_blank_and_bad_json():
    import json

    from claritymed.orchestrator.services.chat_session import _first_user_preview

    session = ChatSession.new("alice")
    session.path.parent.mkdir(parents=True, exist_ok=True)
    with session.path.open("w", encoding="utf-8") as fh:
        fh.write("\n")
        fh.write("not json\n")
        fh.write(json.dumps({"type": "system", "kind": "info", "text": "boot"}) + "\n")
        fh.write(json.dumps({"type": "user", "text": "real message"}) + "\n")
    assert _first_user_preview(session.path) == "real message"
