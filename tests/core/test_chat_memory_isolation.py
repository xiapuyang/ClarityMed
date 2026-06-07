"""Tests for ``LanceChatMemoryStore`` skeleton + isolation contract."""

from __future__ import annotations

import pytest
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from claritymed.context import MissingContextError
from claritymed.errors import InvalidUserIdError
from claritymed.stores.chat_memory import (
    ChatMemoryStore,
    LanceChatMemoryStore,
)
from claritymed.stores.paths import user_chat_memory_dir


def _run_messages(user_text: str, assistant_text: str) -> bytes:
    """Build the exact byte payload pydantic-ai's stream emits for one ask call."""
    messages = [
        ModelRequest(parts=[UserPromptPart(content=user_text)]),
        ModelResponse(parts=[TextPart(content=assistant_text)]),
    ]
    return ModelMessagesTypeAdapter.dump_json(messages)


def test_search_returns_empty_in_v1_stub():
    s = LanceChatMemoryStore("alice")
    assert s.search("headache") == []


def test_for_current_user_requires_context():
    with pytest.raises(MissingContextError):
        ChatMemoryStore.for_current_user()


def test_lance_dirs_are_per_user(trio):
    alice_dir = LanceChatMemoryStore(trio["admin"].user_id).lance_dir
    bob_dir = LanceChatMemoryStore(trio["users"][0].user_id).lance_dir
    assert alice_dir != bob_dir
    assert alice_dir == user_chat_memory_dir("alice")
    assert bob_dir == user_chat_memory_dir("bob")


def test_invalid_user_id_rejected():
    with pytest.raises(InvalidUserIdError):
        LanceChatMemoryStore("../etc")


def test_append_run_messages_then_load_projects_to_turns():
    """One LLM run's bytes append → load reads it back as user+assistant turns."""
    store = LanceChatMemoryStore("alice")
    written = store.append_run_messages_json(_run_messages("hi", "hello there"))
    assert written == 1
    loaded = store.load_recent(k=10)
    assert [t.role for t in loaded] == ["user", "assistant"]
    assert [t.text for t in loaded] == ["hi", "hello there"]


def test_append_runs_across_sessions():
    """Two ask calls in two sessions both land at the tail of one file."""
    store = LanceChatMemoryStore("alice")
    store.append_run_messages_json(_run_messages("day 1 q", "day 1 a"))
    store.append_run_messages_json(_run_messages("day 2 q", "day 2 a"))
    loaded = store.load_recent(k=10)
    assert [t.text for t in loaded] == ["day 1 q", "day 1 a", "day 2 q", "day 2 a"]


def test_load_recent_returns_only_tail():
    store = LanceChatMemoryStore("alice")
    for i in range(20):
        store.append_run_messages_json(_run_messages(f"q{i}", f"a{i}"))
    loaded = store.load_recent(k=3)
    assert [t.text for t in loaded] == ["a18", "q19", "a19"]


def test_load_tolerates_corrupt_line():
    """A garbled line is skipped, not crashed on."""
    store = LanceChatMemoryStore("alice")
    store.append_run_messages_json(_run_messages("good q", "good a"))
    path = store.lance_dir / "messages.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not valid json\n")
    loaded = store.load_recent(k=10)
    assert [t.text for t in loaded] == ["good q", "good a"]


def test_append_empty_bytes_is_noop():
    store = LanceChatMemoryStore("alice")
    assert store.append_run_messages_json(b"") == 0
    assert store.load_recent(k=10) == []


def test_messages_file_is_pydantic_ai_compatible():
    """Saved bytes must roundtrip through pydantic-ai's TypeAdapter unchanged."""
    store = LanceChatMemoryStore("alice")
    store.append_run_messages_json(_run_messages("ping", "pong"))
    line = (store.lance_dir / "messages.jsonl").read_text(encoding="utf-8").strip()
    messages = ModelMessagesTypeAdapter.validate_json(line)
    assert isinstance(messages[0], ModelRequest)
    assert isinstance(messages[1], ModelResponse)
