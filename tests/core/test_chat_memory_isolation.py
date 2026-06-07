"""Tests for ``LanceChatMemoryStore`` skeleton + isolation contract."""

from __future__ import annotations

import pytest

from claritymed.context import MissingContextError
from claritymed.errors import InvalidUserIdError
from claritymed.stores.chat_memory import (
    ChatMemoryStore,
    ChatTurn,
    LanceChatMemoryStore,
)
from claritymed.stores.paths import user_chat_memory_dir


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


def test_save_then_load_recent_roundtrips_turns():
    store = LanceChatMemoryStore("alice")
    turns = [
        ChatTurn(role="user", text="hi"),
        ChatTurn(role="assistant", text="hello there"),
    ]
    written = store.save_turns(turns)
    assert written == 2
    loaded = store.load_recent(k=10)
    assert [t.text for t in loaded] == ["hi", "hello there"]
    assert [t.role for t in loaded] == ["user", "assistant"]


def test_save_appends_across_sessions():
    """A second TUI session's turns land at the tail of the same transcript."""
    store = LanceChatMemoryStore("alice")
    store.save_turns([ChatTurn(role="user", text="day 1")])
    store.save_turns([ChatTurn(role="user", text="day 2")])
    loaded = store.load_recent(k=10)
    assert [t.text for t in loaded] == ["day 1", "day 2"]


def test_load_recent_returns_only_tail():
    store = LanceChatMemoryStore("alice")
    store.save_turns([ChatTurn(role="user", text=str(i)) for i in range(20)])
    loaded = store.load_recent(k=3)
    assert [t.text for t in loaded] == ["17", "18", "19"]


def test_load_recent_tolerates_corrupt_line(tmp_path):
    """A garbled line is skipped, not crashed on."""
    store = LanceChatMemoryStore("alice")
    store.save_turns([ChatTurn(role="user", text="good")])
    # Append a corrupt line by hand.
    path = store.lance_dir / "transcript.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not valid json\n")
    loaded = store.load_recent(k=10)
    assert [t.text for t in loaded] == ["good"]


def test_save_empty_turns_is_noop():
    store = LanceChatMemoryStore("alice")
    assert store.save_turns([]) == 0
    assert store.load_recent(k=10) == []
