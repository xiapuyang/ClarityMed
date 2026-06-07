"""Tests for ``LanceChatMemoryStore`` skeleton + isolation contract."""

from __future__ import annotations

import pytest

from claritymed.context import MissingContextError
from claritymed.errors import InvalidUserIdError
from claritymed.stores.chat_memory import ChatMemoryStore, LanceChatMemoryStore
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
