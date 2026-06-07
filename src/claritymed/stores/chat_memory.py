"""Per-user chat memory store skeleton.

v1 is read-only (architecture A6). The fact-extraction-and-write loop is a
v2 plan, but the per-user directory and search interface live here so the
text_rag plan can wire calls before the real implementation lands.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from claritymed.context import MissingContextError, user_id_ctx
from claritymed.stores.paths import user_chat_memory_dir, validate_user_id

logger = logging.getLogger(__name__)


class ChatChunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)


class ChatMemoryStore(ABC):
    """Per-user chat memory interface. v1 search-only."""

    def __init__(self, user_id: str) -> None:
        self.user_id = validate_user_id(user_id)
        self.lance_dir: Path = user_chat_memory_dir(self.user_id)

    @classmethod
    def for_current_user(cls) -> "ChatMemoryStore":
        uid = user_id_ctx.get()
        if not uid:
            raise MissingContextError("user_id_ctx is not set")
        # Subclasses override __init__; the concrete default is LanceChatMemoryStore.
        return LanceChatMemoryStore(uid)

    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[ChatChunk]: ...


class LanceChatMemoryStore(ChatMemoryStore):
    """Default LanceDB-backed implementation.

    Stub: returns an empty list and logs a warning until the text_rag plan
    wires a real embedder. The point of this skeleton is the isolation
    contract — the per-user directory lives at ``user_chat_memory_dir`` and
    cannot be opened by another user's store.
    """

    def search(self, query: str, k: int = 5) -> list[ChatChunk]:
        logger.warning(
            "chat memory search is a v1 stub (returning []); user=%s query=%r k=%d",
            self.user_id,
            query,
            k,
        )
        return []
