"""Per-user chat memory store skeleton.

Semantic chat memory (LanceDB + embedder) is a v2 plan. v1 ships two
surfaces the TUI needs on day one:

* ``search`` — placeholder for the future semantic recall path. Returns
  ``[]`` until the text_rag plan wires a real embedder.
* ``append_run_messages_json`` / ``load_recent`` — pydantic-ai-native
  persistence. Each ask call's ``stream.all_messages_json()`` is appended
  to ``messages.jsonl`` (one ``[ModelMessage, ...]`` JSON array per line)
  under ``~/.claritymed/data/users/<id>/chat_memory.lance/``. On reopen
  the TUI loads the last K turns and projects them to ``ChatTurn``
  display bubbles. When resume lands, the same file feeds back into
  ``Agent.run(message_history=...)`` with zero format conversion.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from claritymed.context import MissingContextError, user_id_ctx
from claritymed.stores.paths import user_chat_memory_dir, validate_user_id

logger = logging.getLogger(__name__)

Role = Literal["user", "assistant", "system"]


class ChatChunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)


class ChatTurn(BaseModel):
    """One display turn projected from a pydantic-ai message.

    Display-only. Persistence is the raw ModelMessage JSON; this struct
    exists so the TUI does not have to walk pydantic-ai message parts at
    render time.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Role
    text: str
    cancelled: bool = False


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

    @abstractmethod
    def load_recent(self, k: int = 10) -> list[ChatTurn]: ...

    @abstractmethod
    def append_run_messages_json(self, messages_json: bytes) -> int: ...


class LanceChatMemoryStore(ChatMemoryStore):
    """Default LanceDB-backed implementation.

    The semantic ``search`` path is still a stub. The transcript path
    (``append_run_messages_json`` / ``load_recent``) is live.
    """

    def search(self, query: str, k: int = 5) -> list[ChatChunk]:
        logger.warning(
            "chat memory search is a v1 stub (returning []); user=%s query=%r k=%d",
            self.user_id,
            query,
            k,
        )
        return []

    def append_run_messages_json(self, messages_json: bytes) -> int:
        """Append one pydantic-ai run's messages_json (raw bytes) as one line.

        ``messages_json`` is the exact byte payload returned by
        ``StreamedRunResult.all_messages_json()`` — a JSON array of
        ``ModelMessage``. We append it verbatim plus a newline so the file
        is JSON-lines and can be tailed/grepped without parsing the
        nested structure.
        """
        if not messages_json:
            return 0
        path = self._messages_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as fh:
            fh.write(messages_json.rstrip(b"\n"))
            fh.write(b"\n")
        return 1

    def load_recent(self, k: int = 10) -> list[ChatTurn]:
        """Return the last ``k`` display turns projected from messages.jsonl."""
        path = self._messages_path()
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            logger.warning("chat messages read failed: %s", exc)
            return []
        turns: list[ChatTurn] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                messages = ModelMessagesTypeAdapter.validate_json(stripped)
            except ValidationError:
                # Corrupt line — skip rather than blow up the whole load.
                continue
            for msg in messages:
                turn = _project_message(msg)
                if turn is not None:
                    turns.append(turn)
        if k > 0 and len(turns) > k:
            turns = turns[-k:]
        return turns

    def _messages_path(self) -> Path:
        return self.lance_dir / "messages.jsonl"


def _project_message(msg) -> ChatTurn | None:
    """Map one pydantic-ai ``ModelMessage`` to a display ``ChatTurn``.

    Returns ``None`` for messages that have no displayable content —
    system prompts, tool calls, tool returns. The TUI does not surface
    those at startup; they will show up in the right-pane tool log when
    the agent is actively running.
    """
    if isinstance(msg, ModelRequest):
        for part in msg.parts:
            if isinstance(part, UserPromptPart):
                return ChatTurn(role="user", text=str(part.content))
        return None
    if isinstance(msg, ModelResponse):
        text_chunks = [p.content for p in msg.parts if isinstance(p, TextPart)]
        if not text_chunks:
            return None
        return ChatTurn(role="assistant", text="".join(text_chunks))
    return None
