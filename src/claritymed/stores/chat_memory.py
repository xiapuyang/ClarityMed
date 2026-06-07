"""Per-user chat memory store skeleton.

Semantic chat memory (LanceDB + embedder) is a v2 plan. v1 ships two
surfaces the TUI needs on day one:

* ``search`` — placeholder for the future semantic recall path. Returns
  ``[]`` until the text_rag plan wires a real embedder.
* ``load_recent`` / ``save_turns`` — file-backed transcript so the TUI can
  reopen with the prior session visible and the operator can grep
  ``~/.claritymed/data/users/<id>/chat_memory.lance/transcript.jsonl`` for
  audit. The file is JSON-lines so it survives partial writes; the real
  LanceDB store can ingest it later without a separate migration step.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from claritymed.context import MissingContextError, user_id_ctx
from claritymed.stores.paths import user_chat_memory_dir, validate_user_id

logger = logging.getLogger(__name__)

Role = Literal["user", "assistant", "system"]


class ChatChunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)


class ChatTurn(BaseModel):
    """One transcript turn — used by ``load_recent`` / ``save_turns``."""

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
    def save_turns(self, turns: list[ChatTurn]) -> int: ...


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

    def load_recent(self, k: int = 10) -> list[ChatTurn]:
        path = self._transcript_path()
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            logger.warning("chat transcript read failed: %s", exc)
            return []
        recent = lines[-k:] if k > 0 else lines
        turns: list[ChatTurn] = []
        for line in recent:
            line = line.strip()
            if not line:
                continue
            try:
                turns.append(ChatTurn.model_validate(json.loads(line)))
            except (json.JSONDecodeError, ValidationError):
                # Skip a corrupt line rather than refusing to load the
                # whole history — the operator can grep the file to
                # spot what broke.
                continue
        return turns

    def save_turns(self, turns: list[ChatTurn]) -> int:
        if not turns:
            return 0
        path = self._transcript_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for turn in turns:
                fh.write(turn.model_dump_json() + "\n")
        return len(turns)

    def _transcript_path(self) -> Path:
        return self.lance_dir / "transcript.jsonl"
