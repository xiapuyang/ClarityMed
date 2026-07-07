"""Per-session chat transcript: Claude-Code-style JSONL event log.

One ``ChatSession`` owns one ``<session_id>.jsonl`` file under
``data/users/<user_id>/sessions/``. Each line is one atomic event
(``user`` / ``assistant`` / ``system``) carrying its own ``uuid``,
``parentUuid``, ``sessionId``, and ``timestamp``. Assistant events
additionally embed the raw pydantic-ai ``messages`` array, so resuming
a session is a single file read followed by a ``ModelMessagesTypeAdapter``
round-trip — no parts walking, no field-by-field translation.

Design notes:

- ``session_id`` is a uuid4 string we generate ourselves. pydantic-ai's
  ``conversation_id`` is recorded on every assistant event so OTel traces
  (``gen_ai.conversation.id``) can still correlate back, but our
  filenames don't depend on pydantic-ai bootstrapping a conversation_id
  first turn.
- ``new()`` never touches disk; the file is created lazily on the first
  ``append_*`` call. Sessions that never see a turn leave no garbage.
- ``/clear`` semantics: instantiate a new ``ChatSession`` for the same
  user. The old file stays on disk for resume / audit.
- ``append_assistant`` is the integration point for token + latency
  metrics. The same record feeds the in-memory ``message_history`` for
  the next ``Agent.run_stream`` call.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.messages import ModelMessagesTypeAdapter

from claritymed.core.events import DifferentialReady
from claritymed.core.observability.latency import (
    LatencyTrace,
    build_step_records,
    usage_dict,
)
from claritymed.stores.paths import user_sessions_dir, validate_user_id

# Re-exported as the historical name used by audit payload code in
# ``ask_service.py`` (still ``_usage_dict``); rename will follow when
# all call sites are touched.
_usage_dict = usage_dict

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage  # noqa: F401
    from pydantic_ai.usage import RunUsage  # noqa: F401

logger = logging.getLogger(__name__)

SystemEventKind = Literal["start", "clear", "mode", "info"]
SESSION_FILE_SUFFIX = ".jsonl"


class ChatTurn(BaseModel):
    """One displayable turn projected from the JSONL transcript.

    The TUI / webui render these; they're a thin view over the persisted
    events with no parts walking required at render time.

    ``differential`` carries the multi-card renderer's sidecar payload
    when the assistant turn produced one (symptoms plugin runs). It's
    persisted alongside the summary text so cards survive a page
    refresh / session resume — the alternative (only text persisted)
    let cards flash and disappear.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["user", "assistant", "system"]
    text: str
    cancelled: bool = False
    differential: DifferentialReady | None = None


class SessionMeta(BaseModel):
    """Minimal listing record for a session — for resume / history UI."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str
    path: Path
    size_bytes: int
    modified_at: datetime
    preview: str = Field(default="", description="First user prompt, truncated.")


class ChatSession:
    """A single chat session writing to one JSONL file.

    Construct via ``new(user_id)`` for a fresh session or
    ``resume(user_id, session_id)`` to reopen an existing transcript.
    """

    def __init__(
        self,
        *,
        user_id: str,
        session_id: str,
        message_history: list["ModelMessage"] | None = None,
        last_event_uuid: str | None = None,
    ) -> None:
        self.user_id = validate_user_id(user_id)
        self.session_id = session_id
        self._message_history: list[ModelMessage] = list(message_history or [])
        self._last_event_uuid: str | None = last_event_uuid

    # ----- constructors --------------------------------------------------

    @classmethod
    def new(cls, user_id: str) -> "ChatSession":
        """Generate a fresh session_id. No disk I/O until first append."""
        return cls(user_id=user_id, session_id=str(uuid.uuid4()))

    @classmethod
    def resume(cls, user_id: str, session_id: str) -> "ChatSession":
        """Reopen ``<sessions_dir>/<session_id>.jsonl``.

        Rebuilds ``message_history`` from the *last* assistant event's
        embedded messages array, which by pydantic-ai's contract contains
        every message up to that point — no concatenation needed.
        """
        validate_user_id(user_id)
        path = cls._path_for(user_id, session_id)
        message_history: list[ModelMessage] = []
        last_uuid: str | None = None
        if path.exists():
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                logger.warning("session read failed: %s", exc)
                lines = []
            for raw in lines:
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                last_uuid = event.get("uuid") or last_uuid
                if event.get("type") == "assistant":
                    messages = event.get("messages")
                    if messages is not None:
                        try:
                            message_history = list(
                                ModelMessagesTypeAdapter.validate_python(messages)
                            )
                        except Exception:  # noqa: BLE001
                            logger.exception(
                                "failed to rehydrate messages for session=%s",
                                session_id,
                            )
        return cls(
            user_id=user_id,
            session_id=session_id,
            message_history=message_history,
            last_event_uuid=last_uuid,
        )

    # ----- read-side ------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path_for(self.user_id, self.session_id)

    def message_history(self) -> list["ModelMessage"]:
        """Return the in-memory history to feed ``Agent.run_stream``."""
        return list(self._message_history)

    def load_turns(self) -> list[ChatTurn]:
        """Project on-disk events into displayable turns for the UI."""
        path = self.path
        if not path.exists():
            return []
        turns: list[ChatTurn] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            logger.warning("session load_turns failed: %s", exc)
            return []
        for raw in lines:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            kind = event.get("type")
            if kind == "user":
                turns.append(ChatTurn(role="user", text=event.get("text", "")))
            elif kind == "assistant":
                differential = _rehydrate_differential(event.get("differential"))
                turns.append(
                    ChatTurn(
                        role="assistant",
                        text=event.get("text", ""),
                        cancelled=bool(event.get("cancelled", False)),
                        differential=differential,
                    )
                )
            elif kind == "system":
                # Only surface info-kind system events to users; "start" /
                # "clear" are bookkeeping the UI re-emits via toasts.
                if event.get("kind") in ("info",):
                    turns.append(ChatTurn(role="system", text=event.get("text", "")))
        return turns

    @classmethod
    def list_sessions(cls, user_id: str) -> list[SessionMeta]:
        """List sessions for a user, newest first. For resume UI."""
        validate_user_id(user_id)
        sessions_dir = user_sessions_dir(user_id)
        if not sessions_dir.exists():
            return []
        metas: list[SessionMeta] = []
        for file in sessions_dir.iterdir():
            if not file.is_file() or file.suffix != SESSION_FILE_SUFFIX:
                continue
            try:
                stat = file.stat()
            except OSError:
                continue
            preview = _first_user_preview(file)
            metas.append(
                SessionMeta(
                    session_id=file.stem,
                    path=file,
                    size_bytes=stat.st_size,
                    modified_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                    preview=preview,
                )
            )
        metas.sort(key=lambda m: m.modified_at, reverse=True)
        return metas

    # ----- write-side -----------------------------------------------------

    def append_user(self, text: str) -> str:
        return self._append({"type": "user", "text": text})

    def append_assistant(
        self,
        *,
        text: str,
        messages_json: bytes,
        model: str,
        provider_id: str,
        usage: "RunUsage | None",
        latency: "LatencyTrace | None" = None,
        steps: list[dict] | None = None,
        cancelled: bool = False,
        differential: DifferentialReady | None = None,
    ) -> str:
        """Persist an assistant turn and refresh in-memory history.

        ``messages_json`` is the bytes payload from
        ``StreamedRunResult.all_messages_json()``. We embed it parsed so
        each line is valid pretty JSON and the messages array round-trips
        through ``ModelMessagesTypeAdapter`` on resume.

        ``latency`` carries the time-to-first-token, streaming completion,
        and total span. ``steps`` is one record per LLM call inside this
        run (single-step for the current ask agent; multi-step once tools
        land — same schema both ways).
        """
        messages_obj, message_objs, conv_id, run_id = _decode_messages(messages_json)
        if message_objs is not None:
            self._message_history = message_objs
        payload: dict[str, object] = {
            "type": "assistant",
            "text": text,
            "model": model,
            "providerId": provider_id,
            "cancelled": cancelled,
            "messages": messages_obj,
        }
        if latency is not None:
            payload["latency"] = latency.to_dict()
        if conv_id:
            payload["conversationId"] = conv_id
        if run_id:
            payload["runId"] = run_id
        if usage is not None:
            payload["usage"] = _usage_dict(usage)
        if steps:
            payload["steps"] = steps
        if differential is not None:
            # Serialize via model_dump so the entire sidecar payload
            # (cards, session meta, banner_key, per-card citations)
            # round-trips through ``model_validate`` on resume without
            # a bespoke schema mirror.
            payload["differential"] = differential.model_dump(mode="json")
        return self._append(payload)

    def append_system(self, text: str, kind: SystemEventKind = "info") -> str:
        return self._append({"type": "system", "kind": kind, "text": text})

    # ----- internals ------------------------------------------------------

    def _append(self, payload: dict[str, object]) -> str:
        event_uuid = str(uuid.uuid4())
        event: dict[str, object] = {
            **payload,
            "uuid": event_uuid,
            "parentUuid": self._last_event_uuid,
            "sessionId": self.session_id,
            "userId": self.user_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        path = self.path
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False))
            fh.write("\n")
        self._last_event_uuid = event_uuid
        return event_uuid

    @staticmethod
    def _path_for(user_id: str, session_id: str) -> Path:
        # session_id sanity: uuid-ish, no path traversal. Reject anything
        # that isn't safe for a filename — defence in depth, even though
        # we only ever fill it from uuid4 or a previously stored value.
        if not session_id or "/" in session_id or "\\" in session_id:
            raise ValueError(f"invalid session_id: {session_id!r}")
        if session_id.startswith(".") or len(session_id) > 128:
            raise ValueError(f"invalid session_id: {session_id!r}")
        return user_sessions_dir(user_id) / f"{session_id}{SESSION_FILE_SUFFIX}"


def _decode_messages(
    messages_json: bytes,
) -> tuple[list[dict] | None, list["ModelMessage"] | None, str | None, str | None]:
    """Decode ``all_messages_json()`` bytes into both the JSON form (for
    embedding in the event log) and the ``ModelMessage`` form (for the
    in-memory history). Returns ``(messages_obj, message_objs, conv_id,
    run_id)``; any field is ``None`` on decode failure so the assistant
    event still lands with as much information as we have.
    """
    if not messages_json:
        return None, None, None, None
    try:
        messages_obj = json.loads(messages_json.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        logger.exception("invalid messages_json bytes")
        return None, None, None, None
    try:
        message_objs = list(ModelMessagesTypeAdapter.validate_python(messages_obj))
    except Exception:  # noqa: BLE001
        logger.exception("messages_json failed ModelMessagesTypeAdapter")
        message_objs = None
    conv_id: str | None = None
    run_id: str | None = None
    if message_objs:
        first = message_objs[0]
        conv_id = getattr(first, "conversation_id", None)
        run_id = getattr(first, "run_id", None)
    return messages_obj, message_objs, conv_id, run_id


def _rehydrate_differential(raw: object) -> DifferentialReady | None:
    """Re-validate a persisted ``differential`` dict into the sidecar model.

    Persisted events pre-date the multi-card renderer or drift in
    schema (older builds omitted required fields) — treat any
    validation error as "no cards for this turn" rather than failing
    the whole session load. The turn's summary text still renders.
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        return None
    try:
        return DifferentialReady.model_validate(raw)
    except Exception:  # noqa: BLE001
        logger.warning(
            "session load: persisted differential payload failed validation; "
            "dropping cards for this turn",
            exc_info=True,
        )
        return None


def _first_user_preview(path: Path, max_chars: int = 80) -> str:
    try:
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "user":
                    text = str(event.get("text", "")).replace("\n", " ").strip()
                    if len(text) > max_chars:
                        text = text[: max_chars - 1] + "…"
                    return text
    except OSError:
        return ""
    return ""


__all__ = [
    "ChatSession",
    "ChatTurn",
    "LatencyTrace",
    "SessionMeta",
    "build_step_records",
]
