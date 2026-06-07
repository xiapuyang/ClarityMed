"""RequestContext: the serialized form of the three runtime ContextVars.

ContextVars (``claritymed.context``) are the runtime carrier; this is the
audit / cross-process carrier. ``from_context_vars`` / ``apply_to_context_vars``
let test fixtures and audit log replay round-trip a request identity.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from claritymed.context import (
    apply_context,
    get_context_or_raise,
    reset_context,
)


class RequestContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str = Field(pattern=r"^[0-9]{14}[0-9A-F]{8}$")
    user_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,32}$")
    language: Literal["en", "zh"]
    entry: Literal["cli", "api"] = "cli"
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_context_vars(
        cls, entry: Literal["cli", "api"] = "cli"
    ) -> "RequestContext":
        rid, uid, lang = get_context_or_raise()
        return cls(request_id=rid, user_id=uid, language=lang, entry=entry)  # type: ignore[arg-type]

    def apply_to_context_vars(self):
        """Set ContextVars from this snapshot. Returns the reset tokens."""
        return apply_context(self.request_id, self.user_id, self.language)

    @staticmethod
    def reset_context_vars(tokens) -> None:
        reset_context(tokens)
