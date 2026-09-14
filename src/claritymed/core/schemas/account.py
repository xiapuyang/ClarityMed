"""Account: auth metadata + per-user preference. Deliberately not PHI.

``Account`` is persisted as YAML in ``~/.claritymed/data/users/<id>/settings.yaml``.
``Patient`` PHI is persisted separately in SQLite — splitting the two storage
forms keeps an admin who lists Accounts (display names, roles, languages) from
accidentally reading a patient's allergy list.

``role`` is intentionally mutable: ``admin`` may demote themselves, and the
admin module will eventually promote / revoke other accounts. The ``require_admin``
guard (``stores/account.py``) enforces who may write changes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["admin", "user"]
Language = Literal["en", "zh"]
EmergencySensitivity = Literal["strict", "balanced", "lenient", "off"]


class EmergencySettings(BaseModel):
    """Per-user emergency-triage gate preference.

    Nested under ``emergency:`` in ``settings.yaml`` so the gate's
    knobs cluster together. ``sensitivity=='off'`` requires an explicit
    ISO-8601 acknowledgement timestamp — a two-step opt-out that
    survives a YAML hand-edit because the model_validator below rejects
    the file at load time.

    The app default lives in ``configs/emergency.yaml`` and is resolved
    by :func:`claritymed.core.emergency.resolve_sensitivity`. Leaving
    ``sensitivity`` as None here means "use the app default", which is
    the recommended state for most users.
    """

    model_config = ConfigDict(extra="forbid")

    sensitivity: EmergencySensitivity | None = None
    off_acknowledged_at: datetime | None = None

    @model_validator(mode="after")
    def _off_requires_acknowledgement(self) -> "EmergencySettings":
        if self.sensitivity == "off" and self.off_acknowledged_at is None:
            raise ValueError(
                "emergency.sensitivity='off' requires "
                "emergency.off_acknowledged_at (ISO-8601 timestamp). "
                "Turning off the emergency triage gate is a two-step "
                "decision and the timestamp is the evidence the user "
                "made it deliberately."
            )
        return self


class Account(BaseModel):
    """Per-user identity and preferences. No PHI fields, ever."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,32}$")
    display_name: str = Field(min_length=1, max_length=64)
    role: Role = "user"
    language: Language = "en"
    provider_id: str | None = Field(default=None, max_length=64)
    active_system_rag_collections: list[str] = Field(
        default_factory=list,
        description=(
            "User-opted-in system RAG collections (names from "
            "configs/retrieval.yaml system_rag.collections). Empty list "
            "means no system RAG is consulted during ask retrieval."
        ),
    )
    emergency: EmergencySettings = Field(default_factory=EmergencySettings)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
