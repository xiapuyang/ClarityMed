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

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["admin", "user"]
Language = Literal["en", "zh"]


class Account(BaseModel):
    """Per-user identity and preferences. No PHI fields, ever."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(pattern=r"^[a-zA-Z0-9_-]{1,32}$")
    display_name: str = Field(min_length=1, max_length=64)
    role: Role = "user"
    language: Language = "en"
    cloud_provider_opt_in: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
