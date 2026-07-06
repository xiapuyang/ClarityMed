"""Pydantic contracts for the import template directory.

The skill writes a ``template/`` directory containing:

* ``_meta.yaml`` — schema version, source hint, default category,
  enumeration of ``user_ids`` whose per-user files appear alongside it.
* ``<user_id>.yaml`` per user listed in ``_meta.user_ids``, carrying that
  user's ``cases`` and ``facts``.

These models are the on-disk contract. ``template_loader`` walks the
directory, validates each file against the model here, then performs
cross-file invariants (case_id uniqueness, ``user_ids`` set membership)
that no single-file model can express.

All models are ``frozen=True`` + ``extra="forbid"``. Two reasons:

1. Frozen prevents accidental in-place mutation between load and apply
   so a writer can't silently corrupt the validated snapshot.
2. ``extra="forbid"`` defeats the "I'll add a ``user_id:`` field to the
   per-user YAML body for convenience" footgun — the orchestrator must
   only ever derive user_id from the filename (Unit 4 / Unit 8), and
   forbidding extras keeps that the only path.

``FactsBundle.profile`` is a plain ``dict``, not a ``Profile``: the
fact writer (Unit 7) overlays only explicitly-provided keys on top of
the live row. If we hydrated to ``Profile`` here we'd backfill every
unset field with ``None`` and lose the "user provided X explicitly" vs
"user said nothing about X" distinction.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from claritymed.core.schemas.patient import Allergy, Condition, Medication
from claritymed.stores.paths import SLUG_RE, USER_ID_RE


class MetaConfig(BaseModel):
    """``_meta.yaml`` contents — schema version + user_ids enumeration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=1, ge=1)
    created_at: datetime
    source_hint: str | None = Field(default=None, max_length=256)
    default_category: str | None = Field(default=None, max_length=64)
    user_ids: list[str] = Field(min_length=1)

    @field_validator("user_ids")
    @classmethod
    def _validate_user_ids(cls, v: list[str]) -> list[str]:
        for uid in v:
            if not isinstance(uid, str) or not USER_ID_RE.match(uid):
                raise ValueError(f"invalid user_id in _meta.user_ids: {uid!r}")
        if len(v) != len(set(v)):
            raise ValueError("_meta.user_ids contains duplicates")
        return v


class CaseAttachment(BaseModel):
    """One attachment path on the local filesystem.

    The loader (Unit 4) verifies the path exists and is not a symlink;
    that I/O check lives outside the model so fixtures don't have to
    materialize the bytes to validate the shape.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(min_length=1)
    original_filename: str = Field(min_length=1, max_length=255)
    mime: str = Field(min_length=1, max_length=128)


class CaseEntry(BaseModel):
    """One case row inside ``<user_id>.yaml::cases``.

    ``kind`` is REQUIRED — the loader does not default it from
    ``category``. Reasons in the plan's Key Decision §``Manifest.kind`` is
    REQUIRED: ``kind`` is singular (``lab-report``) and used by the chat
    agent's save_record confirmation; ``category`` is the on-disk folder
    (plural — ``lab-reports``). Defaulting one from the other produces a
    parallel "kind = folder name" universe that drifts from the chat
    agent's vocabulary.

    ``category`` is optional in the model so the loader can backfill from
    ``MetaConfig.default_category`` at load time; absence both here AND
    in ``_meta`` is rejected by the loader, not the model.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(min_length=1)
    event_date: date
    title: str = Field(min_length=1, max_length=256)
    kind: str = Field(min_length=1, max_length=64)
    category: str | None = Field(default=None, max_length=64)
    body_md: str = ""
    attachments: list[CaseAttachment] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)

    @field_validator("case_id")
    @classmethod
    def _validate_case_id(cls, v: str) -> str:
        if not SLUG_RE.match(v):
            raise ValueError(
                f"case_id must match SLUG_RE (alnum, '_', '-'; first char "
                f"alnum or '_'; <= 128 chars): {v!r}"
            )
        return v


class FactsBundle(BaseModel):
    """Per-user facts proposed by the skill.

    ``profile`` is a plain dict, intentionally — see module docstring.
    ``allergies`` / ``conditions`` / ``medications`` each validate
    through the canonical patient schema so the importer never has to
    re-derive their constraints.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile: dict[str, Any] | None = None
    allergies: list[Allergy] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)


class UserBundle(BaseModel):
    """One per-user YAML file's body."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cases: list[CaseEntry] = Field(default_factory=list)
    facts: FactsBundle = Field(default_factory=FactsBundle)
