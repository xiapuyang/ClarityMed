"""Versioned, language-aware prompt registry.

One file per prompt name under ``core/prompts/store/<name>.yaml``. Each file
lists one or more ``versions`` with a date, notes, and bilingual templates.
``PromptRegistry.get(name, version="latest", language=None)`` returns the
filled string; missing language or version raises a typed error rather than
falling back silently — prompt selection is part of the evaluation manifest
and must be deterministic.

Why no fallback to a different language: prompts are evaluation variables, and
RAGAS scores differ between zh/en versions of "the same" prompt. Silent fallback
would make scores incomparable across runs.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from claritymed.config import PROMPTS_STORE, default_lang
from claritymed.context import language_ctx

Language = Literal["en", "zh"]


class PromptNotFound(LookupError):
    """No ``<name>.yaml`` in the prompt store."""


class PromptVersionNotFound(LookupError):
    """Named version does not exist for this prompt."""


class PromptVersion(BaseModel):
    """One immutable version of a prompt with both bilingual templates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str
    created_at: date
    notes: str
    languages: dict[Language, str]

    @model_validator(mode="after")
    def _require_both_languages(self) -> "PromptVersion":
        for lang in ("en", "zh"):
            value = self.languages.get(lang)
            if not value or not value.strip():
                msg = f"prompt version {self.version!r} missing language {lang!r}"
                raise ValueError(msg)
        return self


class Prompt(BaseModel):
    """Top-level registry entry: name + description + ordered versions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    versions: list[PromptVersion] = Field(min_length=1)


class PromptRegistry:
    """Eager-load all prompt files in the store directory.

    Loading is eager so a malformed file fails at startup, not on first request.
    """

    def __init__(self, store_dir: Path = PROMPTS_STORE) -> None:
        self.store_dir = store_dir
        self._prompts: dict[str, Prompt] = {}
        self._load_all()

    def _load_all(self) -> None:
        self._prompts.clear()
        if not self.store_dir.exists():
            return
        for path in sorted(self.store_dir.glob("*.yaml")):
            with path.open("r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
            prompt = Prompt.model_validate(raw)
            if prompt.name != path.stem:
                msg = (
                    f"prompt name {prompt.name!r} does not match filename {path.stem!r}"
                )
                raise ValueError(msg)
            self._prompts[prompt.name] = prompt

    def list(self) -> list[str]:
        """Return sorted prompt names available in the store."""
        return sorted(self._prompts)

    def reload(self) -> None:
        """Re-read all files from disk. Admin hot-reload entry."""
        self._load_all()

    def get(
        self,
        name: str,
        version: str = "latest",
        language: Optional[Language] = None,
    ) -> str:
        """Return the filled template string for ``name`` at ``version``.

        ``version="latest"`` returns the version with the most recent
        ``created_at``. ``language`` falls through to ``language_ctx`` then
        ``default_lang()``.
        """
        prompt = self._prompts.get(name)
        if prompt is None:
            raise PromptNotFound(name)

        chosen = self._pick_version(prompt, version)
        lang = self._resolve_language(language)
        return chosen.languages[lang]

    @staticmethod
    def _pick_version(prompt: Prompt, version: str) -> PromptVersion:
        if version == "latest":
            return max(prompt.versions, key=lambda v: v.created_at)
        for v in prompt.versions:
            if v.version == version:
                return v
        raise PromptVersionNotFound(f"{prompt.name}:{version}")

    @staticmethod
    def _resolve_language(language: Optional[Language]) -> Language:
        if language is not None:
            return language
        from_ctx = language_ctx.get()
        if from_ctx in ("en", "zh"):
            return from_ctx  # type: ignore[return-value]
        fallback = default_lang()
        return "zh" if fallback == "zh" else "en"


# Placeholder for tests that need a fresh datetime stamp.
_now = datetime.utcnow
