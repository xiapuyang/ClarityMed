"""Provider catalog: how to reach each model, never the credential itself.

A ``ProviderConfig`` declares HOW to talk to a backend:

* ``kind`` — ``local`` traffic may carry PHI; ``cloud`` traffic must pass the
  PHI guard first. The orchestrator reads this field directly.
* ``api`` — wire protocol. ``openai`` covers Ollama, DeepSeek, Gemini's
  OpenAI-compatible mode, DashScope/Qwen, Kimi, OpenRouter, and most others.
  ``anthropic`` is for the native Anthropic Messages API (and any local
  server emulating it).
* ``api_key_env`` — name of the env var holding the key. The actual key is
  never written to YAML or stored on disk. ``None`` for local backends.

Declaring a cloud entry only says "this option exists." Per-user opt-in
(``Account.cloud_provider_opt_in``) and the admin populating the env var
are still required before any request actually leaves the box.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ProviderKind = Literal["local", "cloud"]
ProviderApi = Literal["openai", "anthropic"]


class ProviderConfig(BaseModel):
    """One row of the provider catalog."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    id: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    kind: ProviderKind
    api: ProviderApi
    base_url: str = Field(min_length=1)
    model: str = Field(min_length=1)
    api_key_env: str | None = Field(default=None, max_length=64)


class ModelsConfig(BaseModel):
    """Parsed ``configs/models.yaml``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    providers: list[ProviderConfig] = Field(min_length=1)
    default_provider: str = Field(min_length=1)

    @model_validator(mode="after")
    def _check_ids_and_default(self) -> "ModelsConfig":
        ids = [p.id for p in self.providers]
        if len(ids) != len(set(ids)):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate provider ids: {dupes}")
        if self.default_provider not in ids:
            raise ValueError(
                f"default_provider {self.default_provider!r} not in providers"
            )
        return self
