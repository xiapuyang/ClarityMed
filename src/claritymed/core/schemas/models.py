"""Provider catalog: declares WHICH backends this deployment will allow.

A ``ProviderConfig`` is intentionally tiny:

* ``id`` — our catalog key. Stored in ``Account.provider_id`` and on every
  audit log line, so it is the single string the rest of the system uses
  to refer to a backend.
* ``kind`` — ``local`` may receive PHI; ``cloud`` must pass the PHI guard
  first. The orchestrator reads this field directly. pydantic-ai has no
  equivalent concept, which is why we keep our own catalog.
* ``model`` — model name. Two shapes, depending on ``base_url``:

      base_url is None  → must be ``"<prefix>:<model>"`` (pydantic-ai's
                          ``KnownModelName`` form, e.g. ``"openai:gpt-4o"``).
                          ``infer_model`` picks the right Model + Provider
                          and reads the conventional env var for the key.
      base_url is set   → bare model name, e.g. ``"qwen3:14b"`` or
                          ``"mlx-community/Llama-3.2-3B"``. The string is
                          forwarded as-is in the request body. No prefix
                          is needed because the endpoint is already pinned.

* ``base_url`` — optional. Set when the endpoint is self-hosted (Ollama,
  MLX server, llama.cpp, LM Studio) or a corporate mirror.
* ``api_key_env`` — optional. Honored **only** when ``base_url`` is set,
  for endpoints that need auth (LM Studio with a token, a Tailscale-fronted
  Ollama, etc.). Missing env in that case raises ``MissingApiKeyError``.
  Stock cloud providers ignore this field — they read their own env vars.
* ``thinking`` — optional reasoning/thinking toggle. Forwarded to
  ``pydantic_ai.settings.ModelSettings.thinking``, which knows how to
  translate the unified value into each vendor's native field
  (Anthropic ``thinking``, OpenAI ``reasoning_effort``, Google
  ``thinking_config``, …). Silently ignored on models that don't
  support reasoning, which is why we keep this one knob instead of
  duplicating per-vendor schemas.

Declaring a cloud entry only says "this option exists." Per-user opt-in
(``Account.cloud_provider_opt_in``) and the admin populating the env var
are still required before any request actually leaves the box.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ProviderKind = Literal["local", "cloud"]
ThinkingLevel = Literal["minimal", "low", "medium", "high", "xhigh"]


class ProviderConfig(BaseModel):
    """One row of the provider catalog."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    id: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    kind: ProviderKind
    model: str = Field(min_length=1)
    base_url: str | None = Field(default=None, min_length=1)
    api_key_env: str | None = Field(default=None, min_length=1, max_length=64)
    thinking: bool | ThinkingLevel | None = Field(default=None)

    @model_validator(mode="after")
    def _check_model_shape(self) -> "ProviderConfig":
        # When no base_url is set, pydantic-ai's ``infer_model`` needs the
        # ``"<prefix>:<model>"`` form to pick the right backend. We enforce
        # the shape here so a typo in YAML fails at load, not at first request.
        if self.base_url is None and ":" not in self.model:
            raise ValueError(
                f"model {self.model!r} needs '<provider>:<model>' format "
                "when no base_url is set (e.g. 'openai:gpt-4o'). "
                "For self-hosted endpoints, set base_url and use a bare "
                "model name."
            )
        if self.base_url is None and self.api_key_env is not None:
            # Stock cloud providers read their own env vars; honoring
            # api_key_env here would create two layers doing the same job.
            raise ValueError(
                f"api_key_env={self.api_key_env!r} is only honored when "
                "base_url is set. Stock cloud providers read their own env "
                "vars (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.) — drop "
                "api_key_env, or set base_url if you really want a custom "
                "endpoint with custom auth."
            )
        return self


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
