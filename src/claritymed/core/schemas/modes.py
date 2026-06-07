"""Interaction mode registry schemas.

Mirrors ``configs/modes.yaml``. Three modes (``ingest`` / ``ask`` / ``rag``)
each declare prompt key, allowed tools, audit event type, and whether the
mode is permitted to call the LLM. The router config under the same file
holds confidence thresholds and rule dictionaries for mode classification.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ModeName = Literal["ingest", "ask", "rag"]


class ModeConfig(BaseModel):
    """Per-mode metadata loaded from ``configs/modes.yaml``."""

    model_config = ConfigDict(extra="forbid")

    prompt_key: str = Field(..., description="Key in prompt registry to load.")
    audit_event_type: str = Field(..., description="Event type written to audit log.")
    allow_llm_inference: bool = Field(
        ...,
        description=(
            "When False the mode must not call LLMClient.chat. When True the "
            "mode may invoke an LLM but its prompt is responsible for enforcing "
            "the project's domain constraints (no numerical interpretation, etc)."
        ),
    )
    tools: list[str] = Field(
        default_factory=list,
        description="White-listed tool function names this mode may invoke.",
    )


class RouterRules(BaseModel):
    """Rule dictionaries used by the deterministic part of the router."""

    model_config = ConfigDict(extra="forbid")

    explicit_prefixes: list[dict] = Field(default_factory=list)
    imperative_verbs_ingest: list[str] = Field(default_factory=list)
    imperative_verbs_rag: list[str] = Field(default_factory=list)
    question_patterns: list[str] = Field(default_factory=list)


class RouterLlmFallback(BaseModel):
    """LLM fallback parameters when rules do not yield a confident decision."""

    model_config = ConfigDict(extra="forbid")

    provider_id: str = Field(..., description="Provider id in models.yaml.")


class RouterConfig(BaseModel):
    """Top-level router config."""

    model_config = ConfigDict(extra="forbid")

    high_threshold: float = Field(0.9, ge=0.0, le=1.0)
    low_threshold: float = Field(0.5, ge=0.0, le=1.0)
    rules: RouterRules
    llm_fallback: RouterLlmFallback


class ModesConfig(BaseModel):
    """Top-level model for ``configs/modes.yaml``."""

    model_config = ConfigDict(extra="forbid")

    modes: dict[ModeName, ModeConfig]
    router: RouterConfig

    def get(self, name: ModeName) -> ModeConfig:
        """Return the ``ModeConfig`` for ``name``."""
        return self.modes[name]
