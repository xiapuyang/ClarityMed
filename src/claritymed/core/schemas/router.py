"""Router config schema.

Mirrors ``configs/router.yaml``. Holds confidence thresholds and rule
dictionaries used by the hybrid mode router (rule-first, LLM fallback).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

ModeName = Literal["ingest", "ask", "rag"]


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
    """Top-level model for ``configs/router.yaml``."""

    model_config = ConfigDict(extra="forbid")

    high_threshold: float = Field(0.9, ge=0.0, le=1.0)
    low_threshold: float = Field(0.5, ge=0.0, le=1.0)
    rules: RouterRules
    llm_fallback: RouterLlmFallback
