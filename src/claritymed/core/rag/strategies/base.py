"""RagStrategy protocol and RetrievalContext value type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from claritymed.core.rag.schemas import EvidenceBundle

ContextLanguage = Literal["en", "zh"]


@dataclass(frozen=True)
class RetrievalContext:
    """Inputs to ``RagStrategy.retrieve`` — caller's request frame."""

    query: str
    user_id: str
    language: ContextLanguage
    user_whitelist: list[str] | None = None
    only_cloud_safe: bool = False


@runtime_checkable
class RagStrategy(Protocol):
    """Top-level RAG approach. v1 ships NaiveHybridStrategy."""

    async def retrieve(self, ctx: RetrievalContext) -> EvidenceBundle: ...
