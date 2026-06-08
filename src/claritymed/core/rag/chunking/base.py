"""Chunker protocol and project-owned chunk types.

The Chunker boundary is intentionally project-owned (not
``llama_index.core.schema.Node``) so the rest of the RAG runtime never
imports LlamaIndex types. A later ``RaptorChunker`` /
``LateChunkingChunker`` plug fits the same protocol without touching any
caller.

``ChunkedDocument`` separates parent chunks (text-only KV stored in the
docstore for prompt context) from child chunks (embedded + searchable in
Qdrant). Each child carries its ``parent_id`` so AskService can hydrate
parent text after retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class RawDocument:
    """Input to a Chunker: one upstream source document."""

    doc_id: str
    text: str
    language: str = "en"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParentChunk:
    """One parent-level chunk. Stored in the docstore; not embedded."""

    parent_id: str
    text: str
    doc_id: str
    parent_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChildChunk:
    """One child-level chunk. Embedded into Qdrant; refers back to a parent."""

    child_id: str
    text: str
    parent_id: str
    doc_id: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkedDocument:
    """Result of chunking one ``RawDocument``: parents + children."""

    parents: list[ParentChunk]
    children: list[ChildChunk]


@runtime_checkable
class Chunker(Protocol):
    """Document → (parents, children) protocol."""

    def chunk(self, doc: RawDocument) -> ChunkedDocument: ...
