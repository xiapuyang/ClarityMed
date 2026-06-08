"""Chunking subsystem.

``Chunker`` protocol + factory. v1 ships ``LlamaIndexParentChildChunker``
(thin adapter around LlamaIndex ``HierarchicalNodeParser`` — handles CJK
sentence boundaries via its secondary regex out of the box).

The Chunker protocol is project-owned so plugging a different upstream
(RAPTOR, late chunking, custom) does not leak LlamaIndex types into
``HybridRetriever`` / ``UserRagStore``.
"""

from claritymed.core.rag.chunking.base import (
    ChildChunk,
    ChunkedDocument,
    Chunker,
    ParentChunk,
    RawDocument,
)
from claritymed.core.rag.chunking.factory import build_chunker
from claritymed.core.rag.chunking.parent_child import LlamaIndexParentChildChunker

__all__ = [
    "ChildChunk",
    "ChunkedDocument",
    "Chunker",
    "LlamaIndexParentChildChunker",
    "ParentChunk",
    "RawDocument",
    "build_chunker",
]
