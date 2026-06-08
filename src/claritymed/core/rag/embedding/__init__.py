"""Embedding subsystem.

``Embedder`` protocol + factory. v1 ships ``BgeM3HttpEmbedder`` (talks to a
HuggingFace text-embeddings-inference / BGE-M3 server). The
``_FastEmbedTestStub`` is reserved for test fixtures — production code
goes through HTTP and fails loud when the server is unreachable, so a 384-
vs 1024-dim mismatch can never silently land in Qdrant.
"""

from claritymed.core.rag.embedding.base import Embedder, SparseVector
from claritymed.core.rag.embedding.bge_m3 import BgeM3HttpEmbedder
from claritymed.core.rag.embedding.factory import build_embedder

__all__ = [
    "BgeM3HttpEmbedder",
    "Embedder",
    "SparseVector",
    "build_embedder",
]
