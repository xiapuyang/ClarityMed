"""RAG strategy: top-level pluggable retrieval approach.

v1 ships ``NaiveHybridStrategy`` (single-shot hybrid + rerank + parent
expand, with optional CRAG-lite grader + 1-shot deterministic rewrite).
Future ids in the same protocol:

* ``agentic`` — multi-turn retrieve/evaluate/retry (Self-RAG, CRAG full,
  FLARE family). Reflection loop is the difference, not the retriever.
* ``hyde`` — Hypothetical Document Embeddings; LLM drafts a hypothetical
  answer, embed *that*, retrieve, reuse top hits as evidence.
* ``graph`` — KG-augmented (MedGraphRAG / KG2RAG family).
* ``raptor`` — Recursive Abstractive Processing.
* ``late_chunking`` — chunk at retrieval time using the embedder's own
  attention.

Strategies share the same ``Retriever`` and pluggable components below,
they differ in how the retrieve loop is structured. NaiveHybridStrategy
is one-shot; AgenticRagStrategy adds a reflection loop.
"""

from claritymed.core.rag.strategies.base import RagStrategy, RetrievalContext
from claritymed.core.rag.strategies.factory import build_strategy
from claritymed.core.rag.strategies.hyde import HydeStrategy
from claritymed.core.rag.strategies.naive_hybrid import NaiveHybridStrategy

__all__ = [
    "HydeStrategy",
    "NaiveHybridStrategy",
    "RagStrategy",
    "RetrievalContext",
    "build_strategy",
]
