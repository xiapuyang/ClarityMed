"""Long-running process entry points.

Lives parallel to ``cli/`` — both packages contain code that owns a
process boundary, but ``cli/`` is short-lived (one user invocation) while
``servers/`` is long-lived (a FastAPI/uvicorn process).

Current members:

* ``embedder`` — BGE-M3 server, TEI-compatible wire format
  (``/embed`` + ``/embed_sparse``). Console script: ``claritymed-embedder``.
* ``reranker`` — bge-reranker-v2-m3 server, TEI ``/rerank``-compatible.
  Console script: ``claritymed-reranker``.

Both depend on the ``rag-server`` optional dep group
(``uv sync --extra rag-server``) — FlagEmbedding + fastapi + uvicorn are
not in the main runtime so an LLM-only deployment stays lean.
"""
