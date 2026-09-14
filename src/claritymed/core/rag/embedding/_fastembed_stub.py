"""Test fixture only — sync fastembed wrapper retained for legacy callers.

**Do not import this in production paths.** ``BgeM3HttpEmbedder`` is the
sole production embedder; routing around it produces a 384-vs-1024 dim
mismatch the moment the user upgrades their corpus.

Kept here because:

1. ``stores/user_rag.py`` was originally built against a sync embed
   interface (Foundation phase). Unit 8 of the RAG plan migrates that
   call site; until then, the test suite uses this stub to avoid pulling
   ``fastembed`` model weights at test time.
2. Several existing ``tests/stores/test_user_rag.py`` cases pass a stub
   embedder directly; preserving the same sync interface keeps those
   tests green during the migration window.

After Unit 8 lands, ``stores/user_rag.py`` uses ``Embedder`` (async) and
this module exists only as a test helper.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_VECTOR_DIM = 384  # fastembed BAAI/bge-small-en-v1.5


class _FastEmbedTestStub:
    """Sync embed wrapper around fastembed. Test fixture only."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_name = model_name
        self._model: Any | None = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding  # imported lazily

            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    def embed(self, text: str) -> list[float]:
        model = self._ensure_model()
        return next(iter(model.embed([text]))).tolist()

    @property
    def dimension(self) -> int:
        return DEFAULT_VECTOR_DIM
