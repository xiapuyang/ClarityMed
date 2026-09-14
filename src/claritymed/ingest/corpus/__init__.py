"""System corpus adapters.

Each ``CorpusSource`` advertises one named system RAG collection plus an
iterator of ``RawDocument`` so the ``rag corpora ingest`` CLI can wire it
to the active chunker + embedder + qdrant store.
"""

from claritymed.ingest.corpus.base import CorpusSource, ingest_corpus
from claritymed.ingest.corpus.statpearls import StatPearlsSource

__all__ = ["CorpusSource", "StatPearlsSource", "ingest_corpus"]
