"""Offline ingest entry points (CLI-driven, not on the request hot path).

Two sub-namespaces:

* ``ingest.corpus`` — system RAG sources (StatPearls, PubMed, DailyMed,
  ...). Each new source adds one ``CorpusSource`` adapter + a YAML
  catalog entry; the ``rag corpora ingest`` CLI command stitches them
  to the shared chunker + embedder + qdrant collection.
* ``ingest.finetune`` — corpora that are **not** indexed (MedDialog-CN,
  Huatuo-26M). Preprocessing emits JSONL fine-tune material under
  ``data/finetune/`` and never touches Qdrant.
"""
