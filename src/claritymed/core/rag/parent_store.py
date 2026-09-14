"""Persistent parent-chunk KV store.

Wraps LlamaIndex ``SimpleDocumentStore`` (the JSON-backed KV in
``llama-index-core``) with project types so callers never see
``TextNode`` directly. One ``ParentStore`` instance = one JSON file:

* ``shared_parent_docstore_path()`` — all system collection parents
  (admin-managed, not PHI).
* ``user_parent_docstore_path(user_id)`` — that user's user_rag parents
  (per-user PHI; isolated by file location, not by filter).

Children live in Qdrant (embedded + searchable); parents live here. When
``HybridRetriever`` returns a chunk to the prompt assembler, ``parent_text``
is populated by ``get_text(parent_id)`` so the LLM sees more context than
just the embedded child window.

Persistence is explicit: callers ``persist()`` after a batch of writes.
Writes that are not persisted survive in memory until the next persist or
process death — this matches LlamaIndex's own semantics.
"""

from __future__ import annotations

from pathlib import Path

from llama_index.core.schema import TextNode
from llama_index.core.storage.docstore import SimpleDocumentStore

from claritymed.core.rag.chunking.base import ParentChunk


class ParentStore:
    """Persistent parent-chunk KV store, backed by SimpleDocumentStore."""

    def __init__(self, persist_path: Path) -> None:
        self._path = persist_path
        if persist_path.exists():
            self._docstore = SimpleDocumentStore.from_persist_path(str(persist_path))
        else:
            persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._docstore = SimpleDocumentStore()

    # --- write ---------------------------------------------------------

    def put(self, parent: ParentChunk) -> None:
        """Upsert one parent. Memory only — call ``persist()`` to flush."""
        self._docstore.add_documents([self._to_node(parent)], allow_update=True)

    def bulk_put(self, parents: list[ParentChunk]) -> int:
        """Upsert many parents. Returns count written."""
        if not parents:
            return 0
        nodes = [self._to_node(p) for p in parents]
        self._docstore.add_documents(nodes, allow_update=True)
        return len(nodes)

    def delete(self, parent_id: str) -> bool:
        """Remove a parent by id. Returns True iff it existed."""
        if not self._docstore.document_exists(parent_id):
            return False
        self._docstore.delete_document(parent_id, raise_error=False)
        return True

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Remove every parent whose ``metadata['doc_id']`` matches.

        Used when the user (or admin) deletes a source document and we
        need to scrub all of its parent chunks atomically.
        """
        to_delete = [
            pid
            for pid, node in self._docstore.docs.items()
            if node.metadata.get("doc_id") == doc_id
        ]
        for pid in to_delete:
            self._docstore.delete_document(pid, raise_error=False)
        return len(to_delete)

    def persist(self) -> None:
        """Flush in-memory state to disk."""
        self._docstore.persist(persist_path=str(self._path))

    # --- read ----------------------------------------------------------

    def get_text(self, parent_id: str) -> str | None:
        """Return parent text, or ``None`` if not found.

        Returning None rather than raising lets retriever code degrade
        gracefully: a missing parent (e.g. docstore not yet ingested for
        this child) is logged + skipped, not a hard failure.
        """
        if not self._docstore.document_exists(parent_id):
            return None
        node = self._docstore.get_node(parent_id, raise_error=False)
        if node is None or not isinstance(node, TextNode):
            return None
        return node.text

    def exists(self, parent_id: str) -> bool:
        return self._docstore.document_exists(parent_id)

    def __len__(self) -> int:
        return len(self._docstore.docs)

    # --- internals -----------------------------------------------------

    @staticmethod
    def _to_node(parent: ParentChunk) -> TextNode:
        metadata = {
            "doc_id": parent.doc_id,
            "parent_index": parent.parent_index,
            **parent.metadata,
        }
        return TextNode(id_=parent.parent_id, text=parent.text, metadata=metadata)
