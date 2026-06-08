"""Parent-child chunker backed by LlamaIndex ``HierarchicalNodeParser``.

LlamaIndex's HierarchicalNodeParser with ``chunk_sizes=[parent, child]``
produces a flat node list where:

* The first ``len(parents)`` nodes are the parent-level chunks (size =
  ``parent_size``).
* The remaining nodes are child-level chunks (size = ``child_size``),
  each carrying ``relationships[NodeRelationship.PARENT]`` pointing to
  its parent node id.

We translate that into our project-owned ``ChunkedDocument`` shape so the
LlamaIndex ``Node`` type does not leak past this adapter.

CJK note: HierarchicalNodeParser uses ``SentenceSplitter`` underneath,
which has a secondary regex (``[^,.;。？！]+[,.;。？！]?``) that splits on
both ASCII and CJK sentence-ending punctuation. Chinese inputs do not
need a separate code path.

Chunk-size semantics: LlamaIndex's chunk_size counts approximate tokens
(via tiktoken). The retrieval.yaml field names ``child_tok`` / ``parent_tok``
match that semantics, so the YAML value flows through unchanged.
"""

from __future__ import annotations

import uuid

from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import HierarchicalNodeParser
from llama_index.core.schema import BaseNode, NodeRelationship

from claritymed.core.rag.chunking.base import (
    ChildChunk,
    ChunkedDocument,
    Chunker,
    ParentChunk,
    RawDocument,
)


class LlamaIndexParentChildChunker(Chunker):
    """Two-level (parent, child) chunker using ``HierarchicalNodeParser``."""

    def __init__(
        self,
        *,
        parent_tok: int,
        child_tok: int,
        overlap_tok: int = 0,
    ) -> None:
        if parent_tok <= child_tok:
            raise ValueError(
                f"parent_tok ({parent_tok}) must exceed child_tok ({child_tok})"
            )
        if overlap_tok < 0:
            raise ValueError(f"overlap_tok must be >= 0, got {overlap_tok}")
        self._parent_tok = parent_tok
        self._child_tok = child_tok
        self._overlap_tok = overlap_tok
        self._parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[parent_tok, child_tok],
            chunk_overlap=overlap_tok,
        )

    def chunk(self, doc: RawDocument) -> ChunkedDocument:
        if not doc.text:
            return ChunkedDocument(parents=[], children=[])

        # Stable id namespace so re-chunking the same doc.text yields the
        # same parent_id / child_id (callers rely on this for idempotent
        # upsert). LlamaIndex node ids default to per-call UUIDs; we
        # synthesize ours from doc_id + ordinal positions instead.
        li_doc = LlamaDocument(
            text=doc.text,
            metadata={"doc_id": doc.doc_id, "language": doc.language, **doc.metadata},
        )
        nodes = self._parser.get_nodes_from_documents([li_doc])

        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []

        # First pass: collect parents in order of appearance.
        parent_id_by_node: dict[str, str] = {}
        for node in nodes:
            if not self._is_parent(node):
                continue
            parent_index = len(parents)
            parent_id = f"{doc.doc_id}#p{parent_index}"
            parent_id_by_node[node.node_id] = parent_id
            parents.append(
                ParentChunk(
                    parent_id=parent_id,
                    text=node.get_content(),
                    doc_id=doc.doc_id,
                    parent_index=parent_index,
                    metadata=dict(doc.metadata),
                )
            )

        # Second pass: collect children with per-parent local index.
        child_counts: dict[str, int] = {}
        for node in nodes:
            if self._is_parent(node):
                continue
            parent_node_id = self._parent_node_id(node)
            if parent_node_id is None or parent_node_id not in parent_id_by_node:
                # Orphan child — should not happen with HierarchicalNodeParser
                # output, but tolerate by skipping rather than crashing the
                # whole ingest run.
                continue
            parent_id = parent_id_by_node[parent_node_id]
            chunk_index = child_counts.get(parent_id, 0)
            child_counts[parent_id] = chunk_index + 1
            children.append(
                ChildChunk(
                    child_id=self._child_id(doc.doc_id, parent_id, chunk_index),
                    text=node.get_content(),
                    parent_id=parent_id,
                    doc_id=doc.doc_id,
                    chunk_index=chunk_index,
                    metadata=dict(doc.metadata),
                )
            )

        return ChunkedDocument(parents=parents, children=children)

    # --- internals ------------------------------------------------------

    @staticmethod
    def _is_parent(node: BaseNode) -> bool:
        """A node is a parent when nothing else points to it as PARENT.

        HierarchicalNodeParser stores PARENT relationships only on
        children; parents carry CHILD relationships pointing down.
        """
        return NodeRelationship.PARENT not in node.relationships

    @staticmethod
    def _parent_node_id(node: BaseNode) -> str | None:
        rel = node.relationships.get(NodeRelationship.PARENT)
        if rel is None:
            return None
        # RelatedNodeInfo or list — HierarchicalNodeParser uses a single ref.
        if isinstance(rel, list):
            rel = rel[0] if rel else None
        return getattr(rel, "node_id", None)

    @staticmethod
    def _child_id(doc_id: str, parent_id: str, chunk_index: int) -> str:
        # Stable id derived from doc_id + parent ordinal + chunk ordinal.
        # Use uuid5 over the deterministic seed so the result is a valid
        # UUID Qdrant accepts as a point id (which it does for str-uuid
        # form). Tests rely on stable ids across rechunking.
        seed = f"{doc_id}::{parent_id}::c{chunk_index}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))
