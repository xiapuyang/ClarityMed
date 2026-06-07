"""Rag mode: ingest documents into the per-user RAG store.

The single non-stub tool here is ``embed_and_store``, which calls
``stores/user_rag.UserRagStore.add_document``. Phase 1 stubs the
chunking / web-fetching steps so the rest of the system can still be
exercised end-to-end before those real pipelines land.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from claritymed.core.prompts.registry import PromptRegistry
from claritymed.core.schemas.receipts import IngestionReceipt

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.models import Model

    from claritymed.stores.user_rag import UserRagStore

RAG_TOOL_NAMES: list[str] = [
    "chunk_document_stub",
    "embed_and_store",
    "tag_phi",
    "fetch_web_link_stub",
]


def chunk_document_stub(text: str, target_chunk_size: int = 500) -> list[str]:
    """Trivial naive chunker — splits on blank lines. Real plan replaces."""
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    return parts or [text]


def fetch_web_link_stub(url: str) -> str:
    return f"[stub: fetch_web_link not implemented for {url}]"


def tag_phi(text: str) -> dict[str, bool]:
    """Naive PHI detector used to flag chunks. Real plan adds NER + heuristics."""
    markers = ("MRN", "13", "@")
    return {"is_phi": any(m in text for m in markers)}


def embed_and_store(
    store: "UserRagStore",
    user_id: str,
    chunks: list[str],
    doc_id: str | None = None,
    public: bool = False,
    metadata: dict | None = None,
) -> IngestionReceipt:
    """Real tool: persist ``chunks`` to the user's RAG collection.

    ``UserRagStore.add_document`` already scrubs each chunk through the PHI
    guard before embedding, so the rag agent does not need to scrub here.
    """
    final_doc_id = doc_id or f"doc-{uuid.uuid4().hex[:12]}"
    written = store.add_document(
        user_id=user_id,
        doc_id=final_doc_id,
        chunks=chunks,
        metadata=metadata,
        public=public,
    )
    return IngestionReceipt(
        doc_id=final_doc_id,
        chunk_count=written,
        embedding_status="ok",
        public=public,
    )


def make_rag_agent(
    model: "Model",
    registry: PromptRegistry | None = None,
    language: str = "en",
) -> "Agent[None, IngestionReceipt]":
    """Build the Pydantic AI agent for rag mode (only used when LLM enabled)."""
    from pydantic_ai import Agent

    reg = registry or PromptRegistry()
    system_prompt = reg.get("rag", language=language)  # type: ignore[arg-type]

    agent = Agent(
        model,
        output_type=IngestionReceipt,
        system_prompt=system_prompt,
    )
    return agent
