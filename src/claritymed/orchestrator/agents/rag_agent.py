"""Rag mode tools: persist user-uploaded text into the per-user RAG store.

The chunking + embedding pipeline lives in ``stores/user_rag.py`` (the
ingest facade) — this module is now a thin async wrapper that the
``RagService`` calls. The previous Phase-1 stubs (``chunk_document_stub``
etc.) were retired in Unit 8 of the RAG plan when real parent-child
chunking + bge-m3 embedding came online.

``tag_phi`` and ``fetch_web_link_stub`` remain as reserved tool names
because the rag agent (LLM-mediated mode) still exposes them — the
deterministic ingest path used by ``RagService`` does not need them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from claritymed.core.prompts.registry import PromptRegistry
from claritymed.core.schemas.receipts import IngestionReceipt
from claritymed.stores.user_rag import generate_doc_id

if TYPE_CHECKING:
    from pydantic_ai import Agent
    from pydantic_ai.models import Model

    from claritymed.stores.user_rag import UserRagStore

RAG_TOOL_NAMES: list[str] = [
    "embed_and_store",
    "tag_phi",
    "fetch_web_link_stub",
]


def fetch_web_link_stub(url: str) -> str:
    """Reserved tool name; LLM-mediated fetch is out of scope for v1."""
    return f"[stub: fetch_web_link not implemented for {url}]"


def tag_phi(text: str) -> dict[str, bool]:
    """Reserved tool name; UserRagStore already scrubs at ingest time."""
    markers = ("MRN", "13", "@")
    return {"is_phi": any(m in text for m in markers)}


async def embed_and_store(
    store: "UserRagStore",
    user_id: str,
    text: str,
    *,
    doc_id: str | None = None,
    public: bool = False,
    metadata: dict | None = None,
) -> IngestionReceipt:
    """Persist ``text`` to the user's RAG store.

    Replaces the Phase-1 ``chunk_document_stub`` + sync ``embed_and_store``
    pair. ``UserRagStore.add_document`` scrubs through ``PhiGuard`` and
    chunks via the configured parent-child chunker before embedding.
    """
    final_doc_id = doc_id or generate_doc_id()
    written = await store.add_document(
        user_id=user_id,
        doc_id=final_doc_id,
        text=text,
        metadata=metadata,
        public=public,
    )
    return IngestionReceipt(
        doc_id=final_doc_id,
        chunk_count=written,
        embedding_status="ok" if written else "stub",
        public=public,
    )


def make_rag_agent(
    model: "Model",
    registry: PromptRegistry | None = None,
    language: str = "en",
) -> "Agent[None, IngestionReceipt]":
    """Build the Pydantic AI agent for rag mode (LLM-mediated path only)."""
    from pydantic_ai import Agent

    reg = registry or PromptRegistry()
    system_prompt = reg.get("rag", language=language)  # type: ignore[arg-type]

    return Agent(
        model,
        output_type=IngestionReceipt,
        system_prompt=system_prompt,
    )
