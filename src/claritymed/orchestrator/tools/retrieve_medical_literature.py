"""Tool: retrieve_medical_literature.

Standalone pydantic-ai tool function that can be registered on any agent
whose deps type is (or is structurally compatible with) ``AskDeps``.

Register on an agent::

    from claritymed.orchestrator.tools.retrieve_medical_literature import (
        retrieve_medical_literature,
    )
    agent.tool(retrieve_medical_literature)
"""

from __future__ import annotations

import logging

from pydantic_ai import RunContext

from claritymed.orchestrator.agents.ask_deps import AskDeps

logger = logging.getLogger(__name__)


async def retrieve_medical_literature(ctx: RunContext[AskDeps], query: str) -> str:
    """Search the medical knowledge base for evidence relevant to the query.

    Call this for medical questions requiring clinical evidence, drug
    information, differential diagnosis, treatment protocols, or
    disease-specific guidance.  Skip for greetings, chitchat, and
    clearly non-medical topics.
    """
    deps = ctx.deps
    from claritymed.orchestrator.services.events import (
        Error,
        RetrievalCompleted,
        RetrievalFiltered,
        RetrievalPending,
        RetrievalStarted,
        ToolCompleted,
        ToolStarted,
    )

    eq = deps.event_queue
    # Always emit ToolStarted so the Steps panel reflects the LLM's call
    # even when RAG is disabled — makes "RAG not configured" visible vs
    # the tool silently not being called at all.
    await eq.put(
        ToolStarted(tool_name="retrieve_medical_literature", args_preview=query[:60])
    )

    if deps.strategy is None:
        await eq.put(
            ToolCompleted(
                tool_name="retrieve_medical_literature", summary="RAG disabled"
            )
        )
        return ""

    from claritymed.core.observability.audit import audit_event
    from claritymed.core.observability.logging import get_access_logger
    from claritymed.core.rag.schemas import load_retrieval_config
    from claritymed.core.rag.strategies.base import RetrievalContext

    # Translate the query to the collection's native language when the
    # session language differs from any cross-lingual collection.  Uses
    # session language (not per-query detection) for a stable rule.
    embedding_query = query
    if deps.translation_service:
        try:
            cfg = load_retrieval_config()
            mismatched = [
                c.language
                for c in cfg.system_rag.collections
                if c.cross_lingual and c.language != deps.language
            ]
            target_lang = (
                max(set(mismatched), key=mismatched.count) if mismatched else None
            )
            if target_lang:
                get_access_logger().info(
                    "translate.query %s→%s", deps.language, target_lang
                )
                await eq.put(
                    ToolStarted(
                        tool_name="translate.query",
                        args_preview=f"translate/{target_lang}",
                    )
                )
                try:
                    from claritymed.core.observability.steps import capture_steps

                    with capture_steps() as translation_steps:
                        embedding_query = (
                            await deps.translation_service.translate_query(
                                query,
                                target_lang=target_lang,  # type: ignore[arg-type]
                            )
                        )
                    for rec in translation_steps:
                        await eq.put(
                            ToolCompleted(
                                tool_name=rec.name,
                                duration_ms=rec.duration_ms,
                                summary=rec.summary
                                or ("failed" if rec.failed else "done"),
                            )
                        )
                    if not translation_steps:
                        await eq.put(
                            ToolCompleted(tool_name="translate.query", summary="done")
                        )
                except Exception:
                    logger.warning("query translation failed, using original")
                    embedding_query = query
                    await eq.put(
                        ToolCompleted(tool_name="translate.query", summary="failed")
                    )
        except Exception:
            logger.warning("could not determine target language for translation")

    only_cloud_safe = (
        getattr(deps.provider_config, "kind", None) == "cloud"
        if deps.provider_config is not None
        else False
    )

    ret_ctx = RetrievalContext(
        query=embedding_query,
        user_id=deps.user_id,
        language=deps.language,  # type: ignore[arg-type]
        user_whitelist=deps.user_whitelist,
        only_cloud_safe=only_cloud_safe,
    )

    await eq.put(RetrievalPending())

    try:
        bundle = await deps.strategy.retrieve(ret_ctx)
    except Exception as exc:
        logger.exception("retrieval failed in retrieve_medical_literature")
        audit_event(
            "rag.retrieval.failed",
            payload={"user_id": deps.user_id, "error": str(exc)[:200]},
        )
        await eq.put(
            Error(error_type="retrieval_failed", message=str(exc), retryable=True)
        )
        return ""

    await eq.put(
        RetrievalStarted(
            active_collections=bundle.trace.active_collections,
            strategy=bundle.trace.strategy,
        )
    )

    chunks_pre = list(bundle.chunks)
    filtered = 0
    if only_cloud_safe:
        from claritymed.orchestrator.phi_guard import PhiGuard

        guard = PhiGuard.from_config()
        safe_chunks, _report = guard.filter_chunks_for_provider(
            chunks_pre, provider_kind="cloud"
        )
        filtered = len(chunks_pre) - len(safe_chunks)
        if filtered > 0:
            await eq.put(
                RetrievalFiltered(
                    total=len(chunks_pre),
                    kept=len(safe_chunks),
                    filtered_phi=filtered,
                    reason="cloud_provider_phi_guard",
                )
            )
    else:
        safe_chunks = chunks_pre

    # Drop chunks below the score threshold to suppress low-relevance results
    # for off-topic queries (e.g. greetings that still get embedded).
    # Prefers rerank_score (cross-encoder) when available.
    try:
        threshold = load_retrieval_config().system_rag.score_threshold
    except Exception:
        threshold = 0.4
    safe_chunks = [
        c
        for c in safe_chunks
        if (c.rerank_score if c.rerank_score is not None else c.score) >= threshold
    ]

    if bundle.trace.rerank_fallback:
        audit_event(
            "rag.rerank.fallback",
            payload={
                "user_id": deps.user_id,
                "collections": bundle.trace.active_collections,
            },
        )

    await eq.put(
        RetrievalCompleted(
            num_chunks=len(safe_chunks),
            fallback_triggered=bundle.trace.fallback_triggered,
            rerank_fallback=bundle.trace.rerank_fallback,
            embed_ms=bundle.trace.embed_ms,
            search_ms=bundle.trace.search_ms,
            rerank_ms=bundle.trace.rerank_ms,
            parent_expand_ms=bundle.trace.parent_expand_ms,
        )
    )

    get_access_logger().info(
        "rag.retrieval collections=%s chunks=%d embed=%dms search=%dms rerank=%dms",
        ",".join(bundle.trace.active_collections),
        len(safe_chunks),
        bundle.trace.embed_ms or 0,
        bundle.trace.search_ms or 0,
        bundle.trace.rerank_ms or 0,
    )
    audit_event(
        "rag.retrieval",
        payload={
            "user_id": deps.user_id,
            "strategy": bundle.trace.strategy,
            "active_collections": bundle.trace.active_collections,
            "num_chunks": len(safe_chunks),
            "chunks": [
                {
                    "collection": c.collection_name,
                    "doc_id": c.doc_id,
                    "doc_title": c.doc_title,
                }
                for c in safe_chunks
            ],
            "filtered_phi": filtered,
            "fallback_triggered": bundle.trace.fallback_triggered,
            "rerank_fallback": bundle.trace.rerank_fallback,
            "embed_ms": bundle.trace.embed_ms,
            "search_ms": bundle.trace.search_ms,
            "rerank_ms": bundle.trace.rerank_ms,
            "parent_expand_ms": bundle.trace.parent_expand_ms,
        },
    )

    deps.retrieved_chunks.extend(safe_chunks)
    return format_evidence(safe_chunks)


def format_evidence(chunks: list) -> str:
    """Format retrieved chunks as a numbered evidence block for the LLM."""
    if not chunks:
        return ""
    lines = ["", "Evidence (cite by [n]):"]
    for i, c in enumerate(chunks, start=1):
        body = c.parent_text or c.text
        src = c.source_uri or c.collection_name or c.source
        lines.append(f"[{i}] ({src}) {body}")
    return "\n".join(lines)
