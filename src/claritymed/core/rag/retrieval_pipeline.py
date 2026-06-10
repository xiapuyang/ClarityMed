"""Shared RAG retrieval pipeline.

Bundles cross-lingual query translation, strategy invocation, cloud-safe
PHI filtering, score-threshold filtering, and audit / event emission so
``RagFeature.pre_invoke`` (deterministic mode) and the
``retrieve_medical_literature`` tool body (tool mode) share one
implementation. The helper has no side effects on ``deps.retrieved_chunks``;
callers extend that list themselves so the data flow is explicit.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from claritymed.core.rag.schemas import load_retrieval_config

if TYPE_CHECKING:
    from claritymed.core.schemas.retrieval import RetrievedChunk
    from claritymed.core.turn_state import TurnState

logger = logging.getLogger(__name__)


async def _translate_query(deps: "TurnState", query: str) -> str:
    """Run cross-lingual query translation when configured.

    Emits ``ToolStarted`` / ``ToolCompleted`` events for the translation
    sub-step so the TUI Steps panel shows it. Returns the original query
    on any failure — translation is best-effort.
    """
    from claritymed.core.observability.logging import get_access_logger
    from claritymed.core.events import ToolCompleted, ToolStarted

    if deps.translation_service is None:
        return query
    try:
        cfg = load_retrieval_config()
        mismatched = [
            c.language
            for c in cfg.system_rag.collections
            if c.cross_lingual and c.language != deps.language
        ]
    except Exception:
        logger.warning("could not determine target language for translation")
        return query

    if not mismatched:
        return query
    target_lang = max(set(mismatched), key=mismatched.count)
    get_access_logger().info("translate.query %s→%s", deps.language, target_lang)
    await deps.event_queue.put(
        ToolStarted(
            tool_name="translate.query",
            args_preview=f"translate/{target_lang}",
        )
    )
    try:
        from claritymed.core.observability.steps import capture_steps

        with capture_steps() as translation_steps:
            translated = await deps.translation_service.translate_query(
                query,
                target_lang=target_lang,  # type: ignore[arg-type]
            )
        for rec in translation_steps:
            await deps.event_queue.put(
                ToolCompleted(
                    tool_name=rec.name,
                    duration_ms=rec.duration_ms,
                    summary=rec.summary or ("failed" if rec.failed else "done"),
                )
            )
        if not translation_steps:
            await deps.event_queue.put(
                ToolCompleted(tool_name="translate.query", summary="done")
            )
        return translated
    except Exception:
        logger.warning("query translation failed, using original")
        await deps.event_queue.put(
            ToolCompleted(tool_name="translate.query", summary="failed")
        )
        return query


async def perform_retrieval(deps: "TurnState", query: str) -> list["RetrievedChunk"]:
    """Run the full retrieval pipeline; return safe chunks for this call.

    Emits ``RetrievalPending`` / ``RetrievalStarted`` / ``RetrievalFiltered``
    / ``RetrievalCompleted`` on ``deps.event_queue`` and writes the
    ``rag.retrieval`` audit event. Pure return — does **not** mutate
    ``deps.retrieved_chunks``; the caller is responsible for accumulating
    the per-call subset into the cumulative list. Tool mode may call
    this 1..N times per turn, and the per-call list is what the tool
    feeds back to the LLM as evidence (numbered [1]..[n]); the cumulative
    deps.retrieved_chunks is what the service uses to render Sources.
    """
    from claritymed.core.observability.audit import audit_event
    from claritymed.core.observability.logging import get_access_logger
    from claritymed.core.rag.strategies.base import RetrievalContext
    from claritymed.core.events import (
        Error,
        RetrievalCompleted,
        RetrievalFiltered,
        RetrievalPending,
        RetrievalStarted,
    )

    if deps.strategy is None:
        return []

    embedding_query = await _translate_query(deps, query)

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

    await deps.event_queue.put(RetrievalPending())

    try:
        bundle = await deps.strategy.retrieve(ret_ctx)
    except Exception as exc:
        logger.exception("retrieval failed")
        audit_event(
            "rag.retrieval.failed",
            payload={"user_id": deps.user_id, "error": str(exc)[:200]},
        )
        await deps.event_queue.put(
            Error(error_type="retrieval_failed", message=str(exc), retryable=True)
        )
        return []

    await deps.event_queue.put(
        RetrievalStarted(
            active_collections=bundle.trace.active_collections,
            strategy=bundle.trace.strategy,
        )
    )

    chunks_pre = list(bundle.chunks)
    filtered = 0
    if only_cloud_safe:
        from claritymed.core.phi.guard import get_default_guard

        guard = get_default_guard()
        safe_chunks, _report = guard.filter_chunks_for_provider(
            chunks_pre, provider_kind="cloud"
        )
        filtered = len(chunks_pre) - len(safe_chunks)
        if filtered > 0:
            await deps.event_queue.put(
                RetrievalFiltered(
                    total=len(chunks_pre),
                    kept=len(safe_chunks),
                    filtered_phi=filtered,
                    reason="cloud_provider_phi_guard",
                )
            )
        # Defense-in-depth: the flag-based filter trusts ingest-time
        # ``is_phi``/``can_cloud`` metadata, which means a chunk that
        # got past ingest with stale flags can carry PHI into the
        # cloud prompt. Run the regex layer over the surviving
        # chunk text + parent_text so the Evidence block can never be
        # the leak channel even if a can_cloud=True flag was wrong.
        for chunk in safe_chunks:
            new_text, _r = guard.scrub_free_text(chunk.text)
            if new_text != chunk.text:
                chunk.text = new_text
            if chunk.parent_text:
                new_parent, _rp = guard.scrub_free_text(chunk.parent_text)
                if new_parent != chunk.parent_text:
                    chunk.parent_text = new_parent
    else:
        safe_chunks = chunks_pre

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

    await deps.event_queue.put(
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

    return safe_chunks


def deduplicate_chunks(chunks: list) -> list:
    """Return chunks with at most one entry per doc_id (first = highest score)."""
    seen: dict[str, None] = {}
    result = []
    for c in chunks:
        if c.doc_id not in seen:
            seen[c.doc_id] = None
            result.append(c)
    return result


def format_evidence(chunks: list, *, cumulative: list | None = None) -> str:
    """Format retrieved chunks as a numbered evidence block for the LLM.

    Deduplicates by doc_id before numbering so citation indices in the
    LLM response match the Sources section shown to the user.

    The parenthesised label is a *user-facing* source identifier
    (``source_uri`` or ``doc_title``). ``collection_name`` is the
    internal vector-store id and is deliberately never exposed — letting
    the LLM see ``statpearls_en`` in evidence trains it to mimic that
    string as a "source" in later turns (observed hallucination).
    Chunks without any human-readable identifier emit an unlabeled
    ``[N] <body>`` line instead.

    When the same turn calls the retrieval tool multiple times, pass
    ``cumulative=deps.retrieved_chunks`` (the union across calls) and
    ``chunks=safe_chunks`` (this call's slice). Numbering is then taken
    from each chunk's position in the cumulative dedup, so ``[2]`` in
    one tool result and ``[2]`` in another result refer to the same
    document — matching what the user finally sees in ``Sources``.
    Without the cumulative arg the function falls back to per-call
    numbering, which is correct only for single-call turns.
    """
    by_doc = deduplicate_chunks(chunks)
    if not by_doc:
        return ""
    indices: dict[str, int]
    if cumulative is not None:
        cumulative_unique = deduplicate_chunks(cumulative)
        indices = {c.doc_id: i + 1 for i, c in enumerate(cumulative_unique)}
    else:
        indices = {c.doc_id: i + 1 for i, c in enumerate(by_doc)}
    lines = ["", "Evidence (cite by [n]):"]
    for c in by_doc:
        n = indices.get(c.doc_id)
        if n is None:
            continue
        body = c.parent_text or c.text
        src = c.source_uri or c.doc_title
        prefix = f"[{n}] ({src}) " if src else f"[{n}] "
        lines.append(f"{prefix}{body}")
    return "\n".join(lines)
