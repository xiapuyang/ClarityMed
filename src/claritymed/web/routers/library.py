"""``/api/v1/library`` — list collections, search, and ingest documents.

The web counterpart to the TUI's ``/library`` modal and ``/upload``
slash command. Three endpoints:

* ``GET  /api/v1/library`` — list system + user collections with
  best-effort chunk counts. Mirrors ``LibraryModal._render_list``.
* ``POST /api/v1/library/search`` — runs the same ``RagStrategy`` the
  chat router uses (``app.state.rag_strategy``); 503 when RAG is off
  in ``retrieval.yaml``.
* ``POST /api/v1/library/ingest`` — builds an :class:`UploadBundle`
  via :func:`build_upload_bundle`, runs ``bundle.validate``, and feeds
  each ``ok`` part through :class:`RagService` part-by-part. Aggregate
  counts come back so the SPA can render the "X added, Y already in
  library" line the TUI shows.

Cross-cutting design points:

* RAG strategy resolution reuses the chat router's
  :func:`_resolve_strategy` cache so the first library hit warms (or
  reads) the same singleton chat does — no double build cost, no
  duplicate qdrant clients.
* Validation failures land as 422 with the upload bundle's reason list
  embedded in ``LibraryIngestValidationError`` so the SPA can map each
  reason (``ocr_pending:scan.png``, ``total_too_short:42/100``, …) to
  the right inline chip warning without re-implementing the gate.
* Public ingest mode: this endpoint marks all parts ``public=True`` to
  match the TUI's behaviour. Users who explicitly type ``/upload`` are
  treated as having consented to share that content with cloud
  models — same posture as the TUI.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from claritymed.core.events import Done, Error, ToolCompleted
from claritymed.core.observability.audit import audit_event
from claritymed.core.rag.strategies.base import RetrievalContext
from claritymed.core.schemas import Account
from claritymed.core.upload.builder import build_upload_bundle
from claritymed.errors import DuplicateDocumentError
from claritymed.web.deps import get_current_user
from claritymed.web.schemas import (
    LibraryIngestPart,
    LibraryIngestRequest,
    LibraryIngestResponse,
    LibraryIngestValidationError,
    LibraryListResponse,
    LibrarySearchChunk,
    LibrarySearchRequest,
    LibrarySearchResponse,
    LibrarySearchTrace,
    LibrarySystemCollection,
    LibraryUserCollection,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["library"])

# Snippet length matches TUI's library modal so the same "first 200
# chars + ellipsis" the user sees in headless mode is what the SPA
# renders for the same query.
_SNIPPET_CHARS = 200
_USER_PREFIX = "user_rag_"


# --- routes -----------------------------------------------------------


@router.get("/library", response_model=LibraryListResponse)
async def list_library(
    request: Request,
    account: Account = Depends(get_current_user),
) -> LibraryListResponse:
    """Return every RAG collection ``HybridRetriever`` would search.

    Layout mirrors ``LibraryModal._render_list``: every entry under
    ``system_rag.collections`` in ``retrieval.yaml`` plus the user's
    own ``user_rag_<uid>`` collection. Chunk counts are best-effort —
    a missing collection or a Qdrant outage produces a ``null`` count
    rather than failing the whole list.

    Strategy cache: the call goes through the chat router's lazy build
    so the first request warms the same singleton chat uses; we read
    the result only to decide whether the system-collection count path
    should hit Qdrant or short-circuit to ``None`` (no server to ask).
    """
    strategy = await _resolve_strategy_via_chat_cache(request, account)
    rag_enabled = strategy is not None

    system_entries = _load_system_rag_collections()
    system_counts = await _system_collection_counts(
        [entry.get("name", "") for entry in system_entries if entry.get("name")],
        rag_enabled=rag_enabled,
    )
    user_count = await _count_user_rag_chunks(account.user_id)

    system_collections: list[LibrarySystemCollection] = []
    for entry in system_entries:
        name = entry.get("name")
        if not name:
            continue
        system_collections.append(
            LibrarySystemCollection(
                name=name,
                language=entry.get("language", "en"),
                authority_tier=int(entry.get("authority_tier", 3)),
                topics=list(entry.get("topics") or []),
                license=entry.get("license"),
                chunk_count=system_counts.get(name),
            )
        )

    return LibraryListResponse(
        system_collections=system_collections,
        user_collection=LibraryUserCollection(
            name=f"{_USER_PREFIX}{account.user_id}",
            chunk_count=user_count,
        ),
        rag_enabled=rag_enabled,
    )


@router.post("/library/search", response_model=LibrarySearchResponse)
async def search_library(
    req: LibrarySearchRequest,
    request: Request,
    account: Account = Depends(get_current_user),
) -> LibrarySearchResponse:
    """Run the live retrieval pipeline and return per-chunk hits + trace.

    503 when ``rag.enabled=false`` in ``retrieval.yaml`` — the cached
    strategy is ``None`` in that case and there is nothing to search.
    Same code path the chat router uses, so a chat-side RAG outage
    surfaces here too.
    """
    strategy = await _resolve_strategy_via_chat_cache(request, account)
    if strategy is None:
        raise HTTPException(status_code=503, detail="Retrieval disabled")

    ctx = RetrievalContext(
        query=req.q,
        user_id=account.user_id,
        language=account.language,  # type: ignore[arg-type]
    )
    bundle = await strategy.retrieve(ctx)

    chunks: list[LibrarySearchChunk] = []
    for chunk in bundle.chunks:
        collection_name = chunk.collection_name or ""
        # The reranker is the source of truth when it produced a score;
        # otherwise fall back to the dense/sparse fusion score so the
        # SPA always has a value to sort by. Mirrors the TUI modal.
        score = chunk.rerank_score if chunk.rerank_score is not None else chunk.score
        chunks.append(
            LibrarySearchChunk(
                tag="USER" if collection_name.startswith(_USER_PREFIX) else "SYS",
                collection_name=collection_name,
                doc_id=chunk.doc_id or "",
                score=float(score) if score is not None else None,
                snippet=_one_line(chunk.text, _SNIPPET_CHARS),
            )
        )

    trace = bundle.trace
    return LibrarySearchResponse(
        chunks=chunks,
        trace=LibrarySearchTrace(
            active_collections=list(trace.active_collections),
            expanded_query=trace.expanded_query,
            embed_ms=trace.embed_ms,
            search_ms=trace.search_ms,
            rerank_ms=trace.rerank_ms,
            parent_expand_ms=trace.parent_expand_ms,
        ),
    )


@router.post(
    "/library/ingest",
    response_model=LibraryIngestResponse,
    responses={422: {"model": LibraryIngestValidationError}},
)
async def ingest_library(
    req: LibraryIngestRequest,
    account: Account = Depends(get_current_user),
) -> LibraryIngestResponse:
    """Build → validate → ingest. Aggregate counts come back to the SPA.

    Validation: empty bundles, ``ocr_pending``/``ocr_failed`` parts,
    sub-floor part / total char counts all land as 422 with the bundle
    reasons embedded so the SPA can re-render the per-attachment chip
    states without polling.

    Ingest: each ``ok`` part runs through one ``RagService.run`` call
    keyed by ``source_uri=part.source_hash`` — that key drives the
    exact-hash dedupe the same way the TUI does, so a re-uploaded part
    raises ``DuplicateDocumentError`` and counts as ``skipped`` rather
    than re-embedded. Per-chunk cosine dedupe lives inside
    ``UserRagStore`` and is invisible at this layer.
    """
    bundle = build_upload_bundle(
        req.text,
        user_id=account.user_id,
        session_id=req.session_id,
    )
    validation = bundle.validate()
    if not validation.ok:
        # Embed the bundle's reason list so the SPA can map per-attachment
        # statuses without polling /attachments/meta a second time.
        raise HTTPException(
            status_code=422,
            detail={
                "detail": "Upload bundle invalid",
                "reasons": list(validation.reasons),
            },
        )

    # Lazy imports — these touch qdrant / embedder construction; we
    # don't want a library list / search request to pay that cost.
    from claritymed.orchestrator.services import RagService
    from claritymed.stores.user_rag import make_user_rag_store

    try:
        store = make_user_rag_store(account.user_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("library ingest: user_rag store build failed")
        raise HTTPException(
            status_code=503, detail=f"User RAG unavailable: {exc}"
        ) from exc

    added_parts = 0
    skipped_parts = 0
    failed_parts = 0
    added_chunks = 0
    per_part: list[LibraryIngestPart] = []

    for part in bundle.parts:
        if part.status != "ok":
            # Validator should have blocked these — defensive only.
            failed_parts += 1
            per_part.append(
                LibraryIngestPart(
                    source=part.source,
                    kind=part.kind,
                    status="failed",
                    error=f"part status={part.status}",
                )
            )
            continue
        service = RagService(store=store)
        try:
            outcome = await _run_one_part(
                service,
                user_id=account.user_id,
                content=part.content,
                source_hash=part.source_hash,
                language=account.language,
                public=req.public,
            )
        except DuplicateDocumentError:
            skipped_parts += 1
            per_part.append(
                LibraryIngestPart(
                    source=part.source,
                    kind=part.kind,
                    status="skipped",
                )
            )
            continue
        except Exception as exc:  # noqa: BLE001
            logger.exception("library ingest: part %s failed", part.source)
            failed_parts += 1
            per_part.append(
                LibraryIngestPart(
                    source=part.source,
                    kind=part.kind,
                    status="failed",
                    error=str(exc)[:200],
                )
            )
            continue

        if outcome["status"] == "skipped":
            skipped_parts += 1
        elif outcome["status"] == "failed":
            failed_parts += 1
        else:
            added_parts += 1
            added_chunks += outcome["chunks"]
        per_part.append(
            LibraryIngestPart(
                source=part.source,
                kind=part.kind,
                status=outcome["status"],
                chunks=outcome["chunks"],
                error=outcome.get("error"),
            )
        )

    audit_event(
        "web.library.ingest",
        payload={
            "user_id": account.user_id,
            "added_parts": added_parts,
            "skipped_parts": skipped_parts,
            "failed_parts": failed_parts,
            "added_chunks": added_chunks,
        },
    )

    return LibraryIngestResponse(
        added_parts=added_parts,
        skipped_parts=skipped_parts,
        failed_parts=failed_parts,
        added_chunks=added_chunks,
        parts=per_part,
    )


# --- helpers ----------------------------------------------------------


async def _run_one_part(
    service: Any,
    *,
    user_id: str,
    content: str,
    source_hash: str,
    language: str,
    public: bool,
) -> dict:
    """Drain one ``RagService.run`` stream and project to an outcome dict.

    Mirrors the TUI's ``_run_rag_upload`` accounting: a Done event with
    ``chunk_count == 0 and skipped_chunk_count > 0`` means every chunk
    cosine-dedup'd against existing content — semantically the same as
    "already in library", so it counts as ``skipped`` not ``added``.

    ``public`` propagates the per-bundle consent flag. ``False`` (the
    default at the request schema) runs PHI scrub at ingest and marks
    chunks ``can_cloud=False``; ``True`` skips the ingest scrub and
    marks them ``can_cloud=True``. Retrieval still runs a regex layer
    over surviving chunks for cloud-bound prompts as defense-in-depth.
    """
    events = service.run(
        content,
        user_id=user_id,
        public=public,
        language=language,
        source_uri=source_hash,
    )
    async for event in events:
        if isinstance(event, ToolCompleted):
            # Carries the embed timing but no terminal status — wait
            # for Done / Error before deciding the outcome.
            continue
        if isinstance(event, Error):
            return {"status": "failed", "chunks": 0, "error": event.message}
        if isinstance(event, Done):
            final = event.final
            if final.chunk_count == 0 and final.skipped_chunk_count > 0:
                return {"status": "skipped", "chunks": 0}
            return {"status": "added", "chunks": final.chunk_count}
    return {"status": "failed", "chunks": 0, "error": "no terminal event"}


def _load_system_rag_collections() -> list[dict]:
    """Read ``configs/retrieval.yaml`` for the system_rag collections list."""
    from claritymed import config as _cfg

    raw = _cfg.load_yaml("retrieval.yaml")
    return list((raw.get("system_rag") or {}).get("collections") or [])


async def _resolve_strategy_via_chat_cache(request: Request, account: Account):
    """Reuse the chat router's lazy strategy build so the cache stays single.

    Defers to ``claritymed.web.routers.chat._resolve_strategy`` so the
    library and chat surfaces share one ``RagStrategy`` — building a
    second one would double the qdrant client count and risk file-lock
    contention against the chat path. ``account.provider_id`` falls
    back to the catalog default the same way the chat router does on a
    cache miss.
    """
    from claritymed.web.routers.chat import _resolve_strategy

    def _model_for_strategy():
        from claritymed.core.llm.model import build_model
        from claritymed.stores.models import resolve_provider

        provider = resolve_provider(account=account, override=None)
        return build_model(provider)

    return await _resolve_strategy(request.app.state, _model_for_strategy)


async def _system_collection_counts(
    names: list[str], *, rag_enabled: bool
) -> dict[str, int | None]:
    """Mirror ``LibraryModal._system_collection_counts``.

    Best-effort point count per system collection via the shared
    Qdrant client. ``None`` when the server has no such collection or
    the count call raises — the row renders as ``?`` in the SPA, same
    as in the TUI modal. When RAG is disabled there is no server to
    ask, so every entry comes back ``None``.
    """
    result: dict[str, int | None] = {n: None for n in names}
    if not names or not rag_enabled:
        return result
    try:
        from claritymed.core.rag import load_retrieval_config
        from claritymed.core.rag.qdrant_store import build_qdrant_client
    except Exception:  # noqa: BLE001
        logger.exception("library list: qdrant imports failed")
        return result

    cfg = load_retrieval_config()
    try:
        aclient = build_qdrant_client(
            url=cfg.qdrant.url,
            api_key_env=cfg.qdrant.api_key_env,
        )
    except Exception:  # noqa: BLE001
        logger.exception("library list: qdrant client build failed")
        return result

    try:
        for name in names:
            try:
                if not await aclient.collection_exists(name):
                    continue
                info = await aclient.count(name, exact=True)
                result[name] = int(info.count)
            except Exception:  # noqa: BLE001
                logger.exception("library list: %s count failed", name)
    finally:
        try:
            await aclient.close()
        except Exception:  # noqa: BLE001
            logger.exception("library list: qdrant client close failed")
    return result


async def _count_user_rag_chunks(user_id: str) -> int | None:
    """Sum of ``chunk_count`` across the user's RAG docs. ``None`` on init."""
    try:
        from claritymed.stores.user_rag import make_user_rag_store

        store = make_user_rag_store(user_id)
        docs = await store.list_documents(user_id)
    except Exception:  # noqa: BLE001
        logger.exception("library list: user_rag count failed for %s", user_id)
        return None
    if not docs:
        return 0
    return sum(int(d.get("chunk_count", 0)) for d in docs)


def _one_line(text: str | None, limit: int) -> str:
    """Collapse newlines + truncate. Same shape the TUI modal uses."""
    if not text:
        return ""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "…"
