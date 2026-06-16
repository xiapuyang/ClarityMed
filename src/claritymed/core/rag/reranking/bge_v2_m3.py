"""HTTP client for a bge-reranker-v2-m3 server (TEI ``/rerank`` shape).

Wire format (TEI ``/rerank``)::

    POST /rerank
    {
      "query": "...",
      "texts": ["d1", "d2", ...],
      "raw_scores": false,
      "return_text": false
    }

    -> [{"index": 1, "score": 0.93}, {"index": 0, "score": 0.21}, ...]

Same fail-loud contract as the embedder: 4xx/5xx, timeout, or invalid
shape → ``RerankerUnreachableError``. ``HybridRetriever`` treats this as
fail-soft (returns un-reranked hits + audit warning) — the soft-fail
boundary lives in the caller, not here.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import httpx

from claritymed.context import request_id_ctx
from claritymed.core.rag.reranking.base import RerankHit, Reranker
from claritymed.errors import MissingApiKeyError, RerankerUnreachableError

if TYPE_CHECKING:
    from claritymed.core.phi.outbound_gate import OutboundTextGate

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 30
DEFAULT_BATCH_SIZE = 32


class BgeRerankerV2M3HttpReranker(Reranker):
    """Async HTTP client for a TEI-compatible bge-reranker-v2-m3 server."""

    def __init__(
        self,
        base_url: str,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        api_key_env: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        scrub_gate: "OutboundTextGate | None" = None,
    ) -> None:
        if not base_url:
            raise ValueError("BgeRerankerV2M3HttpReranker requires base_url")
        self._base_url = base_url.rstrip("/")
        self._batch_size = batch_size
        self._timeout_s = timeout_s
        self._api_key = self._resolve_api_key(api_key_env)
        self._transport = transport
        self._scrub_gate = scrub_gate
        # Shared AsyncClient — lazy on first use, reused across all
        # batches and turns. Building a new client per batch was paying
        # TCP/TLS setup for every retrieval round trip.
        self._async_client: httpx.AsyncClient | None = None

    @staticmethod
    def _resolve_api_key(api_key_env: str | None) -> str | None:
        if not api_key_env:
            return None
        value = os.environ.get(api_key_env)
        if not value:
            raise MissingApiKeyError(
                f"BgeRerankerV2M3HttpReranker needs env var {api_key_env!r} but it is unset"
            )
        return value

    async def rerank(
        self,
        query: str,
        docs: list[str],
        top_k: int,
    ) -> list[RerankHit]:
        if not docs:
            return []
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if self._scrub_gate:
            query = self._scrub_gate.scrub(query)
            docs = self._scrub_gate.scrub_batch(docs)

        all_hits: list[RerankHit] = []
        # Rerank batches independently then merge — TEI accepts an unbounded
        # `texts` array, but bounding the batch keeps server memory predictable.
        for offset, batch in self._batched(docs):
            batch_hits = await self._post_batch(query, batch)
            # Translate batch-local indices back to caller-space indices.
            for hit in batch_hits:
                all_hits.append(RerankHit(index=hit.index + offset, score=hit.score))

        all_hits.sort(key=lambda h: h.score, reverse=True)
        return all_hits[:top_k]

    # --- internals ------------------------------------------------------

    def _batched(self, docs: list[str]) -> list[tuple[int, list[str]]]:
        size = self._batch_size
        return [(i, docs[i : i + size]) for i in range(0, len(docs), size)]

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
        return headers

    def _client(self) -> httpx.AsyncClient:
        if self._async_client is None:

            async def _inject_request_id(request: httpx.Request) -> None:
                rid = request_id_ctx.get()
                if rid:
                    request.headers["X-Request-ID"] = rid

            kwargs: dict[str, Any] = {
                "base_url": self._base_url,
                "timeout": self._timeout_s,
                "headers": self._headers(),
                "event_hooks": {"request": [_inject_request_id]},
            }
            if self._transport is not None:
                kwargs["transport"] = self._transport
            self._async_client = httpx.AsyncClient(**kwargs)
        return self._async_client

    async def aclose(self) -> None:
        """Close the shared AsyncClient. Safe to call multiple times."""
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    async def _post_batch(self, query: str, batch: list[str]) -> list[RerankHit]:
        payload = {"query": query, "texts": batch, "raw_scores": False}
        client = self._client()
        try:
            resp = await client.post("/rerank", json=payload)
        except httpx.HTTPError as exc:
            raise RerankerUnreachableError(
                f"bge-reranker-v2-m3 rerank failed at {self._base_url}: {exc}"
            ) from exc
        if resp.status_code >= 400:
            raise RerankerUnreachableError(
                f"bge-reranker-v2-m3 returned {resp.status_code} "
                f"at {self._base_url}: {resp.text[:200]}"
            )
        data = resp.json()
        return self._parse(data, expected_count=len(batch))

    @staticmethod
    def _parse(data: Any, *, expected_count: int) -> list[RerankHit]:
        if not isinstance(data, list):
            raise RerankerUnreachableError(
                f"bge-reranker-v2-m3 response must be a list, got {type(data).__name__}"
            )
        hits: list[RerankHit] = []
        for entry in data:
            if (
                not isinstance(entry, dict)
                or "index" not in entry
                or "score" not in entry
            ):
                raise RerankerUnreachableError(
                    f"bge-reranker-v2-m3 entry shape invalid: {entry!r}"
                )
            idx = int(entry["index"])
            if not 0 <= idx < expected_count:
                raise RerankerUnreachableError(
                    f"bge-reranker-v2-m3 returned index {idx} out of range "
                    f"[0, {expected_count})"
                )
            hits.append(RerankHit(index=idx, score=float(entry["score"])))
        return hits
