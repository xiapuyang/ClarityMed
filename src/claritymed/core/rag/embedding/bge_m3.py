"""HTTP client for a BGE-M3 embedding server (e.g. HuggingFace TEI).

Two endpoints are used:

* ``POST {base_url}/embed`` with ``{"inputs": [...]}`` → ``list[list[float]]``
  for dense vectors.
* ``POST {base_url}/embed_sparse`` with ``{"inputs": [...]}`` → a list of
  sparse maps. Two shapes are accepted (the TEI v1.x and FlagEmbedding-server
  outputs differ slightly):

  - ``[{"<token_id>": weight, ...}, ...]`` — flat dict per input
  - ``[{"indices": [...], "values": [...]}, ...]`` — explicit arrays

This module is the only place that knows the BGE-M3 wire format. Tests
intercept HTTP via ``httpx.MockTransport``; nothing in the protocol layer
imports ``httpx``.

Fail-loud contract:

* Any non-2xx response, any timeout, any connection error → raises
  ``EmbedderUnreachableError``. We never silently fall back to a CPU
  embedder, because a 384-dim CPU embedding silently written into a
  1024-dim Qdrant collection produces a worse bug than a hard failure.
* ``api_key_env`` declared but env var unset → raises ``MissingApiKeyError``
  the same way custom Ollama endpoints do.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

import httpx

from claritymed.context import request_id_ctx
from claritymed.core.rag.embedding.base import Embedder, SparseVector
from claritymed.errors import EmbedderUnreachableError, MissingApiKeyError

if TYPE_CHECKING:
    from claritymed.core.phi.outbound_gate import OutboundTextGate

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 30
DEFAULT_BATCH_SIZE = 32


class BgeM3HttpEmbedder(Embedder):
    """Async HTTP client for a BGE-M3 dense+sparse server."""

    def __init__(
        self,
        base_url: str,
        dense_dim: int,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        timeout_s: int = DEFAULT_TIMEOUT_S,
        api_key_env: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        scrub_gate: "OutboundTextGate | None" = None,
    ) -> None:
        if not base_url:
            raise ValueError("BgeM3HttpEmbedder requires base_url")
        self._base_url = base_url.rstrip("/")
        self._dense_dim = dense_dim
        self._batch_size = batch_size
        self._timeout_s = timeout_s
        self._api_key = self._resolve_api_key(api_key_env)
        self._transport = transport  # tests inject MockTransport here
        self._scrub_gate = scrub_gate
        # Lazy-initialised on first request and reused for the embedder's
        # lifetime. Building a new AsyncClient per call (the prior
        # behaviour) paid TCP+TLS setup on every retrieval — 3-6 client
        # lifecycles per RAG turn for nothing.
        self._async_client: httpx.AsyncClient | None = None

    @staticmethod
    def _resolve_api_key(api_key_env: str | None) -> str | None:
        if not api_key_env:
            return None
        value = os.environ.get(api_key_env)
        if not value:
            raise MissingApiKeyError(
                f"BgeM3HttpEmbedder needs env var {api_key_env!r} but it is unset"
            )
        return value

    @property
    def dimension(self) -> int:
        return self._dense_dim

    # --- public API -----------------------------------------------------

    async def embed_dense(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._scrub_gate:
            texts = self._scrub_gate.scrub_batch(texts)
        out: list[list[float]] = []
        for batch in self._batched(texts):
            out.extend(await self._post_dense(batch))
        return out

    async def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        if not texts:
            return []
        if self._scrub_gate:
            texts = self._scrub_gate.scrub_batch(texts)
        out: list[SparseVector] = []
        for batch in self._batched(texts):
            out.extend(await self._post_sparse(batch))
        return out

    # --- internals ------------------------------------------------------

    def _batched(self, texts: list[str]) -> list[list[str]]:
        size = self._batch_size
        return [texts[i : i + size] for i in range(0, len(texts), size)]

    def _headers(self) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
        return headers

    def _client(self) -> httpx.AsyncClient:
        """Return the shared AsyncClient, building it on first use.

        Tests that inject a fresh ``transport`` per call (rare) can
        clear ``self._async_client`` to force rebuild; the production
        path reuses one client for the embedder's lifetime so connection
        pooling actually works.
        """
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
                "trust_env": False,
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

    async def _post_dense(self, batch: list[str]) -> list[list[float]]:
        client = self._client()
        try:
            resp = await client.post("/embed", json={"inputs": batch})
        except httpx.HTTPError as exc:
            raise EmbedderUnreachableError(
                f"bge-m3 dense embed failed at {self._base_url}: {exc}"
            ) from exc
        if resp.status_code >= 400:
            raise EmbedderUnreachableError(
                f"bge-m3 dense embed returned {resp.status_code} "
                f"at {self._base_url}: {resp.text[:200]}"
            )
        data = resp.json()
        return self._parse_dense(data, expected_count=len(batch))

    async def _post_sparse(self, batch: list[str]) -> list[SparseVector]:
        client = self._client()
        try:
            resp = await client.post("/embed_sparse", json={"inputs": batch})
        except httpx.HTTPError as exc:
            raise EmbedderUnreachableError(
                f"bge-m3 sparse embed failed at {self._base_url}: {exc}"
            ) from exc
        if resp.status_code >= 400:
            raise EmbedderUnreachableError(
                f"bge-m3 sparse embed returned {resp.status_code} "
                f"at {self._base_url}: {resp.text[:200]}"
            )
        data = resp.json()
        return self._parse_sparse(data, expected_count=len(batch))

    def _parse_dense(self, data: Any, *, expected_count: int) -> list[list[float]]:
        if not isinstance(data, list) or len(data) != expected_count:
            raise EmbedderUnreachableError(
                f"bge-m3 dense response shape invalid: expected list of "
                f"{expected_count} vectors, got {type(data).__name__}"
            )
        for vec in data:
            if not isinstance(vec, list) or len(vec) != self._dense_dim:
                raise EmbedderUnreachableError(
                    f"bge-m3 dense vector dim mismatch: expected {self._dense_dim}, "
                    f"got {len(vec) if isinstance(vec, list) else type(vec).__name__}"
                )
        return data

    def _parse_sparse(self, data: Any, *, expected_count: int) -> list[SparseVector]:
        if not isinstance(data, list) or len(data) != expected_count:
            raise EmbedderUnreachableError(
                f"bge-m3 sparse response shape invalid: expected list of "
                f"{expected_count} entries, got {type(data).__name__}"
            )
        return [self._coerce_sparse(entry) for entry in data]

    @staticmethod
    def _coerce_sparse(entry: Any) -> SparseVector:
        # Shape A: explicit indices/values arrays (qdrant-style).
        if isinstance(entry, dict) and "indices" in entry and "values" in entry:
            idx = entry["indices"]
            val = entry["values"]
            if (
                not isinstance(idx, list)
                or not isinstance(val, list)
                or len(idx) != len(val)
            ):
                raise EmbedderUnreachableError(
                    f"bge-m3 sparse entry indices/values shape invalid: {entry!r}"
                )
            return {int(i): float(v) for i, v in zip(idx, val)}
        # Shape B: flat dict of token_id → weight.
        if isinstance(entry, dict):
            return {int(k): float(v) for k, v in entry.items()}
        raise EmbedderUnreachableError(
            f"bge-m3 sparse entry must be dict, got {type(entry).__name__}"
        )
