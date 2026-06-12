"""MineRU API-based OCR provider.

Uploads the local file to MineRU's standard API (requires a token), polls
until extraction completes, downloads the result ZIP, and returns the
Markdown text from full.md.

API flow:
  1. POST /api/v4/file-urls/batch  → batch_id + presigned file_urls
  2. PUT  <file_url>               → upload raw bytes (no Content-Type)
  3. GET  /api/v4/extract-results/batch/{batch_id}  → poll until done
  4. GET  <full_zip_url>           → download ZIP, extract full.md
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
import uuid
import zipfile
from pathlib import Path

import httpx

from claritymed.core.ocr.base import ExtractResult, OcrError, OcrProvider

logger = logging.getLogger(__name__)

_BASE_URL = "https://mineru.net"
_BATCH_SUBMIT_PATH = "/api/v4/file-urls/batch"
_BATCH_RESULT_PATH = "/api/v4/extract-results/batch/{batch_id}"


class MineRUOcrProvider(OcrProvider):
    """OCR via MineRU API — uploads file, polls, returns Markdown text.

    Supports all MineRU-accepted file types: PDF, Doc/Docx, PPT/PPTx,
    Xls/Xlsx, and images (PNG, JPG, JPEG, WebP, GIF, BMP).

    MineRU is a cloud SaaS (mineru.net); ``is_local = False`` is the
    structural defense that keeps it out of any chain composed with
    ``phi_policy="local-only"``. Constructing it ALSO requires explicit
    env opt-in (``CLARITYMED_ALLOW_MINERU=1``) so a stale ``ocr.yaml``
    listing ``mineru`` in a chain fails fast at startup with a clear
    error rather than silently making a cloud hop mid-extract.
    """

    is_local = False
    """Cloud SaaS — never PHI-safe; chain composer filters it out."""

    label = "mineru"
    # MineRU's API accepts PDF / office / image formats. We leave
    # ``supported_extensions`` as ``None`` (= all) so it acts as the
    # catch-all fallback at the end of a chain.

    def __init__(
        self,
        api_key: str,
        *,
        model_version: str = "vlm",
        poll_interval: float = 3.0,
        poll_timeout: float = 300.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # Env gate. PHI paths are already protected by ``is_local = False``
        # at chain-composition time; this second check makes sure that
        # nothing — including a developer's stray `MineRUOcrProvider(...)`
        # call from a notebook — constructs the provider without a clear
        # acknowledgement.
        import os

        if os.environ.get("CLARITYMED_ALLOW_MINERU") != "1":
            from claritymed.errors import MinerUNotAllowed

            raise MinerUNotAllowed(
                "MinerU is a cloud SaaS (mineru.net); set "
                "CLARITYMED_ALLOW_MINERU=1 to acknowledge this and enable "
                "testing. PHI paths always remain blocked via "
                "is_local=False."
            )
        self._api_key = api_key
        self._model_version = model_version
        self._poll_interval = poll_interval
        self._poll_timeout = poll_timeout
        self._transport = transport

    def _client(self) -> httpx.AsyncClient:
        kwargs: dict = {"timeout": 60.0}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.AsyncClient(**kwargs)

    async def extract_text(self, path: Path) -> ExtractResult:
        """Upload *path* to MineRU, wait for extraction, return Markdown.

        Raises:
            OcrError: On HTTP errors, API errors, or extraction failure.
        """
        if not path.exists():
            raise OcrError(f"Cannot read {path}: file not found")

        async with self._client() as client:
            batch_id, file_url = await self._submit(client, path)
            await self._upload(client, file_url, path)
            zip_url = await self._poll(client, batch_id, path.name)
            text = await self._download_markdown(client, zip_url)
        return ExtractResult(
            text=text, provider_used=self.label, chain_tried=[self.label]
        )

    async def _submit(self, client: httpx.AsyncClient, path: Path) -> tuple[str, str]:
        """Request a presigned upload URL; return (batch_id, file_url)."""
        data_id = uuid.uuid4().hex
        payload = {
            "files": [{"name": path.name, "data_id": data_id}],
            "model_version": self._model_version,
        }
        try:
            resp = await client.post(
                f"{_BASE_URL}{_BATCH_SUBMIT_PATH}",
                json=payload,
                headers=self._auth_headers(),
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OcrError(f"MineRU submit failed for {path.name}: {exc}") from exc

        body = resp.json()
        if body.get("code") != 0:
            raise OcrError(
                f"MineRU submit error for {path.name}: {body.get('msg', 'unknown')}"
            )

        data = body["data"]
        file_urls: list[str] = data.get("file_urls", [])
        if not file_urls:
            raise OcrError(f"MineRU returned no file_urls for {path.name}")

        batch_id: str = data["batch_id"]
        logger.debug("mineru: batch_id=%s for %s", batch_id, path.name)
        return batch_id, file_urls[0]

    async def _upload(
        self, client: httpx.AsyncClient, file_url: str, path: Path
    ) -> None:
        """PUT raw file bytes to the presigned URL."""
        data = path.read_bytes()
        try:
            resp = await client.put(file_url, content=data)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OcrError(f"MineRU upload failed for {path.name}: {exc}") from exc
        logger.debug("mineru: uploaded %d bytes for %s", len(data), path.name)

    async def _poll(
        self, client: httpx.AsyncClient, batch_id: str, file_name: str
    ) -> str:
        """Poll until extraction completes; return full_zip_url."""
        url = f"{_BASE_URL}{_BATCH_RESULT_PATH.format(batch_id=batch_id)}"
        deadline = time.monotonic() + self._poll_timeout

        while True:
            if time.monotonic() >= deadline:
                raise OcrError(
                    f"MineRU polling timed out after {self._poll_timeout}s "
                    f"(batch_id={batch_id})"
                )
            try:
                resp = await client.get(url, headers=self._auth_headers())
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                raise OcrError(f"MineRU poll failed for {file_name}: {exc}") from exc

            body = resp.json()
            if body.get("code") != 0:
                raise OcrError(
                    f"MineRU poll error for {file_name}: {body.get('msg', 'unknown')}"
                )

            results: list[dict] = body["data"].get("extract_result", [])
            if not results:
                await asyncio.sleep(self._poll_interval)
                continue

            entry = results[0]
            state = entry.get("state", "")

            if state == "done":
                zip_url = entry.get("full_zip_url", "")
                if not zip_url:
                    raise OcrError(f"MineRU done but no full_zip_url for {file_name}")
                logger.debug("mineru: extraction done for %s", file_name)
                return zip_url

            if state == "failed":
                raise OcrError(
                    f"MineRU extraction failed for {file_name}: "
                    f"{entry.get('err_msg', 'unknown error')}"
                )

            logger.debug("mineru: state=%s for %s, waiting...", state, file_name)
            await asyncio.sleep(self._poll_interval)

    async def _download_markdown(self, client: httpx.AsyncClient, zip_url: str) -> str:
        """Download the result ZIP and extract full.md."""
        try:
            resp = await client.get(zip_url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise OcrError(f"MineRU ZIP download failed: {exc}") from exc

        try:
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                md_name = next(
                    (n for n in zf.namelist() if n.endswith("full.md")), None
                )
                if md_name is None:
                    raise OcrError(f"MineRU ZIP has no full.md; found: {zf.namelist()}")
                text = zf.read(md_name).decode("utf-8").strip()
        except zipfile.BadZipFile as exc:
            raise OcrError(f"MineRU returned invalid ZIP: {exc}") from exc

        logger.debug("mineru: extracted %d chars from full.md", len(text))
        return text

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}
