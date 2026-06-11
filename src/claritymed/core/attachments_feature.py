"""Deterministic feature plugin that injects OCR'd attachments into the prompt.

When the user pastes an image or uploads a file, ``SessionAttachments``
tracks it and ``OcrWorker`` extracts text into ``ocr.md`` under the blob
directory. Without this plugin the ask-agent would never see that text —
``ChatSession`` only stores the user's typed message, not its attachments.

The plugin runs in ``deterministic`` mode so the OCR'd text is splice-in
prompt context, not a tool the LLM has to discover; missing or in-flight
OCR shows up as a one-line status so the model can decide whether to
defer the answer.
"""

from __future__ import annotations

import logging
from typing import Callable

from claritymed.core.features.base import FeaturePlugin, TurnContext
from claritymed.orchestrator.services.session_attachments import SessionAttachments
from claritymed.stores.blob_store import BlobStore

logger = logging.getLogger(__name__)

_HEADER_EN = "Attachments (extracted by OCR):"
_HEADER_ZH = "附件（OCR 提取的内容）："
_SEP = "\n\n---\n\n"


class AttachmentsFeature(FeaturePlugin):
    """Pre-invoke plugin that renders the session's attachments + OCR text.

    Construction takes a ``get_session_id`` callable so the plugin can
    discover the active session_id at turn time without holding a
    reference to ChatSession (which the orchestrator owns). Returning
    ``None`` from the callable disables the feature for that turn —
    headless test paths use this to skip without a session on disk.
    """

    name = "attachments"
    mode = "deterministic"

    def __init__(self, get_session_id: Callable[[], str | None]) -> None:
        self._get_session_id = get_session_id

    async def pre_invoke(self, ctx: TurnContext) -> str:
        sid = self._get_session_id()
        if sid is None:
            return ""
        user_id = ctx.deps.user_id
        try:
            rows = SessionAttachments(user_id, sid).list()
        except Exception:  # noqa: BLE001
            logger.exception("failed to list session attachments")
            return ""
        if not rows:
            return ""

        blob_store = BlobStore(user_id)
        blocks: list[str] = []
        for att in rows:
            block = _render_one(blob_store, att)
            if block:
                blocks.append(block)
        if not blocks:
            return ""
        header = _HEADER_ZH if ctx.deps.language == "zh" else _HEADER_EN
        return f"{header}\n\n" + _SEP.join(blocks)

    def as_tool(self):
        return None


def _render_one(blob_store: BlobStore, att) -> str:
    """One ``Attachments:`` block per session attachment.

    Status branches are explicit so the model sees ``OCR in progress`` vs
    ``OCR failed: <reason>`` differently — both meaningfully change what
    the model should do next (defer the answer vs ask the user to retry).
    """
    sha_short = att.sha256[:8]
    label = f"- {att.filename} (sha {sha_short})"
    if att.ocr_status == "done":
        try:
            text = blob_store.ocr_path(att.sha256).read_text(encoding="utf-8").strip()
        except OSError:
            return f"{label} — OCR text missing on disk"
        if not text:
            return f"{label} — OCR returned empty text"
        return f"{label}\n\n{text}"
    if att.ocr_status == "pending":
        return f"{label} — OCR in progress"
    if att.ocr_status == "failed":
        reason = att.ocr_reason or "unknown error"
        return f"{label} — OCR failed: {reason}"
    if att.ocr_status == "empty":
        return f"{label} — OCR returned no text"
    return f"{label} — OCR status: {att.ocr_status}"
