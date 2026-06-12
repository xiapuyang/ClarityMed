"""Pre-invoke plugin that expands attachment placeholders inside user input.

When the user pastes an image/file the TUI inserts a short-form placeholder
``[Image sha:XXXXXXXX]`` / ``[File sha:XXXXXXXX]`` (8-char prefix) at the
cursor. At prompt-assembly time this feature substitutes each placeholder
the user *kept* in their text with an inline XML tag carrying the full
sha and the OCR'd content::

    [Image sha:dce847ff]  →  <image sha="<64-char sha>">
                              OCR text...
                              </image>

The full sha matters because downstream ``save_record`` resolves
attachments via ``BlobStore.dir(sha)`` which expects the 64-char digest.
Rendering the full sha inline keeps the LLM-visible reference identical
to the one tools need, avoiding a prefix → full reverse lookup at tool
call time.

Attachments not referenced in this turn's user input are skipped
entirely — they remain in the SessionAttachments index but don't appear
in the LLM prompt. This makes "delete the placeholder" a meaningful UI
gesture: the user can paste an image, change their mind, delete the
placeholder, and the OCR'd content stays out of this turn's prompt.

Status branches render as self-closing tags with ``ocr_status`` as an
attribute, matching the vocabulary already used in
``tool_proposal.yaml`` so the model can branch on status without parsing
free text.
"""

from __future__ import annotations

import logging
import re
from typing import Callable

from claritymed.core.features.base import FeaturePlugin, TurnContext
from claritymed.orchestrator.services.session_attachments import SessionAttachments
from claritymed.stores.blob_store import BlobStore

logger = logging.getLogger(__name__)

# Mirror of the placeholder syntax ``ClarityMedApp._ingest_clipboard_bytes``
# writes into the input bar. Matching both kinds in one pass lets a user
# mix images and files in the same input without the regex caring.
_PLACEHOLDER_RE = re.compile(r"\[(Image|File) sha:([0-9a-f]+)\]")


class AttachmentsFeature(FeaturePlugin):
    """Pre-invoke plugin that inlines OCR'd attachment content at the
    placeholder positions in the user's input.

    Constructed with a ``get_session_id`` callable so the plugin can
    discover the active session_id at turn time without holding a
    reference to ``ChatSession`` (which the orchestrator owns).
    Returning ``None`` from the callable disables the feature for that
    turn — headless test paths use this to skip without a session on
    disk.
    """

    name = "attachments"
    mode = "deterministic"

    def __init__(self, get_session_id: Callable[[], str | None]) -> None:
        self._get_session_id = get_session_id

    async def pre_invoke(self, ctx: TurnContext) -> str:
        """No standalone prompt block.

        Attachments are inlined at the placeholder positions in
        ``ctx.scrubbed`` by the orchestrator's prompt-assembly step
        (see ``AskService._stream_turn``). A separate header / block
        would double-print the OCR text and bloat the prompt.
        """
        return ""

    async def expand_placeholders(self, text: str, ctx: TurnContext) -> str:
        """Substitute every ``[Image sha:XXXXXXXX]`` /
        ``[File sha:XXXXXXXX]`` placeholder in ``text`` with an inline
        ``<image sha="...">OCR</image>`` / ``<file sha="...">OCR</file>``
        tag.

        Placeholders whose 8-char prefix doesn't resolve to a unique
        session attachment are left unchanged — the model sees the raw
        form and can ask the user to disambiguate. An 8-char SHA-256
        prefix collision inside a single session is astronomically
        unlikely but explicitly handled so a freak collision doesn't
        silently rewrite the wrong attachment into the prompt.

        Returns ``text`` unchanged when there is no session, no
        placeholders, or session attachments can't be read.
        """
        sid = self._get_session_id()
        if sid is None or not text:
            return text
        if not _PLACEHOLDER_RE.search(text):
            return text

        user_id = ctx.deps.user_id
        try:
            rows = SessionAttachments(user_id, sid).list()
        except Exception:  # noqa: BLE001
            logger.exception("failed to list session attachments for inline expansion")
            return text
        if not rows:
            return text

        prefix_index: dict[str, list] = {}
        for row in rows:
            prefix_index.setdefault(row.sha256[:8], []).append(row)

        blob_store = BlobStore(user_id)

        def _replace(match: re.Match) -> str:
            kind = match.group(1).lower()  # "image" / "file"
            prefix = match.group(2)
            hits = prefix_index.get(prefix, [])
            if len(hits) != 1:
                return match.group(0)
            return _render_inline_tag(blob_store, kind, hits[0])

        return _PLACEHOLDER_RE.sub(_replace, text)

    def as_tool(self):
        return None


def _render_inline_tag(blob_store: BlobStore, kind: str, att) -> str:
    """One ``<image>`` / ``<file>`` tag for one session attachment.

    Done status with non-empty OCR text → element form with the OCR
    text between opening and closing tags. Every other status →
    self-closing tag with the status as an attribute so the model can
    branch on it without parsing prose. The ``ocr_status`` attribute
    name and values mirror what ``tool_proposal.yaml`` already
    references (``pending`` / ``failed`` / ``empty``) plus an explicit
    ``missing`` value for the "sentinel says done but file is gone"
    corruption case.
    """
    tag = "image" if kind == "image" else "file"
    sha = att.sha256
    if att.ocr_status == "done":
        try:
            ocr = blob_store.read_extracted_text(sha).strip()
        except (OSError, ValueError):
            # OSError: ocr.md / content.<ext> disappeared between
            # sentinel write and prompt assembly. ValueError: sentinel
            # JSON corrupt. Both mean the rendered tag can't carry the
            # text — fall through to the "missing" attribute so the
            # model branches on it instead of seeing a torn payload.
            return f'<{tag} sha="{sha}" ocr_status="missing"/>'
        if not ocr:
            return f'<{tag} sha="{sha}" ocr_status="empty"/>'
        return f'<{tag} sha="{sha}">\n{ocr}\n</{tag}>'
    if att.ocr_status == "pending":
        return f'<{tag} sha="{sha}" ocr_status="pending"/>'
    if att.ocr_status == "failed":
        # Reason text can contain double quotes (e.g. an upstream
        # provider's error message); escape so the tag stays parseable.
        reason = (att.ocr_reason or "unknown").replace('"', "&quot;")
        return f'<{tag} sha="{sha}" ocr_status="failed" reason="{reason}"/>'
    if att.ocr_status == "empty":
        return f'<{tag} sha="{sha}" ocr_status="empty"/>'
    # Unknown / future status — surface verbatim rather than silently
    # dropping so a status-vocab drift fails loudly in the prompt.
    return f'<{tag} sha="{sha}" ocr_status="{att.ocr_status}"/>'
