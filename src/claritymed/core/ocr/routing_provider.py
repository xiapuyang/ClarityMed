"""``RoutingOcrProvider`` — chain-based OCR composer.

Routes ``extract_text`` to a list of providers per file kind. The first
provider that returns text wins; ``OcrError`` from one falls through to
the next.

``phi_policy="local-only"`` filters cloud providers (``is_local=False``)
out of the chain at composition time — defense-in-depth alongside any
per-provider env gate (e.g. MineRU's ``CLARITYMED_ALLOW_MINERU``).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Literal

from claritymed.core.ocr.base import ExtractResult, OcrError, OcrProvider

logger = logging.getLogger(__name__)

_DEFAULT_DOCUMENT_EXTENSIONS = frozenset(
    {".pdf", ".doc", ".docx", ".odt", ".rtf", ".epub", ".ppt", ".pptx", ".xls", ".xlsx"}
)
PhiPolicy = Literal["local-only", "any"]


def _emit_audit(payload: dict[str, Any]) -> None:
    """Emit ocr.extract audit event."""
    try:
        from claritymed.core.observability.audit import audit_event

        audit_event("ocr.extract", payload)
    except Exception:  # noqa: BLE001
        logger.warning("ocr.extract audit failed", exc_info=True)


def _filter_chain_for_policy(
    chain: list[OcrProvider], phi_policy: PhiPolicy
) -> list[OcrProvider]:
    """Drop cloud providers from the chain under ``phi_policy="local-only"``.

    Defense-in-depth alongside the per-provider env gate (MineRU). The
    chain-level filter is the *structural* guarantee — even if every
    other safety mechanism fails, a non-local provider cannot end up in
    a PHI-tagged chain.
    """
    if phi_policy == "any":
        return chain
    return [p for p in chain if getattr(p, "is_local", True)]


class RoutingOcrProvider(OcrProvider):
    """Routes OCR to different backends based on file type.

    Each file kind (document vs image) has its own ordered chain. The
    first provider that returns text wins; ``OcrError`` from one means
    "next provider." ``phi_policy="local-only"`` filters cloud providers
    out at composition time.
    """

    label = "routing"

    def __init__(
        self,
        *,
        document_chain: list[OcrProvider] | None = None,
        image_chain: list[OcrProvider] | None = None,
        phi_policy: PhiPolicy = "any",
        document_extensions: frozenset[str] | None = None,
    ) -> None:
        if document_chain is None and image_chain is None:
            raise ValueError(
                "RoutingOcrProvider: requires document_chain and/or image_chain"
            )

        self._document_chain = _filter_chain_for_policy(
            document_chain or [], phi_policy
        )
        self._image_chain = _filter_chain_for_policy(image_chain or [], phi_policy)
        self._phi_policy: PhiPolicy = phi_policy
        # Callers (factory) can widen the document set with text-only
        # extensions so csv/md/json route to document_chain rather than
        # image_chain. Defaults preserve the historical behavior.
        self._document_extensions = (
            document_extensions
            if document_extensions is not None
            else _DEFAULT_DOCUMENT_EXTENSIONS
        )

    async def extract_text(self, path: Path) -> ExtractResult:
        t0 = time.perf_counter()
        size_bytes = path.stat().st_size if path.exists() else 0
        ext = path.suffix.lower()
        chain = (
            self._document_chain
            if ext in self._document_extensions
            else self._image_chain
        )

        if not chain:
            # The combination of file kind + phi_policy has no provider
            # configured (e.g., image_chain is empty under local-only).
            # Surface explicitly rather than returning "".
            raise OcrError(
                f"no OCR providers available for {path.suffix} "
                f"under phi_policy={self._phi_policy!r}"
            )

        tried_labels: list[str] = []
        last_exc: OcrError | None = None
        for provider in chain:
            # Capability filter: a provider that declares it can't handle
            # this extension is silently skipped and does NOT enter
            # tried_labels. This keeps chain_tried honest — it reflects
            # what was actually attempted, not what was structurally
            # present. ``supported_extensions=None`` means "all extensions"
            # (e.g. MineRU's catch-all).
            supported = provider.supported_extensions
            if supported is not None and ext not in supported:
                continue
            label = provider.label
            tried_labels.append(label)
            try:
                result = await provider.extract_text(path)
            except OcrError as exc:
                last_exc = exc
                continue
            duration_ms = int((time.perf_counter() - t0) * 1000)
            _emit_audit(
                {
                    "status": "ok",
                    "provider": label,
                    "chain_tried": tried_labels,
                    "chain_succeeded": label,
                    "file": path.name,
                    "size_bytes": size_bytes,
                    "chars": len(result.text),
                    "duration_ms": duration_ms,
                    "fallback": len(tried_labels) > 1,
                }
            )
            # Rewrite chain_tried to reflect what *we* walked (failed
            # leaves + winner), not just what the winning leaf returned.
            return ExtractResult(
                text=result.text,
                provider_used=label,
                chain_tried=list(tried_labels),
            )

        duration_ms = int((time.perf_counter() - t0) * 1000)
        # tried_labels may be empty if every provider in the chain
        # declared the extension unsupported — surface that distinctly
        # so operators don't chase a phantom OCR failure.
        if not tried_labels:
            error_msg = (
                f"no chain provider supports extension {ext!r} "
                f"(configured: {[p.label for p in chain]})"
            )
        else:
            error_msg = str(last_exc) if last_exc else "no providers"
        _emit_audit(
            {
                "status": "error",
                "provider": tried_labels[-1] if tried_labels else "unknown",
                "chain_tried": tried_labels,
                "file": path.name,
                "size_bytes": size_bytes,
                "duration_ms": duration_ms,
                "error": error_msg,
                "fallback": len(tried_labels) > 1,
            }
        )
        if not tried_labels:
            raise OcrError(error_msg)
        raise OcrError(
            f"all providers exhausted ({tried_labels}); last error: {last_exc}"
        )
