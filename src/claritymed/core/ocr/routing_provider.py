"""File-type-aware routing OCR provider.

Two construction styles supported:

* Legacy single-provider mode: ``document_provider`` + ``image_default``
  + optional ``image_fallback``. Two-step chain at the image layer; one
  provider at the document layer.
* Chain mode: ``document_chain`` + ``image_chain`` (each is a
  ``list[OcrProvider]``). N-ary fallback: the first provider that
  returns text wins; ``OcrError`` from one means "next provider, please."

Chain mode also accepts ``phi_policy="local-only"`` which filters the
chains to providers with ``is_local = True`` at composition time. Cloud
providers (``MineRUOcrProvider``) are filtered out structurally, so a
buggy ``ocr.yaml`` that puts MineRU on the PHI path becomes "no chain
applicable" rather than "PHI leaked to mineru.net".

Emits one ``ocr.extract`` audit event per call (best-effort — no-op when
the request ContextVars are not set, e.g. in unit tests).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Literal

from claritymed.core.ocr.base import OcrError, OcrProvider

logger = logging.getLogger(__name__)

_DOCUMENT_EXTENSIONS = frozenset(
    {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
)
PhiPolicy = Literal["local-only", "any"]


def _provider_label(provider: OcrProvider) -> str:
    """Short lowercase label derived from class name, e.g. 'mineru' or 'llm'."""
    name = type(provider).__name__  # e.g. "MineRUOcrProvider"
    return (
        name.lower()
        .replace("ocrprovider", "")
        .replace("ocr", "")
        .replace("provider", "")
    )


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

    Two construction styles (use one, not both):

    * Legacy: ``document_provider`` + ``image_default`` + optional
      ``image_fallback``. Single-provider document path; two-step image
      fallback. Kept for back-compat with existing OCR usage.
    * Chain: ``document_chain`` + ``image_chain``. Each is a list; the
      first provider that returns text wins. ``OcrError`` from one
      means "next provider." Optionally ``phi_policy="local-only"`` to
      filter cloud providers out at composition time.
    """

    def __init__(
        self,
        *,
        document_provider: OcrProvider | None = None,
        image_default: OcrProvider | None = None,
        image_fallback: OcrProvider | None = None,
        document_chain: list[OcrProvider] | None = None,
        image_chain: list[OcrProvider] | None = None,
        phi_policy: PhiPolicy = "any",
    ) -> None:
        legacy = (
            document_provider is not None
            or image_default is not None
            or image_fallback is not None
        )
        chain = document_chain is not None or image_chain is not None
        if legacy and chain:
            raise ValueError(
                "RoutingOcrProvider: pass either (document_provider/image_default) "
                "OR (document_chain/image_chain), not both"
            )
        if not legacy and not chain:
            raise ValueError(
                "RoutingOcrProvider: requires either single-provider args or chain args"
            )

        # Normalize legacy → chain shape. Internally we walk one list per
        # file_kind regardless of construction style.
        if legacy:
            self._document_chain = (
                [document_provider] if document_provider is not None else []
            )
            self._image_chain = []
            if image_default is not None:
                self._image_chain.append(image_default)
            if image_fallback is not None:
                self._image_chain.append(image_fallback)
        else:
            self._document_chain = _filter_chain_for_policy(
                document_chain or [], phi_policy
            )
            self._image_chain = _filter_chain_for_policy(image_chain or [], phi_policy)

        self._phi_policy: PhiPolicy = phi_policy

    async def extract_text(self, path: Path) -> str:
        t0 = time.perf_counter()
        size_bytes = path.stat().st_size if path.exists() else 0
        chain = (
            self._document_chain
            if path.suffix.lower() in _DOCUMENT_EXTENSIONS
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
            label = _provider_label(provider)
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
                    "chars": len(result),
                    "duration_ms": duration_ms,
                    "fallback": len(tried_labels) > 1,
                }
            )
            return result

        duration_ms = int((time.perf_counter() - t0) * 1000)
        _emit_audit(
            {
                "status": "error",
                "provider": tried_labels[-1] if tried_labels else "unknown",
                "chain_tried": tried_labels,
                "file": path.name,
                "size_bytes": size_bytes,
                "duration_ms": duration_ms,
                "error": str(last_exc) if last_exc else "no providers",
                "fallback": len(tried_labels) > 1,
            }
        )
        raise OcrError(
            f"all providers exhausted ({tried_labels}); last error: {last_exc}"
        )
