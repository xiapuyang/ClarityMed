"""``RoutingOcrProvider`` — chain-based OCR composer.

Routes ``extract_text`` to a list of providers per file kind. The first
provider that returns text wins; ``OcrError`` from one falls through to
the next.

``phi_policy="local-only"`` filters cloud providers (``is_local=False``)
out of the chain at composition time — defense-in-depth alongside any
per-provider env gate (e.g. MineRU's ``CLARITYMED_ALLOW_MINERU``).
"""

from __future__ import annotations

import contextvars
import logging
import time
from pathlib import Path
from typing import Any, Literal

from claritymed.core.ocr.base import ExtractResult, OcrEmpty, OcrError, OcrProvider

logger = logging.getLogger(__name__)

PhiPolicy = Literal["local-only", "any"]


_current_original_filename: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_original_filename", default=None
)
"""Per-call hint set by the OCR worker so the routing-layer audit
event can record the user-facing filename instead of just the CAS
``content.<ext>`` blob name.

Threading this via ``contextvars`` keeps the ``OcrProvider.extract_text``
signature unchanged — none of the 6 leaf providers need to learn about
a parameter they never use. The worker sets it for the duration of one
extraction and resets in ``finally``."""


def set_original_filename(name: str | None):
    """Set the audit hint; returns the reset token (use in ``finally``).

    Use as::

        token = set_original_filename(job.original_filename)
        try:
            await provider.extract_text(path)
        finally:
            _current_original_filename.reset(token)
    """
    return _current_original_filename.set(name)


def reset_original_filename(token) -> None:
    """Reset the audit hint via the token returned by ``set_original_filename``."""
    _current_original_filename.reset(token)


def _derive_document_extensions(chain: list[OcrProvider]) -> frozenset[str]:
    """Compute the routing-disambiguation set from a chain.

    An extension belongs to ``document_chain`` iff a *non-vision*
    provider in that chain claims it via ``supported_extensions``.
    Vision LLMs (``is_vision=True``) are excluded because their
    extension set reflects "I can ingest these MIMEs", not "I am the
    authoritative handler for these file types"; including them would
    pull every image format into the document side just because an
    LLM happens to be listed there.

    Providers declaring ``supported_extensions=None`` ("all") are
    skipped — they intentionally don't constrain routing.
    """
    result: set[str] = set()
    for provider in chain:
        if provider.is_vision:
            continue
        supported = provider.supported_extensions
        if supported is None:
            continue
        result.update(supported)
    return frozenset(result)


def _emit_audit(payload: dict[str, Any]) -> None:
    """Emit ocr.extract audit event."""
    try:
        from claritymed.core.observability.audit import audit_event

        audit_event("ocr.extract", payload)
    except Exception:  # noqa: BLE001
        logger.warning("ocr.extract audit failed", exc_info=True)


def _with_filename(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach the worker-supplied ``original_filename`` to *payload* if set.

    Returns a new dict (rather than mutating in place) so audit shape
    stays referentially transparent for tests that compare event
    payloads literally.
    """
    name = _current_original_filename.get()
    if name is None:
        return payload
    return {**payload, "original_filename": name}


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
        # Derived from chain composition rather than carried in via a
        # caller-supplied set: each non-vision provider's declared
        # ``supported_extensions`` is the single source of truth for
        # routing. See ``_derive_document_extensions``.
        self._document_extensions = _derive_document_extensions(self._document_chain)

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
        had_real_error = False
        # First leaf-level OcrEmpty whose ``extraction`` carries any
        # modality / is_medical signal wins. The vision LLM is the only
        # leaf that sets these, so in practice this captures its opinion
        # even if a downstream non-vision leaf (rapidocr) re-raises
        # OcrEmpty without a hint — without this, the worker's empty
        # branch would lose the LLM's modality classification.
        best_empty_hint: ExtractResult | None = None
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
            except OcrEmpty as exc:
                last_exc = exc
                hint = getattr(exc, "extraction", None)
                if (
                    best_empty_hint is None
                    and hint is not None
                    and (hint.modality is not None or hint.is_medical is not None)
                ):
                    best_empty_hint = hint
                continue
            except OcrError as exc:
                last_exc = exc
                had_real_error = True
                continue
            duration_ms = int((time.perf_counter() - t0) * 1000)
            _emit_audit(
                _with_filename(
                    {
                        "status": "ok",
                        "provider": label,
                        "chain_tried": tried_labels,
                        "chain_succeeded": label,
                        "blob_filename": path.name,
                        "size_bytes": size_bytes,
                        "chars": len(result.text),
                        "duration_ms": duration_ms,
                        "fallback": len(tried_labels) > 1,
                    }
                )
            )
            # Rewrite chain_tried to reflect what *we* walked (failed
            # leaves + winner), not just what the winning leaf returned.
            # Forward modality/is_medical from the winning provider so
            # _compute_vision_tags can use LLM-OCR's classification even
            # when text extraction succeeded (previously these were dropped
            # here, making the llm-ocr override in _compute_vision_tags
            # unreachable on the success path).
            return ExtractResult(
                text=result.text,
                provider_used=label,
                chain_tried=list(tried_labels),
                modality=result.modality,
                is_medical=result.is_medical,
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
            _emit_audit(
                _with_filename(
                    {
                        "status": "error",
                        "provider": "unknown",
                        "chain_tried": tried_labels,
                        "blob_filename": path.name,
                        "size_bytes": size_bytes,
                        "duration_ms": duration_ms,
                        "error": error_msg,
                        "fallback": False,
                    }
                )
            )
            raise OcrError(error_msg)

        # All providers were tried. Distinguish "image has no text" (every
        # provider returned OcrEmpty) from "something broke" (at least one
        # raised a real OcrError). Only the first case is not an error.
        all_empty = not had_real_error and isinstance(last_exc, OcrEmpty)
        audit_status = "empty" if all_empty else "error"
        error_msg = (
            f"all providers returned no text ({tried_labels})"
            if all_empty
            else f"all providers exhausted ({tried_labels}); last error: {last_exc}"
        )
        _emit_audit(
            _with_filename(
                {
                    "status": audit_status,
                    "provider": tried_labels[-1],
                    "chain_tried": tried_labels,
                    "blob_filename": path.name,
                    "size_bytes": size_bytes,
                    "duration_ms": duration_ms,
                    "error": error_msg,
                    "fallback": len(tried_labels) > 1,
                }
            )
        )
        if all_empty:
            # Carry the chain-level chain_tried plus any leaf-level vision
            # signal so the worker's empty branch can still classify and
            # write modality / is_medical into the sentinel.
            empty_extraction = ExtractResult(
                text="",
                provider_used=tried_labels[-1] if tried_labels else "",
                chain_tried=list(tried_labels),
                modality=best_empty_hint.modality if best_empty_hint else None,
                is_medical=best_empty_hint.is_medical if best_empty_hint else None,
            )
            raise OcrEmpty(error_msg, extraction=empty_extraction)
        raise OcrError(error_msg)
