"""OcrProvider interface.

All OCR backends implement this single method.  Callers never import a
concrete provider directly — they call ``make_ocr_provider()`` from
``claritymed.core.ocr.factory`` and depend only on this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ExtractResult:
    """Result returned by every ``OcrProvider.extract_text`` call.

    Carries the extracted ``text`` plus enough provenance for the
    caller to record *which* provider actually produced it and *which*
    providers were tried first. Leaf providers fill ``provider_used``
    with their own short label and put just that label in
    ``chain_tried``. ``RoutingOcrProvider`` returns the winning leaf's
    result but rewrites ``chain_tried`` to include every leaf it walked
    through (failed + winner), so the ``ocr.json`` sentinel and audit
    log can show real fallback behavior.
    """

    text: str
    provider_used: str
    chain_tried: list[str] = field(default_factory=list)
    modality: str | None = None
    is_medical: bool | None = None


class OcrProvider(ABC):
    """Interface for all OCR backends.

    Every implementation must raise ``OcrError`` (or a subclass) on
    unrecoverable failures so callers can handle them uniformly.

    Subclasses set ``is_local`` (class attribute) to declare whether the
    provider keeps PHI on the machine. The chain composer
    (``RoutingOcrProvider``) uses this flag to filter chains under
    ``phi_policy="local-only"``: cloud providers are dropped from the
    PHI chain regardless of what ``ocr.yaml`` lists. Default ``True``
    because new providers should explicitly opt out of PHI safety, not
    accidentally get treated as cloud.
    """

    is_local: bool = True

    label: str = ""
    """Short lowercase identifier used in audit logs, sentinels, and
    chain tracking. Every concrete provider must set this; routing
    layers and the worker treat it as authoritative."""

    supported_extensions: frozenset[str] | None = None
    """Lowercase file extensions (with leading dot) this provider can
    handle. ``None`` means "all extensions" — used by providers like
    MineRU that accept anything the API supports. Routing layers
    consult this to skip providers that can't handle the input; a
    skipped provider does NOT count as having been tried."""

    is_vision: bool = False
    """``True`` for general-purpose vision LLMs whose
    ``supported_extensions`` reflects "MIME types I can ingest" rather
    than "file types I am the authoritative source for." The router
    excludes such providers from its document-vs-image disambiguation
    so an image format never gets routed to ``document_chain`` just
    because a vision LLM happens to be listed there. Specialized
    backends (pymupdf, marker, pandoc, mineru) leave this ``False``
    — their ``supported_extensions`` IS their authority claim."""

    @abstractmethod
    async def extract_text(self, path: Path) -> ExtractResult:
        """Extract all text from *path*.

        Args:
            path: Absolute or relative path to a PDF or image file.

        Returns:
            ``ExtractResult`` with the extracted text plus the leaf
            label and the (single-element for leaves) chain_tried list.

        Raises:
            OcrError: If extraction fails and the caller should surface it.
        """


class OcrError(RuntimeError):
    """Raised by ``OcrProvider`` implementations on unrecoverable failure."""


class OcrEmpty(OcrError):
    """Raised when OCR ran successfully but found no text in the input.

    Distinguishes "image has no readable text" from "something broke".
    The routing layer re-raises this only when every provider in the chain
    agreed the input was empty and no real error occurred. Callers can
    catch ``OcrEmpty`` before ``OcrError`` to handle the two cases
    differently (e.g. persist ``status="empty"`` without a WARNING log).

    The optional ``extraction`` carries non-text provenance the worker
    still needs on the empty path: ``chain_tried`` (so ``ocr.json`` records
    which leaves were walked) and any ``modality`` / ``is_medical`` signal
    a vision LLM produced before reporting "no text". Without it, an image
    that a vision LLM correctly classified as e.g. an ultrasound but found
    unreadable would lose every routing hint downstream — see
    :class:`~claritymed.orchestrator.services.ocr_worker.OcrWorker` for the
    consumer.
    """

    def __init__(
        self, message: str, *, extraction: ExtractResult | None = None
    ) -> None:
        super().__init__(message)
        self.extraction = extraction
