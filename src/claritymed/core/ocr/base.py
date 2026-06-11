"""OcrProvider interface.

All OCR backends implement this single method.  Callers never import a
concrete provider directly — they call ``make_ocr_provider()`` from
``claritymed.core.ocr.factory`` and depend only on this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


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

    @abstractmethod
    async def extract_text(self, path: Path) -> str:
        """Extract all text from *path* and return it as a plain string.

        Args:
            path: Absolute or relative path to a PDF or image file.

        Returns:
            Extracted text.  May be empty for blank pages.

        Raises:
            OcrError: If extraction fails and the caller should surface it.
        """


class OcrError(RuntimeError):
    """Raised by ``OcrProvider`` implementations on unrecoverable failure."""
