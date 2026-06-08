"""OCR service: file-to-text extraction with a provider interface.

Public API::

    from claritymed.core.ocr import OcrProvider, LLMOcrProvider
    from claritymed.core.ocr.factory import make_ocr_provider
"""

from claritymed.core.ocr.base import OcrProvider
from claritymed.core.ocr.llm_provider import LLMOcrProvider

__all__ = ["OcrProvider", "LLMOcrProvider"]
