"""OCR service: file-to-text extraction with a provider interface.

Public API::

    from claritymed.core.ocr import OcrProvider, LLMOcrProvider, MineRUOcrProvider
    from claritymed.core.ocr.factory import make_ocr_provider
"""

from claritymed.core.ocr.base import OcrProvider
from claritymed.core.ocr.llm_provider import LLMOcrProvider
from claritymed.core.ocr.mineru_provider import MineRUOcrProvider
from claritymed.core.ocr.routing_provider import RoutingOcrProvider

__all__ = ["OcrProvider", "LLMOcrProvider", "MineRUOcrProvider", "RoutingOcrProvider"]
