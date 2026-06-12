"""LLM-based OCR provider using pydantic-ai BinaryContent.

Sends the raw file bytes to a vision-capable LLM and returns the extracted
text.  Supports any file type whose MIME type the target model accepts —
in practice: JPEG, PNG, GIF, WebP, BMP, TIFF, and PDF.

The model is constructed once at provider creation time.  Each
``extract_text`` call creates a fresh ``Agent`` (stateless, no history).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from claritymed.core.ocr.base import ExtractResult, OcrError, OcrProvider

if TYPE_CHECKING:
    from pydantic_ai.models import Model

logger = logging.getLogger(__name__)

_PROMPT_NAME = "ocr"


class LLMOcrProvider(OcrProvider):
    """OCR via a pydantic-ai vision-capable LLM.

    Any model that supports ``BinaryContent`` image/document inputs works
    (GPT-4o, Claude Sonnet, Qwen-VL, etc.).

    ``is_local`` is set per-instance (via the constructor) because the same
    ``LLMOcrProvider`` class wraps both local vision models (Qwen-VL via
    Ollama / MLX) and cloud vision models (GPT-4o). Callers thread the
    real locality from the provider config; default ``True`` keeps the
    PHI-safe failure mode (drop from chain) when locality is unknown.
    """

    label = "llm"
    # Vision LLMs accept PDFs and standard raster images. Office formats
    # (.docx etc.) and pure text files are excluded so the chain skips
    # the LLM hop for them.
    supported_extensions = frozenset(
        {
            ".pdf",
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".gif",
            ".bmp",
            ".tiff",
            ".tif",
        }
    )

    def __init__(self, model: "Model", *, is_local: bool = True) -> None:
        self._model = model
        # Override the class attribute on this instance so chain composers
        # see the right locality without us needing two subclasses.
        self.is_local = is_local

    async def extract_text(self, path: Path) -> ExtractResult:
        """Send *path* to the LLM and return extracted text.

        Raises:
            OcrError: On any failure (file unreadable, model error, or when
                the LLM reports it could not see the document content).
        """
        from pydantic_ai import Agent, BinaryContent

        from claritymed.core.prompts.registry import get_default_registry
        from claritymed.core.schemas.ocr import OcrExtraction

        try:
            binary = BinaryContent.from_path(path)
        except (FileNotFoundError, PermissionError) as exc:
            raise OcrError(f"Cannot read {path}: {exc}") from exc

        prompt = get_default_registry().get(_PROMPT_NAME)
        agent: Agent[None, OcrExtraction] = Agent(
            self._model, system_prompt=prompt, output_type=OcrExtraction
        )
        try:
            result = await agent.run([binary])
        except Exception as exc:
            raise OcrError(f"LLM OCR failed for {path.name}: {exc}") from exc

        extraction = result.output
        if not extraction.success:
            raise OcrError(
                f"LLM could not extract text from {path.name}: "
                f"{extraction.failure_reason or 'no reason given'}"
            )

        text = extraction.text.strip()
        logger.debug("ocr: extracted %d chars from %s", len(text), path.name)
        return ExtractResult(
            text=text, provider_used=self.label, chain_tried=[self.label]
        )
