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

from claritymed.core.ocr.base import OcrError, OcrProvider

if TYPE_CHECKING:
    from pydantic_ai.models import Model

logger = logging.getLogger(__name__)

_PROMPT_NAME = "ocr"


class LLMOcrProvider(OcrProvider):
    """OCR via a pydantic-ai vision-capable LLM.

    Any model that supports ``BinaryContent`` image/document inputs works
    (GPT-4o, Claude Sonnet, Qwen-VL, etc.).
    """

    def __init__(self, model: "Model") -> None:
        self._model = model

    async def extract_text(self, path: Path) -> str:
        """Send *path* to the LLM and return extracted text.

        Raises:
            OcrError: On any failure (file unreadable, model error, etc.).
        """
        from pydantic_ai import Agent, BinaryContent

        from claritymed.core.prompts.registry import PromptRegistry

        try:
            binary = BinaryContent.from_path(path)
        except (FileNotFoundError, PermissionError) as exc:
            raise OcrError(f"Cannot read {path}: {exc}") from exc

        prompt = PromptRegistry().get(_PROMPT_NAME)
        agent: Agent[None, str] = Agent(
            self._model, system_prompt=prompt, output_type=str
        )
        try:
            result = await agent.run([binary])
        except Exception as exc:
            raise OcrError(f"LLM OCR failed for {path.name}: {exc}") from exc

        text = result.output.strip()
        logger.debug("ocr: extracted %d chars from %s", len(text), path.name)
        return text
