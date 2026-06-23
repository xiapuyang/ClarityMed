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
    is_vision = True
    """A vision LLM accepts any visually-renderable input — its
    ``supported_extensions`` enumerates MIMEs it can ingest, not file
    types it claims authority over. Routing skips its claim when
    deciding document_chain vs image_chain; see ``OcrProvider.is_vision``.
    """
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
            OcrError: On any failure (file unreadable, model error, when
                the LLM reports it could not see the document content,
                or when the image trips ``vision.image_limits`` —
                converted from :class:`ImageTooLargeError` /
                :class:`ImageTooSmallError` so the OCR chain treats it
                as a normal provider failure and can fall through to a
                non-vision provider rather than aborting the whole turn).
        """
        from pydantic_ai import Agent, BinaryContent

        from claritymed.config import vision_image_limits
        from claritymed.core.prompts.registry import get_default_registry
        from claritymed.core.schemas.ocr import OcrExtraction
        from claritymed.core.vision.image_guard import validate_image
        from claritymed.errors import ImageTooLargeError, ImageTooSmallError

        # Image guard — runs before we even read the bytes. Bills, latency,
        # and PII surface area all scale with what we send up to a cloud
        # vision LLM; rejecting at the boundary keeps a 50MB DICOM from
        # ever touching the network. File-not-found / permission errors
        # are coalesced with the BinaryContent path below so callers see
        # one canonical OcrError shape regardless of which layer noticed.
        try:
            validate_image(path, vision_image_limits())
            binary = BinaryContent.from_path(path)
        except (ImageTooLargeError, ImageTooSmallError) as exc:
            raise OcrError(f"Image guard rejected {path.name}: {exc}") from exc
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

        # Resolve outcome: prefer explicit status (v2 prompts), fall back to
        # success bool (v1 prompts).  Neither set → treat as failed.
        status = extraction.status
        if status is None:
            if extraction.success is True:
                status = "done"
            elif extraction.success is False:
                status = "failed"
            else:
                status = "failed"

        if status == "failed":
            raise OcrError(
                f"LLM could not extract text from {path.name}: "
                f"{extraction.failure_reason or 'no reason given'}"
            )
        # The "empty" branch still surfaces the vision-LLM's modality /
        # is_medical via OcrEmpty.extraction — a model that read the
        # pixels can correctly classify a blank-text image (e.g. an
        # ultrasound with no overlay text). Dropping that signal is what
        # caused empty-OCR medical images to render as bare <image> tags
        # downstream and the LLM-side routing rules to no-op.
        empty_hint = ExtractResult(
            text="",
            provider_used=self.label,
            chain_tried=[self.label],
            modality=extraction.modality,
            is_medical=extraction.is_medical,
        )
        if status == "empty":
            from claritymed.core.ocr.base import OcrEmpty

            raise OcrEmpty(f"LLM: no text found in {path.name}", extraction=empty_hint)

        text = extraction.text.strip()
        if not text:
            from claritymed.core.ocr.base import OcrEmpty

            raise OcrEmpty(
                f"LLM: extracted empty text from {path.name}",
                extraction=empty_hint,
            )

        logger.debug("ocr: extracted %d chars from %s", len(text), path.name)
        return ExtractResult(
            text=text,
            provider_used=self.label,
            chain_tried=[self.label],
            modality=extraction.modality,
            is_medical=extraction.is_medical,
        )
