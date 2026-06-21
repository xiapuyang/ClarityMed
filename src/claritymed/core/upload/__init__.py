"""Mixed-content upload pipeline.

The /upload command can carry plain text, image attachments, file
attachments, or any mix. ``UploadBundle`` is the single value type that
flows through the pipeline; everything else (modal preview, RAG ingest,
status reporting) reads from it.
"""

from claritymed.core.upload.builder import (
    build_path_mode_bundle,
    build_upload_bundle,
)
from claritymed.core.upload.bundle import (
    PartKind,
    PartStatus,
    UploadBundle,
    UploadPart,
    ValidationResult,
)

__all__ = [
    "PartKind",
    "PartStatus",
    "UploadBundle",
    "UploadPart",
    "ValidationResult",
    "build_path_mode_bundle",
    "build_upload_bundle",
]
