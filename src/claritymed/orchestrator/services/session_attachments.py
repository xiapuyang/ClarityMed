"""Re-export shim — SessionAttachments moved to stores.session_attachments."""

from claritymed.stores.session_attachments import (
    SessionAttachment as SessionAttachment,
    SessionAttachments as SessionAttachments,
    SourceKind as SourceKind,
)

__all__ = ["SessionAttachment", "SessionAttachments", "SourceKind"]
