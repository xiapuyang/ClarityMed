"""Project-wide typed exceptions.

Concrete exception classes let store / orchestrator code be explicit about
what it refuses to do, so callers can branch on intent rather than message
text. Most of these are raised at the trust boundary (CLI / API / cross-user
read), then either bubbled up or converted to a high-uncertainty
``GroundedAnswer`` by the orchestrator.
"""

from __future__ import annotations


class InvalidUserIdError(ValueError):
    """``user_id`` did not match the path-safe regex."""


class UserIdMismatch(ValueError):
    """An entity's ``user_id`` field did not match the store's ``user_id``."""


class PermissionDeniedError(PermissionError):
    """Current account is not allowed to perform this action."""


class PhiViolationError(ValueError):
    """Outbound payload contained PHI that policy refuses to release."""
