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


class UnknownProviderError(KeyError):
    """Resolver was asked for a provider id that is not in ``models.yaml``."""


class MissingApiKeyError(RuntimeError):
    """A provider declared ``api_key_env`` but the env var is unset.

    Only raised for custom endpoints (``base_url`` set). Stock cloud
    providers go through pydantic-ai, which raises its own ``UserError``
    for missing keys — we don't shadow that path.
    """


# --- RAG catalog resolution errors --------------------------------------
#
# Same fail-loud contract as ``UnknownProviderError``: ``configs/retrieval.yaml``
# has an ``active`` id at each catalog section, and a typo there must refuse
# to load rather than silently fall back to a default (the default may not be
# what the operator intended, and certainly is not what a paper experiment
# wants to reproduce).


class UnknownStrategyError(KeyError):
    """``strategies.active`` does not appear in the catalog."""


class UnknownChunkerError(KeyError):
    """``chunker.active`` does not appear in the catalog."""


class UnknownEmbedderError(KeyError):
    """``embedders.active`` does not appear in the catalog."""


class UnknownRerankerError(KeyError):
    """``rerankers.active`` does not appear in the catalog."""


class UnknownTermServiceError(KeyError):
    """``term_service.active`` does not appear in the catalog."""


class UnknownRouterError(KeyError):
    """``router.active`` does not appear in the catalog."""


# --- RAG runtime errors -------------------------------------------------
#
# Embedder is fail-loud (a silently substituted CPU embedder would write
# 384-dim vectors into a 1024-dim Qdrant collection — worse than failure).
# Reranker is fail-soft at the caller (HybridRetriever), but the wire
# layer still surfaces the failure as a typed exception so the caller
# can audit and degrade explicitly.


class EmbedderUnreachableError(RuntimeError):
    """Embedding server returned non-2xx, timed out, or sent a bad shape."""


class RerankerUnreachableError(RuntimeError):
    """Reranker server returned non-2xx, timed out, or sent a bad shape."""
