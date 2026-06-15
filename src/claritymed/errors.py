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


class UserNotFoundError(LookupError):
    """Requested ``user_id`` has no settings.yaml on disk.

    Raised at the CLI boundary when ``--user`` names a user that has never
    been initialised.  Callers should surface this as a plain error message
    directing the operator to run ``claritymed init-user <id>``.
    """


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


class UnknownModeError(KeyError):
    """``rag.mode`` does not appear in the mode registry."""


class DuplicateDocumentError(ValueError):
    """``source_uri`` is already indexed under an existing ``doc_id``.

    Raised by ``UserRagStore.add_document`` when the caller supplies a
    ``source_uri`` that matches a point already in the collection.
    ``existing_doc_id`` names the duplicate so callers can surface it.
    """

    def __init__(self, source_uri: str, existing_doc_id: str) -> None:
        super().__init__(
            f"source_uri already indexed as {existing_doc_id!r}: {source_uri!r}"
        )
        self.source_uri = source_uri
        self.existing_doc_id = existing_doc_id


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


# --- v1 PHI storage + tool dispatcher errors ---------------------------
#
# The single PHI write gate (ToolDispatcher) and the per-event manifest
# storage layer add a handful of typed exceptions that the orchestrator,
# CLI, and TUI all branch on rather than parse error messages.


class NonRetryableLLMError(RuntimeError):
    """Marker base — pydantic-ai's retry loop must not re-issue the call.

    Any LLM-pipeline exception that's deterministic in the *input* (PHI
    leak, schema-incompatible structured output, etc.) should subclass
    this so the framework's retry handler short-circuits and the audit
    log gets one event per actual failure, not one per retry.
    """


class RevisionConflict(RuntimeError):
    """Optimistic-lock failure on ``ManifestStore.update``.

    Caller supplied ``expected_revision`` that did not match the on-disk
    manifest's current revision. The classic concurrent-edit race; the
    safe response is to re-read and retry with a fresh expected revision.
    """


class UnknownSha256(ValueError):
    """A tool referenced a sha256 not present in the user's blob universe.

    Checked at ``ToolDispatcher.gate`` step 2: every sha in tool args must
    be in (current session attachments ∪ ``blobs/<sha>/`` on disk ∪ any
    manifest under the current user's records/library). Cross-user sha
    resolution always fails — the third set is scoped to the auth'd uid.
    """


class PhiLeakDetected(NonRetryableLLMError):
    """``PhiAssertionModel`` (layer 3) refused an outbound cloud call.

    A cloud-bound message stream contained PHI (chunk marker or content
    heuristic hit). Non-retryable so pydantic-ai's retry loop terminates
    rather than re-issuing the same prompt and re-tripping the assertion
    in a flood of audit rows.
    """


class RecordNotFound(LookupError):
    """``ManifestStore.read`` was asked for a slug that does not exist."""


class PathOutsideUserDomain(PermissionError):
    """``record_path`` (or similar) escaped the current user's data root.

    Detected by resolving both the supplied path and the user's records
    root and asserting ``resolved.is_relative_to(root)``. Catches both
    ``../`` traversal and symlink escape — the latter via an explicit
    ``os.lstat`` symlink check at the final path component.
    """


class OcrFailed(RuntimeError):
    """Eager-OCR worker exhausted its provider chain without text.

    Distinct from ``OcrProviderError`` (single-provider failure): this is
    the chain-level "no provider can read this blob" outcome. The blob's
    ``ocr.json`` is still written with ``status="failed"`` so the UI can
    surface state without re-running OCR.
    """


class ApprovalDenied(PermissionError):
    """User denied a tool's approval modal.

    Bubbles out of ``ToolDispatcher.gate`` as a typed exception so the
    LLM gets a structured tool error rather than a swallowed silence.
    """


class OcrProviderError(RuntimeError):
    """One OCR provider failed to extract.

    Recoverable at the routing layer (n-ary fallback): the next provider
    in the chain is tried. Raised by the per-provider ``extract`` impl.
    """


class MinerUNotAllowed(OcrProviderError):
    """Constructing the MinerU provider requires explicit env opt-in.

    MinerU is a cloud SaaS (mineru.net); it never belongs on the PHI
    path. The env gate (``CLARITYMED_ALLOW_MINERU=1``) is for regression
    testing only — ``is_local = False`` keeps PHI chains structurally
    safe even when the env is set.
    """


# --- symptoms feature errors -------------------------------------------
#
# Same fail-loud catalog pattern as the RAG errors above: a typo'd
# ``eligibility.active`` or a config that cross-references a missing
# dataset id raises at load time rather than silently degrading. The
# strategy-level runtime errors (config misuse, dependency unavailable)
# are split so the orchestrator can branch on "won't ever work" vs
# "transiently down".


class UnknownEligibilityStrategyError(KeyError):
    """``eligibility.active`` does not appear in the catalog."""


class UnknownDatasetError(KeyError):
    """A symptoms config field referenced a dataset id not in ``datasets[]``.

    Raised by ``SymptomsConfig`` validators (load time) and never by the
    registry's runtime resolution path — LLM-supplied ``dataset_hint``
    values fall through soft, per KTD-9 in the disease-prediction plan.
    """


class SymptomsServerUnreachableError(RuntimeError):
    """Symptoms server returned non-2xx, timed out, or sent a bad shape.

    Mirrors ``EmbedderUnreachableError``: the orchestrator catches this
    mid-loop and degrades the tool call to a structured ``server_error``
    result so the LLM can fall back to free-text answering.
    """


class MedicalClipUnreachableError(RuntimeError):
    """Medical-clip server (BiomedCLIP) returned non-2xx, timed out, or sent a bad shape.

    The orchestrator's OCR worker catches this and tags the attachment
    with ``modality="unknown", is_medical=null`` so OCR ingest still
    succeeds — the modality classifier is best-effort, not a hard
    dependency of the attachment pipeline (origin §5.5 failure modes).
    """


class VisionServerUnreachableError(RuntimeError):
    """Vision server returned non-2xx, timed out, or sent a bad shape.

    Mirrors ``SymptomsServerUnreachableError``: the tool body catches
    this in the fallback flow (Unit 7) and either advances to the next
    model in ``disease.flow`` or returns ``NoUsableResultError`` when
    the budget is exhausted.
    """


class UnknownDiseaseError(KeyError):
    """``VisionRegistry.route`` could not resolve the requested ``disease_id``.

    Carries the ``available`` list of enabled disease ids so the tool
    body can return a structured dict back to the LLM with actionable
    context (the LLM may pick a different disease and retry).
    """

    def __init__(self, disease_id: str, available: list[str]) -> None:
        self.disease_id = disease_id
        self.available = available
        super().__init__(
            f"disease_id {disease_id!r} not in vision catalog; available: {available!r}"
        )


class VisionCatalogMismatchError(RuntimeError):
    """Boot-time ``/v1/catalog`` cross-check found a server-vs-config drift.

    The server is loading something the config doesn't claim (or vice
    versa). Distinct from ``VisionServerUnreachableError`` because the
    server **is** reachable — it just disagrees. Treated as a hard boot
    failure so an operator notices before users hit it.
    """


class ImageHashMismatchError(ValueError):
    """``image.data_b64`` decoded to bytes whose sha256 disagrees with the claim.

    Catches stale or tampered uploads that try to bypass the
    attachment-ingest modality tag — the server is the ground truth on
    "what bytes are these really". Vision and medical-clip endpoints
    both raise this before any inference work.
    """


class EligibilityStrategyConfigError(RuntimeError):
    """An eligibility strategy was built against an invalid configuration.

    The canonical case is the ``translation`` strategy resolving a
    provider whose ``ProviderConfig.kind`` is ``"cloud"`` — that path
    would send PHI off-device, so construction fails loud rather than
    waiting for the first request.
    """


class EligibilityStrategyUnavailableError(RuntimeError):
    """An eligibility strategy cannot run in the current environment.

    Distinct from ``EligibilityStrategyConfigError``: the config is
    well-formed, but a runtime dependency (LLM provider unreachable,
    ``NoOpTermService`` injected, sidecar JSON missing) makes the
    strategy non-functional. Caller may fall through to the next
    strategy or to free-text answering.
    """
