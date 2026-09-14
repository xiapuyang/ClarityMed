"""Shared OCR → chunk → embed → upsert pipeline for system RAG corpora.

This module is the single implementation behind both entry points that
populate ``system_rag.collections``:

* ``scripts/init_system_rag.py`` — the operator CLI.
* ``POST /admin/rag/collections/upsert`` — the admin SPA endpoint.

Both call :func:`ingest_system_rag` with a :class:`SystemRagIngestRequest`
and get back the same :class:`SystemRagIngestResult`. Keeping the logic
here (rather than duplicating it in the runner) guarantees the web flow
and the CLI walk the same OCR + dedup + centroid path.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from claritymed.core.ocr.factory import make_ocr_provider
from claritymed.core.rag import load_retrieval_config
from claritymed.core.rag.chunking.base import RawDocument
from claritymed.core.rag.chunking.factory import build_chunker
from claritymed.core.rag.embedding.factory import build_embedder
from claritymed.core.rag.parent_store import ParentStore
from claritymed.core.rag.qdrant_store import RagCollectionStore, build_qdrant_client
from claritymed.core.rag.routing.centroid_store import CentroidStore, maybe_refresh
from claritymed.core.schemas.ocr import load_ocr_config
from claritymed.ingest.corpus.base import IngestStats, ingest_corpus
from claritymed.stores.paths import shared_parent_docstore_path, shared_root

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_DENSE_DIM_FALLBACK = 1024

ProgressCallback = Callable[[str], None]


# --- Request / Result -------------------------------------------------------


@dataclass
class SystemRagIngestRequest:
    """All inputs needed to ingest one batch of files into a system corpus.

    The collection ``name`` is the only required field for the append
    case (other metadata is inherited from ``retrieval.yaml``). For a
    new collection, ``language`` / ``authority_tier`` get sane defaults
    (``en`` / ``2``) via :func:`resolve_metadata` so the request can
    still come from a minimal admin form.
    """

    name: str
    files: list[Path]
    topics: list[str] = field(default_factory=list)
    language: str | None = None
    cross_lingual: bool = False
    authority_tier: int | None = None
    license: str | None = None
    source_uri_prefix: str | None = None
    dedupe_cosine_threshold: float = 0.0
    dry_run: bool = False
    limit: int | None = None

    def validate_name(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ValueError(f"name must match {_NAME_RE.pattern}")


@dataclass(frozen=True)
class SystemRagIngestResult:
    """Outcome of one :func:`ingest_system_rag` call.

    ``is_new_collection`` is True iff the name was absent from
    ``retrieval.yaml`` at request time — the caller uses it to decide
    whether to auto-append the snippet to the live config.
    """

    stats: IngestStats
    yaml_snippet: str
    is_new_collection: bool
    centroid_refreshed: bool


# --- Helpers ----------------------------------------------------------------


def slugify_doc_id(stem: str) -> str:
    """Sanitise a filename stem into a stable, comparable doc_id."""
    s = _SLUG_RE.sub("-", stem.lower()).strip("-")
    return s or "doc"


class _GenericSource:
    """Adapter that exposes a pre-built RawDocument list as a CorpusSource."""

    def __init__(self, name: str, docs: list[RawDocument]) -> None:
        self.name = name
        self._docs = docs

    def iter_raw_docs(self) -> Iterator[RawDocument]:
        yield from self._docs


class _NoOpEmbedder:
    """Dry-run embedder stand-in; advertises a dimension but never called."""

    @property
    def dimension(self) -> int:
        return _DENSE_DIM_FALLBACK

    async def embed_dense(self, texts):  # noqa: ANN001
        return [[0.0] * _DENSE_DIM_FALLBACK for _ in texts]

    async def embed_sparse(self, texts):  # noqa: ANN001
        return [{} for _ in texts]


async def _extract_one(
    path: Path,
    *,
    text_extensions: frozenset[str],
    ocr_provider,
) -> tuple[str, str]:
    """Return ``(text, provider_label)`` for a single input path.

    Plain-text extensions bypass the OCR chain (mirroring the paste-time
    fast-path the upload modal uses). Everything else routes through
    ``make_ocr_provider``.
    """
    ext = path.suffix.lower()
    if ext in text_extensions:
        return path.read_text(encoding="utf-8"), "text"
    result = await ocr_provider.extract_text(path)
    return result.text, result.provider_used


async def ocr_files(
    paths: list[Path],
    *,
    collection_name: str,
    language: str,
    on_progress: ProgressCallback | None = None,
) -> list[RawDocument]:
    """OCR every input file and wrap each as a RawDocument.

    Missing files and empty extracts are skipped (logged via
    ``on_progress`` when supplied); the batch survives so one bad PDF
    doesn't kill a 50-doc ingest run.
    """
    ocr_provider = make_ocr_provider()
    text_extensions = frozenset(load_ocr_config().text_extensions)

    docs: list[RawDocument] = []
    seen_ids: set[str] = set()
    for path in paths:
        if not path.exists():
            if on_progress:
                on_progress(f"skip (missing): {path.name}")
            continue
        if on_progress:
            on_progress(f"extract: {path.name}")
        try:
            text, provider_used = await _extract_one(
                path, text_extensions=text_extensions, ocr_provider=ocr_provider
            )
        except Exception as exc:  # noqa: BLE001 — keep the batch alive
            logger.exception("extraction failed: %s", path)
            if on_progress:
                on_progress(f"skip ({exc}): {path.name}")
            continue
        if not text.strip():
            if on_progress:
                on_progress(f"skip (empty extract): {path.name}")
            continue

        # Disambiguate duplicate slugs within one invocation so the
        # resume-by-doc_id contract holds across re-runs.
        doc_id = slugify_doc_id(path.stem)
        suffix = 1
        base = doc_id
        while doc_id in seen_ids:
            suffix += 1
            doc_id = f"{base}-{suffix}"
        seen_ids.add(doc_id)

        docs.append(
            RawDocument(
                doc_id=doc_id,
                text=text,
                language=language,
                metadata={
                    "doc_title": path.stem,
                    "source_uri": str(path.resolve()),
                    "collection": collection_name,
                    "ocr_provider": provider_used,
                },
            )
        )
    return docs


async def _refresh_centroid(
    aclient, name: str, on_progress: ProgressCallback | None = None
) -> bool:
    """Force-refresh the routing centroid for the named collection.

    Returns True on success; logs (best-effort) on failure so the router
    can degrade to its rule-based fallback while still surfacing the
    error in the runner's stdout_tail.
    """
    store = CentroidStore(shared_root() / "centroids")
    try:
        await maybe_refresh(aclient, name, store, force=True)
        if on_progress:
            on_progress(f"centroid refreshed: {name}")
        return True
    except Exception as exc:  # noqa: BLE001 -- best-effort; router degrades
        if on_progress:
            on_progress(f"centroid refresh failed ({exc})")
        return False


def build_yaml_snippet(req: SystemRagIngestRequest) -> str:
    """Render the YAML entry for ``system_rag.collections``.

    Point counts intentionally aren't part of the snippet — the live
    ``corpora list`` queries Qdrant, so a stale yaml number would just
    mislead. The yaml carries routing metadata; the database is the
    source of truth for size.
    """
    # An empty topics list is legitimate now that the centroid router
    # is active — render it inline (``topics: []``) instead of leaving
    # a dangling ``topics:`` key with no children, which downstream
    # YAML readers would coerce to ``None``.
    if req.topics:
        topics_field = "      topics:\n" + "\n".join(
            f"        - {t}" for t in req.topics
        )
    else:
        topics_field = "      topics: []"
    license_line = (
        "      license: null"
        if req.license is None
        else f'      license: "{req.license}"'
    )
    source_uri_line = (
        "      source_uri_prefix: null"
        if req.source_uri_prefix is None
        else f'      source_uri_prefix: "{req.source_uri_prefix}"'
    )
    return (
        f"    - name: {req.name}\n"
        f"      language: {req.language}\n"
        f"      cross_lingual: {'true' if req.cross_lingual else 'false'}\n"
        f"      authority_tier: {req.authority_tier}\n"
        f"{topics_field}\n"
        f"      disease_codes: []\n"
        f"{source_uri_line}\n"
        f"{license_line}"
    )


def resolve_metadata(req: SystemRagIngestRequest) -> bool:
    """Fill missing metadata from ``configs/retrieval.yaml``.

    Returns True when the collection already exists in yaml (append
    mode), False for a brand-new collection.

    Two modes, switched on whether ``req.name`` already exists in
    ``system_rag.collections``:

    * **Append** (existing entry) — inherit topics / language /
      authority_tier / cross_lingual / license / source_uri_prefix from
      the yaml. Explicit values on the request still win (no warnings
      here — that's a CLI concern, surfaced upstream by the script).
    * **New** (no entry yet) — apply the historical defaults (``en``,
      tier 2) for unset fields.
    """
    cfg = load_retrieval_config()
    existing = next(
        (c for c in cfg.system_rag.collections if c.name == req.name),
        None,
    )
    if existing is None:
        if req.language is None:
            req.language = "en"
        if req.authority_tier is None:
            req.authority_tier = 2
        req.source_uri_prefix = None
        return False

    if not req.topics:
        req.topics = list(existing.topics)
    if req.language is None:
        req.language = existing.language
    if req.authority_tier is None:
        req.authority_tier = existing.authority_tier
    # ``cross_lingual`` defaults to False on the dataclass, which we
    # treat as "unspecified" on append so the yaml ``true`` survives.
    if not req.cross_lingual and existing.cross_lingual:
        req.cross_lingual = True
    if req.license is None:
        req.license = existing.license
    req.source_uri_prefix = existing.source_uri_prefix
    return True


# --- Top-level orchestrator -------------------------------------------------


async def ingest_system_rag(
    req: SystemRagIngestRequest,
    *,
    on_progress: ProgressCallback | None = None,
) -> SystemRagIngestResult:
    """Run the full OCR → chunk → embed → dedup → upsert pipeline.

    Side effects:

    1. Calls :func:`resolve_metadata` to fill defaults / inherit from yaml.
    2. OCR every file via the project's OCR provider chain.
    3. Chunk + embed + upsert via :func:`ingest_corpus` (with optional
       cosine-similarity dedup against the target collection).
    4. Persist parent docs to the parent docstore.
    5. Force-refresh the routing centroid for the collection.

    The caller is responsible for editing ``configs/retrieval.yaml``
    based on ``result.is_new_collection`` and ``result.yaml_snippet``.
    """
    req.validate_name()
    is_existing = resolve_metadata(req)

    paths = list(req.files)
    if req.limit is not None:
        paths = paths[: req.limit]

    if on_progress:
        on_progress(f"ocr: {len(paths)} files")

    docs = await ocr_files(
        paths,
        collection_name=req.name,
        language=req.language or "en",
        on_progress=on_progress,
    )
    if not docs:
        raise RuntimeError("no documents to ingest after OCR (all skipped)")

    source = _GenericSource(req.name, docs)
    chunker = build_chunker()
    embedder = _NoOpEmbedder() if req.dry_run else build_embedder()
    cfg = load_retrieval_config()
    aclient = build_qdrant_client(
        url=cfg.qdrant.url, api_key_env=cfg.qdrant.api_key_env
    )
    centroid_refreshed = False
    try:
        store = RagCollectionStore(
            aclient=aclient,
            collection_name=req.name,
            dense_dim=embedder.dimension,
        )
        parent_store = ParentStore(shared_parent_docstore_path())

        if on_progress:
            on_progress("ingest: chunk + embed + upsert")

        stats = await ingest_corpus(
            source,
            chunker=chunker,
            embedder=embedder,
            store=store,
            parent_store=parent_store,
            dry_run=req.dry_run,
            dedupe_threshold=req.dedupe_cosine_threshold,
        )
        if on_progress:
            on_progress(
                f"ingested: {stats.docs_processed} docs / "
                f"{stats.parents_written} parents / "
                f"{stats.children_written} children "
                f"(skipped {stats.docs_skipped}, "
                f"resumed {stats.docs_resumed}, "
                f"deduped {stats.children_deduped})"
            )

        if not req.dry_run and stats.children_written > 0:
            centroid_refreshed = await _refresh_centroid(
                aclient, req.name, on_progress=on_progress
            )
    finally:
        await aclient.close()

    snippet = build_yaml_snippet(req)
    return SystemRagIngestResult(
        stats=stats,
        yaml_snippet=snippet,
        is_new_collection=not is_existing,
        centroid_refreshed=centroid_refreshed,
    )
