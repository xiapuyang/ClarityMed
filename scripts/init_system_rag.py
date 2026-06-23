#!/usr/bin/env python3
"""Initialize a new system RAG collection from one or more documents.

Reuses the same OCR pipeline that ``/upload`` invokes
(:func:`claritymed.core.ocr.factory.make_ocr_provider`) to extract text
from each input file, then runs :func:`ingest_corpus` -- the same
chunk + embed + Qdrant upsert pipeline that ``statpearls`` and
``textbooks`` use -- against a fresh, named collection.

The script is parameterised on ``--name``, ``--topic``, ``--language``,
``--authority-tier``, etc., so it can be re-run for any future doc set
(guidelines, drug labels, ad-hoc cohort PDFs, ...).

Idempotency / append:

* ``RagCollectionStore.ensure_collection`` is create-if-missing -- safe
  to re-run.
* ``doc_id`` is the sanitised filename stem; re-running with the same
  inputs resumes by skipping already-ingested docs, while re-running
  with NEW filenames appends them to the existing collection.
* Optional ``--dedupe-cosine-threshold`` enables per-chunk cosine
  dedup against the target collection (same primitive
  ``stores/user_rag`` uses) -- useful when appending docs that share
  passages with the existing corpus (e.g. successive guideline editions).
* Centroid refresh is idempotent (uses ``maybe_refresh(force=True)``).

Example
-------

::

    uv run python scripts/init_system_rag.py \
        --name ats_idsa_pneumonia_guidelines_en \
        --topic "community-acquired pneumonia" \
        --topic "respiratory infections" \
        --language en \
        --authority-tier 1 \
        --user admin \
        "data/download/Diagnosis and Treatment of Adults with Community-acquired Pneumonia.pdf" \
        "data/download/Infectious Diseases Society of America:American Thoracic Society Consensus Guidelines on the Management of Community-Acquired Pneumonia in Adults.pdf" \
        "data/download/the-management-of-community-acquired-pneumonia-in-adults-with-comorbidities-or-immunocompromising-conditions-an-ats-ers-idsa-eshm-erjev-topic-manuscript-2.25.26.pdf"

After ingest, the script prints a YAML snippet for ``configs/retrieval.yaml``.
The new collection is invisible to the router until that snippet is
appended under ``system_rag.collections`` -- intentionally manual so
config changes stay reviewable.

Optional metadata
-----------------

``--topic`` / ``--language`` / ``--authority-tier`` / ``--license`` are
all optional:

* **Append** -- when ``--name`` matches a collection already declared
  in ``configs/retrieval.yaml``, missing values are inherited from the
  existing yaml entry. Explicit CLI flags still win, but a yellow
  warning is printed on any disagreement so the printed snippet
  doesn't silently downgrade the live config.
* **New** -- ``--language`` defaults to ``en`` and ``--authority-tier``
  to ``2``. ``--topic`` may be omitted entirely now that the active
  ``centroid_classifier`` router scores by query-vector cosine against
  precomputed centroids -- topics survive only as an operator-readable
  label and as a tie-breaker inside the rule-based fallback (where an
  empty list scores ``1.0`` and is never punished).

Error handling
--------------

* Missing input file -> printed warning, skipped (run continues).
* Empty OCR result   -> printed warning, skipped (run continues).
* Bad ``--name``     -> argparse error, exit code 2 (pre-flight).
* Admin gate failure -> ``PermissionDeniedError`` propagates, exit non-zero.
* Qdrant unreachable -> ``ingest_corpus`` raises; propagates, exit non-zero.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from collections.abc import Iterator
from pathlib import Path

from claritymed.cli.common import bootstrap_once, console
from claritymed.cli.entry import inject_context
from claritymed.core.ocr.factory import make_ocr_provider
from claritymed.core.rag import load_retrieval_config
from claritymed.core.rag.chunking.base import RawDocument
from claritymed.core.rag.chunking.factory import build_chunker
from claritymed.core.rag.embedding.factory import build_embedder
from claritymed.core.rag.parent_store import ParentStore
from claritymed.core.rag.qdrant_store import RagCollectionStore, build_qdrant_client
from claritymed.core.rag.routing.centroid_store import CentroidStore, maybe_refresh
from claritymed.core.schemas.ocr import load_ocr_config
from claritymed.ingest.corpus.base import ingest_corpus
from claritymed.stores.account import require_admin
from claritymed.stores.paths import shared_parent_docstore_path, shared_root

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_DENSE_DIM_FALLBACK = 1024


def _slugify_doc_id(stem: str) -> str:
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

    async def embed_dense(self, texts):
        return [[0.0] * _DENSE_DIM_FALLBACK for _ in texts]

    async def embed_sparse(self, texts):
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


async def _ocr_files(
    paths: list[Path],
    *,
    collection_name: str,
    language: str,
) -> list[RawDocument]:
    """OCR every input file and wrap each as a RawDocument."""
    ocr_provider = make_ocr_provider()
    text_extensions = frozenset(load_ocr_config().text_extensions)

    docs: list[RawDocument] = []
    seen_ids: set[str] = set()
    for path in paths:
        if not path.exists():
            console.print(f"[red]skip (missing): {path}[/red]")
            continue
        console.print(f"[dim]extract:[/dim] {path.name}")
        try:
            text, provider_used = await _extract_one(
                path, text_extensions=text_extensions, ocr_provider=ocr_provider
            )
        except Exception as exc:  # noqa: BLE001 -- keep the batch alive
            logger.exception("extraction failed: %s", path)
            console.print(f"[red]skip ({exc}): {path.name}[/red]")
            continue
        if not text.strip():
            console.print(f"[yellow]skip (empty extract): {path.name}[/yellow]")
            continue

        # Disambiguate duplicate slugs within one invocation so the
        # resume-by-doc_id contract holds across re-runs.
        doc_id = _slugify_doc_id(path.stem)
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


async def _refresh_centroid(aclient, name: str) -> None:
    """Force-refresh the routing centroid for the new collection."""
    store = CentroidStore(shared_root() / "centroids")
    try:
        await maybe_refresh(aclient, name, store, force=True)
        console.print(f"[green]centroid refreshed: {name}[/green]")
    except Exception as exc:  # noqa: BLE001 -- best-effort; router degrades to rule-based
        console.print(f"[yellow]centroid refresh failed ({exc})[/yellow]")


def _build_yaml_snippet(args: argparse.Namespace) -> str:
    """Render the YAML entry for ``system_rag.collections``.

    Point counts intentionally aren't part of the snippet -- ``corpora
    list`` queries Qdrant live, so a stale yaml number would just
    mislead. The yaml carries routing metadata; the database is the
    source of truth for size.
    """
    # An empty topics list is legitimate now that the centroid router
    # is active -- render it inline (``topics: []``) instead of leaving
    # a dangling ``topics:`` key with no children, which downstream
    # YAML readers would coerce to ``None``.
    if args.topic:
        topics_field = "      topics:\n" + "\n".join(
            f"        - {t}" for t in args.topic
        )
    else:
        topics_field = "      topics: []"
    license_line = (
        "      license: null"
        if args.license is None
        else f'      license: "{args.license}"'
    )
    # ``source_uri_prefix`` is populated by ``_resolve_metadata``;
    # fall back to ``None`` so the helper stays callable in isolation
    # from unit tests that construct an args namespace by hand.
    source_uri_prefix = getattr(args, "source_uri_prefix", None)
    source_uri_line = (
        "      source_uri_prefix: null"
        if source_uri_prefix is None
        else f'      source_uri_prefix: "{source_uri_prefix}"'
    )
    return (
        f"    - name: {args.name}\n"
        f"      language: {args.language}\n"
        f"      cross_lingual: {'true' if args.cross_lingual else 'false'}\n"
        f"      authority_tier: {args.authority_tier}\n"
        f"{topics_field}\n"
        f"      disease_codes: []\n"
        f"{source_uri_line}\n"
        f"{license_line}"
    )


def _print_yaml_snippet(args: argparse.Namespace) -> None:
    """Print a YAML snippet for ``configs/retrieval.yaml``.

    The router only includes collections declared in
    ``system_rag.collections`` -- printing this is the explicit handoff
    to the operator so config changes stay reviewable.
    """
    snippet = _build_yaml_snippet(args)
    console.print()
    console.print(
        "[bold]Next step:[/bold] add (or update) this entry in "
        "[cyan]configs/retrieval.yaml[/cyan] under "
        "[cyan]system_rag.collections[/cyan]:"
    )
    console.print(snippet)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="init_system_rag",
        description=(
            "Initialise a new system RAG collection from one or more "
            "documents (reuses the /upload OCR pipeline)."
        ),
    )
    p.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="One or more document files (PDF, image, txt, md, ...)",
    )
    p.add_argument(
        "--name",
        required=True,
        help=(
            f"Qdrant collection name. Must match {_NAME_RE.pattern}. "
            "Once data is written this string becomes the on-disk index "
            "name -- renaming requires a data migration."
        ),
    )
    p.add_argument(
        "--topic",
        action="append",
        default=[],
        metavar="TOPIC",
        help=(
            "Repeatable; sets system_rag.collections[].topics. Optional: "
            "the active centroid_classifier router ignores topics. Kept "
            "as an operator-readable label and a rule-based-fallback "
            "tie-breaker. On append, inherited from yaml if omitted."
        ),
    )
    p.add_argument(
        "--language",
        choices=("en", "zh"),
        default=None,
        help=(
            "Collection language. Defaults to 'en' for a new collection; "
            "inherited from configs/retrieval.yaml when --name matches "
            "an existing entry (append mode)."
        ),
    )
    p.add_argument(
        "--cross-lingual",
        action="store_true",
        help="Mark the collection as cross-lingual for the router.",
    )
    p.add_argument(
        "--authority-tier",
        type=int,
        choices=(1, 2, 3),
        default=None,
        help=(
            "1=top-tier (guidelines), 2=textbook, 3=other. Defaults to 2 "
            "for a new collection; inherited from yaml when appending."
        ),
    )
    p.add_argument(
        "--license",
        default=None,
        help="License string written to the printed YAML snippet.",
    )
    p.add_argument(
        "--user",
        default=None,
        help="Admin user id (defaults to ContextVar / first registered user).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of input files processed (smoke test).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="OCR + chunk only; skip Qdrant write and centroid refresh.",
    )
    p.add_argument(
        "--dedupe-cosine-threshold",
        type=float,
        default=0.0,
        metavar="FLOAT",
        help=(
            "Per-chunk cosine-similarity dedup against the target collection. "
            "0.92-0.95 is the typical range (matches /upload's default). "
            "<= 0 disables (default) -- recommended when appending docs that "
            "may overlap an existing edition / guideline."
        ),
    )
    args = p.parse_args(argv)
    if not _NAME_RE.match(args.name):
        p.error(f"--name must match {_NAME_RE.pattern}")
    # --topic is deferred to `_resolve_metadata` so append-mode can
    # inherit topics from the existing yaml entry without re-typing.
    return args


def _resolve_metadata(args: argparse.Namespace) -> None:
    """Fill missing metadata fields from configs/retrieval.yaml.

    Two modes, switched on whether ``args.name`` already exists in
    ``system_rag.collections``:

    * **Append** (existing entry) -- inherit topics / language /
      authority_tier / cross_lingual / license / source_uri_prefix from
      the yaml. Explicit CLI flags still win, but each disagreement is
      surfaced as a yellow warning so the printed snippet doesn't
      silently downgrade live config.
    * **New** (no entry yet) -- apply the historical defaults (en,
      tier 2) for unset fields. ``--topic`` is *optional*: the active
      ``centroid_classifier`` router scores by query-vector cosine
      against precomputed centroids, so topics no longer drive
      selection. They survive only as an operator-readable label and
      as a tie-breaker inside the rule-based fallback (which
      ``_topic_score`` already neutralises -- an empty topic list
      scores 1.0, i.e. never punished).

    ``--cross-lingual`` uses ``store_true``, so ``False`` is
    indistinguishable from "unspecified". On append we treat ``False``
    as unspecified and let a yaml ``true`` survive -- otherwise every
    append would silently flip cross-lingual collections off.
    """
    cfg = load_retrieval_config()
    existing = next(
        (c for c in cfg.system_rag.collections if c.name == args.name),
        None,
    )
    if existing is None:
        if args.language is None:
            args.language = "en"
        if args.authority_tier is None:
            args.authority_tier = 2
        args.source_uri_prefix = None
        return

    if not args.topic:
        args.topic = list(existing.topics)
    elif list(args.topic) != list(existing.topics):
        console.print(
            "[yellow]warning:[/yellow] --topic overrides yaml "
            f"({list(existing.topics)!r} -> {list(args.topic)!r}); "
            "review the printed snippet before pasting."
        )
    if args.language is None:
        args.language = existing.language
    elif args.language != existing.language:
        console.print(
            f"[yellow]warning:[/yellow] --language {args.language!r} "
            f"differs from yaml ({existing.language!r})."
        )
    if args.authority_tier is None:
        args.authority_tier = existing.authority_tier
    elif args.authority_tier != existing.authority_tier:
        console.print(
            f"[yellow]warning:[/yellow] --authority-tier "
            f"{args.authority_tier} differs from yaml "
            f"({existing.authority_tier})."
        )
    if not args.cross_lingual and existing.cross_lingual:
        args.cross_lingual = True
    if args.license is None:
        args.license = existing.license
    elif args.license != existing.license:
        console.print(
            f"[yellow]warning:[/yellow] --license overrides yaml "
            f"({existing.license!r} -> {args.license!r})."
        )
    args.source_uri_prefix = existing.source_uri_prefix


async def _amain(args: argparse.Namespace) -> int:
    paths = list(args.paths)
    if args.limit is not None:
        paths = paths[: args.limit]

    docs = await _ocr_files(paths, collection_name=args.name, language=args.language)
    if not docs:
        console.print("[red]no documents to ingest[/red]")
        return 1

    source = _GenericSource(args.name, docs)
    chunker = build_chunker()
    embedder = _NoOpEmbedder() if args.dry_run else build_embedder()
    cfg = load_retrieval_config()
    aclient = build_qdrant_client(
        url=cfg.qdrant.url, api_key_env=cfg.qdrant.api_key_env
    )
    store = RagCollectionStore(
        aclient=aclient,
        collection_name=args.name,
        dense_dim=embedder.dimension,
    )
    parent_store = ParentStore(shared_parent_docstore_path())

    stats = await ingest_corpus(
        source,
        chunker=chunker,
        embedder=embedder,
        store=store,
        parent_store=parent_store,
        dry_run=args.dry_run,
        dedupe_threshold=args.dedupe_cosine_threshold,
    )
    console.print(
        f"[green]{stats.source}: {stats.docs_processed} docs / "
        f"{stats.parents_written} parents / {stats.children_written} "
        f"children (skipped {stats.docs_skipped}, "
        f"resumed {stats.docs_resumed}, deduped {stats.children_deduped})[/green]"
    )
    if not args.dry_run and stats.children_written > 0:
        await _refresh_centroid(aclient, args.name)
    _print_yaml_snippet(args)
    return 0


def main(argv: list[str] | None = None) -> int:
    bootstrap_once()
    args = _parse_args(argv)
    # Resolve metadata before the admin gate so config errors (e.g.
    # missing --topic on a new collection) surface immediately without
    # touching the user store.
    _resolve_metadata(args)
    with inject_context(
        user_id=args.user,
        command=f"init_system_rag name={args.name!r}",
        check_user_exists=True,
    ):
        require_admin()
        return asyncio.run(_amain(args))


if __name__ == "__main__":
    sys.exit(main())
