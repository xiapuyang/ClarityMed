#!/usr/bin/env python3
"""Initialize a new system RAG collection from one or more documents.

Thin CLI wrapper around :func:`claritymed.ingest.system_rag.ingest_system_rag`
— the same shared pipeline that the admin SPA's
``POST /admin/rag/collections/upsert`` endpoint runs.

Reuses the same OCR pipeline that ``/upload`` invokes
(:func:`claritymed.core.ocr.factory.make_ocr_provider`) to extract text
from each input file, then runs ``ingest_corpus`` -- the same chunk +
embed + Qdrant upsert pipeline that ``statpearls`` and ``textbooks`` use
-- against a fresh, named collection.

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

    uv run python scripts/init_system_rag.py \\
        --name ats_idsa_pneumonia_guidelines_en \\
        --topic "community-acquired pneumonia" \\
        --topic "respiratory infections" \\
        --language en \\
        --authority-tier 1 \\
        --user admin \\
        "data/download/Diagnosis and Treatment of Adults with Community-acquired Pneumonia.pdf"

After ingest, the script prints a YAML snippet for ``configs/retrieval.yaml``.
The new collection is invisible to the router until that snippet is
appended under ``system_rag.collections`` -- intentionally manual so
config changes stay reviewable when invoked from the CLI. The admin
SPA endpoint auto-appends instead.

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
import sys
from pathlib import Path

from claritymed.cli.common import bootstrap_once, console
from claritymed.cli.entry import inject_context
from claritymed.ingest.system_rag import (
    _NAME_RE,
    SystemRagIngestRequest,
    build_yaml_snippet,
    ingest_system_rag,
    resolve_metadata,
)
from claritymed.stores.account import require_admin

logger = logging.getLogger(__name__)


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
    return args


def _request_from_args(args: argparse.Namespace) -> SystemRagIngestRequest:
    return SystemRagIngestRequest(
        name=args.name,
        files=list(args.paths),
        topics=list(args.topic),
        language=args.language,
        cross_lingual=args.cross_lingual,
        authority_tier=args.authority_tier,
        license=args.license,
        dedupe_cosine_threshold=args.dedupe_cosine_threshold,
        dry_run=args.dry_run,
        limit=args.limit,
    )


def _warn_on_yaml_overrides(
    cli_args: argparse.Namespace, resolved: SystemRagIngestRequest
) -> None:
    """Print yellow warnings when CLI explicit values differ from yaml.

    The shared resolver silently lets CLI values win — that's the right
    default for the web flow (the SPA already shows the merged form),
    but the CLI operator should see when their flags override yaml so
    the printed snippet doesn't downgrade live config without notice.
    """
    from claritymed.core.rag import load_retrieval_config

    cfg = load_retrieval_config()
    existing = next(
        (c for c in cfg.system_rag.collections if c.name == resolved.name),
        None,
    )
    if existing is None:
        return
    if cli_args.topic and list(cli_args.topic) != list(existing.topics):
        console.print(
            "[yellow]warning:[/yellow] --topic overrides yaml "
            f"({list(existing.topics)!r} -> {list(cli_args.topic)!r})."
        )
    if cli_args.language is not None and cli_args.language != existing.language:
        console.print(
            f"[yellow]warning:[/yellow] --language {cli_args.language!r} "
            f"differs from yaml ({existing.language!r})."
        )
    if (
        cli_args.authority_tier is not None
        and cli_args.authority_tier != existing.authority_tier
    ):
        console.print(
            f"[yellow]warning:[/yellow] --authority-tier "
            f"{cli_args.authority_tier} differs from yaml "
            f"({existing.authority_tier})."
        )
    if cli_args.license is not None and cli_args.license != existing.license:
        console.print(
            f"[yellow]warning:[/yellow] --license overrides yaml "
            f"({existing.license!r} -> {cli_args.license!r})."
        )


def _print_yaml_snippet(snippet: str) -> None:
    """Print a YAML snippet for ``configs/retrieval.yaml``.

    The CLI prints the snippet for manual paste; the admin SPA endpoint
    appends it programmatically. Keeping the CLI manual preserves the
    reviewable-config posture that operators expect from a shell tool.
    """
    console.print()
    console.print(
        "[bold]Next step:[/bold] add (or update) this entry in "
        "[cyan]configs/retrieval.yaml[/cyan] under "
        "[cyan]system_rag.collections[/cyan]:"
    )
    console.print(snippet)


async def _amain(args: argparse.Namespace) -> int:
    req = _request_from_args(args)
    _warn_on_yaml_overrides(args, req)
    try:
        result = await ingest_system_rag(
            req,
            on_progress=lambda line: console.print(f"[dim]{line}[/dim]"),
        )
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    stats = result.stats
    console.print(
        f"[green]{stats.source}: {stats.docs_processed} docs / "
        f"{stats.parents_written} parents / {stats.children_written} "
        f"children (skipped {stats.docs_skipped}, "
        f"resumed {stats.docs_resumed}, deduped {stats.children_deduped})[/green]"
    )
    if result.centroid_refreshed:
        console.print(f"[green]centroid refreshed: {req.name}[/green]")
    _print_yaml_snippet(result.yaml_snippet)
    return 0


def main(argv: list[str] | None = None) -> int:
    bootstrap_once()
    args = _parse_args(argv)
    # Resolve metadata *outside* the admin gate so config errors (e.g.
    # an invalid --name) surface immediately without touching the user
    # store. The shared pipeline calls resolve_metadata() again inside
    # ingest_system_rag(); the call is idempotent.
    req = _request_from_args(args)
    resolve_metadata(req)
    # Hydrate the CLI namespace from the resolved request so legacy
    # tests asserting on argparse fields still pass.
    args.topic = list(req.topics)
    args.language = req.language
    args.cross_lingual = req.cross_lingual
    args.authority_tier = req.authority_tier
    args.license = req.license
    args.source_uri_prefix = req.source_uri_prefix
    with inject_context(
        user_id=args.user,
        command=f"init_system_rag name={args.name!r}",
        check_user_exists=True,
    ):
        require_admin()
        return asyncio.run(_amain(args))


def _build_yaml_snippet(args: argparse.Namespace) -> str:
    """Legacy CLI wrapper around :func:`build_yaml_snippet`.

    Kept so the existing unit tests (which build argparse Namespaces by
    hand) continue to assert on the same output.
    """
    req = SystemRagIngestRequest(
        name=args.name,
        files=[],
        topics=list(getattr(args, "topic", []) or []),
        language=getattr(args, "language", None) or "en",
        cross_lingual=bool(getattr(args, "cross_lingual", False)),
        authority_tier=int(getattr(args, "authority_tier", 2) or 2),
        license=getattr(args, "license", None),
        source_uri_prefix=getattr(args, "source_uri_prefix", None),
    )
    return build_yaml_snippet(req)


def _resolve_metadata(args: argparse.Namespace) -> None:
    """Legacy CLI wrapper around :func:`resolve_metadata`.

    The shared resolver silently lets CLI values win; the CLI's warning
    behaviour is in :func:`_warn_on_yaml_overrides`. This wrapper
    preserves the old warning side-effect so tests that capture stdout
    still see the yellow text.

    Builds a metadata-only request (no ``files`` / ``paths`` needed) so
    legacy callers can pass argparse Namespaces that haven't been
    through the full CLI ``_parse_args`` flow.
    """
    req = SystemRagIngestRequest(
        name=args.name,
        files=[],
        topics=list(getattr(args, "topic", []) or []),
        language=getattr(args, "language", None),
        cross_lingual=bool(getattr(args, "cross_lingual", False)),
        authority_tier=getattr(args, "authority_tier", None),
        license=getattr(args, "license", None),
    )
    _warn_on_yaml_overrides(args, req)
    resolve_metadata(req)
    args.topic = list(req.topics)
    args.language = req.language
    args.cross_lingual = req.cross_lingual
    args.authority_tier = req.authority_tier
    args.license = req.license
    args.source_uri_prefix = req.source_uri_prefix


if __name__ == "__main__":
    sys.exit(main())
