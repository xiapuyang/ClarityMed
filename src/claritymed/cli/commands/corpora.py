"""``claritymed rag corpora`` — admin commands for system RAG corpora.

Three concerns live here:

* The ``corpora`` Typer sub-app (``list``, ``ingest``, ``refresh-centroid``,
  ``migrate-payload``) — registered under ``rag`` by ``cli.commands.rag``.
* The startup hook that refreshes routing centroids before the TUI takes
  over the terminal — called from ``cli.commands.tui`` so the first query
  inside the TUI never races a half-built centroid file.
* The ``_NoOpEmbedder`` dry-run stand-in used by ``corpora ingest``.

All commands require admin role and run synchronously; the per-command
async bodies live inside each function so they never start a loop
outside the Typer call site.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from claritymed.cli.common import console, run_async
from claritymed.cli.entry import inject_context

# Cap concurrent Qdrant calls during the TUI-startup centroid refresh.
# Each task issues one ``count`` + one ``scroll(limit=500)`` against
# Qdrant; 4 in flight keeps the server unsaturated while still
# parallelising a small (~2-10 collection) catalog. Raise only if Qdrant
# + network can clearly handle more concurrent scrolls.
_CENTROID_REFRESH_CONCURRENCY = 4


corpora_app = typer.Typer(help="Manage system RAG corpora (admin).")


@corpora_app.command("list")
def corpora_list() -> None:
    """List system corpora declared in ``configs/retrieval.yaml``.

    ``size=`` is queried live from Qdrant (``aclient.count(exact=True)``)
    so it never drifts from on-disk truth -- the yaml carries metadata,
    not a counter. A missing collection or Qdrant outage renders the
    cell as ``?`` rather than failing the listing.
    """
    from claritymed.core.rag.qdrant_store import build_qdrant_client
    from claritymed.core.rag.schemas import load_retrieval_config

    cfg = load_retrieval_config()
    collections = list(cfg.system_rag.collections)

    async def _live_counts() -> dict[str, int | None]:
        counts: dict[str, int | None] = {c.name: None for c in collections}
        if not collections:
            return counts
        aclient = build_qdrant_client(
            url=cfg.qdrant.url, api_key_env=cfg.qdrant.api_key_env
        )
        try:
            for c in collections:
                try:
                    if not await aclient.collection_exists(c.name):
                        continue
                    info = await aclient.count(c.name, exact=True)
                    counts[c.name] = int(info.count)
                except Exception:  # noqa: BLE001 — single-collection failures shouldn't blank the list
                    counts[c.name] = None
        finally:
            await aclient.close()
        return counts

    counts = run_async(_live_counts())
    for c in collections:
        size_repr = "?" if counts.get(c.name) is None else str(counts[c.name])
        console.print(
            f"  [bold]{c.name}[/bold]  lang={c.language}  tier={c.authority_tier}  "
            f"size={size_repr}  topics={c.topics}"
        )


@corpora_app.command("ingest")
def corpora_ingest(
    name: str = typer.Argument(..., help="Corpus name (e.g. statpearls)"),
    user: str | None = typer.Option(None, "--user", "-u"),
    limit: int | None = typer.Option(None, "--limit", help="Max docs to ingest"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Parse + chunk only"),
    raw_dir: str | None = typer.Option(
        None, "--raw-dir", help="Override raw corpus path"
    ),
) -> None:
    """Ingest a system corpus into Qdrant + parent docstore.

    Admin only. Run from the host once the raw download is in place.
    """
    from claritymed.core.rag import load_retrieval_config
    from claritymed.core.rag.chunking.factory import build_chunker
    from claritymed.core.rag.embedding.factory import build_embedder
    from claritymed.core.rag.parent_store import ParentStore
    from claritymed.core.rag.qdrant_store import RagCollectionStore, build_qdrant_client
    from claritymed.ingest.corpus.base import ingest_corpus
    from claritymed.ingest.corpus.statpearls import StatPearlsSource
    from claritymed.ingest.corpus.textbooks import TextbooksSource
    from claritymed.stores.account import require_admin
    from claritymed.stores.paths import (
        shared_knowledge_raw_dir,
        shared_parent_docstore_path,
    )

    _KNOWN_CORPORA = {"statpearls", "textbooks"}

    with inject_context(
        user_id=user, command=f"corpora.ingest name={name!r}", check_user_exists=True
    ) as (
        _,
        _uid,
        _,
    ):
        require_admin()
        if name not in _KNOWN_CORPORA:
            console.print(f"[red]Unknown corpus: {name}[/red]")
            console.print(f"[dim]Available: {', '.join(sorted(_KNOWN_CORPORA))}[/dim]")
            raise typer.Exit(code=2)

        root = Path(raw_dir) if raw_dir else shared_knowledge_raw_dir() / name
        if name == "statpearls":
            source = StatPearlsSource(root)
        else:
            source = TextbooksSource(root)
        chunker = build_chunker()
        embedder = build_embedder() if not dry_run else _NoOpEmbedder()
        cfg = load_retrieval_config()
        aclient = build_qdrant_client(
            url=cfg.qdrant.url,
            api_key_env=cfg.qdrant.api_key_env,
        )
        store = RagCollectionStore(
            aclient=aclient,
            collection_name=source.name,
            dense_dim=embedder.dimension,
        )
        parent_store = ParentStore(shared_parent_docstore_path())

        async def _run() -> None:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            # Try to size the bar. StatPearls source is filesystem-backed;
            # counting jsonl files is cheap. Falls back to indeterminate
            # mode (no ETA) when source can't be sized in advance.
            try:
                total = sum(1 for _ in source.iter_raw_docs())
            except Exception:  # noqa: BLE001 — best-effort sizing only
                total = None
            if limit is not None:
                total = min(total, limit) if total else limit

            columns = [
                TextColumn("[bold]{task.fields[name]}[/bold]"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn(
                    "[dim]{task.fields[parents]}p / {task.fields[children]}c[/dim]"
                ),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ]
            with Progress(*columns, console=console, transient=False) as bar:
                task_id = bar.add_task(
                    "ingest",
                    total=total,
                    name=source.name,
                    parents=0,
                    children=0,
                )

                def _on_doc(s) -> None:
                    # Bar advances on both new + resumed docs so it
                    # tracks "docs reached" not just "docs embedded";
                    # otherwise a fully-resumed run shows the bar
                    # frozen at 0 even though work is happening.
                    bar.update(
                        task_id,
                        completed=s.docs_processed + s.docs_resumed,
                        parents=s.parents_written,
                        children=s.children_written,
                    )

                stats = await ingest_corpus(
                    source,
                    chunker=chunker,
                    embedder=embedder,
                    store=store,
                    parent_store=parent_store,
                    limit=limit,
                    dry_run=dry_run,
                    on_doc=_on_doc,
                )

            console.print(
                f"[green]{stats.source}: {stats.docs_processed} docs / "
                f"{stats.parents_written} parents / {stats.children_written} "
                f"children (skipped {stats.docs_skipped}, "
                f"resumed {stats.docs_resumed})[/green]"
            )

            if not dry_run and stats.children_written > 0:
                await _refresh_centroid_for(aclient, source.name, console)

        run_async(_run())


async def _refresh_centroid_for(aclient, collection_name: str, con) -> None:
    """Compute/refresh the centroid for one system collection."""
    from claritymed.core.rag.routing.centroid_store import CentroidStore, maybe_refresh
    from claritymed.stores.paths import shared_root

    store = CentroidStore(shared_root() / "centroids")
    try:
        refreshed = await maybe_refresh(aclient, collection_name, store)
        if refreshed:
            con.print(f"[dim]centroid refreshed: {collection_name}[/dim]")
        else:
            con.print(f"[dim]centroid up-to-date: {collection_name}[/dim]")
    except Exception as exc:
        con.print(f"[yellow]centroid refresh failed ({exc}); continuing[/yellow]")


def refresh_system_centroids_on_startup() -> None:
    """Synchronously refresh every system-RAG centroid before launching the TUI.

    No-op unless ``rag.enabled`` is true AND the active router is
    ``centroid_classifier`` — otherwise centroids are never consulted at
    query time. For each collection, ``maybe_refresh`` covers both
    "centroid file missing" and "delta >= threshold" with one call.

    Per-collection exceptions are caught so a flaky Qdrant doesn't block
    the TUI launch — the centroid router degrades gracefully to the rule-
    based fallback when a centroid is absent.
    """
    from claritymed.core.rag import load_retrieval_config
    from claritymed.core.rag.qdrant_store import build_qdrant_client
    from claritymed.core.rag.routing.centroid_store import CentroidStore, maybe_refresh
    from claritymed.stores.paths import shared_root

    cfg = load_retrieval_config()
    if not cfg.rag.enabled:
        return
    if cfg.router.resolved().id != "centroid_classifier":
        return

    collections = list(cfg.system_rag.collections)
    if not collections:
        return

    console.print(
        f"[dim]refreshing centroids for {len(collections)} system collection(s) "
        f"(concurrency={_CENTROID_REFRESH_CONCURRENCY})…[/dim]"
    )

    async def _run() -> None:
        aclient = build_qdrant_client(
            url=cfg.qdrant.url,
            api_key_env=cfg.qdrant.api_key_env,
        )
        store = CentroidStore(shared_root() / "centroids")
        sem = asyncio.Semaphore(_CENTROID_REFRESH_CONCURRENCY)

        async def _one(name: str) -> None:
            async with sem:
                try:
                    refreshed = await maybe_refresh(aclient, name, store)
                except Exception as exc:  # noqa: BLE001
                    console.print(
                        f"[yellow]centroid refresh failed for {name} "
                        f"({exc}); continuing[/yellow]"
                    )
                    return
                if refreshed:
                    console.print(f"[green]centroid refreshed: {name}[/green]")
                else:
                    console.print(f"[dim]centroid up-to-date: {name}[/dim]")

        await asyncio.gather(*(_one(meta.name) for meta in collections))

    run_async(_run())


@corpora_app.command("refresh-centroid")
def corpora_refresh_centroid(
    name: str = typer.Argument(..., help="Corpus name (e.g. statpearls)"),
    user: str | None = typer.Option(None, "--user", "-u"),
) -> None:
    """Recompute the routing centroid for an existing system collection.

    Run this after bulk ingests or when adding a new collection to ensure
    the embedding-based router has an up-to-date centroid. Safe to re-run
    at any time — always forces a recompute regardless of delta.
    """
    from claritymed.core.rag import load_retrieval_config
    from claritymed.core.rag.qdrant_store import build_qdrant_client
    from claritymed.core.rag.routing.centroid_store import CentroidStore, maybe_refresh
    from claritymed.ingest.corpus.statpearls import StatPearlsSource
    from claritymed.ingest.corpus.textbooks import TextbooksSource
    from claritymed.stores.paths import shared_knowledge_raw_dir, shared_root

    _CORPUS_SOURCES = {
        "statpearls": lambda: StatPearlsSource(
            shared_knowledge_raw_dir() / "statpearls"
        ),
        "textbooks": lambda: TextbooksSource(shared_knowledge_raw_dir() / "textbooks"),
    }

    with inject_context(
        user_id=user,
        command=f"corpora.refresh-centroid name={name!r}",
        check_user_exists=True,
    ):
        if name not in _CORPUS_SOURCES:
            console.print(f"[red]Unknown corpus: {name!r}[/red]")
            console.print(f"[dim]Available: {', '.join(sorted(_CORPUS_SOURCES))}[/dim]")
            raise typer.Exit(code=2)

        collection_name = _CORPUS_SOURCES[name]().name
        cfg = load_retrieval_config()
        aclient = build_qdrant_client(
            url=cfg.qdrant.url,
            api_key_env=cfg.qdrant.api_key_env,
        )
        store = CentroidStore(shared_root() / "centroids")

        async def _run() -> None:
            try:
                await maybe_refresh(aclient, collection_name, store, force=True)
                console.print(f"[green]centroid refreshed: {collection_name}[/green]")
            except Exception as exc:
                console.print(f"[red]failed: {exc}[/red]")
                raise typer.Exit(code=1) from exc

        run_async(_run())


class _NoOpEmbedder:
    """Dry-run embedder stand-in; advertises a dimension but never called."""

    @property
    def dimension(self) -> int:
        return 1024

    async def embed_dense(self, texts):
        return [[0.0] * 1024 for _ in texts]

    async def embed_sparse(self, texts):
        return [{} for _ in texts]


@corpora_app.command("migrate-payload")
def corpora_migrate_payload(
    name: str = typer.Argument(..., help="Corpus name (e.g. statpearls)"),
    user: str | None = typer.Option(None, "--user", "-u"),
) -> None:
    """Patch Qdrant payloads in-place — no re-embedding needed.

    Fixes metadata fields on existing points (e.g. source_uri format,
    adding doc_title) without touching vectors. Use this after an ingest
    that left stale payload values; much faster than a full re-ingest.
    """
    from claritymed.core.rag import load_retrieval_config
    from claritymed.core.rag.qdrant_store import build_qdrant_client
    from claritymed.stores.account import require_admin

    with inject_context(
        user_id=user,
        command=f"corpora.migrate-payload name={name!r}",
        check_user_exists=True,
    ) as (_, _uid, _):
        require_admin()
        if name != "statpearls":
            console.print(f"[red]Unknown corpus: {name}[/red]")
            raise typer.Exit(code=2)

        from claritymed.ingest.corpus.statpearls import (
            COLLECTION_NAME,
            _nbk_uri,
        )

        cfg = load_retrieval_config()
        aclient = build_qdrant_client(
            url=cfg.qdrant.url,
            api_key_env=cfg.qdrant.api_key_env,
        )

        async def _run() -> None:
            from qdrant_client.models import PointIdsList
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TaskProgressColumn,
                TextColumn,
                TimeElapsedColumn,
            )

            # Count total points upfront for a determinate bar.
            count_result = await aclient.count(
                collection_name=COLLECTION_NAME, exact=True
            )
            total = count_result.count

            patched = 0
            scanned = 0
            offset = None

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TextColumn("[cyan]{task.fields[patched]} patched"),
                console=console,
                transient=False,
            ) as bar:
                task_id = bar.add_task(
                    f"migrate {COLLECTION_NAME}",
                    total=total,
                    patched=0,
                )
                while True:
                    results, next_offset = await aclient.scroll(
                        collection_name=COLLECTION_NAME,
                        with_payload=True,
                        with_vectors=False,
                        limit=256,
                        offset=offset,
                    )
                    if not results:
                        break

                    for point in results:
                        payload = dict(point.payload or {})
                        doc_id = payload.get("doc_id", "")
                        src_uri = payload.get("source_uri")
                        new_payload: dict = {}
                        if src_uri and "/article-" in src_uri:
                            new_payload["source_uri"] = _nbk_uri(doc_id)
                        if "doc_title" not in payload and payload.get("title"):
                            new_payload["doc_title"] = payload["title"]
                        if new_payload:
                            await aclient.set_payload(
                                collection_name=COLLECTION_NAME,
                                payload=new_payload,
                                points=PointIdsList(points=[point.id]),
                            )
                            patched += 1

                    scanned += len(results)
                    bar.update(task_id, completed=scanned, patched=patched)

                    if next_offset is None:
                        break
                    offset = next_offset

            console.print(
                f"[green]Done — scanned {scanned} points, patched {patched}.[/green]"
            )

        asyncio.run(_run())
