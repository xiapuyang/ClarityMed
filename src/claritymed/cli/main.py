"""Typer CLI entry point: ``claritymed ask|ingest|rag``.

The CLI is the headless surface that mirrors the TUI's actions one-shot.
Both go through the same service layer — the CLI never touches an Agent
directly so any service-level change (audit, scrub, event schema) reaches
both surfaces with one edit.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer
from rich.console import Console

from claritymed import config as _cfg
from claritymed.cli.entry import inject_context
from claritymed.core.llm.model import build_model
from claritymed.core.observability.logging import setup_logging
from claritymed.orchestrator.services import (
    AskService,
    Done,
    Error,
    IngestService,
    RagService,
    TokenChunk,
    ToolCompleted,
    ToolStarted,
)
from claritymed.stores.models import load_models, resolve_provider
from claritymed.stores.user_rag import make_user_rag_store

app = typer.Typer(
    name="claritymed",
    help="ClarityMed CLI — ingest / ask / rag (headless mirror of the TUI).",
    no_args_is_help=True,
)
console = Console()

_BOOTSTRAPPED = False

# File extensions that require OCR conversion before ingestion.
_OCR_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
}


def _bootstrap_once() -> None:
    """Idempotent CLI bootstrap: load ~/.claritymed/.env then init the three loggers.

    Called from every subcommand. The first call wins; subsequent calls in the
    same process are no-ops. Tests that need a fresh state can re-enter via
    ``setup_logging`` directly.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    _cfg.load_env_file()
    setup_logging(script_name="claritymed", console_level=None)
    _BOOTSTRAPPED = True


@app.callback()
def _cli_root() -> None:
    """Root callback — runs before every subcommand."""
    _bootstrap_once()


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _prefetch_models() -> None:
    """Download in-process model weights to HF cache before the TUI starts.

    Fails fast if ``privacy_filter.enabled=true`` but the required packages
    are not installed — a missing package is a misconfiguration, not a
    recoverable runtime condition.
    """
    from claritymed.core.scrub.service import ScrubService

    svc = ScrubService.from_config()
    if not svc._config.privacy_filter.enabled:
        return
    try:
        svc.check_runtime_deps()
    except ImportError as exc:
        print(f"startup error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    model = svc._config.privacy_filter.model_name
    print(f"Checking model cache: {model}")
    ok = svc.ensure_downloaded()
    if ok:
        print(f"  ✓ {model} ready")
    else:
        print(f"  ✗ {model} download failed", file=sys.stderr)
        raise SystemExit(1)


def _run_async(coro):
    return asyncio.run(coro)


def _maybe_build_strategy():
    """Return a RagStrategy when ``rag.enabled=true``, else ``None``.

    Lazy-imports the retrieval factory so a disabled-RAG ``ask`` does not
    pay the import cost of qdrant / llama-index. Fail-loud once enabled —
    a missing dependency raises rather than silently degrading to LLM-only.
    """
    from claritymed.core.rag import load_retrieval_config

    cfg = load_retrieval_config()
    if not cfg.rag.enabled:
        return None
    from claritymed.core.rag import build_hybrid_retriever
    from claritymed.core.rag.strategies import build_strategy

    retriever = build_hybrid_retriever(cfg)
    return build_strategy(
        retriever,
        config=cfg.strategies,
        max_evidence=cfg.rag.max_evidence,
    )


# --- ask ----------------------------------------------------------------


@app.command()
def ask(
    question: str = typer.Argument(..., help="The medical question to ask."),
    user: str | None = typer.Option(None, "--user", "-u"),
    language: str | None = typer.Option(None, "--lang", "-l"),
    provider_id: str | None = typer.Option(None, "--provider", "-p"),
) -> None:
    """Stream a grounded answer to ``question``."""

    async def _run() -> None:
        from claritymed.orchestrator.services import ChatSession

        with inject_context(
            user_id=user,
            language=language,
            command=f"ask q={question[:60]!r}",
            check_user_exists=True,
        ) as (_, uid, lang):
            provider = resolve_provider(override=provider_id)
            model = build_model(provider)
            from claritymed.core.translation import make_translation_provider

            strategy = _maybe_build_strategy()
            service = AskService(
                model=model,
                language=lang,
                chat_session=ChatSession.new(uid),
                provider_id=provider.id,
                model_name=provider.model,
                strategy=strategy,
                provider_config=provider,
                translation_service=make_translation_provider(model),
            )

            async for event in service.run(question, user_id=uid):
                if isinstance(event, TokenChunk):
                    console.print(event.text, end="")
                elif isinstance(event, Error):
                    _stderr(f"\n[error] {event.error_type}: {event.message}")
                    raise typer.Exit(code=1)
                elif isinstance(event, Done):
                    console.print()  # newline after streaming text

    _run_async(_run())


# --- ingest -------------------------------------------------------------


ingest_app = typer.Typer(help="Ingest personal info, history, or reports.")
app.add_typer(ingest_app, name="ingest")


@ingest_app.command("profile")
def ingest_profile(
    field: str = typer.Argument(..., help="key=value, e.g. allergy=penicillin"),
    user: str | None = typer.Option(None, "--user", "-u"),
    language: str | None = typer.Option(None, "--lang", "-l"),
) -> None:
    """Save a single profile field."""

    async def _run() -> None:
        with inject_context(
            user_id=user,
            language=language,
            command=f"ingest.profile {field!r}",
            check_user_exists=True,
        ) as (_, uid, _):
            service = IngestService()
            async for event in service.run(field, user_id=uid):
                if isinstance(event, ToolStarted):
                    console.print(f"[dim]→ {event.tool_name}[/dim]")
                elif isinstance(event, ToolCompleted):
                    console.print(f"[dim]✓ {event.tool_name}: {event.summary}[/dim]")
                elif isinstance(event, Done):
                    console.print(f"[green]{event.final.summary}[/green]")
                elif isinstance(event, Error):
                    _stderr(f"[error] {event.message}")
                    raise typer.Exit(code=1)

    _run_async(_run())


# --- rag ----------------------------------------------------------------


rag_app = typer.Typer(help="Manage reference materials (per-user RAG).")
app.add_typer(rag_app, name="rag")


@rag_app.command("add")
def rag_add(
    path: str = typer.Argument(..., help="Local file path to add."),
    user: str | None = typer.Option(None, "--user", "-u"),
    language: str | None = typer.Option(None, "--lang", "-l"),
    public: bool = typer.Option(
        False,
        "--public",
        help="Mark as public reference (skips PHI scrub, allows cloud).",
    ),
) -> None:
    """Add a document to the user's personal RAG store."""

    async def _run() -> None:
        with inject_context(
            user_id=user,
            language=language,
            command=f"rag.add path={path!r}",
            check_user_exists=True,
        ) as (_, uid, _):
            file_path = Path(path)
            if file_path.suffix.lower() in _OCR_EXTENSIONS:
                from claritymed.core.ocr.base import OcrError
                from claritymed.core.ocr.factory import make_ocr_provider

                ocr = make_ocr_provider()
                try:
                    text = await ocr.extract_text(file_path)
                except OcrError as exc:
                    _stderr(f"[error] OCR failed: {exc}")
                    raise typer.Exit(code=1) from exc
            else:
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
            from claritymed.errors import DuplicateDocumentError

            store = make_user_rag_store(uid)
            service = RagService(store=store)
            try:
                async for event in service.run(
                    text,
                    user_id=uid,
                    public=public,
                    source_uri=str(file_path.resolve()),
                ):
                    if isinstance(event, ToolStarted):
                        console.print(f"[dim]→ {event.tool_name}[/dim]")
                    elif isinstance(event, ToolCompleted):
                        console.print(
                            f"[dim]✓ {event.tool_name}: {event.summary}[/dim]"
                        )
                    elif isinstance(event, Done):
                        console.print(
                            f"[green]doc_id={event.final.doc_id} "
                            f"chunks={event.final.chunk_count}[/green]"
                        )
                    elif isinstance(event, Error):
                        _stderr(f"[error] {event.message}")
                        raise typer.Exit(code=1)
            except DuplicateDocumentError as exc:
                console.print(
                    f"[yellow]already indexed as {exc.existing_doc_id} — skipped[/yellow]"
                )

    _run_async(_run())


corpora_app = typer.Typer(help="Manage system RAG corpora (admin).")
rag_app.add_typer(corpora_app, name="corpora")


@corpora_app.command("list")
def corpora_list() -> None:
    """List system corpora declared in ``configs/retrieval.yaml``."""
    from claritymed.core.rag.schemas import load_retrieval_config

    cfg = load_retrieval_config()
    for c in cfg.system_rag.collections:
        console.print(
            f"  [bold]{c.name}[/bold]  lang={c.language}  tier={c.authority_tier}  "
            f"size={c.size_chunks}  topics={c.topics}"
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
    from pathlib import Path

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

        _run_async(_run())


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
    import asyncio

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


@rag_app.command("list")
def rag_list(
    user: str | None = typer.Option(None, "--user", "-u"),
) -> None:
    """List documents in the user's personal RAG store."""

    async def _run() -> None:
        with inject_context(
            user_id=user, command="rag.list", check_user_exists=True
        ) as (_, uid, _):
            store = make_user_rag_store(uid)
            docs = await store.list_documents(uid)
            if not docs:
                console.print("[dim]No documents.[/dim]")
                return
            from rich.table import Table

            table = Table(show_header=True, header_style="bold", box=None)
            table.add_column("doc_id", style="cyan", no_wrap=True)
            table.add_column("chunks", justify="right")
            table.add_column("phi", justify="center")
            table.add_column("ingested_at", style="dim", no_wrap=True)
            table.add_column("source", style="dim")
            table.add_column("preview")
            for d in docs:
                phi_mark = (
                    "[yellow]PHI[/yellow]" if d["is_phi"] else "[green]pub[/green]"
                )
                src = d.get("source_uri") or "—"
                table.add_row(
                    d["doc_id"],
                    str(d["chunk_count"]),
                    phi_mark,
                    (d["ingested_at"] or "")[:19],
                    src,
                    d["preview"].replace("\n", " "),
                )
            console.print(table)

    _run_async(_run())


@rag_app.command("show")
def rag_show(
    doc_id: str = typer.Argument(..., help="doc_id to inspect (from 'rag list')."),
    user: str | None = typer.Option(None, "--user", "-u"),
) -> None:
    """Print all chunks for a document in the user's RAG store."""

    async def _run() -> None:
        with inject_context(
            user_id=user, command=f"rag.show {doc_id}", check_user_exists=True
        ) as (_, uid, _):
            store = make_user_rag_store(uid)
            chunks = await store.get_chunks(uid, doc_id)
            if not chunks:
                console.print(f"[yellow]No chunks found for doc_id={doc_id!r}[/yellow]")
                return
            phi_label = (
                "[yellow]PHI[/yellow]"
                if chunks[0]["is_phi"]
                else "[green]public[/green]"
            )
            console.print(
                f"[bold]{doc_id}[/bold]  {phi_label}  {len(chunks)} chunk(s)\n"
            )
            for c in chunks:
                console.print(f"[dim]── chunk {c['chunk_index']} ──[/dim]")
                console.print(c["text"])
                console.print()

    _run_async(_run())


@rag_app.command("rm")
def rag_rm(
    doc_id: str = typer.Argument(..., help="doc_id to delete (from 'rag list')."),
    user: str | None = typer.Option(None, "--user", "-u"),
) -> None:
    """Delete a document from the user's personal RAG store."""

    async def _run() -> None:
        with inject_context(
            user_id=user, command=f"rag.rm {doc_id}", check_user_exists=True
        ) as (_, uid, _):
            store = make_user_rag_store(uid)
            await store.delete_document(uid, doc_id)
            console.print(f"[green]Deleted {doc_id}[/green]")

    _run_async(_run())


finetune_app = typer.Typer(help="Preprocess fine-tune corpora (NOT indexed in RAG).")
app.add_typer(finetune_app, name="finetune")


@finetune_app.command("preprocess")
def finetune_preprocess(
    name: str = typer.Argument(..., help="Corpus name (e.g. meddialog_cn)"),
    input_dir: str = typer.Option(..., "--input", help="Raw corpus directory"),
    output_dir: str = typer.Option(..., "--output", help="JSONL output directory"),
    limit: int | None = typer.Option(None, "--limit", help="Cap dialogs processed"),
) -> None:
    """Clean a fine-tune corpus into Alpaca-style JSONL splits.

    These corpora are NOT indexed in Qdrant — they are training material
    only. Output goes under ``data/finetune/<name>/`` by convention.
    """
    from pathlib import Path

    from claritymed.ingest.finetune.meddialog_cn import preprocess_meddialog_cn

    if name != "meddialog_cn":
        console.print(f"[red]Unknown fine-tune corpus: {name}[/red]")
        raise typer.Exit(code=2)

    stats = preprocess_meddialog_cn(Path(input_dir), Path(output_dir), limit=limit)
    console.print(
        f"[green]meddialog_cn: total={stats.total_dialogs} kept={stats.kept} "
        f"(no_answer={stats.dropped_no_answer} short={stats.dropped_short_answer} "
        f"long_q={stats.dropped_long_question} dedup={stats.dropped_dedup})[/green]"
    )
    console.print(
        f"  train={stats.written_train}  val={stats.written_val}  "
        f"test={stats.written_test}"
    )


@app.command()
def tui(
    user: str | None = typer.Option(None, "--user", "-u"),
    language: str | None = typer.Option(None, "--lang", "-l"),
    provider_id: str | None = typer.Option(None, "--provider", "-p"),
) -> None:
    """Launch the Textual TUI."""
    from claritymed.cli.tui import ClarityMedApp
    from claritymed.errors import UnknownProviderError

    # Resolve the provider up front so a typo (`--provider oMLX`) fails
    # cleanly to stderr instead of opening the TUI and exploding on the
    # first submit. Matches the project rule: provider resolution is loud,
    # never silent.
    try:
        provider = resolve_provider(override=provider_id)
    except UnknownProviderError as exc:
        valid = ", ".join(p.id for p in load_models().providers)
        _stderr(f"[error] {exc}")
        _stderr(f"  valid provider ids: {valid}")
        raise typer.Exit(code=1) from exc

    from claritymed.cli.entry import _resolve_language, _resolve_user_id

    resolved_lang = _resolve_language(language)
    resolved_user, _ = _resolve_user_id(user)

    # Pre-download in-process models before the TUI takes over the terminal.
    # Currently only openai/privacy-filter (BGE embedder/reranker are served
    # separately and must be downloaded via `uv run hf download`).
    _prefetch_models()

    ClarityMedApp(
        user_id=resolved_user,
        language=resolved_lang,
        provider_id=provider.id,
    ).run()


prompts_app = typer.Typer(
    name="prompts",
    help="Sync YAML prompts with a Phoenix instance.",
    no_args_is_help=True,
)
app.add_typer(prompts_app, name="prompts")


def _print_sync_report(report) -> None:
    for entry in report.entries:
        marker = {
            "pushed": "[green]→[/green]",
            "pulled": "[green]←[/green]",
            "skipped": "[dim]·[/dim]",
            "missing": "[yellow]?[/yellow]",
            "error": "[red]✗[/red]",
        }.get(entry.action, "?")
        console.print(
            f"  {marker} {entry.prompt_name}({entry.language}) "
            f"-> {entry.phoenix_name}  {entry.detail}"
        )
    changed = len(report.changed)
    errors = len(report.errors)
    console.print(
        f"\n[bold]{report.direction}[/bold] "
        f"{'(dry-run) ' if report.dry_run else ''}"
        f"changed={changed} errors={errors}"
    )


@prompts_app.command("push")
def prompts_push(
    name: str | None = typer.Argument(None, help="Filter to one prompt name."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report changes without writing to Phoenix."
    ),
) -> None:
    """Push local YAML prompts (latest version) to Phoenix as the ``production`` tag."""
    from claritymed.core.prompts.phoenix_sync import push

    try:
        report = push(name=name, dry_run=dry_run)
    except Exception as exc:  # noqa: BLE001
        _stderr(f"[error] {exc}")
        raise typer.Exit(code=1) from exc
    _print_sync_report(report)
    if report.errors:
        raise typer.Exit(code=1)


@prompts_app.command("diff")
def prompts_diff(
    name: str | None = typer.Argument(None, help="Filter to one prompt name."),
    color: bool = typer.Option(
        True, "--color/--no-color", help="Colorize diff output."
    ),
) -> None:
    """Show a unified diff between local YAML and Phoenix production-tagged prompts.

    Same content shape as ``push`` / ``pull`` reads — one entry per
    ``(name, language)`` pair. Exit code is 0 when everything matches,
    1 when at least one entry differs or errors so this can gate CI.
    """
    from claritymed.core.prompts.phoenix_sync import diff

    try:
        report = diff(name=name)
    except Exception as exc:  # noqa: BLE001
        _stderr(f"[error] {exc}")
        raise typer.Exit(code=1) from exc

    for entry in report.entries:
        marker = {
            "same": "[dim]·[/dim]",
            "differs": "[yellow]Δ[/yellow]",
            "remote_missing": "[yellow]?[/yellow]",
            "error": "[red]✗[/red]",
        }.get(entry.action, "?")
        console.print(
            f"  {marker} {entry.prompt_name}({entry.language}) "
            f"-> {entry.phoenix_name}  {entry.detail}"
        )
        if entry.action == "differs":
            if color:
                from rich.syntax import Syntax

                console.print(
                    Syntax(
                        "\n".join(entry.unified_diff),
                        "diff",
                        theme="ansi_dark",
                        background_color="default",
                        word_wrap=True,
                    )
                )
            else:
                for line in entry.unified_diff:
                    console.print(line)

    differs = len(report.differs)
    errors = len(report.errors)
    console.print(f"\n[bold]diff[/bold] differs={differs} errors={errors}")
    if differs or errors:
        raise typer.Exit(code=1)


@prompts_app.command("pull")
def prompts_pull(
    name: str | None = typer.Argument(None, help="Filter to one prompt name."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Report changes without writing YAML."
    ),
    into_new_version: bool = typer.Option(
        False,
        "--into-new-version",
        help="Append a new YAML version instead of overwriting the latest in place.",
    ),
    new_version_name: str | None = typer.Option(
        None,
        "--version-name",
        help="Override the new version label (implies --into-new-version). "
        "Default auto-bumps to vN+1.",
    ),
) -> None:
    """Pull Phoenix ``production``-tagged prompts back into YAML.

    Default: overwrite the latest version in place (compact diff). Use
    ``--into-new-version`` to append a fresh version block instead, and
    ``--version-name v1.1`` to tag it explicitly.
    """
    from claritymed.core.prompts.phoenix_sync import pull

    try:
        report = pull(
            name=name,
            dry_run=dry_run,
            into_new_version=into_new_version,
            new_version_name=new_version_name,
        )
    except Exception as exc:  # noqa: BLE001
        _stderr(f"[error] {exc}")
        raise typer.Exit(code=1) from exc
    _print_sync_report(report)
    if report.errors:
        raise typer.Exit(code=1)


audit_app = typer.Typer(
    name="audit",
    help="Inspect the structured audit log.",
    no_args_is_help=True,
)
app.add_typer(audit_app, name="audit")


@audit_app.command("grep")
def audit_grep(
    trace_id: str | None = typer.Option(
        None, "--trace-id", help="Filter by OTel trace id."
    ),
    request_id: str | None = typer.Option(
        None, "--request-id", help="Filter by request id."
    ),
    user_id: str | None = typer.Option(None, "--user-id", help="Filter by user id."),
    kind: str | None = typer.Option(
        None, "--kind", help="Filter by audit kind, e.g. mode.ask."
    ),
    since: str | None = typer.Option(
        None, "--since", help="Lower bound on created_at (ISO 8601 prefix match)."
    ),
    until: str | None = typer.Option(
        None, "--until", help="Upper bound on created_at (ISO 8601 prefix match)."
    ),
    limit: int | None = typer.Option(None, "--limit", help="Stop after N matches."),
    json_out: bool = typer.Option(
        False,
        "--json",
        help="Emit one JSON object per line (drops the formatter prefix).",
    ),
) -> None:
    """Grep audit.log* JSONL by trace / request / user / kind / time window.

    Reads every rotated ``audit.log*`` under ``CLARITYMED_LOG_DIR``, parses
    the JSON payload from each line, and prints lines whose payload matches
    every supplied filter. Files are walked in modification order so output
    is roughly chronological even across rotations.
    """
    import json
    from pathlib import Path

    log_dir = Path(_cfg.LOG_DIR)
    if not log_dir.exists():
        _stderr(f"[error] log dir does not exist: {log_dir}")
        raise typer.Exit(code=1)
    files = sorted(log_dir.glob("audit.log*"), key=lambda p: p.stat().st_mtime)
    if not files:
        _stderr(f"[error] no audit.log* files in {log_dir}")
        raise typer.Exit(code=1)

    matched = 0
    for fp in files:
        try:
            fh = fp.open(encoding="utf-8")
        except OSError as exc:
            _stderr(f"[warn] cannot open {fp}: {exc}")
            continue
        with fh:
            for raw in fh:
                line = raw.rstrip("\n")
                idx = line.find("{")
                if idx < 0:
                    continue
                try:
                    event = json.loads(line[idx:])
                except json.JSONDecodeError:
                    continue
                if trace_id and event.get("trace_id") != trace_id:
                    continue
                if request_id and event.get("request_id") != request_id:
                    continue
                if user_id and event.get("user_id") != user_id:
                    continue
                if kind and event.get("kind") != kind:
                    continue
                created = event.get("created_at", "")
                if since and created < since:
                    continue
                if until and created > until:
                    continue
                if json_out:
                    print(json.dumps(event, ensure_ascii=False))
                else:
                    print(line)
                matched += 1
                if limit is not None and matched >= limit:
                    return
    if matched == 0:
        _stderr("[info] no matches")
        raise typer.Exit(code=1)


@app.command("init-user")
def init_user_cmd(
    user_id: str = typer.Argument(
        ..., help="User ID to create (alphanumeric, hyphens, underscores)."
    ),
    display_name: str | None = typer.Option(
        None, "--name", "-n", help="Display name (defaults to user_id)."
    ),
) -> None:
    """Create a user account and initialise their data directory.

    The first user created on a fresh install is automatically promoted to
    admin.  Subsequent users receive the ``user`` role.

    Idempotent — safe to run more than once for the same user_id.
    """
    from claritymed.stores.account import init_user

    _bootstrap_once()
    account = init_user(user_id, display_name=display_name)
    console.print(
        f"[green]✓[/green] user=[bold]{account.user_id}[/bold]  "
        f"role={account.role}  display_name={account.display_name!r}"
    )


# Convenience subcommand to list known modes — useful for shell completion.
@app.command()
def modes() -> None:
    """List configured interaction modes."""
    cfg = _cfg.load_modes_config()
    for name, mode in cfg.modes.items():
        console.print(
            f"[bold]{name}[/bold]  llm={mode.allow_llm_inference}  tools={len(mode.tools)}"
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
