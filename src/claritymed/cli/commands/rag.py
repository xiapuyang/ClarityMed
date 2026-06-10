"""``claritymed rag`` — per-user RAG store management.

Holds the user-facing subcommands (``add``, ``list``, ``show``, ``rm``)
and wires in the admin ``corpora`` sub-app from
``cli.commands.corpora``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from claritymed.cli.commands.corpora import corpora_app
from claritymed.cli.common import console, run_async
from claritymed.cli.entry import inject_context
from claritymed.orchestrator.services import (
    Done,
    Error,
    RagService,
    ToolCompleted,
    ToolStarted,
)
from claritymed.stores.user_rag import make_user_rag_store

logger = logging.getLogger(__name__)

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


rag_app = typer.Typer(help="Manage reference materials (per-user RAG).")
rag_app.add_typer(corpora_app, name="corpora")


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
                    logger.error("OCR failed: %s", exc)
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
                        logger.error("%s", event.message)
                        raise typer.Exit(code=1)
            except DuplicateDocumentError as exc:
                console.print(
                    f"[yellow]already indexed as {exc.existing_doc_id} — skipped[/yellow]"
                )

    run_async(_run())


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

    run_async(_run())


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

    run_async(_run())


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

    run_async(_run())
