"""``claritymed finetune`` — preprocess fine-tune corpora.

Fine-tune corpora are NOT indexed in Qdrant — they are training material
only. Output is JSONL splits under ``data/finetune/<name>/`` by
convention; the trainer is out of scope for this CLI.
"""

from __future__ import annotations

from pathlib import Path

import typer

from claritymed.cli.common import console

finetune_app = typer.Typer(help="Preprocess fine-tune corpora (NOT indexed in RAG).")


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
