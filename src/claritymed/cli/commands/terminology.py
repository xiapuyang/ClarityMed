"""``claritymed terminology`` — inspect the shared terminology catalog."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import typer
from rich.table import Table

from claritymed.cli.common import console
from claritymed.stores.paths import shared_terminology_jsonl

logger = logging.getLogger(__name__)

terminology_app = typer.Typer(help="Inspect the shared terminology catalog.")


@terminology_app.command("summary")
def summary(
    path: str | None = typer.Option(
        None,
        "--path",
        "-p",
        help="Path to concepts.jsonl. Defaults to shared terminology dir.",
    ),
) -> None:
    """Print concept and alias counts broken down by type, source, and language."""
    target = Path(path) if path else shared_terminology_jsonl()

    if not target.exists():
        logger.error(
            "terminology file not found: %s -- "
            "run: uv run python scripts/init_terminology.py --seed",
            target,
        )
        raise typer.Exit(code=1)

    concept_count = 0
    total_aliases = 0
    type_counts: Counter[str] = Counter()
    lang_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    with target.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            concept_count += 1
            type_counts[rec.get("type", "?")] += 1
            for alias in rec.get("aliases", []):
                total_aliases += 1
                lang_counts[alias.get("language", "?")] += 1
                source_counts[alias.get("source", "?")] += 1

    console.print(f"\n[bold]Terminology catalog:[/bold] {target}")
    console.print(
        f"  concepts: [cyan]{concept_count:>10,}[/cyan]  "
        f"aliases: [cyan]{total_aliases:>10,}[/cyan]\n"
    )

    _print_table(
        "By type",
        headers=("type", "concepts", "%"),
        rows=[
            (t, f"{n:,}", f"{100 * n / concept_count:.1f}")
            for t, n in type_counts.most_common()
        ],
    )
    _print_table(
        "By source",
        headers=("source", "aliases", "%"),
        rows=[
            (s, f"{n:,}", f"{100 * n / total_aliases:.1f}")
            for s, n in source_counts.most_common()
        ],
    )
    _print_table(
        "By language",
        headers=("lang", "aliases", "%"),
        rows=[
            (lang, f"{n:,}", f"{100 * n / total_aliases:.1f}")
            for lang, n in lang_counts.most_common()
        ],
    )


def _print_table(title: str, headers: tuple[str, ...], rows: list[tuple]) -> None:
    table = Table(title=title, show_header=True, header_style="bold", box=None)
    for h in headers:
        table.add_column(
            h, justify="right" if h not in ("type", "source", "lang") else "left"
        )
    for row in rows:
        table.add_row(*row)
    console.print(table)
    console.print()
