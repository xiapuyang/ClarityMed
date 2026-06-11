"""Typer CLI entry point: wires every subcommand module into one app.

The CLI is the headless surface that mirrors the TUI's actions one-shot.
Both go through the same service layer — the CLI never touches an Agent
directly so any service-level change (audit, scrub, event schema)
reaches both surfaces with one edit.

Each subcommand lives in its own module under ``cli/commands/``:

* ``ask`` — stream a grounded answer
* ``ingest`` — save profile / history fields without an LLM
* ``rag`` (+ nested ``corpora``) — per-user RAG store and admin corpora
* ``finetune`` — preprocess fine-tune corpora
* ``tui`` — launch the Textual TUI
* ``prompts`` — sync YAML prompts with Phoenix
* ``audit`` — inspect the structured audit log
* ``init-user`` — create an account
* ``eval`` — lm-eval-harness MCQA runner (optional extra)

This file is the only place that knows the whole tree. Adding a new
subcommand is one import + one ``add_typer`` / ``command`` registration.
"""

from __future__ import annotations

import logging
import typer

from claritymed.cli.commands.admin import init_user_cmd
from claritymed.cli.commands.ask import ask
from claritymed.cli.commands.audit import audit_app
from claritymed.cli.commands.finetune import finetune_app
from claritymed.cli.commands.ingest import ingest_app
from claritymed.cli.commands.prompts import prompts_app
from claritymed.cli.commands.rag import rag_app
from claritymed.cli.commands.terminology import terminology_app
from claritymed.cli.commands.tool import tool_app
from claritymed.cli.commands.tui import tui
from claritymed.cli.common import bootstrap_once

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="claritymed",
    help="ClarityMed CLI — ingest / ask / rag (headless mirror of the TUI).",
    no_args_is_help=True,
)


@app.callback()
def _cli_root() -> None:
    """Root callback — runs before every subcommand."""
    bootstrap_once()


# Top-level commands.
app.command()(ask)
app.command()(tui)
app.command("init-user")(init_user_cmd)

# Sub-apps.
app.add_typer(ingest_app, name="ingest")
app.add_typer(rag_app, name="rag")
app.add_typer(terminology_app, name="terminology")
app.add_typer(finetune_app, name="finetune")
app.add_typer(prompts_app, name="prompts")
app.add_typer(audit_app, name="audit")
# v1 PHI ingest tools — headless mirror of the LLM tool surface.
app.add_typer(tool_app, name="tool")

# ``eval`` sub-app — installed only when the ``evals`` optional extra is
# present. Importing it pulls in lm-eval transitively (torch, datasets),
# which we don't want forced on plain ``claritymed ask`` users. Hidden
# behind a try/except so the rest of the CLI still works without the
# extra; a missing extra prints a remediation hint at first invocation.
_eval_import_error: str | None = None
try:
    from claritymed.cli.commands.eval import eval_app

    app.add_typer(eval_app, name="eval")
except ImportError as _exc:  # pragma: no cover — install-time gate
    _eval_import_error = str(_exc)

    @app.command("eval", hidden=True)
    def _eval_stub() -> None:
        """Placeholder when ``uv sync --extra evals`` has not been run."""
        logger.error(
            "claritymed eval requires the `evals` extra: "
            "uv sync --extra evals (import failed: %s)",
            _eval_import_error,
        )
        raise typer.Exit(code=2)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
