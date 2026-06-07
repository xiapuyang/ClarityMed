"""Textual TUI surface for ClarityMed.

The TUI is the primary interactive entry point (the Typer subcommands are the
headless mirror). It calls the same ``orchestrator.services`` layer as the CLI
so any service-level change reaches both surfaces with a single edit.
"""

from claritymed.cli.tui.app import ClarityMedApp

__all__ = ["ClarityMedApp"]
