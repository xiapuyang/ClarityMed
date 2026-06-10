"""Per-subcommand modules for the ``claritymed`` Typer CLI.

Each module owns one Typer ``app`` (or one top-level command function)
and exposes it for the root ``cli.main`` to wire into the global app.
Splitting per subcommand keeps any one file small enough to read top to
bottom and lets a focused fix touch just the surface area it needs.
"""
