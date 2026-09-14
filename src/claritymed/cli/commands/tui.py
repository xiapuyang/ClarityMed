"""``claritymed tui`` — launch the Textual TUI.

Resolves the provider up front so a typo or a cloud-without-opt-in fails
to stderr instead of opening the TUI and crashing on the first submit.
Centroid refresh + model prefetch happen synchronously here so the first
query inside the TUI never races a half-built centroid file or a
half-downloaded ONNX model.
"""

from __future__ import annotations

import logging
import typer

from claritymed.cli.commands.corpora import refresh_system_centroids_on_startup
from claritymed.cli.common import (
    CLIEmergencySensitivity,
    prefetch_models,
    try_load_account,
)
from claritymed.stores.models import load_models, resolve_provider

logger = logging.getLogger(__name__)


def tui(
    user: str | None = typer.Option(None, "--user", "-u"),
    language: str | None = typer.Option(None, "--lang", "-l"),
    provider_id: str | None = typer.Option(None, "--provider", "-p"),
    emergency_sensitivity: CLIEmergencySensitivity | None = typer.Option(
        None,
        "--emergency-sensitivity",
        case_sensitive=False,
        help=(
            "Override the emergency triage gate sensitivity for this TUI "
            "session: strict | balanced | lenient. To disable the gate "
            "entirely, set CLARITYMED_FORCE_EMERGENCY_GATE=off (master env "
            "switch — downgrades to 'lenient' at runtime) or write "
            "emergency.sensitivity='off' with off_acknowledged_at to "
            "data/users/<uid>/settings.yaml. 'off' is intentionally not "
            "accepted here because the CLI flag bypasses the settings.yaml "
            "two-step acknowledgement safeguard."
        ),
    ),
) -> None:
    """Launch the Textual TUI."""
    from claritymed.cli.entry import _resolve_language, _resolve_user_id
    from claritymed.cli.tui import ClarityMedApp
    from claritymed.errors import UnknownProviderError

    resolved_lang = _resolve_language(language)
    resolved_user, _ = _resolve_user_id(user)

    # Resolve the provider up front so a typo (`--provider oMLX`) fails
    # cleanly to stderr instead of opening the TUI and exploding on the
    # first submit. Matches the project rule: provider resolution is
    # loud, never silent.
    account = try_load_account(resolved_user)
    try:
        provider = resolve_provider(override=provider_id, account=account)
    except UnknownProviderError as exc:
        valid = ", ".join(p.id for p in load_models().providers)
        msg = f"{exc}  valid provider ids: {valid}"
        logger.error(msg)
        typer.echo(msg, err=True)
        raise typer.Exit(code=1) from exc

    # Pre-download in-process models before the TUI takes over the
    # terminal. Currently only openai/privacy-filter — BGE embedder /
    # reranker are served separately and downloaded via
    # ``uv run hf download``.
    prefetch_models()

    # Recompute any missing or stale routing centroids before the UI
    # takes over the terminal — blocking, so the first query inside the
    # TUI never races a half-built centroid file.
    refresh_system_centroids_on_startup()

    ClarityMedApp(
        user_id=resolved_user,
        language=resolved_lang,
        provider_id=provider.id,
        emergency_sensitivity_override=(
            emergency_sensitivity.value if emergency_sensitivity else None
        ),
    ).run()
