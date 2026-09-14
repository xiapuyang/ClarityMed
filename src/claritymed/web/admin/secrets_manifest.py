"""Declared schema for ``~/.claritymed/.env``.

The admin UI introspects ``EXPECTED_SECRETS`` to render the secrets editor:
each entry names a key, a human-readable label, and a one-line hint about
where the key is used. Values are never stored here — the manifest is
metadata only, and the editor writes to the env file via
:func:`claritymed.web.admin.secrets.atomic_write_env`.

The set is hand-maintained; ``configs/models.yaml`` may declare additional
``api_key_env`` entries for self-hosted endpoints and the admin module
cross-checks at startup so drift surfaces as a warning, not silent loss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SecretCategory = Literal["jwt", "provider", "endpoint", "observability"]


@dataclass(frozen=True)
class SecretSpec:
    """One row in the admin secrets editor.

    Attributes:
        key: Environment variable name as it appears in ``~/.claritymed/.env``.
        category: Grouping for UI sectioning.
        label: Human-readable label (en).
        hint: One-line explanation of where the key is consumed.
        required: When True, an absent value blocks the UI with a banner.
    """

    key: str
    category: SecretCategory
    label: str
    hint: str
    required: bool = False


# Seed manifest — extended as new providers / observability sinks land.
EXPECTED_SECRETS: dict[str, SecretSpec] = {
    s.key: s
    for s in [
        SecretSpec(
            key="CLARITYMED_JWT_SECRET",
            category="jwt",
            label="JWT signing secret",
            hint="Used to sign session cookies. Rotate to force re-login.",
            required=True,
        ),
        SecretSpec(
            key="OPENAI_API_KEY",
            category="provider",
            label="OpenAI",
            hint="Used by any provider with model prefix 'openai:'.",
        ),
        SecretSpec(
            key="ANTHROPIC_API_KEY",
            category="provider",
            label="Anthropic",
            hint="Used by any provider with model prefix 'anthropic:'.",
        ),
        SecretSpec(
            key="DEEPSEEK_API_KEY",
            category="provider",
            label="DeepSeek",
            hint="Used by any provider with model prefix 'deepseek:'.",
        ),
        SecretSpec(
            key="GEMINI_API_KEY",
            category="provider",
            label="Google Gemini",
            hint="Used by any provider with model prefix 'google-gla:'.",
        ),
        SecretSpec(
            key="MOONSHOTAI_API_KEY",
            category="provider",
            label="Moonshot AI",
            hint="Used by any provider with model prefix 'moonshotai:'.",
        ),
        SecretSpec(
            key="OPENROUTER_API_KEY",
            category="provider",
            label="OpenRouter",
            hint="Used by any provider with model prefix 'openrouter:'.",
        ),
        SecretSpec(
            key="ALIBABA_API_KEY",
            category="provider",
            label="Alibaba (Qwen cloud)",
            hint="Used by any provider with model prefix 'alibaba:'.",
        ),
        SecretSpec(
            key="PHOENIX_API_KEY",
            category="observability",
            label="Phoenix prompts / tracing",
            hint="Used when tracing.enabled and prompts push/pull.",
        ),
    ]
}


def expected_keys() -> frozenset[str]:
    """Return the canonical key set as a frozenset for membership checks."""
    return frozenset(EXPECTED_SECRETS.keys())
