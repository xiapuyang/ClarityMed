"""Load ``configs/models.yaml`` and resolve which provider is active.

Resolution priority (highest first):

    1. ``override`` — an explicit argument from the CLI (``--provider``) or
       a programmatic caller. Strongest because the human just said it.
    2. ``Account.provider_id`` — the per-user preference stored in
       ``settings.yaml``. Survives across requests.
    3. ``ModelsConfig.default_provider`` — the catalog-wide fallback from
       ``models.yaml``. The "we did nothing" answer.

Every candidate is validated against the catalog before being returned, so a
typo in ``settings.yaml`` raises ``UnknownProviderError`` instead of silently
falling through to the default (which would mask the misconfiguration).
"""

from __future__ import annotations

import os

from claritymed import config as _cfg
from claritymed.core.schemas import Account, ModelsConfig, ProviderConfig
from claritymed.errors import UnknownProviderError

# pydantic-ai reads these env vars for each model prefix (no base_url path).
# Derived from pydantic-ai's provider source; update when new providers land.
_PREFIX_ENV: dict[str, str | list[str]] = {
    "openai": "OPENAI_API_KEY",
    "openai-chat": "OPENAI_API_KEY",
    "openai-responses": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "google": "GEMINI_API_KEY",
    "google-vertex": "GOOGLE_APPLICATION_CREDENTIALS",
    "alibaba": ["ALIBABA_API_KEY", "DASHSCOPE_API_KEY"],
    "moonshotai": "MOONSHOTAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "CO_API_KEY",
    "groq": "GROQ_API_KEY",
}


def load_models() -> ModelsConfig:
    """Parse ``configs/models.yaml``.

    Uses the process-wide ``load_yaml`` cache; call ``_cfg.reload_configs()``
    to force a re-read (e.g. after an admin edits the file).
    """
    raw = _cfg.load_yaml("models.yaml")
    return ModelsConfig.model_validate(raw)


def resolve_provider(
    *,
    override: str | None = None,
    account: Account | None = None,
) -> ProviderConfig:
    """Pick the active provider for this request.

    A typo at any layer is surfaced — we never silently downgrade to the
    default, because hiding a misconfigured per-user setting would let
    a user accidentally fall back to a different backend.

    Resolution order: ``override`` (CLI ``--provider``) > ``account.provider_id``
    > ``ModelsConfig.default_provider``. Cloud reachability is gated by
    env-key presence (``is_provider_available``) and the catalog ``kind`` —
    this resolver only picks; it does not enforce per-user cloud policy.
    """
    models = load_models()
    catalog = {p.id: p for p in models.providers}

    candidates: list[tuple[str, str | None]] = [
        ("override", override),
        ("account", account.provider_id if account else None),
        ("default", models.default_provider),
    ]
    for source, candidate in candidates:
        if not candidate:
            continue
        if candidate not in catalog:
            raise UnknownProviderError(
                f"{source} requested provider {candidate!r}, "
                f"which is not in models.yaml"
            )
        return catalog[candidate]

    # Unreachable: ModelsConfig requires default_provider to be set.
    raise UnknownProviderError("no provider could be resolved")


def is_provider_available(provider: ProviderConfig) -> bool:
    """Return True if the required credentials are present in the environment.

    Self-hosted (base_url set): available when api_key_env is absent or its
    env var is non-empty.  Stock cloud (no base_url): available when the env
    var pydantic-ai would read for the model prefix is non-empty.

    Unknown prefixes return ``False``. Earlier versions returned True
    optimistically and deferred the verdict to pydantic-ai — which then
    failed with a confusing message at the LLM call site. Treating an
    unrecognised prefix as "not available" surfaces the typo at provider
    selection time (CLI ``--provider``, eval auto-pick) where the operator
    can read the error message in context.
    """
    if provider.base_url is not None:
        if provider.api_key_env is None:
            return True
        return bool(os.environ.get(provider.api_key_env))

    prefix = provider.model.split(":")[0]
    keys = _PREFIX_ENV.get(prefix)
    if keys is None:
        return False
    if isinstance(keys, str):
        keys = [keys]
    return any(bool(os.environ.get(k)) for k in keys)


def list_available_providers() -> list[ProviderConfig]:
    """Return catalog providers whose credentials are present in the environment."""
    return [p for p in load_models().providers if is_provider_available(p)]


# Local-provider health endpoints checked by ``pick_reachable_provider``. The
# probe is the same set the e2e fixture (``tests/e2e/conftest.py``) uses, so
# both call sites agree on what "reachable" means without one drifting from
# the other. Order is preference: try omlx first (MLX server with the larger
# Qwen model), fall back to Ollama (default catalog entry).
_LOCAL_HEALTH_URLS: tuple[tuple[str, str], ...] = (
    ("omlx", "http://127.0.0.1:8000/health"),
    ("ollama", "http://127.0.0.1:11434/api/version"),
)


def _local_service_up(url: str, timeout_s: float = 2.0) -> bool:
    """HTTP GET ``url`` and return True iff it responds 200 in ``timeout_s``."""
    try:
        import httpx

        return httpx.get(url, timeout=timeout_s).status_code == 200
    except Exception:  # noqa: BLE001
        return False


def pick_reachable_provider() -> ProviderConfig | None:
    """Return the first local provider whose server responds + creds are set.

    Mirrors the resolution order in the e2e provider fixture so a
    ``claritymed eval run medqa`` run with no ``--provider`` flag lands on the
    same backend the e2e suite would use. Returns ``None`` when no probed
    local provider is reachable — callers fall back to the catalog default
    (which may itself be unreachable; that's the caller's concern).

    Cloud providers are intentionally not probed: probing implies a network
    call, and a half-up cloud endpoint reachable from CI but not from the
    operator's box is exactly the kind of brittleness we want to avoid in
    a default-picker.
    """
    catalog = {p.id: p for p in load_models().providers}
    for pid, url in _LOCAL_HEALTH_URLS:
        provider = catalog.get(pid)
        if provider is None:
            continue
        if not is_provider_available(provider):
            continue
        if _local_service_up(url):
            return provider
    return None
