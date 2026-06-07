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

from claritymed import config as _cfg
from claritymed.core.schemas import Account, ModelsConfig, ProviderConfig
from claritymed.errors import UnknownProviderError


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
    default, because hiding a misconfigured per-user setting would let a
    cloud-opted-in user accidentally fall back to a different backend.
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
