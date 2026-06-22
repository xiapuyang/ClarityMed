"""``GET /api/v1/providers`` — catalog projection for the model picker.

Returns one entry per provider in ``configs/models.yaml``, decorated
with frontend-friendly fields the YAML deliberately does not carry:

* ``display_name`` — human label (currently ``id``; pluggable once the
  catalog grows a label field).
* ``family`` — coarse bucket (claude / openai / gemini / qwen / …) used
  by the UI to group entries.
* ``context_window`` — best-guess from the static map in
  :mod:`claritymed.web.context_window`. The frontend renders a ratio
  next to the picker ("12k / 128k") only when this is > 0.
* ``available`` — does the env-key gate (or local reachability) say the
  provider can actually be invoked right now? The UI dims unavailable
  entries instead of hiding them so the user understands which
  credentials are missing.

The response also carries ``default_provider_id`` (catalog default)
and ``current_provider_id`` (this user's persisted preference, or the
default when unset) so the SPA can render the picker state without a
second round-trip.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from claritymed.core.schemas import Account
from claritymed.stores.models import (
    is_provider_available,
    load_models,
)
from claritymed.core.llm.context_window import estimate_context_window, model_family
from claritymed.web.deps import get_current_user
from claritymed.web.schemas import ProviderListResponse, ProviderResponse

router = APIRouter(prefix="/api/v1", tags=["providers"])


@router.get("/providers", response_model=ProviderListResponse)
async def list_providers(
    account: Account = Depends(get_current_user),
) -> ProviderListResponse:
    """Return the full provider catalog plus user/default selection."""
    models = load_models()
    entries: list[ProviderResponse] = []
    for p in models.providers:
        # ``thinking`` may be omitted (None) / bool / Literal string. Surface
        # whatever the YAML declared so the UI can render an indicator chip
        # without re-deriving the value.
        entries.append(
            ProviderResponse(
                id=p.id,
                kind=p.kind,
                model=p.model,
                family=model_family(p.model),
                display_name=p.id,
                context_window=estimate_context_window(p.model),
                available=is_provider_available(p),
                thinking=p.thinking,
            )
        )
    return ProviderListResponse(
        providers=entries,
        default_provider_id=models.default_provider,
        current_provider_id=account.provider_id or models.default_provider,
    )
