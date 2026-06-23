"""Shared local-provider pick for the gate's extractor and composer.

Both LLMs in the gate must run on a local provider (PHI never leaves
the box — the safety net's own pipeline cannot be the leak). The
selection logic is identical for both, so it lives here once.

Pick policy: walk ``models.yaml`` for the first ``kind: local`` entry
whose credentials resolve. We do not health-probe — that adds a
network call to cold paths, and the LLM's first ``.run()`` will
surface a downed server anyway (which :meth:`EmergencyTriage.assess`
catches and turns into ``routine_noop``).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai.models import Model

logger = logging.getLogger(__name__)


def build_local_gate_model(component: str) -> "Model | None":
    """Return a built local model, or ``None`` if no local provider is usable.

    ``component`` is purely for the log message ("extractor" / "composer")
    so an operator can tell which gate piece is dormant.
    """
    from claritymed.core.llm import build_model
    from claritymed.errors import MissingApiKeyError
    from claritymed.stores.models import load_models

    models = load_models()
    local_providers = [p for p in models.providers if p.kind == "local"]
    if not local_providers:
        logger.warning(
            "emergency gate: no kind=local provider in models.yaml; "
            "%s stays unwired (gate will always return routine_noop)",
            component,
        )
        return None

    for provider in local_providers:
        try:
            model = build_model(provider)
        except MissingApiKeyError:
            logger.warning(
                "emergency gate: local provider %r declares api_key_env "
                "but it is unset; trying next local provider for %s",
                provider.id,
                component,
            )
            continue
        logger.info(
            "emergency gate: %s wired to local provider %r",
            component,
            provider.id,
        )
        return model

    logger.warning(
        "emergency gate: every local provider has an unmet api_key_env; "
        "%s stays unwired (gate will always return routine_noop)",
        component,
    )
    return None
