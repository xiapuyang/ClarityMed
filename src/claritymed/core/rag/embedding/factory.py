"""Resolve the active ``Embedder`` from ``configs/retrieval.yaml``.

Mirrors ``stores/models.py:resolve_provider`` in shape: catalog +
active id, ``Unknown<X>Error`` on miss. Tests can pass an ``EmbedderConfig``
directly so they don't need to redirect the YAML loader.
"""

from __future__ import annotations

from claritymed.core.phi.outbound_gate import make_outbound_gate, resolve_phi_kind
from claritymed.core.rag.embedding.base import Embedder
from claritymed.core.rag.embedding.bge_m3 import BgeM3HttpEmbedder
from claritymed.core.rag.schemas import EmbedderConfig, load_retrieval_config
from claritymed.errors import UnknownEmbedderError


def build_embedder(config: EmbedderConfig | None = None) -> Embedder:
    """Instantiate the active embedder from the retrieval config.

    Args:
        config: Optional config override. When ``None``, loads from
            ``configs/retrieval.yaml``.

    Returns:
        Embedder ready for async use.

    Raises:
        UnknownEmbedderError: Active id does not match a supported impl.
            (Unknown id within the YAML catalog is caught earlier by the
            ``EmbedderConfig`` validator; this guards against catalog
            entries the factory has not been taught to instantiate yet.)
    """
    cfg = config or load_retrieval_config().embedders
    entry = cfg.resolved()
    if entry.id == "bge_m3_http":
        return BgeM3HttpEmbedder(
            base_url=entry.base_url,
            dense_dim=entry.dense_dim,
            batch_size=entry.batch_size,
            timeout_s=entry.timeout_s,
            api_key_env=entry.api_key_env,
            scrub_gate=make_outbound_gate(
                resolve_phi_kind(entry.phi_kind, entry.base_url)
            ),
        )
    raise UnknownEmbedderError(
        f"build_embedder has no factory branch for id={entry.id!r}"
    )
