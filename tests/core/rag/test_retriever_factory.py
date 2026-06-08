"""Unit tests for ``build_hybrid_retriever`` and the ``rag.enabled`` switch.

These tests live below the wiring layer — the CLI/TUI tests cover that
the flag is *honored*; here we cover that the factory wires the right
components when called.
"""

from __future__ import annotations

from claritymed.core.rag import build_hybrid_retriever, load_retrieval_config
from claritymed.core.rag.retriever import HybridRetriever
from claritymed.core.rag.schemas import RagBootstrapConfig, TermServiceConfig


def test_rag_bootstrap_defaults_to_disabled():
    """A fresh RagBootstrapConfig defaults to off — RAG must be opt-in."""
    cfg = RagBootstrapConfig()
    assert cfg.enabled is False
    assert cfg.max_evidence == 5


def test_default_yaml_has_rag_disabled():
    """The shipped ``configs/retrieval.yaml`` must keep ``rag.enabled=false``.

    Flipping the default would force every user to have the embedder /
    reranker servers running just to use ``claritymed ask``.
    """
    cfg = load_retrieval_config()
    assert cfg.rag.enabled is False


def test_env_var_can_enable_rag(monkeypatch):
    """``CLARITYMED_RAG_ENABLED=true`` flips on without editing YAML."""
    for value in ("1", "true", "TRUE", "yes", "on"):
        monkeypatch.setenv("CLARITYMED_RAG_ENABLED", value)
        assert load_retrieval_config().rag.enabled is True


def test_env_var_can_disable_rag(monkeypatch, tmp_path):
    """``CLARITYMED_RAG_ENABLED=false`` overrides a yaml-enabled config too."""
    for value in ("0", "false", "FALSE", "no", "off"):
        monkeypatch.setenv("CLARITYMED_RAG_ENABLED", value)
        assert load_retrieval_config().rag.enabled is False


def test_env_var_typo_falls_back_to_yaml(monkeypatch):
    """An unrecognized value is ignored — typos must not silently flip RAG."""
    monkeypatch.setenv("CLARITYMED_RAG_ENABLED", "ture")  # common typo
    assert load_retrieval_config().rag.enabled is False  # yaml default


def test_default_yaml_has_qdrant_url():
    """Shipped yaml points at the local Docker port — server-only model."""
    cfg = load_retrieval_config()
    assert cfg.qdrant.url == "http://localhost:6333"
    assert cfg.qdrant.api_key_env is None


def test_qdrant_url_env_var_overrides_yaml(monkeypatch):
    """``CLARITYMED_QDRANT_URL`` overrides yaml — convenient for dev to
    point at a remote / alternate-port server without editing config."""
    monkeypatch.setenv("CLARITYMED_QDRANT_URL", "http://qdrant.staging:6333")
    assert load_retrieval_config().qdrant.url == "http://qdrant.staging:6333"


def test_qdrant_url_env_var_empty_leaves_yaml(monkeypatch):
    """Empty / whitespace env var must not zero-out yaml — only set values win."""
    monkeypatch.setenv("CLARITYMED_QDRANT_URL", "   ")
    assert load_retrieval_config().qdrant.url == "http://localhost:6333"


def test_build_hybrid_retriever_returns_real_retriever():
    """The factory wires every dependency without touching the network.

    Construction-time fail-loud only fires for unknown active ids; HTTP
    endpoints are dialed on the first ``retrieve`` call, not here. So a
    successful call to ``build_hybrid_retriever`` proves the catalog +
    factories agree without needing a live embedder server. We swap the
    term service to ``none`` so the test doesn't require a UMLS export.
    """
    cfg = load_retrieval_config()
    # Pick the existing ``none`` catalog entry rather than mutating shape.
    none_entry = next(e for e in cfg.term_service.catalog if e.id == "none")
    patched = cfg.model_copy(
        update={
            "term_service": TermServiceConfig(active="none", catalog=[none_entry]),
        }
    )
    retriever = build_hybrid_retriever(patched)
    assert isinstance(retriever, HybridRetriever)
