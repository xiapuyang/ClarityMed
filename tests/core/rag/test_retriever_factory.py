"""Unit tests for ``build_hybrid_retriever`` and the ``rag.enabled`` switch.

These tests live below the wiring layer — the CLI/TUI tests cover that
the flag is *honored*; here we cover that the factory wires the right
components when called.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from claritymed.core.rag import build_hybrid_retriever, load_retrieval_config
from claritymed.core.rag.retriever import HybridRetriever
from claritymed.core.rag.schemas import RagBootstrapConfig, TermServiceConfig
from claritymed.errors import UnknownRouterError


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


def test_router_env_var_overrides_yaml(monkeypatch):
    """``CLARITYMED_RAG_ROUTER`` flips ``router.active`` without editing YAML.

    Lets dev / eval pin a router per shell without committing config churn
    (e.g. ``CLARITYMED_RAG_ROUTER=centroid_classifier uv run claritymed ask ...``).
    """
    monkeypatch.setenv("CLARITYMED_RAG_ROUTER", "centroid_classifier")
    assert load_retrieval_config().router.active == "centroid_classifier"


def test_router_env_var_empty_leaves_yaml(monkeypatch):
    """Empty / whitespace value is a no-op — mirrors the QDRANT_URL contract.

    Compares against a baseline load with the env unset so this test is
    independent of which router id ships in yaml.
    """
    monkeypatch.delenv("CLARITYMED_RAG_ROUTER", raising=False)
    baseline = load_retrieval_config().router.active
    monkeypatch.setenv("CLARITYMED_RAG_ROUTER", "   ")
    assert load_retrieval_config().router.active == baseline


def test_router_env_var_unknown_id_fails_loud(monkeypatch):
    """Unknown id is caught by ``RouterConfig._resolve_active`` — no silent fallback.

    The override path must not weaken the existing fail-loud guarantee:
    a typo here would otherwise route every query to a router that
    doesn't exist.
    """
    monkeypatch.setenv("CLARITYMED_RAG_ROUTER", "nope")
    with pytest.raises(UnknownRouterError):
        load_retrieval_config()


def test_mode_env_var_overrides_yaml(monkeypatch):
    """``CLARITYMED_RAG_MODE`` flips ``rag.mode`` without editing YAML.

    Eval/dev A/B between deterministic and tool mode is the prime use
    case — one shell var, no config churn.
    """
    monkeypatch.setenv("CLARITYMED_RAG_MODE", "tool")
    assert load_retrieval_config().rag.mode == "tool"


def test_mode_env_var_empty_leaves_yaml(monkeypatch):
    """Empty / whitespace value is a no-op — matches the other RAG envs.

    Baselines against an env-unset load so the test doesn't fossilize the
    current yaml mode default.
    """
    monkeypatch.delenv("CLARITYMED_RAG_MODE", raising=False)
    baseline = load_retrieval_config().rag.mode
    monkeypatch.setenv("CLARITYMED_RAG_MODE", "   ")
    assert load_retrieval_config().rag.mode == baseline


def test_mode_env_var_invalid_fails_loud(monkeypatch):
    """Typos must not silently route to an undefined mode.

    Pydantic's ``Literal["deterministic", "tool", "agentic"]`` validator
    rejects anything else at load time — no extra check needed.
    """
    monkeypatch.setenv("CLARITYMED_RAG_MODE", "turbo")
    with pytest.raises(ValidationError):
        load_retrieval_config()


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
