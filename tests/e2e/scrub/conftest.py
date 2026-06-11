"""E2E scrub fixtures — requires the ONNX privacy-filter model cached locally.

Run with:
    uv run pytest tests/e2e/scrub -v --no-cov

Required:
    - onnxruntime installed (uv sync --extra privacy-filter)
    - openai/privacy-filter ONNX weights cached in the HuggingFace hub
      (triggered automatically on first ``uv run claritymed tui`` run, or
       manually: uv run python -c "from claritymed.core.scrub.service import
       ScrubService; ScrubService.from_config().ensure_downloaded()")
"""

from __future__ import annotations

import pytest

from claritymed.core.scrub.service import ScrubService


@pytest.fixture(scope="session", autouse=True)
def require_local_services() -> None:
    """Override parent conftest — scrub e2e only needs the ONNX model, not HTTP services."""


@pytest.fixture(autouse=True)
def _seed_terminology() -> None:
    """Override parent conftest — not needed for scrub e2e tests."""


@pytest.fixture(scope="session", autouse=True)
def require_privacy_filter_model() -> None:
    """Skip all tests in this directory if the ONNX model is not available locally."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        pytest.skip(
            "onnxruntime not installed; run: uv sync --extra privacy-filter",
            allow_module_level=True,
        )

    from claritymed.core.scrub.service import ScrubService

    svc = ScrubService.from_config()
    if not svc._config.privacy_filter.enabled:
        pytest.skip(
            "privacy_filter.enabled=false in configs/safety.yaml",
            allow_module_level=True,
        )

    onnx_file = svc._config.privacy_filter.onnx_file
    if not onnx_file:
        pytest.skip(
            "ONNX path required for e2e tests (onnx_file is null in config)",
            allow_module_level=True,
        )

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=svc._config.privacy_filter.model_name,
            local_files_only=True,
            allow_patterns=[
                "config.json",
                "tokenizer*.json",
                "special_tokens_map.json",
                onnx_file,
                f"{onnx_file}_data",
            ],
        )
    except (ImportError, Exception) as exc:
        pytest.skip(
            f"privacy-filter model not cached locally ({exc}); download with:\n"
            '  uv run python -c "from claritymed.core.scrub.service import '
            'ScrubService; ScrubService.from_config().ensure_downloaded()"',
            allow_module_level=True,
        )


@pytest.fixture(autouse=True)
def _request_context():
    """Seed request ContextVars for every e2e scrub test.

    ``_emit_scrub_audit`` calls ``audit_event``, which requires
    request_id / user_id / language to be set — otherwise it raises
    MissingContextError and the broad except in ``_emit_scrub_audit`` silently
    swallows it, leaving no scrub.privacy_filter entry in the audit log.
    """
    from claritymed.context import apply_context, new_request_id, reset_context

    tokens = apply_context(new_request_id(), "e2e-test", "en")
    yield
    reset_context(tokens)


@pytest.fixture(scope="module")
def scrub_service() -> ScrubService:
    """Return a fully-loaded ScrubService with the real ONNX pipeline."""
    svc = ScrubService.from_config()
    pipe = svc._get_pipeline()
    assert pipe is not None, "ONNX pipeline failed to load — check model cache"
    return svc


@pytest.fixture(scope="module")
def model_only_scrub_service(scrub_service: ScrubService) -> ScrubService:
    """ScrubService with no regex rules — exercises only the model layer.

    Reuses the already-loaded pipeline from ``scrub_service`` so weights are
    not loaded twice. Use this fixture for tests that must assert on
    ``model_hit_types``; without it the regex pre-pass removes email and phone
    before the model sees them, leaving model_hits=0.
    """
    from claritymed.core.scrub.service import ScrubConfig

    config = ScrubConfig(
        free_text_patterns=[],
        privacy_filter=scrub_service._config.privacy_filter,
    )
    svc = ScrubService(config)
    svc._pipeline = scrub_service._pipeline
    svc._pipeline_tried = True
    return svc
