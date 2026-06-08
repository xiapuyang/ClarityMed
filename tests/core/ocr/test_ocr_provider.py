"""Tests for the OCR service: schema, factory, and LLMOcrProvider."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai.models.test import TestModel

from claritymed.core.ocr.base import OcrError, OcrProvider
from claritymed.core.ocr.llm_provider import LLMOcrProvider
from claritymed.core.schemas.ocr import LLMOcrConfig, OcrConfig, load_ocr_config


# --- schema validation ---------------------------------------------------


def test_ocr_config_defaults():
    cfg = OcrConfig()
    assert cfg.provider == "llm"
    assert cfg.llm is None


def test_llm_ocr_config_inline_model():
    cfg = LLMOcrConfig(model="openai:gpt-4o")
    assert cfg.provider_id is None
    assert cfg.base_url is None
    assert cfg.api_key_env is None


def test_llm_ocr_config_provider_id():
    cfg = LLMOcrConfig(provider_id="omlx")
    assert cfg.provider_id == "omlx"
    assert cfg.model is None


def test_llm_ocr_config_inline_with_base_url():
    cfg = LLMOcrConfig(model="qwen-vl:7b", base_url="http://127.0.0.1:11434/v1")
    assert cfg.base_url == "http://127.0.0.1:11434/v1"


def test_llm_ocr_config_rejects_both_provider_id_and_model():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="either provider_id or model"):
        LLMOcrConfig(provider_id="omlx", model="openai:gpt-4o")


def test_llm_ocr_config_rejects_neither():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="either provider_id"):
        LLMOcrConfig()


def test_ocr_config_rejects_extra_fields():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        OcrConfig.model_validate({"provider": "llm", "unknown_key": "oops"})


# --- load_ocr_config -----------------------------------------------------


def test_load_ocr_config_reads_yaml():
    cfg = load_ocr_config()
    assert cfg.provider == "llm"
    assert cfg.llm is not None
    assert cfg.llm.provider_id == "omlx"


# --- OcrProvider interface -----------------------------------------------


def test_ocr_provider_is_abstract():
    with pytest.raises(TypeError):
        OcrProvider()  # type: ignore[abstract]


# --- LLMOcrProvider ------------------------------------------------------


async def test_llm_ocr_provider_returns_extracted_text(tmp_path: Path):
    fake_file = tmp_path / "report.png"
    fake_file.write_bytes(b"\x89PNG\r\n")

    provider = LLMOcrProvider(TestModel(custom_output_text="  Blood pressure 120/80  "))
    result = await provider.extract_text(fake_file)
    assert result == "Blood pressure 120/80"


async def test_llm_ocr_provider_raises_ocr_error_on_missing_file(tmp_path: Path):
    provider = LLMOcrProvider(TestModel(custom_output_text="x"))
    with pytest.raises(OcrError, match="Cannot read"):
        await provider.extract_text(tmp_path / "nonexistent.pdf")


async def test_llm_ocr_provider_wraps_model_errors(tmp_path: Path):
    fake_file = tmp_path / "scan.jpg"
    fake_file.write_bytes(b"\xff\xd8\xff")

    with patch.object(
        LLMOcrProvider,
        "extract_text",
        new=AsyncMock(side_effect=OcrError("LLM OCR failed")),
    ):
        provider = LLMOcrProvider(TestModel(custom_output_text="x"))
        with pytest.raises(OcrError):
            await provider.extract_text(fake_file)


# --- factory: inline model path ------------------------------------------


def test_make_ocr_provider_inline_base_url(monkeypatch):
    """Inline model + base_url builds LLMOcrProvider without API key."""
    import claritymed.core.schemas.ocr as _ocr_schema

    monkeypatch.setattr(
        _ocr_schema,
        "load_ocr_config",
        lambda: OcrConfig(
            provider="llm",
            llm=LLMOcrConfig(model="qwen-vl:7b", base_url="http://127.0.0.1:11434/v1"),
        ),
    )
    from claritymed.core.ocr.factory import make_ocr_provider

    assert isinstance(make_ocr_provider(), LLMOcrProvider)


def test_make_ocr_provider_inline_with_api_key_env(monkeypatch):
    """api_key_env is read from the environment when base_url is set."""
    import claritymed.core.schemas.ocr as _ocr_schema

    monkeypatch.setenv("TEST_OCR_KEY", "sk-test")
    monkeypatch.setattr(
        _ocr_schema,
        "load_ocr_config",
        lambda: OcrConfig(
            provider="llm",
            llm=LLMOcrConfig(
                model="qwen-vl:7b",
                base_url="http://127.0.0.1:11434/v1",
                api_key_env="TEST_OCR_KEY",
            ),
        ),
    )
    from claritymed.core.ocr.factory import make_ocr_provider

    assert isinstance(make_ocr_provider(), LLMOcrProvider)


def test_make_ocr_provider_missing_api_key_env_raises(monkeypatch):
    import claritymed.core.schemas.ocr as _ocr_schema
    from claritymed.errors import MissingApiKeyError

    monkeypatch.delenv("MISSING_OCR_KEY", raising=False)
    monkeypatch.setattr(
        _ocr_schema,
        "load_ocr_config",
        lambda: OcrConfig(
            provider="llm",
            llm=LLMOcrConfig(
                model="qwen-vl:7b",
                base_url="http://127.0.0.1:11434/v1",
                api_key_env="MISSING_OCR_KEY",
            ),
        ),
    )
    from claritymed.core.ocr.factory import make_ocr_provider

    with pytest.raises(MissingApiKeyError):
        make_ocr_provider()


# --- factory: provider_id path -------------------------------------------


def test_make_ocr_provider_from_catalog(monkeypatch):
    """provider_id resolves the model via the models.yaml catalog."""
    import claritymed.core.schemas.ocr as _ocr_schema

    monkeypatch.setenv("OMLX_API_KEY", "sk-test")
    monkeypatch.setattr(
        _ocr_schema,
        "load_ocr_config",
        lambda: OcrConfig(provider="llm", llm=LLMOcrConfig(provider_id="omlx")),
    )
    from claritymed.core.ocr.factory import make_ocr_provider

    assert isinstance(make_ocr_provider(), LLMOcrProvider)


def test_make_ocr_provider_from_catalog_unknown_id_raises(monkeypatch):
    """UnknownProviderError is raised when provider_id is not in models.yaml."""
    import claritymed.core.schemas.ocr as _ocr_schema
    from claritymed.errors import UnknownProviderError

    monkeypatch.setattr(
        _ocr_schema,
        "load_ocr_config",
        lambda: OcrConfig(
            provider="llm",
            llm=LLMOcrConfig(provider_id="no-such-provider"),
        ),
    )
    from claritymed.core.ocr.factory import make_ocr_provider

    with pytest.raises(UnknownProviderError):
        make_ocr_provider()


def test_make_ocr_provider_raises_on_missing_llm_section(monkeypatch):
    import claritymed.core.schemas.ocr as _ocr_schema

    monkeypatch.setattr(
        _ocr_schema,
        "load_ocr_config",
        lambda: OcrConfig(provider="llm", llm=None),
    )
    from claritymed.core.ocr.factory import make_ocr_provider

    with pytest.raises(ValueError, match="provider_id or model"):
        make_ocr_provider()
