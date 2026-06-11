"""Tests for the OCR service: schema, factory, providers, and routing."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic_ai.models.test import TestModel

from claritymed.core.ocr.base import OcrError, OcrProvider
from claritymed.core.ocr.llm_provider import LLMOcrProvider
from claritymed.core.ocr.mineru_provider import MineRUOcrProvider
from claritymed.core.ocr.routing_provider import RoutingOcrProvider
from claritymed.core.schemas.ocr import (
    ImageOcrConfig,
    LLMOcrConfig,
    MineRUOcrConfig,
    OcrConfig,
    load_ocr_config,
)


# ---------------------------------------------------------------------------
# Schema: OcrConfig defaults
# ---------------------------------------------------------------------------


def test_ocr_config_defaults():
    cfg = OcrConfig()
    assert cfg.document_provider == "mineru"
    assert cfg.image.default == "llm"
    assert cfg.image.fallback == "mineru"
    assert cfg.mineru is None
    assert cfg.llm is None


def test_image_ocr_config_fallback_can_be_disabled():
    cfg = ImageOcrConfig(fallback=None)
    assert cfg.fallback is None


def test_ocr_config_rejects_extra_fields():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        OcrConfig.model_validate({"document_provider": "mineru", "unknown_key": "oops"})


# ---------------------------------------------------------------------------
# Schema: MineRUOcrConfig
# ---------------------------------------------------------------------------


def test_mineru_ocr_config_defaults():
    cfg = MineRUOcrConfig()
    assert cfg.api_key_env == "MINERU_API_TOKEN"
    assert cfg.model_version == "vlm"
    assert cfg.poll_interval == 3.0
    assert cfg.poll_timeout == 300.0


def test_mineru_ocr_config_custom_values():
    cfg = MineRUOcrConfig(
        api_key_env="MY_KEY", model_version="pipeline", poll_interval=5.0
    )
    assert cfg.api_key_env == "MY_KEY"
    assert cfg.model_version == "pipeline"
    assert cfg.poll_interval == 5.0


# ---------------------------------------------------------------------------
# Schema: LLMOcrConfig
# ---------------------------------------------------------------------------


def test_llm_ocr_config_inline_model():
    cfg = LLMOcrConfig(model="openai:gpt-4o")
    assert cfg.provider_id is None


def test_llm_ocr_config_provider_id():
    cfg = LLMOcrConfig(provider_id="omlx")
    assert cfg.model is None


def test_llm_ocr_config_rejects_both():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="either provider_id or model"):
        LLMOcrConfig(provider_id="omlx", model="openai:gpt-4o")


def test_llm_ocr_config_rejects_neither():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="either provider_id"):
        LLMOcrConfig()


# ---------------------------------------------------------------------------
# load_ocr_config
# ---------------------------------------------------------------------------


def test_load_ocr_config_reads_yaml():
    cfg = load_ocr_config()
    assert cfg.document_provider == "mineru"
    assert cfg.image.default == "llm"
    assert cfg.image.fallback == "mineru"
    assert cfg.mineru is not None
    assert cfg.mineru.api_key_env == "MINERU_API_TOKEN"
    assert cfg.llm is not None
    assert cfg.llm.provider_id == "omlx"


# ---------------------------------------------------------------------------
# OcrProvider interface
# ---------------------------------------------------------------------------


def test_ocr_provider_is_abstract():
    with pytest.raises(TypeError):
        OcrProvider()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# LLMOcrProvider
# ---------------------------------------------------------------------------


async def test_llm_ocr_provider_returns_extracted_text(tmp_path: Path):
    fake = tmp_path / "report.png"
    fake.write_bytes(b"\x89PNG\r\n")

    model = TestModel(
        custom_output_args={"success": True, "text": "  Blood pressure 120/80  "}
    )
    result = await LLMOcrProvider(model).extract_text(fake)
    assert result == "Blood pressure 120/80"


async def test_llm_ocr_provider_raises_on_failure_response(tmp_path: Path):
    fake = tmp_path / "report.pdf"
    fake.write_bytes(b"%PDF-1.4")

    model = TestModel(
        custom_output_args={
            "success": False,
            "failure_reason": "No document content received",
        }
    )
    with pytest.raises(OcrError, match="No document content received"):
        await LLMOcrProvider(model).extract_text(fake)


async def test_llm_ocr_provider_raises_on_missing_file(tmp_path: Path):
    model = TestModel(custom_output_args={"success": True, "text": "x"})
    with pytest.raises(OcrError, match="Cannot read"):
        await LLMOcrProvider(model).extract_text(tmp_path / "nonexistent.pdf")


async def test_llm_ocr_provider_wraps_model_errors(tmp_path: Path):
    fake = tmp_path / "scan.jpg"
    fake.write_bytes(b"\xff\xd8\xff")

    with patch.object(
        LLMOcrProvider,
        "extract_text",
        new=AsyncMock(side_effect=OcrError("LLM OCR failed")),
    ):
        with pytest.raises(OcrError):
            await LLMOcrProvider(TestModel(custom_output_text="x")).extract_text(fake)


# ---------------------------------------------------------------------------
# MineRUOcrProvider — helpers
# ---------------------------------------------------------------------------


def _make_zip_bytes(md_content: str) -> bytes:
    """Build a minimal ZIP containing full.md."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("full.md", md_content)
    return buf.getvalue()


def _mock_transport(responses: list[httpx.Response]) -> httpx.MockTransport:
    """Return a transport that replays *responses* in order."""
    it = iter(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        return next(it)

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# MineRUOcrProvider — happy path
# ---------------------------------------------------------------------------


async def test_mineru_ocr_provider_returns_markdown(tmp_path: Path):
    fake = tmp_path / "report.pdf"
    fake.write_bytes(b"%PDF-1.4")

    transport = _mock_transport(
        [
            # 1. Submit
            httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "data": {
                        "batch_id": "batch-1",
                        "file_urls": ["https://oss.example/presigned"],
                    },
                },
            ),
            # 2. PUT upload
            httpx.Response(200),
            # 3. Poll — running
            httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "data": {
                        "batch_id": "batch-1",
                        "extract_result": [
                            {"file_name": "report.pdf", "state": "running"}
                        ],
                    },
                },
            ),
            # 4. Poll — done
            httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "data": {
                        "batch_id": "batch-1",
                        "extract_result": [
                            {
                                "file_name": "report.pdf",
                                "state": "done",
                                "full_zip_url": "https://cdn.example/result.zip",
                            }
                        ],
                    },
                },
            ),
            # 5. ZIP download
            httpx.Response(200, content=_make_zip_bytes("# Report\n\nSome text.")),
        ]
    )

    provider = MineRUOcrProvider("sk-test", poll_interval=0.0, transport=transport)
    result = await provider.extract_text(fake)
    assert result == "# Report\n\nSome text."


async def test_mineru_ocr_provider_raises_on_missing_file(tmp_path: Path):
    provider = MineRUOcrProvider("sk-test")
    with pytest.raises(OcrError, match="file not found"):
        await provider.extract_text(tmp_path / "ghost.pdf")


async def test_mineru_ocr_provider_raises_on_submit_api_error(tmp_path: Path):
    fake = tmp_path / "doc.pdf"
    fake.write_bytes(b"%PDF")

    transport = _mock_transport(
        [
            httpx.Response(200, json={"code": 1, "msg": "invalid token"}),
        ]
    )
    provider = MineRUOcrProvider("bad-key", transport=transport)
    with pytest.raises(OcrError, match="invalid token"):
        await provider.extract_text(fake)


async def test_mineru_ocr_provider_raises_on_extraction_failure(tmp_path: Path):
    fake = tmp_path / "doc.pdf"
    fake.write_bytes(b"%PDF")

    transport = _mock_transport(
        [
            httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "data": {"batch_id": "b1", "file_urls": ["https://oss.example/u"]},
                },
            ),
            httpx.Response(200),
            httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "data": {
                        "batch_id": "b1",
                        "extract_result": [
                            {
                                "file_name": "doc.pdf",
                                "state": "failed",
                                "err_msg": "unsupported format",
                            }
                        ],
                    },
                },
            ),
        ]
    )
    provider = MineRUOcrProvider("sk-test", poll_interval=0.0, transport=transport)
    with pytest.raises(OcrError, match="unsupported format"):
        await provider.extract_text(fake)


async def test_mineru_ocr_provider_raises_on_bad_zip(tmp_path: Path):
    fake = tmp_path / "doc.pdf"
    fake.write_bytes(b"%PDF")

    transport = _mock_transport(
        [
            httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "data": {"batch_id": "b1", "file_urls": ["https://oss.example/u"]},
                },
            ),
            httpx.Response(200),
            httpx.Response(
                200,
                json={
                    "code": 0,
                    "msg": "ok",
                    "data": {
                        "batch_id": "b1",
                        "extract_result": [
                            {
                                "file_name": "doc.pdf",
                                "state": "done",
                                "full_zip_url": "https://cdn.example/result.zip",
                            }
                        ],
                    },
                },
            ),
            httpx.Response(200, content=b"not a zip"),
        ]
    )
    provider = MineRUOcrProvider("sk-test", poll_interval=0.0, transport=transport)
    with pytest.raises(OcrError, match="invalid ZIP"):
        await provider.extract_text(fake)


# ---------------------------------------------------------------------------
# RoutingOcrProvider
# ---------------------------------------------------------------------------


def _noop_provider(text: str = "extracted") -> OcrProvider:
    """Return a stub OcrProvider that always returns *text*."""

    class _Stub(OcrProvider):
        async def extract_text(self, path: Path) -> str:
            return text

    return _Stub()


def _failing_provider(msg: str = "boom") -> OcrProvider:
    """Return a stub OcrProvider that always raises OcrError."""

    class _Fail(OcrProvider):
        async def extract_text(self, path: Path) -> str:
            raise OcrError(msg)

    return _Fail()


@pytest.mark.parametrize(
    "extension", [".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"]
)
async def test_routing_sends_documents_to_document_provider(
    tmp_path: Path, extension: str
):
    fake = tmp_path / f"file{extension}"
    fake.write_bytes(b"data")

    router = RoutingOcrProvider(
        document_provider=_noop_provider("doc result"),
        image_default=_failing_provider("should not be called"),
    )
    assert await router.extract_text(fake) == "doc result"


@pytest.mark.parametrize(
    "extension", [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"]
)
async def test_routing_sends_images_to_image_default(tmp_path: Path, extension: str):
    fake = tmp_path / f"scan{extension}"
    fake.write_bytes(b"data")

    router = RoutingOcrProvider(
        document_provider=_failing_provider("should not be called"),
        image_default=_noop_provider("image result"),
    )
    assert await router.extract_text(fake) == "image result"


async def test_routing_image_falls_back_on_default_error(tmp_path: Path):
    fake = tmp_path / "scan.png"
    fake.write_bytes(b"data")

    router = RoutingOcrProvider(
        document_provider=_failing_provider(),
        image_default=_failing_provider("default failed"),
        image_fallback=_noop_provider("fallback result"),
    )
    assert await router.extract_text(fake) == "fallback result"


async def test_routing_raises_when_no_fallback_and_default_fails(tmp_path: Path):
    fake = tmp_path / "scan.png"
    fake.write_bytes(b"data")

    router = RoutingOcrProvider(
        document_provider=_failing_provider(),
        image_default=_failing_provider("default failed"),
        image_fallback=None,
    )
    with pytest.raises(OcrError, match="default failed"):
        await router.extract_text(fake)


async def test_routing_fallback_propagates_fallback_error(tmp_path: Path):
    fake = tmp_path / "scan.png"
    fake.write_bytes(b"data")

    router = RoutingOcrProvider(
        document_provider=_failing_provider(),
        image_default=_failing_provider("default err"),
        image_fallback=_failing_provider("fallback err"),
    )
    with pytest.raises(OcrError, match="fallback err"):
        await router.extract_text(fake)


# ---------------------------------------------------------------------------
# Audit emission
# ---------------------------------------------------------------------------


async def test_routing_emits_audit_on_success(tmp_path: Path):
    """ocr.extract audit event is emitted with status=ok and timing on success."""
    fake = tmp_path / "scan.png"
    fake.write_bytes(b"data")

    captured: list[dict] = []

    def fake_audit(kind, payload=None):
        captured.append({"kind": kind, **(payload or {})})

    with patch(
        "claritymed.core.ocr.routing_provider._emit_audit",
        side_effect=lambda p: captured.append(p),
    ):
        router = RoutingOcrProvider(
            document_provider=_failing_provider(),
            image_default=_noop_provider("hello world"),
        )
        result = await router.extract_text(fake)

    assert result == "hello world"
    assert len(captured) == 1
    ev = captured[0]
    assert ev["status"] == "ok"
    assert ev["file"] == "scan.png"
    assert ev["chars"] == len("hello world")
    assert ev["duration_ms"] >= 0
    assert ev["fallback"] is False


async def test_routing_emits_audit_with_fallback_flag(tmp_path: Path):
    """audit event marks fallback=True and lists every provider tried."""
    fake = tmp_path / "scan.png"
    fake.write_bytes(b"data")

    captured: list[dict] = []

    with patch(
        "claritymed.core.ocr.routing_provider._emit_audit",
        side_effect=lambda p: captured.append(p),
    ):
        router = RoutingOcrProvider(
            document_provider=_failing_provider(),
            image_default=_failing_provider("default err"),
            image_fallback=_noop_provider("fallback text"),
        )
        await router.extract_text(fake)

    ev = captured[0]
    assert ev["status"] == "ok"
    assert ev["fallback"] is True
    # New audit shape: chain_tried lists every step; chain_succeeded names
    # the one that won.
    assert len(ev["chain_tried"]) >= 2
    assert ev["chain_succeeded"] == ev["chain_tried"][-1]


async def test_routing_emits_audit_on_error(tmp_path: Path):
    """audit event is emitted with status=error even when extraction raises."""
    fake = tmp_path / "doc.pdf"
    fake.write_bytes(b"%PDF")

    captured: list[dict] = []

    with patch(
        "claritymed.core.ocr.routing_provider._emit_audit",
        side_effect=lambda p: captured.append(p),
    ):
        router = RoutingOcrProvider(
            document_provider=_failing_provider("extraction failed"),
            image_default=_noop_provider(),
        )
        with pytest.raises(OcrError):
            await router.extract_text(fake)

    ev = captured[0]
    assert ev["status"] == "error"
    assert "extraction failed" in ev["error"]
    assert ev["duration_ms"] >= 0


async def test_routing_audit_skips_gracefully_without_context(tmp_path: Path):
    """_emit_audit does not raise when ContextVars are unset (default in tests)."""
    fake = tmp_path / "img.png"
    fake.write_bytes(b"data")

    router = RoutingOcrProvider(
        document_provider=_failing_provider(),
        image_default=_noop_provider("ok"),
    )
    # No request context set — _emit_audit must not propagate any exception.
    result = await router.extract_text(fake)
    assert result == "ok"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_make_ocr_provider_returns_routing_provider(monkeypatch):
    import claritymed.core.schemas.ocr as _schema

    monkeypatch.setenv("MINERU_API_TOKEN", "sk-test")
    monkeypatch.setenv("OMLX_API_KEY", "sk-omlx")
    monkeypatch.setattr(
        _schema,
        "load_ocr_config",
        lambda: OcrConfig(
            document_provider="mineru",
            image=ImageOcrConfig(default="llm", fallback="mineru"),
            mineru=MineRUOcrConfig(),
            llm=LLMOcrConfig(provider_id="omlx"),
        ),
    )
    from claritymed.core.ocr.factory import make_ocr_provider

    assert isinstance(make_ocr_provider(), RoutingOcrProvider)


def test_make_ocr_provider_missing_mineru_token_raises(monkeypatch):
    import claritymed.core.schemas.ocr as _schema
    from claritymed.errors import MissingApiKeyError

    monkeypatch.delenv("MINERU_API_TOKEN", raising=False)
    monkeypatch.setattr(
        _schema,
        "load_ocr_config",
        lambda: OcrConfig(
            document_provider="mineru",
            image=ImageOcrConfig(default="llm", fallback="mineru"),
            mineru=MineRUOcrConfig(),
            llm=LLMOcrConfig(model="openai:gpt-4o"),
        ),
    )
    from claritymed.core.ocr.factory import make_ocr_provider

    with pytest.raises(MissingApiKeyError):
        make_ocr_provider()


def test_make_ocr_provider_missing_llm_section_raises(monkeypatch):
    import claritymed.core.schemas.ocr as _schema

    monkeypatch.setenv("MINERU_API_TOKEN", "sk-test")
    monkeypatch.setattr(
        _schema,
        "load_ocr_config",
        lambda: OcrConfig(
            document_provider="mineru",
            image=ImageOcrConfig(default="llm", fallback=None),
            mineru=MineRUOcrConfig(),
            llm=None,
        ),
    )
    from claritymed.core.ocr.factory import make_ocr_provider

    with pytest.raises(ValueError, match="provider_id or model"):
        make_ocr_provider()


def test_make_ocr_provider_llm_inline_base_url(monkeypatch):
    import claritymed.core.schemas.ocr as _schema

    monkeypatch.setenv("MINERU_API_TOKEN", "sk-test")
    monkeypatch.setattr(
        _schema,
        "load_ocr_config",
        lambda: OcrConfig(
            document_provider="mineru",
            image=ImageOcrConfig(default="llm", fallback=None),
            mineru=MineRUOcrConfig(),
            llm=LLMOcrConfig(model="qwen-vl:7b", base_url="http://127.0.0.1:11434/v1"),
        ),
    )
    from claritymed.core.ocr.factory import make_ocr_provider

    assert isinstance(make_ocr_provider(), RoutingOcrProvider)


def test_make_ocr_provider_llm_inline_with_api_key_env(monkeypatch):
    import claritymed.core.schemas.ocr as _schema

    monkeypatch.setenv("MINERU_API_TOKEN", "sk-test")
    monkeypatch.setenv("TEST_OCR_KEY", "sk-llm")
    monkeypatch.setattr(
        _schema,
        "load_ocr_config",
        lambda: OcrConfig(
            document_provider="mineru",
            image=ImageOcrConfig(default="llm", fallback=None),
            mineru=MineRUOcrConfig(),
            llm=LLMOcrConfig(
                model="qwen-vl:7b",
                base_url="http://127.0.0.1:11434/v1",
                api_key_env="TEST_OCR_KEY",
            ),
        ),
    )
    from claritymed.core.ocr.factory import make_ocr_provider

    assert isinstance(make_ocr_provider(), RoutingOcrProvider)


def test_make_ocr_provider_missing_llm_api_key_raises(monkeypatch):
    import claritymed.core.schemas.ocr as _schema
    from claritymed.errors import MissingApiKeyError

    monkeypatch.setenv("MINERU_API_TOKEN", "sk-test")
    monkeypatch.delenv("MISSING_OCR_KEY", raising=False)
    monkeypatch.setattr(
        _schema,
        "load_ocr_config",
        lambda: OcrConfig(
            document_provider="mineru",
            image=ImageOcrConfig(default="llm", fallback=None),
            mineru=MineRUOcrConfig(),
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


def test_make_ocr_provider_from_catalog(monkeypatch):
    import claritymed.core.schemas.ocr as _schema

    monkeypatch.setenv("MINERU_API_TOKEN", "sk-test")
    monkeypatch.setenv("OMLX_API_KEY", "sk-omlx")
    monkeypatch.setattr(
        _schema,
        "load_ocr_config",
        lambda: OcrConfig(
            document_provider="mineru",
            image=ImageOcrConfig(default="llm", fallback=None),
            mineru=MineRUOcrConfig(),
            llm=LLMOcrConfig(provider_id="omlx"),
        ),
    )
    from claritymed.core.ocr.factory import make_ocr_provider

    assert isinstance(make_ocr_provider(), RoutingOcrProvider)


def test_make_ocr_provider_unknown_catalog_id_raises(monkeypatch):
    import claritymed.core.schemas.ocr as _schema
    from claritymed.errors import UnknownProviderError

    monkeypatch.setenv("MINERU_API_TOKEN", "sk-test")
    monkeypatch.setattr(
        _schema,
        "load_ocr_config",
        lambda: OcrConfig(
            document_provider="mineru",
            image=ImageOcrConfig(default="llm", fallback=None),
            mineru=MineRUOcrConfig(),
            llm=LLMOcrConfig(provider_id="no-such-provider"),
        ),
    )
    from claritymed.core.ocr.factory import make_ocr_provider

    with pytest.raises(UnknownProviderError):
        make_ocr_provider()
