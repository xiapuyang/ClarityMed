"""Tests for the OCR service: schema, factory, providers, and routing."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from pydantic_ai.models.test import TestModel

from claritymed.core.ocr.base import ExtractResult, OcrEmpty, OcrError, OcrProvider
from claritymed.core.ocr.llm_provider import LLMOcrProvider
from claritymed.core.ocr.mineru_provider import MineRUOcrProvider
from claritymed.core.ocr.routing_provider import RoutingOcrProvider
from claritymed.core.schemas.ocr import (
    ChainEntry,
    LLMOcrConfig,
    MineRUOcrConfig,
    OcrConfig,
    load_ocr_config,
)


# ---------------------------------------------------------------------------
# Schema: OcrConfig
# ---------------------------------------------------------------------------


def test_ocr_config_requires_at_least_one_chain():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="at least one of"):
        OcrConfig()


def test_ocr_config_accepts_document_chain_only():
    cfg = OcrConfig(document_chain=[ChainEntry(name="pymupdf")])
    assert cfg.document_chain[0].name == "pymupdf"
    assert cfg.image_chain == []


def test_ocr_config_text_extensions_default_populated():
    cfg = OcrConfig(document_chain=[ChainEntry(name="pymupdf")])
    # Defaults sourced from _DEFAULT_TEXT_EXTENSIONS.
    assert ".md" in cfg.text_extensions
    assert ".csv" in cfg.text_extensions
    assert all(e.startswith(".") for e in cfg.text_extensions)


def test_ocr_config_text_extensions_normalize_case_and_dot():
    cfg = OcrConfig(
        document_chain=[ChainEntry(name="pymupdf")],
        text_extensions=["MD", ".CSV", "json"],
    )
    # Lowercase + leading dot — both inserted, both lowercased.
    assert cfg.text_extensions == [".md", ".csv", ".json"]


def test_ocr_config_rejects_extra_fields():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        OcrConfig.model_validate(
            {
                "document_chain": [{"name": "pymupdf"}],
                "unknown_key": "oops",
            }
        )


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
    assert cfg.document_chain  # populated chain from configs/ocr.yaml
    assert cfg.image_chain
    assert cfg.mineru is not None
    assert cfg.mineru.api_key_env == "MINERU_API_TOKEN"
    assert cfg.llm is not None
    assert cfg.llm.provider_id == "omlx"


def test_load_ocr_config_raises_when_yaml_empty(monkeypatch):
    from unittest.mock import MagicMock

    import claritymed.config as _cfg_module
    from claritymed.core.schemas.ocr import load_ocr_config

    mock_load_yaml = MagicMock(return_value=None)
    monkeypatch.setattr(_cfg_module, "load_yaml", mock_load_yaml)
    with pytest.raises(ValueError, match="missing or empty"):
        load_ocr_config()


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
    assert result.text == "Blood pressure 120/80"
    assert result.provider_used == "llm"
    assert result.chain_tried == ["llm"]


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


async def test_llm_ocr_provider_status_done_returns_text(tmp_path: Path):
    """v2 path: status='done' returns text and threads modality/is_medical."""
    fake = tmp_path / "ct.png"
    fake.write_bytes(b"\x89PNG\r\n")

    model = TestModel(
        custom_output_args={
            "status": "done",
            "text": "CT头颅平扫：未见明显异常",
            "modality": "ct",
            "is_medical": True,
        }
    )
    result = await LLMOcrProvider(model).extract_text(fake)
    assert result.text == "CT头颅平扫：未见明显异常"
    assert result.modality == "ct"
    assert result.is_medical is True


async def test_llm_ocr_provider_status_empty_raises_ocr_empty(tmp_path: Path):
    """v2 path: status='empty' → OcrEmpty (not OcrError)."""
    fake = tmp_path / "blank.png"
    fake.write_bytes(b"\x89PNG\r\n")

    model = TestModel(custom_output_args={"status": "empty", "text": ""})
    with pytest.raises(OcrEmpty, match="no text found"):
        await LLMOcrProvider(model).extract_text(fake)


async def test_llm_ocr_provider_empty_carries_modality_hint(tmp_path: Path):
    """status='empty' with a modality classification → hint travels via OcrEmpty.

    A vision LLM that read the pixels but found no readable text can still
    classify modality and is_medical. That signal must survive the empty
    path so the worker's empty branch can populate ocr.json — otherwise
    OCR-blank medical images render as bare ``<image>`` tags and the
    LLM-side routing rules in detect_disease_from_image_tool can't fire.
    """
    fake = tmp_path / "us.png"
    fake.write_bytes(b"\x89PNG\r\n")

    model = TestModel(
        custom_output_args={
            "status": "empty",
            "text": "",
            "modality": "ultrasound",
            "is_medical": True,
        }
    )
    with pytest.raises(OcrEmpty) as excinfo:
        await LLMOcrProvider(model).extract_text(fake)
    hint = excinfo.value.extraction
    assert hint is not None
    assert hint.modality == "ultrasound"
    assert hint.is_medical is True
    assert hint.chain_tried == ["llm"]
    assert hint.text == ""


async def test_llm_ocr_provider_status_failed_raises_ocr_error(tmp_path: Path):
    """v2 path: status='failed' → OcrError."""
    fake = tmp_path / "bad.pdf"
    fake.write_bytes(b"%PDF-1.4")

    model = TestModel(
        custom_output_args={
            "status": "failed",
            "failure_reason": "Unsupported format",
        }
    )
    with pytest.raises(OcrError, match="Unsupported format"):
        await LLMOcrProvider(model).extract_text(fake)


async def test_llm_ocr_provider_empty_text_after_done_raises_ocr_empty(tmp_path: Path):
    """status='done' but text is blank after strip → OcrEmpty."""
    fake = tmp_path / "blank.png"
    fake.write_bytes(b"\x89PNG\r\n")

    model = TestModel(custom_output_args={"status": "done", "text": "   "})
    with pytest.raises(OcrEmpty, match="empty text"):
        await LLMOcrProvider(model).extract_text(fake)


async def test_llm_ocr_provider_raises_when_agent_run_throws(tmp_path: Path):
    """Exception from Agent.run is wrapped in OcrError."""
    from unittest.mock import AsyncMock, patch

    fake = tmp_path / "scan.png"
    fake.write_bytes(b"\x89PNG\r\n")

    model = TestModel(custom_output_args={"status": "done", "text": "x"})
    with patch(
        "pydantic_ai.Agent.run", new=AsyncMock(side_effect=RuntimeError("timeout"))
    ):
        with pytest.raises(OcrError, match="LLM OCR failed"):
            await LLMOcrProvider(model).extract_text(fake)


async def test_llm_ocr_provider_no_status_no_success_raises_ocr_error(tmp_path: Path):
    """Neither status nor success set → treated as failed."""
    fake = tmp_path / "scan.png"
    fake.write_bytes(b"\x89PNG\r\n")

    # Both fields default to None — the fallback else-branch fires
    model = TestModel(custom_output_args={})
    with pytest.raises(OcrError):
        await LLMOcrProvider(model).extract_text(fake)


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
    assert result.text == "# Report\n\nSome text."
    assert result.provider_used == "mineru"
    assert result.chain_tried == ["mineru"]


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
# RoutingOcrProvider — routing by file kind
# ---------------------------------------------------------------------------


def _noop_provider(
    text: str = "extracted",
    *,
    label: str = "stub",
    supported: frozenset[str] | None = None,
) -> OcrProvider:
    """Return a stub OcrProvider that always returns *text*.

    ``supported`` becomes ``supported_extensions``. Defaults to ``None``
    (= "all") for tests that don't care about routing; tests covering
    document-vs-image routing pass an explicit set so the stub
    participates in ``_derive_document_extensions``.
    """

    class _Stub(OcrProvider):
        async def extract_text(self, path: Path) -> ExtractResult:
            return ExtractResult(text=text, provider_used=label, chain_tried=[label])

    _Stub.label = label
    _Stub.supported_extensions = supported
    return _Stub()


def _failing_provider(
    msg: str = "boom",
    *,
    label: str = "fail",
    supported: frozenset[str] | None = None,
) -> OcrProvider:
    """Return a stub OcrProvider that always raises OcrError."""

    class _Fail(OcrProvider):
        async def extract_text(self, path: Path) -> ExtractResult:
            raise OcrError(msg)

    _Fail.label = label
    _Fail.supported_extensions = supported
    return _Fail()


_DOC_EXTS = frozenset({".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"})


@pytest.mark.parametrize("extension", sorted(_DOC_EXTS))
async def test_routing_sends_documents_to_document_chain(
    tmp_path: Path, extension: str
):
    fake = tmp_path / f"file{extension}"
    fake.write_bytes(b"data")

    router = RoutingOcrProvider(
        document_chain=[_noop_provider("doc result", label="doc", supported=_DOC_EXTS)],
        image_chain=[_failing_provider("should not be called", label="img")],
    )
    result = await router.extract_text(fake)
    assert result.text == "doc result"
    assert result.provider_used == "doc"


@pytest.mark.parametrize(
    "extension", [".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"]
)
async def test_routing_sends_images_to_image_chain(tmp_path: Path, extension: str):
    fake = tmp_path / f"scan{extension}"
    fake.write_bytes(b"data")

    router = RoutingOcrProvider(
        document_chain=[_failing_provider("should not be called", label="doc")],
        image_chain=[_noop_provider("image result", label="img")],
    )
    result = await router.extract_text(fake)
    assert result.text == "image result"
    assert result.provider_used == "img"


async def test_routing_image_falls_back_on_default_error(tmp_path: Path):
    fake = tmp_path / "scan.png"
    fake.write_bytes(b"data")

    router = RoutingOcrProvider(
        document_chain=[_failing_provider(label="doc")],
        image_chain=[
            _failing_provider("default failed", label="img1"),
            _noop_provider("fallback result", label="img2"),
        ],
    )
    result = await router.extract_text(fake)
    assert result.text == "fallback result"
    assert result.provider_used == "img2"
    assert result.chain_tried == ["img1", "img2"]


async def test_routing_raises_when_chain_exhausted(tmp_path: Path):
    fake = tmp_path / "scan.png"
    fake.write_bytes(b"data")

    router = RoutingOcrProvider(
        document_chain=[_failing_provider(label="doc")],
        image_chain=[_failing_provider("default failed", label="img")],
    )
    with pytest.raises(OcrError, match="all providers exhausted"):
        await router.extract_text(fake)


# ---------------------------------------------------------------------------
# Audit emission
# ---------------------------------------------------------------------------


async def test_routing_emits_audit_on_success(tmp_path: Path):
    """ocr.extract audit event is emitted with status=ok and timing on success."""
    fake = tmp_path / "scan.png"
    fake.write_bytes(b"data")

    captured: list[dict] = []

    with patch(
        "claritymed.core.ocr.routing_provider._emit_audit",
        side_effect=lambda p: captured.append(p),
    ):
        router = RoutingOcrProvider(
            document_chain=[_failing_provider(label="doc")],
            image_chain=[_noop_provider("hello world", label="img")],
        )
        result = await router.extract_text(fake)

    assert result.text == "hello world"
    assert len(captured) == 1
    ev = captured[0]
    assert ev["status"] == "ok"
    # Field was renamed from ``file`` to ``blob_filename`` to make
    # room for an honest ``original_filename`` field carrying the
    # user-facing name. The on-disk blob name (``content.<ext>``
    # for production, ``scan.png`` for these fake tmp_path fixtures)
    # remains here for ops correlation.
    assert ev["blob_filename"] == "scan.png"
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
            document_chain=[_failing_provider(label="doc")],
            image_chain=[
                _failing_provider("default err", label="img1"),
                _noop_provider("fallback text", label="img2"),
            ],
        )
        await router.extract_text(fake)

    ev = captured[0]
    assert ev["status"] == "ok"
    assert ev["fallback"] is True
    # chain_tried lists every step; chain_succeeded names the one that won.
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
            document_chain=[
                _failing_provider(
                    "extraction failed", label="doc", supported=frozenset({".pdf"})
                )
            ],
            image_chain=[_noop_provider(label="img")],
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
        document_chain=[_failing_provider(label="doc")],
        image_chain=[_noop_provider("ok", label="img")],
    )
    # No request context set — _emit_audit must not propagate any exception.
    result = await router.extract_text(fake)
    assert result.text == "ok"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _chain_cfg(**overrides) -> OcrConfig:
    """Build a minimal valid OcrConfig with the given chains/llm/mineru."""
    base = {
        "document_chain": [ChainEntry(name="mineru")],
        "image_chain": [ChainEntry(name="llm")],
        "mineru": MineRUOcrConfig(),
        "llm": LLMOcrConfig(provider_id="omlx"),
    }
    base.update(overrides)
    return OcrConfig(**base)


def test_make_ocr_provider_returns_routing_provider(monkeypatch):
    import claritymed.core.schemas.ocr as _schema

    monkeypatch.setenv("MINERU_API_TOKEN", "sk-test")
    monkeypatch.setenv("OMLX_API_KEY", "sk-omlx")
    monkeypatch.setattr(_schema, "load_ocr_config", _chain_cfg)
    from claritymed.core.ocr.factory import make_ocr_provider

    assert isinstance(make_ocr_provider(), RoutingOcrProvider)


def test_make_ocr_provider_missing_mineru_token_raises(monkeypatch):
    import claritymed.core.schemas.ocr as _schema
    from claritymed.errors import MissingApiKeyError

    monkeypatch.delenv("MINERU_API_TOKEN", raising=False)
    monkeypatch.setattr(
        _schema,
        "load_ocr_config",
        lambda: _chain_cfg(llm=LLMOcrConfig(model="openai:gpt-4o")),
    )
    from claritymed.core.ocr.factory import make_ocr_provider

    with pytest.raises(MissingApiKeyError):
        make_ocr_provider()


def test_make_ocr_provider_skips_mineru_when_envgate_missing(monkeypatch, caplog):
    """Chain construction must not blow up when one entry's provider
    can't be constructed (e.g. CLARITYMED_ALLOW_MINERU missing).

    The factory treats ``MinerUNotAllowed`` / ``ImportError`` as per-entry
    skips, leaving a usable (possibly shorter) chain.
    """
    import logging

    import claritymed.core.schemas.ocr as _schema

    monkeypatch.delenv("CLARITYMED_ALLOW_MINERU", raising=False)
    monkeypatch.setenv("MINERU_API_TOKEN", "sk-test")
    monkeypatch.setattr(
        _schema,
        "load_ocr_config",
        lambda: _chain_cfg(
            document_chain=[ChainEntry(name="mineru")],
            image_chain=[
                ChainEntry(name="llm"),
                ChainEntry(name="mineru"),
            ],
            llm=LLMOcrConfig(model="qwen-vl:7b", base_url="http://127.0.0.1:11434/v1"),
        ),
    )
    from claritymed.core.ocr.factory import make_ocr_provider

    with caplog.at_level(logging.INFO, logger="claritymed.core.ocr.factory"):
        provider = make_ocr_provider()

    assert isinstance(provider, RoutingOcrProvider)
    # mineru entry skipped → empty document chain.
    assert provider._document_chain == []
    # image chain: llm kept, mineru skipped.
    assert len(provider._image_chain) == 1
    skip_msgs = [r.message for r in caplog.records if "skipping" in r.message]
    assert any("mineru" in m for m in skip_msgs), skip_msgs


def test_make_ocr_provider_missing_llm_section_raises(monkeypatch):
    import claritymed.core.schemas.ocr as _schema

    monkeypatch.setenv("MINERU_API_TOKEN", "sk-test")
    monkeypatch.setattr(
        _schema,
        "load_ocr_config",
        lambda: _chain_cfg(llm=None),
    )
    from claritymed.core.ocr.factory import make_ocr_provider

    with pytest.raises(ValueError, match="provider_id/model"):
        make_ocr_provider()


def test_make_ocr_provider_llm_inline_base_url(monkeypatch):
    import claritymed.core.schemas.ocr as _schema

    monkeypatch.setenv("MINERU_API_TOKEN", "sk-test")
    monkeypatch.setattr(
        _schema,
        "load_ocr_config",
        lambda: _chain_cfg(
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
        lambda: _chain_cfg(
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
        lambda: _chain_cfg(
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
    monkeypatch.setattr(_schema, "load_ocr_config", _chain_cfg)
    from claritymed.core.ocr.factory import make_ocr_provider

    assert isinstance(make_ocr_provider(), RoutingOcrProvider)


def test_make_ocr_provider_unknown_catalog_id_raises(monkeypatch):
    import claritymed.core.schemas.ocr as _schema
    from claritymed.errors import UnknownProviderError

    monkeypatch.setenv("MINERU_API_TOKEN", "sk-test")
    monkeypatch.setattr(
        _schema,
        "load_ocr_config",
        lambda: _chain_cfg(llm=LLMOcrConfig(provider_id="no-such-provider")),
    )
    from claritymed.core.ocr.factory import make_ocr_provider

    with pytest.raises(UnknownProviderError):
        make_ocr_provider()
