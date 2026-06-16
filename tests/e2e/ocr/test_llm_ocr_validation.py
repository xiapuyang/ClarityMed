"""Validate LLM OCR's structured fields (status / modality / is_medical).

The OCR e2e in :mod:`test_filetype_matrix` only checks that the LLM
chain *extracts* text — it doesn't verify the classifier fields v2 of
the prompt added (status, modality, is_medical). Those fields drive
downstream behavior:

* ``is_medical=False`` is the gate the vision tool uses to refuse
  non-medical images (R6 in the vision plan).
* ``modality`` is the value the modality hard gate (KTD-V3) compares
  against the loaded vision model.
* ``status="empty"`` and ``status="failed"`` change the OCR worker's
  retry logic.

A regression on any of these silently breaks a downstream contract;
this benchmark catches that drift by running the live vision LLM
against a small matrix of fixtures with known ground truth.

Skip unless an LLM provider is reachable (uses ``e2e_provider_id``).
Run explicitly::

    uv run pytest tests/e2e/ocr/test_llm_ocr_validation.py -v --no-cov
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import pytest

logger = logging.getLogger(__name__)


# --- expected-result schema ------------------------------------------------


Status = Literal["done", "empty", "failed"]


@dataclass(frozen=True)
class OcrExpectation:
    """Ground-truth fields for one OCR fixture.

    ``None`` on a field means "don't assert"; lets a case skip a
    dimension the LLM is allowed to be soft on (e.g. modality on a
    blank page, where ``null`` is also acceptable per the prompt).
    """

    name: str
    expected_status: Status
    expected_modality: str | None  # exact match when set
    expected_is_medical: bool | None  # exact match when set
    # Substrings that MUST appear in the extracted text (when status=done).
    must_contain: tuple[str, ...] = ()
    # Substrings that MUST NOT appear (watermarks, decorative stamps).
    must_not_contain: tuple[str, ...] = ()


# --- fixtures (programmatic + checked-in PNG) ------------------------------


def _make_blank_png(tmp_path: Path) -> Path:
    """Pure white 800x600 PNG — no glyphs, no marks.

    Goal: probe ``status="empty"`` — content is visible (the page) but
    no readable text.
    """
    from PIL import Image

    p = tmp_path / "blank.png"
    Image.new("RGB", (800, 600), color="white").save(str(p), format="PNG")
    return p


def _make_receipt_png(tmp_path: Path) -> Path:
    """Synthetic restaurant receipt — clearly non-medical.

    Goal: probe ``is_medical=False`` on a document that LLMs sometimes
    misclassify as medical because it contains numbers and dates.
    """
    from PIL import Image, ImageDraw

    p = tmp_path / "receipt.png"
    img = Image.new("RGB", (520, 360), color="white")
    draw = ImageDraw.Draw(img)
    lines = [
        "BLUE BOTTLE COFFEE",
        "123 Mission St, San Francisco",
        "Date: 2026-06-15  14:32",
        "------------------------",
        "Cappuccino           5.50",
        "Croissant            4.25",
        "------------------------",
        "Subtotal:            9.75",
        "Tax:                 0.85",
        "TOTAL:              10.60",
        "Thank you!",
    ]
    y = 20
    for line in lines:
        draw.text((20, y), line, fill="black")
        y += 28
    img.save(str(p), format="PNG")
    return p


def _make_watermarked_doc_png(tmp_path: Path) -> Path:
    """Document with a large diagonal "CONFIDENTIAL" watermark.

    Goal: probe the watermark-exclusion rule. The extracted text must
    contain the body content but NOT the watermark word.
    """
    from PIL import Image, ImageDraw

    p = tmp_path / "watermarked.png"
    img = Image.new("RGB", (640, 480), color="white")
    draw = ImageDraw.Draw(img)
    # Body text.
    body = [
        "Patient Discharge Summary",
        "Name: John Doe   DOB: 1980-04-22",
        "Diagnosis: Hypertension",
        "Prescription: Lisinopril 10mg daily",
    ]
    y = 60
    for line in body:
        draw.text((40, y), line, fill="black")
        y += 32
    # Light-gray watermark overlay (smaller than full diagonal to fit
    # Pillow's default font, but visually obvious to a vision LLM).
    draw.text((140, 220), "CONFIDENTIAL", fill=(220, 220, 220))
    img.save(str(p), format="PNG")
    return p


# --- fixture matrix --------------------------------------------------------


_FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "fixtures" / "vision"


def _real_ultrasound() -> Path | None:
    """Return the first real BUSI image in fixtures; ``None`` if absent.

    The fixture dir holds placeholders alongside real images — we filter
    by suffix + the ``_PLACEHOLDER`` naming convention.
    """
    for p in sorted((_FIXTURE_ROOT / "busi").glob("*")):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            return p
    return None


def _real_report_overlay() -> Path | None:
    for p in sorted((_FIXTURE_ROOT / "report_overlay").glob("*")):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            return p
    return None


@dataclass(frozen=True)
class _Case:
    spec: OcrExpectation
    make: Callable[[Path], Path]  # tmp_path → path-to-fixture


def _make_real_ultrasound(_tmp_path: Path) -> Path:
    p = _real_ultrasound()
    if p is None:
        pytest.skip(
            "no real ultrasound fixture in tests/fixtures/vision/busi/ — "
            "see tests/fixtures/vision/README.md for sourcing"
        )
    return p


def _make_real_report(_tmp_path: Path) -> Path:
    p = _real_report_overlay()
    if p is None:
        pytest.skip(
            "no real report-overlay fixture in tests/fixtures/vision/report_overlay/"
        )
    return p


_CASES: list[_Case] = [
    _Case(
        spec=OcrExpectation(
            name="ultrasound_busi",
            expected_status="done",
            expected_modality="ultrasound",
            expected_is_medical=True,
        ),
        make=_make_real_ultrasound,
    ),
    _Case(
        spec=OcrExpectation(
            name="report_overlay_busi",
            expected_status="done",
            expected_modality=None,  # ultrasound with overlay text — accept either
            expected_is_medical=True,
        ),
        make=_make_real_report,
    ),
    _Case(
        spec=OcrExpectation(
            name="restaurant_receipt",
            expected_status="done",
            expected_modality=None,  # prompt allows null when not medical
            expected_is_medical=False,
            must_contain=("TOTAL",),
        ),
        make=_make_receipt_png,
    ),
    _Case(
        spec=OcrExpectation(
            name="blank_page",
            expected_status="empty",
            expected_modality=None,
            expected_is_medical=None,
        ),
        make=_make_blank_png,
    ),
    _Case(
        spec=OcrExpectation(
            name="watermarked_doc",
            expected_status="done",
            expected_modality=None,  # accept document/photo/unknown
            expected_is_medical=True,
            must_contain=("Hypertension",),
            must_not_contain=("CONFIDENTIAL",),
        ),
        make=_make_watermarked_doc_png,
    ),
]


# --- helpers ---------------------------------------------------------------


def _build_llm_ocr_provider(provider_id: str):
    """Construct an :class:`LLMOcrProvider` against the chosen catalog provider.

    Mirrors what :func:`ocr.factory.build_provider_chain` does when it
    reads the catalog ``provider_id`` from ``ocr.yaml``, but skips the
    config indirection so this test stands alone.
    """
    from claritymed.core.llm.model import build_model
    from claritymed.core.ocr.llm_provider import LLMOcrProvider
    from claritymed.stores.models import resolve_provider

    provider = resolve_provider(override=provider_id)
    model = build_model(provider)
    return LLMOcrProvider(model, is_local=provider.kind == "local")


def _assert_expectation(
    expectation: OcrExpectation,
    *,
    status: str,
    modality: str | None,
    is_medical: bool | None,
    text: str,
) -> None:
    """Assert each declared dimension; ``None`` fields are skipped."""
    assert status == expectation.expected_status, (
        f"{expectation.name}: status={status!r} expected "
        f"{expectation.expected_status!r}"
    )
    if expectation.expected_modality is not None:
        assert modality == expectation.expected_modality, (
            f"{expectation.name}: modality={modality!r} expected "
            f"{expectation.expected_modality!r}"
        )
    if expectation.expected_is_medical is not None:
        assert is_medical == expectation.expected_is_medical, (
            f"{expectation.name}: is_medical={is_medical!r} expected "
            f"{expectation.expected_is_medical!r}"
        )
    for must in expectation.must_contain:
        assert must in text, (
            f"{expectation.name}: expected substring {must!r} missing from "
            f"extracted text: {text[:200]!r}"
        )
    for must_not in expectation.must_not_contain:
        assert must_not not in text, (
            f"{expectation.name}: forbidden substring {must_not!r} "
            f"present in extracted text: {text[:200]!r}"
        )


# --- the test --------------------------------------------------------------


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.spec.name)
@pytest.mark.asyncio
async def test_llm_ocr_returns_expected_fields(
    case: _Case, e2e_provider_id: str, tmp_path: Path
) -> None:
    """Live LLM OCR on a known fixture; assert structured fields match.

    Each case writes its fixture into ``tmp_path``, runs the LLM OCR
    provider against it, and verifies status/modality/is_medical plus
    optional substring constraints. When ``status="empty"`` is asserted
    we accept the provider raising :class:`OcrEmpty` as the equivalent
    of ``status="empty"`` — that's what the provider does in code.
    """
    from claritymed.core.ocr.base import OcrEmpty, OcrError

    fixture = case.make(tmp_path)
    provider = _build_llm_ocr_provider(e2e_provider_id)
    try:
        result = await provider.extract_text(fixture)
    except OcrEmpty as exc:
        # ``status="empty"`` is raised as OcrEmpty by the provider so the
        # chain can fall through; the test must accept that as the
        # success signal for the blank-page case.
        if case.spec.expected_status == "empty":
            logger.info(
                "%s: provider raised OcrEmpty (expected): %s", case.spec.name, exc
            )
            return
        raise
    except OcrError as exc:
        if case.spec.expected_status == "failed":
            logger.info(
                "%s: provider raised OcrError (expected): %s", case.spec.name, exc
            )
            return
        pytest.fail(f"{case.spec.name}: unexpected OcrError: {exc}")

    _assert_expectation(
        case.spec,
        status="done",
        modality=result.modality,
        is_medical=result.is_medical,
        text=result.text,
    )
