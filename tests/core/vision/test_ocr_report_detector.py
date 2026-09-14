"""Tests for the OCR-report heuristic (KTD-V6).

Pure function — no network, no disk, no model. Exercises the length
floor, English + Chinese marker matching, language scoping, and the
all-languages fallback the OCR worker relies on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from claritymed.core.vision.ocr_report_detector import (
    DEFAULT_MIN_CHARS,
    has_structured_report,
    load_ocr_report_config,
)


# --- shared fixtures -----------------------------------------------------

_EN_MARKERS = ["findings", "impression", "conclusion", "diagnosis", "recommendation"]
_ZH_MARKERS = ["所见", "印象", "结论", "诊断", "建议"]
_MARKERS = {"en": _EN_MARKERS, "zh": _ZH_MARKERS}


def _padded(text: str, length: int) -> str:
    """Inflate ``text`` to ``length`` chars with neutral padding.

    Keeps marker-keyword presence stable while letting tests dial the
    length floor up and down without rewriting the body each time.
    """
    if len(text) >= length:
        return text
    return text + ("x" * (length - len(text)))


# --- length floor --------------------------------------------------------


def test_below_min_chars_short_circuits_to_false() -> None:
    text = "FINDINGS: small mass."
    assert has_structured_report(text, "en", min_chars=200, markers=_MARKERS) is False


def test_whitespace_padded_text_still_below_floor() -> None:
    text = "FINDINGS\n\n" + (" " * 1000)
    # Stripped length is 8 → way under floor; the spaces don't help.
    assert has_structured_report(text, "en", min_chars=200, markers=_MARKERS) is False


def test_at_min_chars_with_marker_returns_true() -> None:
    text = _padded("findings: lesion in upper outer quadrant.", 200)
    assert has_structured_report(text, "en", min_chars=200, markers=_MARKERS) is True


# --- English markers -----------------------------------------------------


@pytest.mark.parametrize("marker", _EN_MARKERS)
def test_each_english_marker_fires_override(marker: str) -> None:
    text = _padded(f"... {marker} appear consistent with ...", 250)
    assert has_structured_report(text, "en", min_chars=200, markers=_MARKERS) is True


def test_uppercase_marker_matches_case_insensitively() -> None:
    text = _padded("FINDINGS: nodule 8mm.", 220)
    assert has_structured_report(text, "en", min_chars=200, markers=_MARKERS) is True


def test_text_with_no_english_marker_is_false() -> None:
    text = _padded("a long prose paragraph about nothing in particular.", 250)
    assert has_structured_report(text, "en", min_chars=200, markers=_MARKERS) is False


# --- Chinese markers -----------------------------------------------------


@pytest.mark.parametrize("marker", _ZH_MARKERS)
def test_each_chinese_marker_fires_override(marker: str) -> None:
    text = _padded(f"超声检查{marker}: 右乳低回声结节", 220)
    assert has_structured_report(text, "zh", min_chars=200, markers=_MARKERS) is True


# --- language scoping ----------------------------------------------------


def test_language_scopes_marker_search_when_specified() -> None:
    # Chinese-only marker in the text, but we scope to 'en' — should miss.
    text = _padded("超声检查所见: 右乳低回声结节", 250)
    assert has_structured_report(text, "en", min_chars=200, markers=_MARKERS) is False


def test_language_none_checks_all_configured_marker_sets() -> None:
    # Worker default: language=None lets a ZH report fire even when no
    # locale is known. This is the user-story for "EN session, ZH scan".
    text = _padded("超声检查所见: 右乳低回声结节", 250)
    assert has_structured_report(text, None, min_chars=200, markers=_MARKERS) is True


def test_unknown_language_falls_back_to_all_languages() -> None:
    # A language argument that isn't in markers (e.g. "ja") falls back
    # to checking every set — defensive against a config drift where
    # the locale label changes between symptoms and vision configs.
    text = _padded("Impression: benign cyst, 6mm", 250)
    assert has_structured_report(text, "ja", min_chars=200, markers=_MARKERS) is True


# --- known false-positive (plan §5.6 / scenarios) -----------------------


def test_textbook_prose_with_findings_word_is_a_false_positive() -> None:
    """Documented acceptable tradeoff — the conservative wrong answer.

    The LLM uses the OCR text path instead of running the vision tool
    on a textbook excerpt. KTD-V6 takes this hit deliberately: the
    inverse failure (a real radiology report slips through the gate
    and gets a small model's opinion attached) is the worse error.
    """
    text = _padded(
        "Many studies report findings on chest radiography that suggest "
        "early-stage pneumonia in pediatric populations, although the ",
        500,
    )
    assert has_structured_report(text, "en", min_chars=200, markers=_MARKERS) is True


# --- empty marker config --------------------------------------------------


def test_empty_markers_returns_false_for_any_input() -> None:
    text = _padded("findings: anything", 500)
    assert has_structured_report(text, None, min_chars=200, markers={}) is False


# --- load_ocr_report_config ----------------------------------------------


def test_load_ocr_report_config_missing_file_returns_defaults(tmp_path: Path) -> None:
    cfg = load_ocr_report_config(tmp_path / "no-such-vision.yaml")
    assert cfg["min_chars"] == DEFAULT_MIN_CHARS
    assert cfg["markers"] == {}


def test_load_ocr_report_config_reads_block(tmp_path: Path) -> None:
    path = tmp_path / "vision.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "ocr_report": {
                    "min_chars": 100,
                    "markers": {
                        "en": ["findings"],
                        "zh": ["所见"],
                    },
                },
                # Other blocks the loader must not stumble on.
                "diseases": [],
            }
        ),
        encoding="utf-8",
    )
    cfg = load_ocr_report_config(path)
    assert cfg == {
        "min_chars": 100,
        "markers": {"en": ["findings"], "zh": ["所见"]},
    }


def test_load_ocr_report_config_repo_default_loads(tmp_path: Path) -> None:
    """Regression: the shipped configs/vision.yaml is loadable."""
    from claritymed.config import CONFIGS_DIR

    cfg = load_ocr_report_config(CONFIGS_DIR / "vision.yaml")
    # The shipped file has both EN and ZH marker buckets — sanity check
    # so a future config rewrite doesn't silently drop one.
    assert set(cfg["markers"].keys()) >= {"en", "zh"}
    assert cfg["min_chars"] > 0
