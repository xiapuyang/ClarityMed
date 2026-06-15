"""OCR-report heuristic — KTD-V6.

When an uploaded image is itself a photograph or scan of a clinician's
typed report (FINDINGS / IMPRESSION / 所见 / 印象), the vision tool must
stay out of the way: the radiologist's reading is authoritative and our
small disease-specific model would just be noise. This module is the
detector that fires that override.

Two-part heuristic, deliberately cheap so it runs in the OCR worker's
hot path with zero network calls:

1. **Length floor.** Below ``min_chars`` of stripped text, the image is
   almost certainly not a report (UI screenshots, photo overlays, blank
   forms). Default 200 — calibrated against typical FINDINGS-section
   length.
2. **Keyword match.** Any localized marker keyword present in the
   lowercased text (``findings``, ``impression``, ``所见``, ``印象`` …)
   flips the override on.

The signature accepts ``language`` so callers that already know the
user's locale can scope the check; passing ``None`` (the OCR-worker
default — the worker has no reliable language signal at extraction
time) checks every configured marker set. This is intentional: a user
in an ``en`` session may still upload a Chinese radiology scan, and we
want the override to fire either way.

False positives are acceptable, false negatives are not. A textbook
excerpt that prose-mentions "findings" will trigger the override and
the LLM answers from text — the conservative wrong call. The reverse
(a real report slips through the gate and we run a model on it) would
make ClarityMed look authoritative on something a radiologist already
diagnosed; the plan's KTD-V6 explicitly accepts the false-positive
tradeoff (plan §"Test scenarios" + brainstorm §5.6).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_MIN_CHARS = 200


def has_structured_report(
    text: str,
    language: str | None,
    *,
    min_chars: int,
    markers: dict[str, list[str]],
) -> bool:
    """Return True when ``text`` looks like a clinician's typed report.

    Args:
        text: OCR'd text. Whitespace-stripped before the length check
            so a page-of-image-with-a-newline doesn't qualify.
        language: Locale to scope the keyword search to, e.g. ``"en"`` /
            ``"zh"``. ``None`` (worker default) checks every configured
            language so cross-locale uploads still fire the override.
        min_chars: Length floor below which the check short-circuits to
            False. Sourced from ``configs/vision.yaml::ocr_report.min_chars``.
        markers: ``{language: [keyword, ...]}`` map from
            ``configs/vision.yaml::ocr_report.markers``. Keywords are
            matched case-insensitively against ``text``.

    Returns:
        True iff the length floor is cleared AND at least one marker
        keyword from the relevant language(s) appears in the lowercased
        text.
    """
    if len(text.strip()) < min_chars:
        return False
    if language and language in markers:
        candidates = list(markers[language])
    else:
        # No language hint, or the hint names a language with no
        # configured markers — fall back to every configured set so a
        # ZH report inside an EN session still trips the override.
        candidates = [
            keyword
            for language_markers in markers.values()
            for keyword in language_markers
        ]
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in candidates)


def load_ocr_report_config(vision_yaml_path: Path) -> dict[str, Any]:
    """Read just the ``ocr_report`` block from ``configs/vision.yaml``.

    Returned shape: ``{"min_chars": int, "markers": {lang: [str, ...]}}``.
    Designed for the OCR worker's bootstrap path — no need to load the
    full VisionConfig (which would pull in diseases, models, servers).

    Returns an empty config (`min_chars=DEFAULT_MIN_CHARS`, `markers={}`)
    when the file is missing so the worker can still run in environments
    that haven't shipped the vision config yet (CI smoke tests, very
    early development). A missing config means the detector returns
    False for every input — the override is effectively off, which is
    the safe default (vision tool will still get called normally).
    """
    if not vision_yaml_path.exists():
        return {"min_chars": DEFAULT_MIN_CHARS, "markers": {}}
    raw = yaml.safe_load(vision_yaml_path.read_text(encoding="utf-8")) or {}
    block = raw.get("ocr_report") or {}
    return {
        "min_chars": int(block.get("min_chars", DEFAULT_MIN_CHARS)),
        "markers": dict(block.get("markers") or {}),
    }


__all__ = [
    "DEFAULT_MIN_CHARS",
    "has_structured_report",
    "load_ocr_report_config",
]
