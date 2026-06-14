"""The three Unit 13 prompt YAMLs load via :class:`PromptRegistry` and
honour OSS-hygiene + bilingual + content invariants."""

from __future__ import annotations

import re

import pytest

from claritymed.core.prompts.registry import PromptRegistry

_SYMPTOM_PROMPT_NAMES = (
    "predict_disease_from_symptoms_tool",
    "translate_complaint_to_en",
    "symptoms_final_reply",
)


@pytest.fixture(scope="module")
def registry() -> PromptRegistry:
    return PromptRegistry()


@pytest.mark.parametrize("name", _SYMPTOM_PROMPT_NAMES)
def test_prompt_loads(name: str, registry: PromptRegistry) -> None:
    """Every Unit 13 prompt must be discoverable by the registry."""
    assert name in registry.list(), (
        f"{name!r} not loaded; expected ``store/{name}.yaml``"
    )


@pytest.mark.parametrize("name", _SYMPTOM_PROMPT_NAMES)
def test_both_languages_present(name: str, registry: PromptRegistry) -> None:
    """The registry's PromptVersion validator already enforces bilingual,
    but we assert here so a regression surfaces at the test layer with a
    clearer message than ``PromptVersion`` validation error."""
    for lang in ("en", "zh"):
        rendered = registry.get(name, language=lang)
        assert rendered.strip(), f"{name!r} language {lang!r} is empty"


def test_tool_description_routes_on_representative_conditions(
    registry: PromptRegistry,
) -> None:
    """The tool description must enumerate enough DDXPlus coverage for
    the LLM to know whether to call. A regression that drops the
    conditions list (e.g. an over-zealous tightening pass) would silently
    lose routing signal."""
    en = registry.get("predict_disease_from_symptoms_tool", language="en")
    for needle in ("chest pain", "abdominal pain"):
        assert needle in en.lower(), f"tool description missing {needle!r}"


def test_translation_prompt_has_no_personal_identifiers(
    registry: PromptRegistry,
) -> None:
    """OSS hygiene per CLAUDE.md — the prompt must not contain a personal
    email, a personal name, or a phone number that would survive into a
    public GitHub mirror."""
    for lang in ("en", "zh"):
        text = registry.get("translate_complaint_to_en", language=lang)
        assert not re.search(r"[\w.]+@[\w.]+", text), (
            f"email-like pattern in translate_complaint_to_en ({lang})"
        )
        # No 10-digit phone runs (US-style); allow short numbers because
        # "120" is the local emergency number used in symptoms_final_reply.
        assert not re.search(r"\b\d{10}\b", text), (
            f"long digit run in translate_complaint_to_en ({lang})"
        )


def test_final_reply_encodes_all_four_tiers(registry: PromptRegistry) -> None:
    """The tier-by-tier playbook is the structural defence against
    LLM-side safety drift (KTD-2). Dropping a tier from the prompt would
    let that tier's reply structure silently regress.

    Match case-insensitively — the playbook headers are uppercase
    (``CRITICAL``, ``URGENT``, ...) but the SeverityTier literal is
    title-case. Either spelling is fine as long as the tier is mentioned.
    """
    en = registry.get("symptoms_final_reply", language="en").lower()
    for tier in ("critical", "urgent", "moderate", "mild"):
        assert tier in en, f"final-reply prompt missing tier {tier!r}"


def test_final_reply_zh_uses_local_emergency_number(
    registry: PromptRegistry,
) -> None:
    """ZH prompt must reference the Chinese emergency dispatch (120) or
    equivalent local phrasing, not US "911" — culturally adjusted
    phrasing is a deliberate KTD-3 acceptance criterion."""
    zh = registry.get("symptoms_final_reply", language="zh")
    assert "120" in zh or "急救" in zh
    assert "911" not in zh


def test_final_reply_documents_special_branches(registry: PromptRegistry) -> None:
    """The decision-table branches (cap, cancel, override, expired,
    server_error) must all appear in the prompt — the LLM cannot
    handle a branch it does not know exists."""
    en = registry.get("symptoms_final_reply", language="en")
    for branch in (
        "hit_cap",
        "severity_override",
        "meets_confidence_threshold",
        "session_expired",
        "server_error",
    ):
        assert branch in en, f"final-reply prompt missing branch {branch!r}"
