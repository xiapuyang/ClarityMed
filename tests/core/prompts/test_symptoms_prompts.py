"""The three Unit 13 prompt YAMLs load via :class:`PromptRegistry` and
honour OSS-hygiene + bilingual + content invariants."""

from __future__ import annotations

import re

import pytest

from claritymed.core.prompts.registry import PromptRegistry

_SYMPTOM_PROMPT_NAMES = (
    "predict_disease_from_symptoms_tool",
    "translate_complaint",
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
        text = registry.get("translate_complaint", language=lang)
        assert not re.search(r"[\w.]+@[\w.]+", text), (
            f"email-like pattern in translate_complaint ({lang})"
        )
        # No 10-digit phone runs (US-style); allow short numbers because
        # "120" is the local emergency number used in symptoms_final_reply.
        assert not re.search(r"\b\d{10}\b", text), (
            f"long digit run in translate_complaint ({lang})"
        )


def test_final_reply_v2_references_severity_tier_and_escalation(
    registry: PromptRegistry,
) -> None:
    """Structural guarantees of the v2 (multi-card) prompt.

    v1 encoded a per-tier playbook because the LLM composed the entire
    tier-appropriate reply. v2 delegates per-tier prose to the static
    conditions catalog rendered on cards; the LLM's job shrinks to a
    short summary above the cards. The v2 structural invariants are:

    * The LLM must acknowledge the top card's severity tier as a
      concept (rather than restating the card's report).
    * The LLM must close with an escalation-trigger sentence.
    * The LLM must NOT list condition names, probabilities, or ICD
      codes (those live on the cards).

    A regression here would let the LLM either duplicate card content
    (noise) or drop the safety-net summary shape (miss the escalation
    trigger). Case-insensitive match — the prompt formatting varies.
    """
    en = registry.get("symptoms_final_reply", language="en").lower()
    zh = registry.get("symptoms_final_reply", language="zh")
    # Reference to severity tier as a concept — the summary points at
    # the top card's tier without restating the card.
    assert "severity" in en or "tier" in en
    assert "严重度" in zh
    # Escalation trigger closing sentence.
    assert "escalation" in en or "come back" in en
    assert "升级触发" in zh or "尽早" in zh
    # Explicit prohibition on duplicating card content.
    assert "cards" in en, "v2 prompt must reference the cards contract"
    assert "卡片" in zh, "v2 中文 prompt 必须提到卡片契约"


def test_final_reply_v2_cites_evidence_by_index_only(
    registry: PromptRegistry,
) -> None:
    """v2 summary integrates RAG Evidence via ``[n]`` citations.

    The multi-card renderer places catalog citations on each card, but
    the summary above the cards still gets the deterministically-spliced
    ``Evidence (cite by [n]):`` block. The prompt must instruct the LLM
    to (a) cite by ``[n]`` when Evidence supports a claim, (b) only use
    indices that appear in the Evidence block, and (c) never restate
    source titles inline (the ``Sources: [n]`` list below the summary
    already renders them). Regression would either strip the citation
    channel from the summary (evidence-less prose) or reopen the door
    to fabricated indices / inline title restatement.
    """
    en = registry.get("symptoms_final_reply", language="en")
    zh = registry.get("symptoms_final_reply", language="zh")
    # The [n] citation channel must be named explicitly.
    assert "[n]" in en and "Evidence" in en
    assert "[n]" in zh and ("Evidence" in zh or "证据" in zh or "文献" in zh)
    # Anti-fabrication invariant.
    en_lower = en.lower()
    assert "never invent" in en_lower or "fabricat" in en_lower
    assert "凭空" in zh or "没有" in zh
    # Anti-inline-title-restatement invariant.
    assert "restate" in en_lower or "restating" in en_lower
    assert "复述" in zh


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
