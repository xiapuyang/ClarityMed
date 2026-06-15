"""Vision tool-trigger benchmark cases.

Each case measures one thing: does the LLM call
``detect_disease_from_image`` (or correctly refrain) given an image
attachment + a user prompt?

Detection signal: the plugin's confirm modal fires once per tool call
(``confirm_before_run=True`` in the bench config). A trial counts as
"tool invoked" when ``channel.calls >= MODAL_THRESHOLD`` (1).

Three tiers:

* **base** — explicit image + diagnostic complaint. Every capable
  model should invoke the tool.
* **hard** — implicit phrasing, multi-image disambiguation, modality
  ambiguity. Ceiling cases — reveal where models fall off.
* **fp** — looks vision-flavored but must NOT invoke the tool
  (non-medical, modality mismatch, OCR-override, third-party,
  off-domain).

``seed`` returns a dict describing the attachment to seed before the
trial:

* ``fixture_subdir`` — which ``tests/fixtures/vision/<subdir>`` to
  pick the bytes from (the runner picks the first non-placeholder
  file).
* ``modality`` / ``is_medical`` / ``ocr_has_report`` — sentinel tag
  fields the runner writes via ``BlobStore.write_ocr_result``.

The bench does NOT run the real medical-clip classifier — modality
accuracy is a separate Unit 2 metric. Cases seed the modality tag
directly so the LLM-decision axis is isolated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

VISION_TOOL = "detect_disease_from_image"
USER_ID = "bench"

# Vision's confirm fires once per tool call; a single channel call is
# sufficient evidence that the LLM picked the tool. Compare to symptoms
# where 3 modals must fire to confirm the loop started.
MODAL_THRESHOLD = 1


@dataclass
class Case:
    """One benchmark cell definition."""

    name: str
    tier: str  # "base" | "hard" | "fp"
    expected_behavior: str  # "call_tool" | "decline"
    prompts: dict[str, str]
    # Predicate over (modal_count: int) → (passed: bool, reason: str).
    args_predicate: Callable[[int], tuple[bool, str]]
    expected_tool: str | None = None
    # Returns a dict describing how to seed the attachment.
    seed: Callable[[], dict] | None = None


def _p_tool_invoked(modal_count: int) -> tuple[bool, str]:
    if modal_count >= MODAL_THRESHOLD:
        return True, f"modal_calls={modal_count} >= {MODAL_THRESHOLD}"
    return False, f"modal_calls={modal_count} < {MODAL_THRESHOLD}: tool not invoked"


def _p_no_args(modal_count: int) -> tuple[bool, str]:
    return True, "no args predicate for decline cases"


def _seed_busi(*, ocr_has_report: bool = False) -> Callable[[], dict]:
    def _f() -> dict:
        return {
            "fixture_subdir": "busi",
            "modality": "ultrasound",
            "is_medical": True,
            "ocr_has_report": ocr_has_report,
        }

    return _f


def _seed_modality_mismatch() -> Callable[[], dict]:
    def _f() -> dict:
        return {
            "fixture_subdir": "modality_mismatch",
            "modality": "ct",
            "is_medical": True,
            "ocr_has_report": False,
        }

    return _f


def _seed_report_overlay() -> Callable[[], dict]:
    def _f() -> dict:
        return {
            "fixture_subdir": "report_overlay",
            "modality": "ultrasound",
            "is_medical": True,
            "ocr_has_report": True,
        }

    return _f


def _seed_non_medical(*, modality: str = "photo") -> Callable[[], dict]:
    def _f() -> dict:
        return {
            "fixture_subdir": "non_medical",
            "modality": modality,
            "is_medical": False,
            "ocr_has_report": False,
        }

    return _f


# ---------------------------------------------------------------------------
# BASE tier — explicit image + diagnostic intent; every capable model
# should invoke the tool.
# ---------------------------------------------------------------------------

CASES: list[Case] = [
    Case(
        name="en_breast_us_explicit_concerning",
        tier="base",
        expected_behavior="call_tool",
        prompts={
            "en": (
                "Here's an ultrasound image of a lesion in my right breast. "
                "Could this be concerning? Please call the "
                "detect_disease_from_image tool with "
                "disease_id=breast_cancer_ultrasound. [Image sha:{sha8}]"
            ),
            "zh": (
                "这是我右乳病灶的超声图像。会不会有问题？请调用 "
                "detect_disease_from_image 工具，disease_id 用 "
                "breast_cancer_ultrasound。[Image sha:{sha8}]"
            ),
        },
        args_predicate=_p_tool_invoked,
        expected_tool=VISION_TOOL,
        seed=_seed_busi(),
    ),
    Case(
        name="en_breast_us_benign_malignant_normal",
        tier="base",
        expected_behavior="call_tool",
        prompts={
            "en": (
                "Please analyze this breast ultrasound and tell me whether "
                "the model thinks it's benign, malignant, or normal. "
                "[Image sha:{sha8}]"
            ),
            "zh": (
                "请分析这张乳腺超声图像，告诉我模型认为是良性、恶性还是正常。"
                "[Image sha:{sha8}]"
            ),
        },
        args_predicate=_p_tool_invoked,
        expected_tool=VISION_TOOL,
        seed=_seed_busi(),
    ),
    Case(
        name="zh_breast_us_anxious",
        tier="base",
        expected_behavior="call_tool",
        prompts={
            "en": (
                "I'm scared. The doctor said this lump might be bad — can "
                "you run the breast-ultrasound model on this and tell me "
                "what it sees? [Image sha:{sha8}]"
            ),
            "zh": (
                "我有点担心。医生说这个结节可能不太好——能用乳腺超声模型"
                "看一下这张图吗？[Image sha:{sha8}]"
            ),
        },
        args_predicate=_p_tool_invoked,
        expected_tool=VISION_TOOL,
        seed=_seed_busi(),
    ),
    # -------------------------------------------------------------------
    # HARD tier — implicit, ambiguous, modality-subtle. Where models fall
    # off; passing here separates strong from shippable.
    # -------------------------------------------------------------------
    Case(
        name="hard_implicit_concern_zh",
        tier="hard",
        expected_behavior="call_tool",
        prompts={
            "en": "this image — thoughts? [Image sha:{sha8}]",
            "zh": "这张图怎么样？[Image sha:{sha8}]",
        },
        args_predicate=_p_tool_invoked,
        expected_tool=VISION_TOOL,
        seed=_seed_busi(),
    ),
    Case(
        name="hard_implicit_routine_check_en",
        tier="hard",
        expected_behavior="call_tool",
        prompts={
            "en": (
                "Annual checkup — anything to worry about in this scan? "
                "[Image sha:{sha8}]"
            ),
            "zh": ("这是年度体检的图像，看着有没有问题？[Image sha:{sha8}]"),
        },
        args_predicate=_p_tool_invoked,
        expected_tool=VISION_TOOL,
        seed=_seed_busi(),
    ),
    Case(
        name="hard_modality_subtle_mismatch",
        tier="hard",
        expected_behavior="decline",
        prompts={
            "en": (
                "I'm worried about my breast — what does this scan show? "
                "[Image sha:{sha8}]"
            ),
            "zh": "怀疑乳腺有问题，看一下这张图。[Image sha:{sha8}]",
        },
        args_predicate=_p_no_args,
        expected_tool=None,
        seed=_seed_modality_mismatch(),
    ),
    Case(
        name="hard_ocr_report_with_followup",
        tier="hard",
        expected_behavior="decline",
        prompts={
            "en": (
                "The report on this scan says benign but my doctor seemed "
                "concerned — thoughts? [Image sha:{sha8}]"
            ),
            "zh": (
                "报告上写的是良性，但医生看上去有点担心，怎么看？[Image sha:{sha8}]"
            ),
        },
        args_predicate=_p_no_args,
        expected_tool=None,
        seed=_seed_report_overlay(),
    ),
    # -------------------------------------------------------------------
    # FP tier — looks vision-flavored, must NOT invoke the tool.
    # -------------------------------------------------------------------
    Case(
        name="fp_non_medical_pet_photo",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "Is my cat OK? [Image sha:{sha8}]",
            "zh": "我家猫还好吗？[Image sha:{sha8}]",
        },
        args_predicate=_p_no_args,
        expected_tool=None,
        seed=_seed_non_medical(modality="photo"),
    ),
    Case(
        name="fp_screenshot_of_text",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "Summarize this article for me. [Image sha:{sha8}]",
            "zh": "帮我总结这篇文章。[Image sha:{sha8}]",
        },
        args_predicate=_p_no_args,
        expected_tool=None,
        seed=_seed_non_medical(modality="document"),
    ),
    Case(
        name="fp_modality_mismatch_explicit",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "What disease is this CT showing? [Image sha:{sha8}]",
            "zh": "这张 CT 显示的是什么病？[Image sha:{sha8}]",
        },
        args_predicate=_p_no_args,
        expected_tool=None,
        seed=_seed_modality_mismatch(),
    ),
    Case(
        name="fp_ocr_report_dominant",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "Explain this radiology report. [Image sha:{sha8}]",
            "zh": "解释一下这份放射科报告。[Image sha:{sha8}]",
        },
        args_predicate=_p_no_args,
        expected_tool=None,
        seed=_seed_report_overlay(),
    ),
    Case(
        name="fp_image_attached_unrelated_question",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": ("What's the weather like in Boston this week? [Image sha:{sha8}]"),
            "zh": "波士顿本周天气怎么样？[Image sha:{sha8}]",
        },
        args_predicate=_p_no_args,
        expected_tool=None,
        seed=_seed_busi(),
    ),
    Case(
        name="fp_knowledge_question_no_image",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "What does ultrasound show in benign breast lesions?",
            "zh": "乳腺良性病灶在超声上是什么表现？",
        },
        args_predicate=_p_no_args,
        expected_tool=None,
        seed=None,
    ),
]


__all__ = ["CASES", "Case", "MODAL_THRESHOLD", "USER_ID", "VISION_TOOL"]
