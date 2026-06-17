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
    """One benchmark cell definition.

    ``expected_behavior`` is one of:

    * ``"call_tool"``   — LLM must invoke ``detect_disease_from_image``;
                          predicate sees ``modal_count`` and asserts the
                          plugin's confirm modal fired.
    * ``"decline"``     — LLM must NOT invoke the tool AND must NOT emit
                          an ``ask_user_question`` modal; any modal fire
                          counts as a false positive.
    * ``"ask_clarification"`` — LLM must emit a disambig
                          ``ask_user_question`` (Rules 4 / 5 / 6) instead
                          of either calling the tool or staying silent.
                          Predicate is :func:`_p_disambig_asked` which
                          inspects the first modal's option count.
    """

    name: str
    tier: str  # "base" | "hard" | "fp"
    expected_behavior: str  # "call_tool" | "decline" | "ask_clarification"
    prompts: dict[str, str]
    # Predicate signature varies by behavior:
    #   call_tool / decline → (modal_count) → (passed, reason)
    #   ask_clarification   → (modal_count, first_modal_options) → (passed, reason)
    # The runner inspects ``expected_behavior`` and dispatches accordingly.
    args_predicate: Callable[..., tuple[bool, str]]
    expected_tool: str | None = None
    # Returns a dict describing how to seed the attachment.
    seed: Callable[[], dict] | None = None


# Min options on the first ask_user_question payload that we treat as a
# "disambig modal" (LLM-issued, one option per enabled disease) versus the
# plugin's confirm modal (always exactly 2 options: yes / no). With 4+
# enabled diseases the gap is wide enough that option count is a robust
# discriminator; if the catalog ever shrinks below this threshold the
# bench must switch to a stricter signal (e.g. comparing option labels to
# vision.confirm.<id>.yes_label).
DISAMBIG_MIN_OPTIONS = 3


def _p_tool_invoked(modal_count: int) -> tuple[bool, str]:
    if modal_count >= MODAL_THRESHOLD:
        return True, f"modal_calls={modal_count} >= {MODAL_THRESHOLD}"
    return False, f"modal_calls={modal_count} < {MODAL_THRESHOLD}: tool not invoked"


def _p_no_args(modal_count: int) -> tuple[bool, str]:
    return True, "no args predicate for decline cases"


def _p_disambig_asked(modal_count: int, first_modal_options: int) -> tuple[bool, str]:
    """Pass when the LLM emitted a disambig modal — not the confirm modal.

    Rule 4 / 5 / 6 cases expect the model to call ``ask_user_question``
    with a disease (or modality) picker — option count grows with the
    catalog. The plugin's own confirm modal has exactly 2 options
    (yes / no), so ``>= DISAMBIG_MIN_OPTIONS`` cleanly separates "LLM
    asked the user" from "LLM jumped to the tool" (which would have
    surfaced the confirm modal instead).
    """
    if modal_count == 0:
        return False, "no modal fired: LLM neither called the tool nor asked"
    if first_modal_options >= DISAMBIG_MIN_OPTIONS:
        return (
            True,
            f"disambig modal (options={first_modal_options} >= {DISAMBIG_MIN_OPTIONS})",
        )
    return (
        False,
        f"first modal has {first_modal_options} options "
        f"(< {DISAMBIG_MIN_OPTIONS}): looks like the confirm modal, not "
        "the disambig — the LLM called the tool instead of asking",
    )


def _seed_busi(*, ocr_has_report: bool = False) -> Callable[[], dict]:
    def _f() -> dict:
        return {
            "fixture_subdir": "busi",
            "modality": "ultrasound",
            "is_medical": True,
            "ocr_has_report": ocr_has_report,
        }

    return _f


def _seed_busi_empty_ocr() -> Callable[[], dict]:
    """Seed BUSI fixture with ``status="empty"`` + full classifier tagging.

    Reproduces the v3 prompt's worked-example tag shape::

        <image modality="ultrasound" is_medical="true"
               ocr_has_report="false" ocr_status="empty"/>

    Diagnostic ultrasounds rarely contain printed text, so OCR
    legitimately returns empty by design. v2 of the prompt produced
    prose explanations on this exact tag shape (the field-reported
    failure); v3 names empty OCR as NORMAL for diagnostic scans and
    branches on user intent (Rule 3 vs Rule 4). Differs from
    ``_seed_busi()`` only in ``status="empty"`` — the load-bearing
    axis the v3 prompt addresses.
    """

    def _f() -> dict:
        return {
            "fixture_subdir": "busi",
            "status": "empty",
            "modality": "ultrasound",
            "is_medical": True,
            "ocr_has_report": False,
        }

    return _f


def _seed_chest_ct(*, ocr_has_report: bool = False) -> Callable[[], dict]:
    def _f() -> dict:
        return {
            "fixture_subdir": "chest_ct",
            "modality": "ct",
            "is_medical": True,
            "ocr_has_report": ocr_has_report,
        }

    return _f


def _seed_chest_ct_inline_ocr(*, text: str) -> Callable[[], dict]:
    """Seed chest CT with non-report inline OCR text.

    Tags chest CT + ``ocr_has_report=false`` (Rule 3 territory) and
    renders ``text`` between open/close tags. Distinct from
    ``_seed_chest_ct()`` which collapses to ``ocr_status="empty"``
    because its OCR text is empty — here OCR succeeded but the text
    is non-report (watermark, image caption, footer, etc.) so the
    rendered tag carries inline body content. The LLM has to read past
    the body and route on the attributes.

    Reproduces two field traces where vague Chinese prompts
    (``"分析这个影像"``) correctly triggered the tool despite junk
    inline OCR — one with stock-photo watermarks, one with a brief
    descriptive caption.
    """

    def _f() -> dict:
        return {
            "fixture_subdir": "chest_ct",
            "modality": "ct",
            "is_medical": True,
            "ocr_has_report": False,
            "text": text,
        }

    return _f


def _seed_skin(*, ocr_has_report: bool = False) -> Callable[[], dict]:
    def _f() -> dict:
        return {
            "fixture_subdir": "skin",
            "modality": "dermoscopy",
            "is_medical": True,
            "ocr_has_report": ocr_has_report,
        }

    return _f


def _seed_histopath(*, ocr_has_report: bool = False) -> Callable[[], dict]:
    """Seed a histopathology image. Same fixture subdir powers both the
    lung and colon histopath cases — the prompt's disease_id is what
    steers the LLM toward the right per-organ model."""

    def _f() -> dict:
        return {
            "fixture_subdir": "histopath",
            "modality": "histopathology",
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


def _seed_unknown_modality(fixture_subdir: str = "busi") -> Callable[[], dict]:
    """Seed an image with ``modality="unknown"`` (medical-clip low conf).

    Mirrors the worker's ``medical_clip_unreachable`` posture: when the
    classifier can't decide, the sentinel still carries ``modality`` as
    the literal ``"unknown"`` so the LLM can branch on Rule 5 instead of
    seeing a bare tag. ``is_medical=True`` so Rule 1 doesn't pre-empt.
    """

    def _f() -> dict:
        return {
            "fixture_subdir": fixture_subdir,
            "modality": "unknown",
            "is_medical": True,
            "ocr_has_report": False,
        }

    return _f


def _seed_bare_tag(fixture_subdir: str = "busi") -> Callable[[], dict]:
    """Seed an image with ocr_status="empty" and NO classifier attrs.

    Reproduces the worst-case path Rule 6 exists to handle: every OCR
    leaf came back empty AND no vision-LLM hint was attached AND
    medical-clip was unreachable (or unconfigured). The rendered tag is
    bare — no modality, no is_medical, no ocr_has_report. The LLM must
    emit ``ask_user_question`` with the full disease list rather than
    guess from the user's text alone (since the catalog is two-
    dimensional and guessing the wrong modality wastes a turn on a
    ``modality_mismatch`` envelope).
    """

    def _f() -> dict:
        return {
            "fixture_subdir": fixture_subdir,
            "status": "empty",
            # None = sentinel writer omits the field → bare attribute on
            # the rendered <image> tag, which is the whole point.
            "modality": None,
            "is_medical": None,
            "ocr_has_report": None,
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
    # -------------------------------------------------------------------
    # Chest CT (``lung_cancer_chest_ct``) — TP/FN/FP coverage.
    # Mirrors the breast-ultrasound tiers so the bench can compute
    # FP/FN/TP rates per disease, not just aggregated across vision.
    # -------------------------------------------------------------------
    Case(
        name="en_chest_ct_explicit_nodule",
        tier="base",
        expected_behavior="call_tool",
        prompts={
            "en": (
                "Here's a chest CT axial slice. Could the nodule be "
                "concerning? Please call the detect_disease_from_image "
                "tool with disease_id=lung_cancer_chest_ct. "
                "[Image sha:{sha8}]"
            ),
            "zh": (
                "这是一张胸部 CT 轴位切片。这个结节会不会有问题？请调用 "
                "detect_disease_from_image 工具，disease_id 用 "
                "lung_cancer_chest_ct。[Image sha:{sha8}]"
            ),
        },
        args_predicate=_p_tool_invoked,
        expected_tool=VISION_TOOL,
        seed=_seed_chest_ct(),
    ),
    Case(
        name="en_chest_ct_subtype_question",
        tier="base",
        expected_behavior="call_tool",
        prompts={
            "en": (
                "Can the model tell me whether this lung lesion is "
                "adenocarcinoma, large cell carcinoma, squamous cell "
                "carcinoma, or normal? [Image sha:{sha8}]"
            ),
            "zh": (
                "模型能告诉我这个肺部病灶是腺癌、大细胞癌、鳞状细胞癌"
                "还是正常吗？[Image sha:{sha8}]"
            ),
        },
        args_predicate=_p_tool_invoked,
        expected_tool=VISION_TOOL,
        seed=_seed_chest_ct(),
    ),
    Case(
        name="hard_chest_ct_implicit_anxious",
        tier="hard",
        expected_behavior="call_tool",
        prompts={
            "en": (
                "Doc said I should follow up on this scan — what do you "
                "see? [Image sha:{sha8}]"
            ),
            "zh": "医生让我复查这张片子，你看到什么？[Image sha:{sha8}]",
        },
        args_predicate=_p_tool_invoked,
        expected_tool=VISION_TOOL,
        seed=_seed_chest_ct(),
    ),
    Case(
        name="fp_chest_ct_modality_mismatch_us",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": ("This is a chest CT, right? What does it show? [Image sha:{sha8}]"),
            "zh": "这是一张胸部 CT 吧？显示了什么？[Image sha:{sha8}]",
        },
        args_predicate=_p_no_args,
        expected_tool=None,
        # Seed an ultrasound image; LLM might still try to call chest_ct
        # because the prompt says "CT" — must refuse on modality grounds.
        seed=_seed_busi(),
    ),
    Case(
        name="fp_chest_ct_ocr_report",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": (
                "The radiology report on this chest CT mentions a "
                "nodule — please summarize. [Image sha:{sha8}]"
            ),
            "zh": (
                "这张胸部 CT 的放射科报告里提到一个结节，请帮我总结一下。"
                "[Image sha:{sha8}]"
            ),
        },
        args_predicate=_p_no_args,
        expected_tool=None,
        seed=_seed_report_overlay(),
    ),
    # -------------------------------------------------------------------
    # Skin lesion (``skin_cancer_dermoscopy``) — TP/FN/FP coverage.
    # The disease ships ``enabled: false`` until weights are promoted;
    # the bench still exercises tool-trigger decisions against the live
    # plugin (which sees the disease in the catalog regardless of
    # enabled state — enabled only gates routing).
    # -------------------------------------------------------------------
    Case(
        name="en_skin_explicit_mole_concern",
        tier="base",
        expected_behavior="call_tool",
        prompts={
            "en": (
                "Dermoscopy image of a mole on my arm. Could this be "
                "concerning? Please call the detect_disease_from_image "
                "tool with disease_id=skin_cancer_dermoscopy. "
                "[Image sha:{sha8}]"
            ),
            "zh": (
                "这是我胳膊上一颗痣的皮肤镜图像。会不会有问题？请调用 "
                "detect_disease_from_image 工具，disease_id 用 "
                "skin_cancer_dermoscopy。[Image sha:{sha8}]"
            ),
        },
        args_predicate=_p_tool_invoked,
        expected_tool=VISION_TOOL,
        seed=_seed_skin(),
    ),
    Case(
        name="en_skin_melanoma_question",
        tier="base",
        expected_behavior="call_tool",
        prompts={
            "en": (
                "Please analyze this skin lesion image and tell me "
                "whether the model thinks it could be melanoma or "
                "benign. [Image sha:{sha8}]"
            ),
            "zh": (
                "请分析这张皮损图像，告诉我模型认为它更像黑色素瘤"
                "还是良性。[Image sha:{sha8}]"
            ),
        },
        args_predicate=_p_tool_invoked,
        expected_tool=VISION_TOOL,
        seed=_seed_skin(),
    ),
    Case(
        name="hard_skin_implicit_change_zh",
        tier="hard",
        expected_behavior="call_tool",
        prompts={
            "en": (
                "This spot has changed color recently — opinion? [Image sha:{sha8}]"
            ),
            "zh": "这个斑最近颜色变了，怎么看？[Image sha:{sha8}]",
        },
        args_predicate=_p_tool_invoked,
        expected_tool=VISION_TOOL,
        seed=_seed_skin(),
    ),
    Case(
        name="fp_skin_modality_mismatch_us",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": ("Is this skin patch suspicious? [Image sha:{sha8}]"),
            "zh": "这块皮肤可疑吗？[Image sha:{sha8}]",
        },
        args_predicate=_p_no_args,
        expected_tool=None,
        # Seed an ultrasound image; LLM should refuse since dermoscopy
        # gate (KTD-V3) sees an ultrasound modality tag.
        seed=_seed_busi(),
    ),
    Case(
        name="fp_skin_non_medical_arm_photo",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "Cool tattoo I just got. [Image sha:{sha8}]",
            "zh": "刚做的新纹身，你看怎么样？[Image sha:{sha8}]",
        },
        args_predicate=_p_no_args,
        expected_tool=None,
        seed=_seed_non_medical(modality="photo"),
    ),
    # FN-prone implicit (mirror of ``hard_chest_ct_implicit_anxious``):
    # the user defers to a clinician without naming the lesion, the
    # disease, or the tool. A model that only triggers on explicit
    # diagnostic verbs ("analyze", "diagnose") will under-fire here.
    Case(
        name="hard_skin_implicit_observe_en",
        tier="hard",
        expected_behavior="call_tool",
        prompts={
            "en": (
                "My dermatologist said to keep an eye on this one — "
                "what do you see? [Image sha:{sha8}]"
            ),
            "zh": "皮肤科医生让我留意一下这个，你看怎么样？[Image sha:{sha8}]",
        },
        args_predicate=_p_tool_invoked,
        expected_tool=VISION_TOOL,
        seed=_seed_skin(),
    ),
    # Knowledge-only TN: no image attached, generic dermoscopy concept
    # question. The disease tool requires an image attachment to run —
    # firing it here would be a clean false positive. Mirror of the
    # generic ``fp_knowledge_question_no_image`` but phrased for skin.
    Case(
        name="fp_skin_knowledge_only_dermoscopy",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": (
                "What dermoscopic features distinguish melanoma from a benign nevus?"
            ),
            "zh": "皮肤镜下黑色素瘤和良性痣怎么区分？",
        },
        args_predicate=_p_no_args,
        expected_tool=None,
        seed=None,
    ),
    # Multi-class subtype phrasing — mirror of
    # ``en_chest_ct_subtype_question``. The skin_lesion dataset is the
    # ISIC 9-class set (melanoma / BCC / SCC / nevus / actinic keratosis
    # / benign keratosis / dermatofibroma / vascular / suspicious). A
    # prompt that names a few of the classes is the strongest TP signal
    # short of explicit tool naming.
    Case(
        name="en_skin_isic_subtype_question",
        tier="base",
        expected_behavior="call_tool",
        prompts={
            "en": (
                "Could you check whether this lesion is more consistent "
                "with melanoma, basal cell carcinoma, a benign nevus, or "
                "actinic keratosis? [Image sha:{sha8}]"
            ),
            "zh": (
                "麻烦帮我看一下这处皮损更像黑色素瘤、基底细胞癌、"
                "良性痣还是日光性角化？[Image sha:{sha8}]"
            ),
        },
        args_predicate=_p_tool_invoked,
        expected_tool=VISION_TOOL,
        seed=_seed_skin(),
    ),
    # -------------------------------------------------------------------
    # Lung histopathology (``lung_cancer_histopathology``) — TP/FN/FP
    # coverage. Both lung_cancer_chest_ct and lung_cancer_histopathology
    # are "lung cancer" diseases; the disambiguation lives entirely in
    # the image modality. The cases here exercise the LLM's ability to
    # route a histopath image to the histopath tool rather than the CT
    # tool. The disease ships ``enabled: false`` until weights are
    # promoted; the bench still exercises tool-trigger decisions against
    # the live plugin (which sees the disease in the catalog regardless
    # of enabled state).
    # -------------------------------------------------------------------
    Case(
        name="en_lung_histopath_explicit_biopsy",
        tier="base",
        expected_behavior="call_tool",
        prompts={
            "en": (
                "Here's a lung histopathology slide from my biopsy. "
                "Could this tissue be concerning? Please call the "
                "detect_disease_from_image tool with "
                "disease_id=lung_cancer_histopathology. [Image sha:{sha8}]"
            ),
            "zh": (
                "这是我活检的肺组织病理切片。这个组织会不会有问题？请调用 "
                "detect_disease_from_image 工具，disease_id 用 "
                "lung_cancer_histopathology。[Image sha:{sha8}]"
            ),
        },
        args_predicate=_p_tool_invoked,
        expected_tool=VISION_TOOL,
        seed=_seed_histopath(),
    ),
    # Multi-class subtype phrasing — mirrors en_chest_ct_subtype_question
    # for the histopath side. The strongest TP signal short of explicit
    # tool naming: the user names the histopath labels directly.
    Case(
        name="en_lung_histopath_subtype_question",
        tier="base",
        expected_behavior="call_tool",
        prompts={
            "en": (
                "Can the model tell me whether this lung biopsy tissue "
                "looks like adenocarcinoma, squamous cell carcinoma, or "
                "healthy lung tissue? [Image sha:{sha8}]"
            ),
            "zh": (
                "模型能告诉我这张肺活检组织更像腺癌、鳞状细胞癌"
                "还是正常肺组织吗？[Image sha:{sha8}]"
            ),
        },
        args_predicate=_p_tool_invoked,
        expected_tool=VISION_TOOL,
        seed=_seed_histopath(),
    ),
    # FN-prone implicit — the user defers to a clinician without naming
    # the disease or the tool. A model that only triggers on explicit
    # diagnostic verbs under-fires here.
    Case(
        name="hard_lung_histopath_implicit_pathologist",
        tier="hard",
        expected_behavior="call_tool",
        prompts={
            "en": (
                "The pathologist sent over this lung slide — what do "
                "you see? [Image sha:{sha8}]"
            ),
            "zh": "病理科送来这张肺切片，你看到什么？[Image sha:{sha8}]",
        },
        args_predicate=_p_tool_invoked,
        expected_tool=VISION_TOOL,
        seed=_seed_histopath(),
    ),
    # FP — same disease (lung cancer) but a CT image, not a histopath
    # slide. The LLM should refuse on modality grounds (histopath tool
    # accepts ``histopathology``, not ``ct``) and route to the chest_ct
    # tool instead. This case asserts the tool is NOT called by the
    # histopath path; aggregate "any vision tool fires" would pass for
    # the wrong reason.
    Case(
        name="fp_lung_histopath_wrong_modality_ct",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": (
                "Could you check this lung tissue for cancer with the "
                "histopathology model? [Image sha:{sha8}]"
            ),
            "zh": (
                "麻烦用组织病理模型检查一下这块肺组织有没有癌变？[Image sha:{sha8}]"
            ),
        },
        args_predicate=_p_no_args,
        expected_tool=None,
        seed=_seed_chest_ct(),
    ),
    # FP — pathology context but the image carries the pathologist's
    # report already (KTD-V6 OCR override). The tool must stay out.
    Case(
        name="fp_lung_histopath_ocr_report",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": (
                "The pathology report on this lung slide mentions "
                "carcinoma — can you summarize? [Image sha:{sha8}]"
            ),
            "zh": (
                "这张肺切片的病理报告里提到癌变，请帮我总结一下。[Image sha:{sha8}]"
            ),
        },
        args_predicate=_p_no_args,
        expected_tool=None,
        seed=_seed_report_overlay(),
    ),
    # Knowledge-only TN: no image attached. The disease tool requires
    # an image to run — firing it here would be a clean false positive.
    Case(
        name="fp_lung_histopath_knowledge_only",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": (
                "What histological features distinguish lung "
                "adenocarcinoma from squamous cell carcinoma?"
            ),
            "zh": "病理上肺腺癌和肺鳞状细胞癌怎么区分？",
        },
        args_predicate=_p_no_args,
        expected_tool=None,
        seed=None,
    ),
    # -------------------------------------------------------------------
    # Colon histopathology (``colon_cancer_histopathology``) — TP/FN/FP
    # coverage. Sister section to lung_histopath. The same fixture
    # subdir feeds both diseases; the disease_id in the prompt steers
    # the LLM toward the right per-organ tool.
    # -------------------------------------------------------------------
    Case(
        name="en_colon_histopath_explicit_biopsy",
        tier="base",
        expected_behavior="call_tool",
        prompts={
            "en": (
                "Here's a colon histopathology slide from my biopsy. "
                "Could this tissue be concerning? Please call the "
                "detect_disease_from_image tool with "
                "disease_id=colon_cancer_histopathology. [Image sha:{sha8}]"
            ),
            "zh": (
                "这是我活检的结肠组织病理切片。这个组织会不会有问题？请调用 "
                "detect_disease_from_image 工具，disease_id 用 "
                "colon_cancer_histopathology。[Image sha:{sha8}]"
            ),
        },
        args_predicate=_p_tool_invoked,
        expected_tool=VISION_TOOL,
        seed=_seed_histopath(),
    ),
    Case(
        name="en_colon_histopath_subtype_question",
        tier="base",
        expected_behavior="call_tool",
        prompts={
            "en": (
                "Can the model tell me whether this colon biopsy tissue "
                "looks like adenocarcinoma or healthy colon tissue? "
                "[Image sha:{sha8}]"
            ),
            "zh": (
                "模型能告诉我这张结肠活检组织更像腺癌还是正常"
                "结肠组织吗？[Image sha:{sha8}]"
            ),
        },
        args_predicate=_p_tool_invoked,
        expected_tool=VISION_TOOL,
        seed=_seed_histopath(),
    ),
    Case(
        name="hard_colon_histopath_implicit_followup",
        tier="hard",
        expected_behavior="call_tool",
        prompts={
            "en": (
                "GI doc said to follow up on this colon slide — "
                "thoughts? [Image sha:{sha8}]"
            ),
            "zh": "消化科医生让我复查这张结肠切片，怎么看？[Image sha:{sha8}]",
        },
        args_predicate=_p_tool_invoked,
        expected_tool=VISION_TOOL,
        seed=_seed_histopath(),
    ),
    # FP — non-medical photo with a colon-cancer-flavored prompt. The
    # LLM should refuse on the not-medical sentinel (no image_modality
    # match, no medical=True).
    Case(
        name="fp_colon_histopath_non_medical",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": (
                "Can you check this picture for signs of colon cancer? "
                "[Image sha:{sha8}]"
            ),
            "zh": ("能帮我看看这张照片有没有结肠癌的迹象吗？[Image sha:{sha8}]"),
        },
        args_predicate=_p_no_args,
        expected_tool=None,
        seed=_seed_non_medical(modality="photo"),
    ),
    # FP — looks like a histopath prompt but the user names "lung", not
    # "colon". The LLM should pick the lung tool, not the colon tool.
    # We can't directly observe "which disease_id was picked" from the
    # confirm-modal count alone, so this case is a NEGATIVE for the
    # colon tool (the args_predicate doesn't distinguish — this case is
    # primarily a fail-loud safeguard against modality drift in fixture
    # generation rather than a strict bench gate).
    Case(
        name="fp_colon_histopath_wrong_organ_in_prompt",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": (
                "Please call the colon histopathology model on this "
                "lung biopsy slide. [Image sha:{sha8}]"
            ),
            "zh": ("请用结肠组织病理模型分析这张肺活检切片。[Image sha:{sha8}]"),
        },
        args_predicate=_p_no_args,
        expected_tool=None,
        seed=_seed_histopath(),
    ),
    # Knowledge-only TN: no image, generic histology concept question.
    Case(
        name="fp_colon_histopath_knowledge_only",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": (
                "What histological features distinguish colon "
                "adenocarcinoma from healthy colonic mucosa?"
            ),
            "zh": "病理上结肠腺癌和正常结肠黏膜怎么区分？",
        },
        args_predicate=_p_no_args,
        expected_tool=None,
        seed=None,
    ),
    # -------------------------------------------------------------------
    # v3 prompt worked examples — same tag (tagged ultrasound +
    # ``ocr_status="empty"``), two intents. This is the field-reported
    # failure shape where v2 of the tool prompt produced Markdown
    # checkboxes / prose explanations instead of calling either tool.
    # v3 splits the routing explicitly:
    #
    #   * STRONG diagnostic intent → Rule 3 → call detect_disease_from_image
    #   * VAGUE intent → Rule 4 → call ask_user_question
    #
    # Both cases use ``expected_behavior="call_tool"`` with
    # ``_p_tool_invoked`` so the bench counts "any modal fired" as
    # success — the regression v3 prevents is prose-instead-of-modal,
    # and the ultrasound modality only covers one disease in today's
    # catalog (``breast_cancer_ultrasound``), so Rule 4's disambig
    # would have just one option and collapses cleanly into Rule 3.
    # ``tool_invoked_rate`` vs ``disambig_rate`` in the summary show
    # which path the model picked.
    # -------------------------------------------------------------------
    Case(
        name="base_breast_us_strong_empty_ocr",
        tier="base",
        expected_behavior="call_tool",
        prompts={
            "en": "Is this breast lesion concerning? [Image sha:{sha8}]",
            "zh": "分析这张乳房超声。[Image sha:{sha8}]",
        },
        args_predicate=_p_tool_invoked,
        expected_tool=VISION_TOOL,
        seed=_seed_busi_empty_ocr(),
    ),
    Case(
        name="hard_breast_us_vague_empty_ocr",
        tier="hard",
        expected_behavior="call_tool",
        prompts={
            "en": "thoughts? [Image sha:{sha8}]",
            "zh": "分析这个。[Image sha:{sha8}]",
        },
        args_predicate=_p_tool_invoked,
        expected_tool=VISION_TOOL,
        seed=_seed_busi_empty_ocr(),
    ),
    # Field-traced shapes: vague Chinese prompt + chest CT + non-empty,
    # non-report OCR (``ocr_has_report=false``). The LLM must read past
    # the inline text and fire Rule 3 on the ``modality`` /
    # ``is_medical`` attributes. Two variants of non-report OCR:
    #   1. Stock-photo watermark — repeated brand junk.
    #   2. Descriptive caption — brief modality-naming text that could
    #      tempt the model to over-explain in prose instead of firing.
    # A model that anchors on inline text over attributes will
    # under-fire on both. Distinct from the ``status="empty"`` v3
    # worked examples — here OCR succeeded.
    Case(
        name="hard_chest_ct_watermark_ocr",
        tier="hard",
        expected_behavior="call_tool",
        prompts={
            "en": "Analyze this scan. [Image sha:{sha8}]",
            "zh": "分析这个影像 [Image sha:{sha8}]",
        },
        args_predicate=_p_tool_invoked,
        expected_tool=VISION_TOOL,
        seed=_seed_chest_ct_inline_ocr(text="SCIENCEPHOTOLIBRARY\n" * 6),
    ),
    Case(
        name="hard_chest_ct_caption_ocr",
        tier="hard",
        expected_behavior="call_tool",
        prompts={
            "en": "Analyze this scan. [Image sha:{sha8}]",
            "zh": "分析这个影像 [Image sha:{sha8}]",
        },
        args_predicate=_p_tool_invoked,
        expected_tool=VISION_TOOL,
        seed=_seed_chest_ct_inline_ocr(text="CT Chest Axial Image"),
    ),
    # Vague Chinese prompt on a dermoscopy image with empty OCR. Per
    # Rule 4 the LLM must call ``ask_user_question`` rather than fire
    # the tool — even though dermoscopy maps to a single covered
    # disease (where a strict Rule 4 collapse would re-route to Rule
    # 3). The field-traced behavior asks the user via subtype options;
    # ``_p_disambig_asked`` is option-count based and accepts either
    # disease- or subtype-shaped option lists. The win this case locks
    # in is "didn't barge into the tool on a vague prompt".
    Case(
        name="hard_skin_vague_clarification_zh",
        tier="hard",
        expected_behavior="ask_clarification",
        prompts={
            "en": "Analyze this photo. [Image sha:{sha8}]",
            "zh": "分析这个照片 [Image sha:{sha8}]",
        },
        args_predicate=_p_disambig_asked,
        expected_tool=None,
        seed=_seed_skin(),
    ),
    # -------------------------------------------------------------------
    # ask_clarification — Rule 5 / Rule 6 disambig paths.
    #
    # Rule 5 (``modality="unknown"``): classifier was low-confidence.
    # Rule 6 (bare tag): no ``modality`` AND no ``is_medical`` attrs at
    #   all — every signal source failed at ingest time.
    # In both cases the LLM should emit a disambig ``ask_user_question``
    # rather than gamble on a tool call that's likely to bounce on the
    # KTD-V3 modality gate. The :func:`_p_disambig_asked` predicate
    # accepts only modal payloads whose option count looks like a
    # disease list (one per enabled disease) — exactly what the prompt
    # tells the LLM to construct from ``vision.disambig.disease``.
    # -------------------------------------------------------------------
    Case(
        name="rule5_modality_unknown_clear_disease_cue",
        tier="hard",
        expected_behavior="ask_clarification",
        prompts={
            "en": (
                "Could you analyze this breast ultrasound for me? [Image sha:{sha8}]"
            ),
            "zh": "麻烦帮我分析一下这张乳房彩超。[Image sha:{sha8}]",
        },
        args_predicate=_p_disambig_asked,
        expected_tool=None,
        seed=_seed_unknown_modality(),
    ),
    Case(
        name="rule5_modality_unknown_vague_prompt",
        tier="hard",
        expected_behavior="ask_clarification",
        prompts={
            "en": "What does this scan show? [Image sha:{sha8}]",
            "zh": "这张片子显示了什么？[Image sha:{sha8}]",
        },
        args_predicate=_p_disambig_asked,
        expected_tool=None,
        seed=_seed_unknown_modality(),
    ),
    # Rule 6 — bare tag with a STRONG textual cue. The temptation for the
    # LLM is to call breast_cancer_ultrasound directly because the user
    # named the modality + organ. Rule 6 forbids that — without a tag
    # attribute confirming modality the LLM must disambig, since a wrong
    # guess wastes a turn on modality_mismatch. This is the exact
    # scenario that ran the user's original "dead image" turn (empty OCR
    # + medical-clip unreachable + clear prompt).
    Case(
        name="rule6_bare_tag_breast_cue",
        tier="hard",
        expected_behavior="ask_clarification",
        prompts={
            "en": "Please analyze this breast ultrasound. [Image sha:{sha8}]",
            "zh": "请分析这张乳房彩超。[Image sha:{sha8}]",
        },
        args_predicate=_p_disambig_asked,
        expected_tool=None,
        seed=_seed_bare_tag(fixture_subdir="busi"),
    ),
    # Rule 6 — bare tag with a STRONG CT cue. Same logic with a
    # different organ + modality so the bench can detect Rule-6
    # regressions that happen to leave the breast path correct.
    Case(
        name="rule6_bare_tag_chest_ct_cue",
        tier="hard",
        expected_behavior="ask_clarification",
        prompts={
            "en": (
                "What does this chest CT show — any lung nodules? [Image sha:{sha8}]"
            ),
            "zh": "这张胸部 CT 看到肺结节了吗？[Image sha:{sha8}]",
        },
        args_predicate=_p_disambig_asked,
        expected_tool=None,
        seed=_seed_bare_tag(fixture_subdir="chest_ct"),
    ),
    # Rule 6 — bare tag with a VAGUE prompt. No textual modality cue
    # either, so this is the cleanest disambig case: nothing — sentinel
    # or text — narrows the disease.
    Case(
        name="rule6_bare_tag_vague_prompt",
        tier="hard",
        expected_behavior="ask_clarification",
        prompts={
            "en": "Could you take a look at this image? [Image sha:{sha8}]",
            "zh": "帮我看一下这张图。[Image sha:{sha8}]",
        },
        args_predicate=_p_disambig_asked,
        expected_tool=None,
        seed=_seed_bare_tag(fixture_subdir="busi"),
    ),
    # Rule 6 FP guard — bare tag with a NON-MEDICAL prompt. Even
    # without modality / is_medical attrs, an obviously non-clinical
    # question must not trigger the disambig modal (or the tool). The
    # LLM should treat the bare tag as "no useful classifier signal" AND
    # the prompt as "not asking about a medical scan", and answer in
    # plain text instead. This is the negative complement of
    # rule6_bare_tag_vague_prompt — both have bare tags, only one
    # warrants a disambig.
    Case(
        name="rule6_bare_tag_non_medical_prompt",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "Is my cat OK in this picture? [Image sha:{sha8}]",
            "zh": "照片里我家猫还好吗？[Image sha:{sha8}]",
        },
        args_predicate=_p_no_args,
        expected_tool=None,
        seed=_seed_bare_tag(fixture_subdir="non_medical"),
    ),
]


__all__ = [
    "CASES",
    "Case",
    "DISAMBIG_MIN_OPTIONS",
    "MODAL_THRESHOLD",
    "USER_ID",
    "VISION_TOOL",
]
