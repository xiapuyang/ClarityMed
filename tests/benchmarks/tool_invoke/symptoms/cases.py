"""Benchmark cases for ``predict_disease_from_symptoms`` tool-trigger evaluation.

Each case measures one thing: does the LLM decide to call
``predict_disease_from_symptoms`` (or correctly refrain) given this input?
Detection is via prompt-channel modal call count rather than the approval
channel used by ingest tools — the symptoms plugin drives modals directly,
never reaching the ingest approval gate.

Three tiers:

* **base** — explicit, multi-symptom diagnostic complaints with enough
  DDXPlus-scope evidence that any capable model should invoke the tool.
  Floor: every model we consider shippable must clear this.

* **hard** — implicit or vague presentations where the intent is ambiguous.
  A model that asks for clarification or guesses a free-text reply is
  not wrong per se; but a model that routes to the tool is better. Ceiling
  cases — they reveal where models start falling off.

* **fp** — false-positive guard: inputs that look medical but should NOT
  trigger ``predict_disease_from_symptoms``. General knowledge questions,
  ingest-only prompts, third-party subjects, historical symptoms, and
  off-domain requests. A trial passes when the symptoms tool is not invoked.

``expected_behavior`` values used here:

* ``"call_tool"`` — symptoms tool must be invoked (modal calls ≥ threshold).
* ``"decline"``   — symptoms tool must NOT be invoked (modal calls < threshold).
  Other tools (ingest, ask_user_question) firing is not penalised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

SYMPTOMS_TOOL = "predict_disease_from_symptoms"
USER_ID = "bench"

# Matches the e2e test threshold: the symptoms plugin drives ≥3 modals
# (confirm + initial questions + ≥1 follow-up) for any real tool invocation.
MODAL_THRESHOLD = 3


@dataclass
class Case:
    """One benchmark cell definition."""

    name: str
    tier: str  # "base" | "hard" | "fp"
    expected_behavior: str  # "call_tool" | "decline"
    prompts: dict[str, str]
    # For call_tool cases: predicate on the raw modal_call_count (int).
    # For decline cases: unused, always trivially True.
    args_predicate: Callable[[int], tuple[bool, str]]
    expected_tool: Optional[str] = None
    seed: Optional[Callable[[], dict]] = None


def _p_tool_invoked(modal_count: int) -> tuple[bool, str]:
    """Pass when the symptoms plugin generated enough modals to confirm invocation."""
    if modal_count >= MODAL_THRESHOLD:
        return True, f"modal_calls={modal_count} >= {MODAL_THRESHOLD}"
    return False, f"modal_calls={modal_count} < {MODAL_THRESHOLD}: tool not invoked"


def _p_no_args(modal_count: int) -> tuple[bool, str]:
    return True, "no args predicate for decline cases"


# ---------------------------------------------------------------------------
# BASE tier — explicit multi-symptom complaints; every capable model should
# invoke the symptoms tool here.
# ---------------------------------------------------------------------------

CASES: list[Case] = [
    Case(
        name="zh_cough_dyspnea",
        tier="base",
        expected_behavior="call_tool",
        expected_tool=SYMPTOMS_TOOL,
        prompts={
            "zh": "我咳嗽，呼吸困难，这两天一直这样。",
            "en": "I've been coughing and having difficulty breathing for the past two days.",
        },
        args_predicate=_p_tool_invoked,
    ),
    Case(
        name="zh_chest_fever_aches",
        tier="base",
        expected_behavior="call_tool",
        expected_tool=SYMPTOMS_TOOL,
        prompts={
            "zh": "我胸口疼、发烧、全身酸痛，已经持续了两天，有点担心会不会是什么严重的问题。",
            "en": "I have chest pain, fever, and body aches for two days. Worried it might be serious.",
        },
        args_predicate=_p_tool_invoked,
    ),
    Case(
        name="en_breath_chest_dizziness",
        tier="base",
        expected_behavior="call_tool",
        expected_tool=SYMPTOMS_TOOL,
        prompts={
            "en": "I've had shortness of breath, chest pain, and dizziness since yesterday morning. Could this be serious?",
            "zh": "我从昨天早上开始就呼吸困难、胸口疼、头晕，这会不会有什么大问题？",
        },
        args_predicate=_p_tool_invoked,
    ),
    Case(
        name="zh_headache_nausea_fever",
        tier="base",
        expected_behavior="call_tool",
        expected_tool=SYMPTOMS_TOOL,
        prompts={
            "zh": "我头疼、恶心、发烧三天了，想知道可能是什么病。",
            "en": "I've had a headache, nausea, and fever for three days. Want to know what disease it could be.",
        },
        args_predicate=_p_tool_invoked,
    ),
    # ---------------------------------------------------------------------------
    # HARD tier — vague or indirect complaints; strong models route to the tool.
    # ---------------------------------------------------------------------------
    # zh_implicit_unwell / zh_vague_discomfort / en_multiple_vague_symptoms
    # were originally hard/call_tool.  Moved to fp/decline because the
    # symptoms tool only covers the narrow DDXPlus domain (49 acute ED
    # conditions) — vague expressions without named symptoms fall outside
    # scope.  Eligibility rejects them silently; the LLM should answer
    # with free text.  ask_user_question or a free-text clarifying reply
    # are both acceptable outcomes.
    Case(
        name="zh_implicit_unwell",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "zh": "最近一直感觉身体不太对，有好几个症状，想检查一下可能是什么原因。",
            "en": "Lately something feels off — I have several symptoms and want to figure out what might be wrong.",
        },
        args_predicate=_p_no_args,
    ),
    Case(
        name="en_implicit_feeling_terrible",
        tier="hard",
        expected_behavior="call_tool",
        expected_tool=SYMPTOMS_TOOL,
        prompts={
            "en": "I've been feeling really terrible the past few days — exhausted, short of breath, and my chest feels tight.",
            "zh": "这几天感觉很难受，特别累、呼吸困难，胸口也很紧。",
        },
        args_predicate=_p_tool_invoked,
    ),
    Case(
        name="zh_vague_discomfort",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "zh": "我身体有点不舒服，有些地方感觉异常，不确定是不是需要担心。",
            "en": "I feel a bit unwell, something seems off, not sure if I should be worried.",
        },
        args_predicate=_p_no_args,
    ),
    Case(
        name="en_multiple_vague_symptoms",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "Multiple things going on at once — fatigue, some pain, not feeling right. Could you help figure this out?",
            "zh": "同时有好几个症状——疲劳、有些疼痛、整体状态不对——能帮我分析一下吗？",
        },
        args_predicate=_p_no_args,
    ),
    # ---------------------------------------------------------------------------
    # FP tier — should NOT trigger predict_disease_from_symptoms.
    # ---------------------------------------------------------------------------
    # ---- Knowledge / factual questions (no personal complaint) ----
    Case(
        name="zh_aspirin_knowledge",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "zh": "阿司匹林能降低心脏病发作的风险吗？",
            "en": "Can aspirin reduce the risk of a heart attack?",
        },
        args_predicate=_p_no_args,
    ),
    Case(
        name="en_metformin_side_effects",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "What are the common side effects of metformin?",
            "zh": "二甲双胍常见的副作用有哪些？",
        },
        args_predicate=_p_no_args,
    ),
    Case(
        name="en_what_is_hypertension",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "What is hypertension and how is it diagnosed?",
            "zh": "什么是高血压？怎么诊断的？",
        },
        args_predicate=_p_no_args,
    ),
    # Hypothetical / educational — phrased about "someone" or
    # as a general "what diseases cause X" knowledge question.
    # No personal complaint is being made.
    Case(
        name="fp_hypothetical_disease_query",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "What diseases commonly present with both cough and fever at the same time?",
            "zh": "哪些疾病会同时出现咳嗽和发烧这两种症状？",
        },
        args_predicate=_p_no_args,
    ),
    # ---- Ingest-only prompts (should route to ingest tools, not DDX) ----
    Case(
        name="zh_allergy_save",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "zh": "我对青霉素过敏，请帮我记录一下。",
            "en": "I'm allergic to penicillin, please record that for me.",
        },
        args_predicate=_p_no_args,
    ),
    Case(
        name="zh_weight_update",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "zh": "请帮我更新一下体重，我现在68公斤。",
            "en": "Please update my weight — I'm now 68 kg.",
        },
        args_predicate=_p_no_args,
    ),
    # ---- Off-domain / scheduling ----
    Case(
        name="en_doctor_appointment",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "How do I book an appointment with a specialist?",
            "zh": "我怎么预约专科医生？",
        },
        args_predicate=_p_no_args,
    ),
    # ---- Third-party subject — symptoms belong to someone else ----
    # Triggering DDX for another person's symptoms pollutes the user's
    # own diagnostic context and risks giving advice about a non-patient.
    Case(
        name="fp_third_party_symptoms",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "My mom has had a cough and low-grade fever for the past week — what do you think it could be?",
            "zh": "我妈妈低烧咳嗽已经一周了，你觉得可能是什么原因？",
        },
        args_predicate=_p_no_args,
    ),
    # ---- Historical / resolved symptoms — past event, not active complaint ----
    # The user is reporting something that already happened and resolved.
    # Running DDX on historical resolved symptoms is misleading and unhelpful.
    Case(
        name="fp_past_resolved_symptoms",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "Last month I had chest pain and shortness of breath for a few days, but it cleared up on its own.",
            "zh": "上个月我胸口疼、呼吸困难了几天，后来自己好了。",
        },
        args_predicate=_p_no_args,
    ),
    # ---- Lab result interpretation — numeric values, not subjective symptoms ----
    # "WBC of 11.2" is a lab value question, not a symptom complaint.
    # The correct response is to explain what the value means, not to
    # launch a multi-modal DDX intake flow.
    Case(
        name="fp_lab_result_question",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "My blood test came back with WBC of 11.2 — is that high? Should I be worried?",
            "zh": "我血常规白细胞是 11.2，这偏高吗？需要担心吗？",
        },
        args_predicate=_p_no_args,
    ),
    # ---- Recovery / improvement update — no active complaint ----
    # The user is reporting that symptoms are gone, not presenting new ones.
    Case(
        name="fp_recovery_update",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "Just wanted to say I'm feeling a lot better now — the fever and cough from last week are basically gone.",
            "zh": "就是想说我现在好多了，上周那次发烧咳嗽基本都好了。",
        },
        args_predicate=_p_no_args,
    ),
    # ---- Drug side-effect as knowledge question ----
    # "Is coughing a known side effect of lisinopril?" is a drug-property
    # question. The user is not asking for a differential diagnosis of their
    # cough — they already have a plausible cause. Running DDX here would
    # ignore the context and launch an unnecessary intake flow.
    Case(
        name="fp_drug_side_effect_query",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "I started lisinopril last week and noticed a dry cough. Is coughing a known side effect?",
            "zh": "我上周开始吃赖诺普利，发现有点干咳。咳嗽是这个药的已知副作用吗？",
        },
        args_predicate=_p_no_args,
    ),
    # ---- Trivial / self-attributed sensation ----
    # A minor sensation with an obvious self-explanation ("probably from
    # sleeping awkwardly") is not a diagnostic request. Firing DDX here
    # would be over-eager and erode user trust.
    Case(
        name="fp_minor_trivial_symptom",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "My left hand is a bit stiff this morning — probably from sleeping awkwardly.",
            "zh": "今早左手有点僵，可能是睡姿不对。",
        },
        args_predicate=_p_no_args,
    ),
    # ---- Lifestyle / burnout attribution ----
    # The user attributes their fatigue to an obvious lifestyle cause
    # (overwork). They are not seeking a differential diagnosis.
    Case(
        name="fp_lifestyle_fatigue",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "I've been exhausted lately — working 12-hour days all week. Definitely just burnout.",
            "zh": "最近一直很疲惫，每天上班十几个小时。肯定就是过劳了。",
        },
        args_predicate=_p_no_args,
    ),
]
