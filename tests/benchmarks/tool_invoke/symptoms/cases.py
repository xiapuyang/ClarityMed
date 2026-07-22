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

Instability analysis — the classic flu triad case
--------------------------------------------------

The prompt "I have fever, cough, and muscle aches for 3 days" (see
:data:`flu_triad_3d`) trips the tool *unstably* across models even though
it's a textbook DDXPlus-scope presentation (influenza / URTI /
early-pneumonia are all covered). Five compounding causes:

1. **Pre-training override.** Small models "know the answer" —
   fever+cough+myalgia+72h is the canonical influenza pattern in every
   USMLE and patient-education corpus. The model skips the structured
   loop and drops straight into free-text advice ("sounds like the flu,
   rest, hydrate, watch for red flags"). Larger reasoning models are
   more willing to defer to the tool; small local models are the ones
   that skip.
2. **Red-flag drought.** The tool description says "multi-symptom
   presentation plausibly in scope", but the flu triad has no
   respiratory-distress marker (no SOB, no chest pain, no hemoptysis)
   and no specificity anchor (no productive/colored sputum, no chills
   qualifier). The model reads the presentation as low-severity and
   the tool as "for serious/uncertain cases", so it declines.
3. **Framing ambiguity.** The prompt states symptoms but doesn't
   explicitly *ask* — no "what could this be?", "is this serious?",
   "should I see a doctor?". The tool description keys off diagnostic
   asking; a bare symptom report registers as reporting/venting rather
   than a diagnostic ask.
4. **Duration cue.** "3 days" is short enough to read as
   self-limiting viral illness. Models frequently reply "give it
   another few days" instead of routing to DDX.
5. **Tool-prompt vs user-prompt language mismatch** (see
   ``CLARITYMED_TOOL_PROMPT_LANG``). Small models weight tool
   descriptions less when they're in a different language from the
   user turn, further weakening the "call the tool" signal.

The base-tier cases below deliberately stress these five axes: some
cases fix (1) by adding pneumonia-specific evidence, others fix (3) by
explicitly asking, others fix (2) by adding a red flag. Comparing pass
rates across cases isolates which axis a given model is losing on.
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
    """One benchmark cell definition.

    Bump ``revision`` in place whenever ``prompts`` or ``args_predicate``
    semantics change for an existing ``name`` — cross-run comparisons
    join on ``(name, revision)``.
    """

    name: str
    tier: str  # "base" | "hard" | "fp"
    expected_behavior: str  # "call_tool" | "decline"
    prompts: dict[str, str]
    # For call_tool cases: predicate on the raw modal_call_count (int).
    # For decline cases: unused, always trivially True.
    args_predicate: Callable[[int], tuple[bool, str]]
    expected_tool: Optional[str] = None
    seed: Optional[Callable[[], dict]] = None
    revision: int = 1


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
    # ---------------------------------------------------------------------------
    # Cold / flu / pneumonia focus block
    #
    # DDXPlus covers Influenza, URTI, Bronchitis, Pneumonia, and Bronchiolitis
    # in this space — the exact acute upper- / lower-respiratory infections a
    # user would show up with in an ED-adjacent triage app. The base + hard
    # cases below stress the five instability axes documented at the top of
    # this file. The fp cases below stress adjacency traps: cold/flu/pneumonia
    # language that is *not* a personal diagnostic ask.
    # ---------------------------------------------------------------------------
    # ---- hard: verbatim source prompt — original instability observation ----
    # The exact bare-report wording that motivated this focus block. Kept
    # unedited (no trailing "what could this be?", no framing) so cross-run
    # comparisons stay honest — this cell's pass rate IS the answer to
    # "does the tool fire on this specific input?".
    Case(
        name="flu_triad_3d_verbatim",
        tier="hard",
        expected_behavior="call_tool",
        expected_tool=SYMPTOMS_TOOL,
        prompts={
            "en": "I have fever, cough, and muscle aches for 3 days",
            "zh": "我发烧、咳嗽、肌肉酸痛三天了",
        },
        args_predicate=_p_tool_invoked,
    ),
    # ---- base: classic flu triad — the exact instability case ----
    # Fever + cough + diffuse muscle aches (E_91 + E_201 + E_144) for a
    # 3-day acute window. Textbook Influenza / URTI presentation and firmly
    # in DDXPlus scope. Small models skip the tool because "they already
    # know the answer"; that skip is exactly what this bench measures.
    Case(
        name="flu_triad_3d",
        tier="base",
        expected_behavior="call_tool",
        expected_tool=SYMPTOMS_TOOL,
        prompts={
            "en": "I have fever, cough, and muscle aches for 3 days — what could this be?",
            "zh": "我发烧、咳嗽、全身酸痛已经三天了，这可能是什么病？",
        },
        args_predicate=_p_tool_invoked,
    ),
    # ---- base: same triad but framed as a bare report (no explicit ask) ----
    # Fixes instability axis (3) *inversely*: this variant STRIPS the
    # explicit "what could this be?" so we can measure how much of the
    # tool-skip is driven by framing vs the pre-training override. Comparing
    # pass rate here vs ``flu_triad_3d`` isolates the framing effect.
    Case(
        name="flu_triad_3d_bare_report",
        tier="hard",
        expected_behavior="call_tool",
        expected_tool=SYMPTOMS_TOOL,
        prompts={
            "en": "I've had a fever, cough, and body aches for 3 days now.",
            "zh": "我已经发烧、咳嗽、全身酸痛三天了。",
        },
        args_predicate=_p_tool_invoked,
    ),
    # ---- base: pneumonia red-flag pattern — productive colored sputum ----
    # E_91 (fever) + E_77 (colored/abundant sputum) + E_220 (pleuritic pain).
    # Fixes instability axis (2): the productive-sputum + pleuritic-pain pair
    # is a high-specificity pneumonia signal that should reliably trip the
    # "this needs a differential" heuristic even in small models.
    Case(
        name="pneumonia_productive_cough_fever",
        tier="base",
        expected_behavior="call_tool",
        expected_tool=SYMPTOMS_TOOL,
        prompts={
            "en": "I've had a fever for four days, my cough is bringing up yellow-green phlegm, and it hurts when I breathe in deeply.",
            "zh": "我发烧四天了，咳嗽有黄绿色的痰，深呼吸的时候胸口疼。",
        },
        args_predicate=_p_tool_invoked,
    ),
    # ---- base: cold-to-pneumonia escalation with SOB red flag ----
    # E_66 (significant SOB) added to fever+cough baseline. Complications of a
    # simple URTI are exactly the "should I see a doctor?" scenario the tool
    # exists for. Skipping the tool here is a real safety miss, not just a
    # taste miss.
    Case(
        name="cold_worsening_to_sob",
        tier="base",
        expected_behavior="call_tool",
        expected_tool=SYMPTOMS_TOOL,
        prompts={
            "en": "I caught a cold about five days ago and it seemed to be getting better, but now I'm short of breath just walking to the kitchen and my fever came back today. Should I be worried?",
            "zh": "五天前得了感冒，本来快好了，今天又开始发烧，走到厨房就气喘。这个情况严重吗？",
        },
        args_predicate=_p_tool_invoked,
    ),
    # ---- base: severe influenza — high fever + chills + severe myalgia ----
    # E_94 (chills) + E_144 (diffuse muscle pain) + E_91 (fever). Adds the
    # chills qualifier that distinguishes flu from a common cold; a capable
    # tool caller should read this as "differential worth running".
    Case(
        name="flu_high_fever_chills_myalgia",
        tier="base",
        expected_behavior="call_tool",
        expected_tool=SYMPTOMS_TOOL,
        prompts={
            "en": "I've had a high fever with chills and shaking, plus severe muscle aches all over, since yesterday. Is this the flu or something worse?",
            "zh": "从昨天开始高烧发冷、浑身打颤，全身肌肉都很酸疼。是流感还是别的什么？",
        },
        args_predicate=_p_tool_invoked,
    ),
    # ---- base: pneumonia + hemoptysis (major red flag) ----
    # E_45 (coughing up blood) is a high-specificity respiratory signal that
    # DDXPlus's Pneumonia / Bronchiectasis / Tuberculosis branches all read.
    # If any case makes a model call the tool this one should.
    Case(
        name="pneumonia_hemoptysis_fever",
        tier="base",
        expected_behavior="call_tool",
        expected_tool=SYMPTOMS_TOOL,
        prompts={
            "en": "I've been running a fever and coughing for a week, and today there was a streak of blood in what I coughed up.",
            "zh": "发烧咳嗽一周了，今天咳出来的痰里带了一点血丝。",
        },
        args_predicate=_p_tool_invoked,
    ),
    # ---- hard: URTI-scope but no fever, no red flag ----
    # Sore throat + nasal congestion + cough (E_97 + E_181 + E_201). Still in
    # DDXPlus URTI scope but without fever / systemic markers, small models
    # will read as "give it a few days" and skip. Capable models should still
    # offer the DDX loop.
    Case(
        name="urti_sore_throat_congestion",
        tier="hard",
        expected_behavior="call_tool",
        expected_tool=SYMPTOMS_TOOL,
        prompts={
            "en": "My throat has been sore for four days, my nose is stuffy, and I've been coughing on and off. Any idea what's going on?",
            "zh": "喉咙疼四天了，鼻塞、间歇性咳嗽，你觉得可能是什么？",
        },
        args_predicate=_p_tool_invoked,
    ),
    # ---- hard: chills + aches + fatigue with NO measured fever ----
    # Instability axis (1) + (2): the model's pre-training bias reads
    # "no fever = not flu = don't bother with DDX" even though DDXPlus's
    # E_91 (fever) is a felt-OR-measured question. This case exposes the
    # felt-vs-measured gap; capable models should still offer the loop.
    Case(
        name="flu_chills_no_measured_fever",
        tier="hard",
        expected_behavior="call_tool",
        expected_tool=SYMPTOMS_TOOL,
        prompts={
            "en": "The last two days I've been getting the chills, my body aches all over, and I'm exhausted. I haven't taken my temperature but I feel hot.",
            "zh": "这两天一直发冷、全身酸疼、特别累。没量体温，但感觉自己在发烫。",
        },
        args_predicate=_p_tool_invoked,
    ),
    # ---- hard: post-cold plateau, mild chest tightness ----
    # A cold that didn't resolve cleanly + mild chest tightness is the
    # slow-onset-pneumonia / post-viral-bronchitis pattern. Framed vaguely so
    # small models default to "you'll be fine in a few days". Capable models
    # should still route to DDX.
    Case(
        name="post_cold_chest_tight",
        tier="hard",
        expected_behavior="call_tool",
        expected_tool=SYMPTOMS_TOOL,
        prompts={
            "en": "I had a cold about ten days ago that I thought was gone, but I still have a lingering cough and my chest feels a bit tight when I take a deep breath.",
            "zh": "十天前感冒过一次，本以为好了，但一直咳嗽没停，深呼吸的时候胸口有点闷。",
        },
        args_predicate=_p_tool_invoked,
    ),
    # ---- fp: cold vs flu knowledge question, no personal complaint ----
    Case(
        name="fp_cold_vs_flu_knowledge",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "What's the actual difference between a cold and the flu? I get them mixed up.",
            "zh": "感冒和流感到底有什么区别？我总是分不清。",
        },
        args_predicate=_p_no_args,
    ),
    # ---- fp: vaccination timing — public-health question, not diagnostic ----
    Case(
        name="fp_flu_vaccine_timing",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "When is the best time of year to get the flu shot?",
            "zh": "流感疫苗一般什么时候打最合适？",
        },
        args_predicate=_p_no_args,
    ),
    # ---- fp: OTC cold-medicine recommendation — drug-advice ask, no DDX ----
    Case(
        name="fp_cold_medicine_recommendation",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "Which over-the-counter cold medicine works best for a stuffy nose and mild headache?",
            "zh": "鼻塞加轻度头疼，哪种非处方感冒药最有效？",
        },
        args_predicate=_p_no_args,
    ),
    # ---- fp: child has the flu — third-party + prevention question ----
    # Two independent decline signals stacked: the sick person is not the
    # user, and the ask is prevention rather than diagnosis. A model that
    # calls the tool here is misreading both.
    Case(
        name="fp_kid_has_flu_prevention",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "My daughter came down with the flu yesterday. What can I do to keep from catching it myself?",
            "zh": "我女儿昨天得了流感，我怎么做才能不被她传染？",
        },
        args_predicate=_p_no_args,
    ),
    # ---- fp: recently recovered from flu — resolved, no active complaint ----
    Case(
        name="fp_flu_recovered_last_week",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "I had the flu last week — high fever, aches, the whole thing. I'm basically back to normal now, just checking if that all sounds like the flu.",
            "zh": "上周得了流感——高烧、酸疼，全套症状。现在基本恢复了，就是想确认一下当时那些症状是不是流感。",
        },
        args_predicate=_p_no_args,
    ),
    # ---- fp: chronic cough — outside acute DDXPlus scope ----
    # DDXPlus is an acute-ED corpus. A 2-month mild cough with no fever is
    # a chronic-cough workup (post-viral, GERD, asthma, drug-induced) that
    # sits outside the tool's window. The tool description explicitly names
    # "chronic skin condition management" as out-of-scope; chronic cough is
    # the same shape of miss.
    Case(
        name="fp_chronic_cough_2mo",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "I've had a mild dry cough for about two months now. No fever, no chest pain, energy is fine. It just won't fully go away.",
            "zh": "轻微的干咳大概两个月了。没发烧、没胸口疼、精力也正常，就是一直断不干净。",
        },
        args_predicate=_p_no_args,
    ),
    # ---- fp: family-history worry, no personal symptoms ----
    # Anxiety framed around a relative's outcome ≠ personal diagnostic ask.
    Case(
        name="fp_pneumonia_family_worry",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "My grandfather passed away from pneumonia a few years ago. Now that my son has a cold I'm worried — is it common for a cold to turn into pneumonia?",
            "zh": "我爷爷几年前是肺炎去世的。现在我儿子感冒了我特别担心——感冒转成肺炎常见吗？",
        },
        args_predicate=_p_no_args,
    ),
    # ---- fp: vaccine side effect — expected reaction, not a symptom ask ----
    # Post-vaccination arm soreness with an obvious attributed cause. A model
    # that fires DDX here is ignoring the causal context in the same turn.
    Case(
        name="fp_flu_shot_arm_soreness",
        tier="fp",
        expected_behavior="decline",
        prompts={
            "en": "I got the flu shot yesterday and my arm is sore where they gave the injection. Is that normal?",
            "zh": "昨天打了流感疫苗，注射的那条胳膊有点疼。这正常吗？",
        },
        args_predicate=_p_no_args,
    ),
]
