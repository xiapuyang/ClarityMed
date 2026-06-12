"""Benchmark cases — predicate-style.

Each case carries enough information for *both* the runner (predicate
grading) and the judge (semantic grading), so the JSONL written by the
runner is self-contained: ``judge.py`` does not need to import or rerun
anything from this file beyond reading the row.

Three tiers:

* **base**  — explicit "use the X tool" prompts, mirror existing e2e.
  Floor for what every model should hit.
* **hard**  — implicit / paraphrased / unit-converted / multi-tool / ask.
  Where ceilings actually differ.
* **fp**    — false-positive: prompts that should NOT trigger any
  ingest tool (knowledge questions, chitchat, retrieve-vs-save
  ambiguity). The trial passes when no destructive tool fires.

``expected_behavior`` encodes the success criterion the runner uses:

* ``"call_tool"``  — ``expected_tool`` must be called and ``args_predicate``
  must pass on its args.
* ``"call_tools"`` — every name in ``expected_tools`` must be called
  (order doesn't matter).
* ``"decline"``    — no tool in ``INGEST_TOOLS`` may be called.
                     ``ask_user_question`` is allowed (a polite "are you
                     sure?" is fine).
* ``"ask"``        — ``ask_user_question`` must be called at least once.

Existing pytest is not touched; this is the benchmark's own set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from claritymed.stores.manifest_store import ManifestStore

USER_ID = "bench"

INGEST_TOOLS: frozenset[str] = frozenset(
    {
        "save_allergy",
        "save_medication",
        "save_condition",
        "update_profile_field",
        "save_record",
        "save_to_library",
        "delete_record",
    }
)
ASK_USER_TOOL = "ask_user_question"


@dataclass
class Case:
    """One benchmark cell definition."""

    name: str
    tier: str  # "base" | "hard" | "fp"
    expected_behavior: str  # "call_tool" | "call_tools" | "decline" | "ask"
    prompts: dict[str, str]
    args_predicate: Callable[[dict], tuple[bool, str]]
    expected_tool: Optional[str] = None
    expected_tools: list[str] = field(default_factory=list)
    seed: Optional[Callable[[], dict]] = None


# --- predicates ------------------------------------------------------
#
# Each returns ``(ok, reason)``. ``reason`` is recorded in the JSONL so a
# downstream eyeballing run can see why a predicate flagged a row. Keep
# predicates short and string-based — the judge handles nuance.


def _str(args: dict, key: str) -> str:
    v = args.get(key)
    return v if isinstance(v, str) else ""


def _p_substance_penicillin(args: dict) -> tuple[bool, str]:
    s = _str(args, "substance").lower()
    if "penicillin" in s or "青霉素" in s:
        return True, f"substance={s!r}"
    return False, f"substance does not look like penicillin: {s!r}"


def _p_med_metformin(args: dict) -> tuple[bool, str]:
    n = _str(args, "name").lower()
    if "metformin" in n or "二甲双胍" in n:
        return True, f"name={n!r}"
    return False, f"name does not look like metformin: {n!r}"


def _p_condition_diabetes(args: dict) -> tuple[bool, str]:
    d = _str(args, "display").lower()
    if "diabet" in d or "糖尿" in d:
        return True, f"display={d!r}"
    return False, f"display does not look like diabetes: {d!r}"


def _p_weight_kg_plausible(args: dict) -> tuple[bool, str]:
    if args.get("field") != "weight_kg":
        return False, f"field={args.get('field')!r}, expected weight_kg"
    try:
        v = float(args.get("value"))
    except (TypeError, ValueError):
        return False, f"value not numeric: {args.get('value')!r}"
    if 30 <= v <= 250:
        return True, f"value={v}kg"
    return False, f"value out of plausible range: {v}"


def _p_weight_kg_converted_from_160lbs(args: dict) -> tuple[bool, str]:
    """160 lbs ≈ 72.6 kg; accept [70, 75] as 'converted correctly'."""
    if args.get("field") != "weight_kg":
        return False, f"field={args.get('field')!r}"
    try:
        v = float(args.get("value"))
    except (TypeError, ValueError):
        return False, f"value not numeric: {args.get('value')!r}"
    if 70 <= v <= 75:
        return True, f"value={v}kg (160 lbs ≈ 72.6 kg)"
    if 155 <= v <= 165:
        return False, f"value={v} looks like lbs, not converted to kg"
    return False, f"value={v} not a plausible conversion of 160 lbs"


def _p_save_record_checkup(args: dict) -> tuple[bool, str]:
    """Hard-tier predicate — prompts no longer hand-feed 'checkups',
    so we only require the args be coherent: category + title present.
    Whether the chosen category label is the *best* one is judged by
    the LLM judge, not the predicate."""
    cat = _str(args, "category")
    title = _str(args, "title")
    if not cat:
        return False, "category empty"
    if not title:
        return False, "title empty"
    return True, f"category={cat!r}, title={title!r}"


def _p_save_to_library_hypertension(args: dict) -> tuple[bool, str]:
    """Loose predicate — prompts no longer use the word 'library', so
    the only requirement is a non-empty title that mentions the topic
    (hypertension / 高血压). The semantic 'is this really library-worthy?'
    call is left to the judge."""
    title = _str(args, "title").lower()
    if not title:
        return False, "title empty"
    if "hypertension" in title or "高血压" in title:
        return True, f"title={title!r}"
    return False, f"title={title!r} does not mention hypertension"


def _p_delete_record_match_seed(args: dict) -> tuple[bool, str]:
    rp = _str(args, "record_path")
    ck = _str(args, "confirm_kind")
    if rp and ck:
        return True, f"record_path={rp!r} confirm_kind={ck!r}"
    return False, f"missing record_path / confirm_kind: {args!r}"


def _p_args_present(args: dict) -> tuple[bool, str]:
    """Catch-all — used for cases graded by behavior, not args content
    (multi-tool, decline, ask, fp)."""
    return True, "no per-args predicate"


# --- seed (delete_record) -------------------------------------------


def _seed_delete_target() -> dict:
    store = ManifestStore(USER_ID, "records")
    category = "exam-reports"
    slug = "2026-06-01-checkup-bench"
    store.create(
        category,
        slug,
        {
            "kind": "exam-report",
            "title": "Annual checkup",
            "date": "2026-06-01",
            "provider": "Dr. Bench",
            "attachments": [],
            "extracted_labs": [],
            "tags": [],
            "notes": "Seeded for delete_record benchmark.",
        },
    )
    return {"record_path": f"{category}/{slug}", "confirm_kind": "exam-report"}


# --- cases -----------------------------------------------------------

CASES: list[Case] = [
    # ---- BASE: explicit, floor measurement -------------------------
    Case(
        name="save_allergy",
        tier="base",
        expected_behavior="call_tool",
        expected_tool="save_allergy",
        args_predicate=_p_substance_penicillin,
        prompts={
            "en": (
                "Use the save_allergy tool now to record an allergy: "
                "substance=penicillin, severity=severe, source=self_report. "
                "Do not ask follow-up questions; call the tool and confirm."
            ),
            "zh": (
                "现在调用 save_allergy 工具登记一条过敏：substance=penicillin, "
                "severity=severe, source=self_report。不要追问，直接调用工具并确认。"
            ),
        },
    ),
    Case(
        name="save_medication",
        tier="base",
        expected_behavior="call_tool",
        expected_tool="save_medication",
        args_predicate=_p_med_metformin,
        prompts={
            "en": (
                "Use the save_medication tool now: "
                "name=metformin, dose=500 mg, frequency=twice daily. "
                "Do not ask follow-up questions; call the tool and confirm."
            ),
            "zh": (
                "现在调用 save_medication 工具：name=metformin, dose=500 mg, "
                "frequency=twice daily。不要追问，直接调用工具并确认。"
            ),
        },
    ),
    Case(
        name="save_condition",
        tier="base",
        expected_behavior="call_tool",
        expected_tool="save_condition",
        args_predicate=_p_condition_diabetes,
        prompts={
            "en": (
                "Use the save_condition tool now: "
                "display='type 2 diabetes', onset_date=2020-01-01. "
                "Do not ask follow-up questions; call the tool and confirm."
            ),
            "zh": (
                "现在调用 save_condition 工具："
                "display='type 2 diabetes', onset_date=2020-01-01。"
                "不要追问，直接调用工具并确认。"
            ),
        },
    ),
    Case(
        name="update_profile_weight",
        tier="base",
        expected_behavior="call_tool",
        expected_tool="update_profile_field",
        args_predicate=_p_weight_kg_plausible,
        prompts={
            "en": (
                "My weight is now 72.5 kg. Please record that update. "
                "Use the update_profile_field tool — do not ask follow-up "
                "questions, call it and confirm."
            ),
            "zh": (
                "我现在的体重是 72.5 kg，请记录更新。"
                "调用 update_profile_field 工具，不要追问，直接调用并确认。"
            ),
        },
    ),
    Case(
        name="save_record",
        tier="base",
        expected_behavior="call_tool",
        expected_tool="save_record",
        args_predicate=_p_save_record_checkup,
        prompts={
            "en": (
                "I had a checkup yesterday. Please save it. "
                "Use save_record with these args: "
                'category="checkups", kind="checkup", title="Annual checkup". '
                "Leave attachments, extracted_labs, tags, notes, provider, date "
                "at their defaults — do not pass them. "
                "Call the tool and confirm; do not ask follow-up questions."
            ),
            "zh": (
                "我昨天做了体检，请保存。调用 save_record，参数："
                'category="checkups", kind="checkup", title="Annual checkup"。'
                "attachments、extracted_labs、tags、notes、provider、date 保持默认，不要传。"
                "直接调用工具并确认；不要追问。"
            ),
        },
    ),
    Case(
        name="save_to_library",
        tier="base",
        expected_behavior="call_tool",
        expected_tool="save_to_library",
        args_predicate=_p_save_to_library_hypertension,
        prompts={
            "en": (
                "Please save a library entry. "
                'Use save_to_library with title="2024 Hypertension Guideline". '
                "Leave attachments, authors, year, tags, public at their defaults "
                "— do not pass them. "
                "Call the tool and confirm; do not ask follow-up questions."
            ),
            "zh": (
                "请保存一条文库条目。调用 save_to_library，"
                'title="2024 Hypertension Guideline"。'
                "attachments、authors、year、tags、public 保持默认，不要传。"
                "直接调用工具并确认；不要追问。"
            ),
        },
    ),
    Case(
        name="delete_record",
        tier="base",
        expected_behavior="call_tool",
        expected_tool="delete_record",
        args_predicate=_p_delete_record_match_seed,
        seed=_seed_delete_target,
        prompts={
            "en": (
                "Use the delete_record tool now: "
                "record_path={record_path}, confirm_kind={confirm_kind}. "
                "Do not ask follow-up questions; call the tool and confirm."
            ),
            "zh": (
                "现在调用 delete_record 工具："
                "record_path={record_path}, confirm_kind={confirm_kind}。"
                "不要追问，直接调用工具并确认。"
            ),
        },
    ),
    # ---- HARD: implicit / inferred / multi-tool / ask ---------------
    Case(
        # No "please add this to my chart" — natural statement only.
        # A capable model recognises this as a record-worthy fact and
        # offers/calls save_allergy on its own.
        name="implicit_allergy",
        tier="hard",
        expected_behavior="call_tool",
        expected_tool="save_allergy",
        args_predicate=_p_substance_penicillin,
        prompts={
            "en": (
                "Heads up — I'm severely allergic to penicillin. "
                "Last time I took it I went into anaphylaxis."
            ),
            "zh": ("提醒一下，我对青霉素严重过敏。上次吃了之后引发了过敏性休克。"),
        },
    ),
    Case(
        # No "please update my profile" — the unit conversion is the
        # whole test. User just shares a number.
        name="weight_unit_conversion",
        tier="hard",
        expected_behavior="call_tool",
        expected_tool="update_profile_field",
        args_predicate=_p_weight_kg_converted_from_160lbs,
        prompts={
            "en": "I weighed in at 160 pounds this morning.",
            "zh": "今早我称的体重是 160 磅。",
        },
    ),
    Case(
        # No "please record both" — model must recognise this is TWO
        # facts (condition + medication) on its own.
        name="multi_condition_med",
        tier="hard",
        expected_behavior="call_tools",
        expected_tools=["save_condition", "save_medication"],
        args_predicate=_p_args_present,
        prompts={
            "en": (
                "I was diagnosed with type 2 diabetes back in 2020, "
                "and I've been on metformin 500 mg twice daily ever since."
            ),
            "zh": (
                "我 2020 年被诊断为 2 型糖尿病，"
                "从那时起一直在吃二甲双胍 500 mg 一日两次。"
            ),
        },
    ),
    Case(
        # Hinted twin — "please record both" explicitly tells the
        # model it's two distinct facts. Pair lets us measure how
        # much help that single phrase provides.
        name="multi_condition_med_hinted",
        tier="hard",
        expected_behavior="call_tools",
        expected_tools=["save_condition", "save_medication"],
        args_predicate=_p_args_present,
        prompts={
            "en": (
                "I was diagnosed with type 2 diabetes back in 2020, "
                "and I've been on metformin 500 mg twice daily ever since. "
                "Please record both."
            ),
            "zh": (
                "我 2020 年被诊断为 2 型糖尿病，"
                "从那时起一直在吃二甲双胍 500 mg 一日两次。请把这两条都记下来。"
            ),
        },
    ),
    Case(
        name="decline_thinking_aloud",
        tier="hard",
        expected_behavior="decline",
        args_predicate=_p_args_present,
        prompts={
            "en": (
                "I'm just thinking out loud about whether I should start metformin. "
                "Don't add anything to my record yet — I haven't decided."
            ),
            "zh": (
                "我只是随口想想要不要开始吃二甲双胍。先别往我的病历里加任何东西——我还没决定。"
            ),
        },
    ),
    Case(
        # No "please file it under checkups" — model picks its own
        # category. Pair-mate of ``implicit_save_record_hinted`` for
        # measuring how much the prompt nudge buys.
        name="implicit_save_record",
        tier="hard",
        expected_behavior="call_tool",
        expected_tool="save_record",
        args_predicate=_p_save_record_checkup,
        prompts={
            "en": "I had my annual checkup yesterday.",
            "zh": "昨天做了年度体检。",
        },
    ),
    Case(
        # Hinted twin — "please file it under checkups" tells the
        # model both *that* to save and *where*. Lets the CSV compare
        # hinted vs natural delta per model.
        name="implicit_save_record_hinted",
        tier="hard",
        expected_behavior="call_tool",
        expected_tool="save_record",
        args_predicate=_p_save_record_checkup,
        prompts={
            "en": "My annual checkup was yesterday — please file it under checkups.",
            "zh": "昨天做了年度体检，请把它归档到 checkups 类别下。",
        },
    ),
    Case(
        # No "save it in my library" — natural statement, model has
        # to infer that "want to keep handy" → save_to_library.
        name="implicit_save_to_library",
        tier="hard",
        expected_behavior="call_tool",
        expected_tool="save_to_library",
        args_predicate=_p_save_to_library_hypertension,
        prompts={
            "en": (
                "I came across a really solid 2024 hypertension guideline "
                "and want to keep it handy for future reference."
            ),
            "zh": ("我找到一份很好的 2024 年高血压指南，想留着以后参考。"),
        },
    ),
    Case(
        # Hinted twin — explicit "save in my library" picks the tool
        # for the model. A/B with the natural version above.
        name="implicit_save_to_library_hinted",
        tier="hard",
        expected_behavior="call_tool",
        expected_tool="save_to_library",
        args_predicate=_p_save_to_library_hypertension,
        prompts={
            "en": (
                "I found a great 2024 hypertension guideline I want to keep handy. "
                "Save it in my library."
            ),
            "zh": (
                "我找到一份很好的 2024 年高血压指南，想留作参考。请存到我的文库里。"
            ),
        },
    ),
    Case(
        # Drug class, not specific drug → model should ask which one.
        name="ask_drug_class",
        tier="hard",
        expected_behavior="ask",
        args_predicate=_p_args_present,
        prompts={
            "en": (
                "Please add to my chart that I take a beta blocker for blood pressure."
            ),
            "zh": ("请加到我的病历里，我在吃 beta blocker（β 受体阻滞剂）控制血压。"),
        },
    ),
    Case(
        # Ambiguous "the result from yesterday" — what kind? Should ask.
        name="ask_ambiguous_record",
        tier="hard",
        expected_behavior="ask",
        args_predicate=_p_args_present,
        prompts={
            "en": "Save the result from yesterday into my records.",
            "zh": "把昨天的那份结果保存到我的记录里。",
        },
    ),
    Case(
        # No content at all — model must ask "what would you like to save?"
        # The wrong move is hallucinating a save_record with invented args.
        name="ask_empty_save",
        tier="hard",
        expected_behavior="ask",
        args_predicate=_p_args_present,
        prompts={
            "en": "Can you save this to my records, please?",
            "zh": "帮我把这个保存到记录里好吗？",
        },
    ),
    Case(
        # Delete by description, not record_path. The user can't say
        # "checkups/2026-03-12-annual" out loud — model must ask which
        # one (or list candidates back). Hallucinating a record_path
        # would be a destructive misfire.
        name="ask_delete_by_description",
        tier="hard",
        expected_behavior="ask",
        args_predicate=_p_args_present,
        prompts={
            "en": "Please delete my last lab report from my records.",
            "zh": "把我上次那份化验单从记录里删掉。",
        },
    ),
    Case(
        # Allergy without severity. SaveAllergyArgs requires the literal
        # severity field; a careful model should ask rather than picking
        # one. Saving with "mild" or "moderate" silently is the wrong
        # move because severity drives clinical decisions downstream.
        name="ask_allergy_no_severity",
        tier="hard",
        expected_behavior="ask",
        args_predicate=_p_args_present,
        prompts={
            "en": (
                "Add to my chart that I'm allergic to peanuts — "
                "I get a reaction every time."
            ),
            "zh": ("请加到我的病历里，我对花生过敏——每次接触都有反应。"),
        },
    ),
    Case(
        # Vague history: "a few chronic conditions" — which ones?
        # Model must enumerate via ask, not guess.
        name="ask_vague_conditions",
        tier="hard",
        expected_behavior="ask",
        args_predicate=_p_args_present,
        prompts={
            "en": (
                "I've dealt with a few chronic conditions over the years. "
                "Can you add them to my profile so my doctors can see them?"
            ),
            "zh": ("我这些年有几种慢性病。能加到我的档案里，让医生们能看到吗？"),
        },
    ),
    # ---- FP: must NOT trigger an ingest tool ------------------------
    Case(
        name="fp_knowledge_alt",
        tier="fp",
        expected_behavior="decline",
        args_predicate=_p_args_present,
        prompts={
            "en": "What does it mean if my ALT is elevated on a blood test?",
            "zh": "血液化验里 ALT 偏高代表什么？",
        },
    ),
    Case(
        name="fp_chitchat",
        tier="fp",
        expected_behavior="decline",
        args_predicate=_p_args_present,
        prompts={
            "en": "Hi, can you hear me?",
            "zh": "你好，能听见我说话吗？",
        },
    ),
    Case(
        name="fp_retrieve_not_save",
        tier="fp",
        expected_behavior="decline",
        args_predicate=_p_args_present,
        prompts={
            "en": (
                "What was my cholesterol value from the last checkup I had on file?"
            ),
            "zh": "我档案里最近一次体检的胆固醇值是多少？",
        },
    ),
]
