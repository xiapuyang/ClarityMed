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
* ``"decline"``           — no tool in ``INGEST_TOOLS`` may be called.
                            ``ask_user_question`` is allowed (a polite
                            "are you sure?" is fine).
* ``"ask_tool"``          — ``ask_user_question`` MUST be called. Used
                            when the clarification has a small enumerable
                            answer space (severity tier, record kind,
                            specific drug within a class) where the
                            structured picker is the contract. A plain-
                            text question grades as ``text_ask_only``
                            (failure) because it loses the structure.
* ``"ask_tool_or_text"``  — open-ended clarification: either the
                            structured tool OR a plain-text question is
                            accepted. Passes when no ingest tool fires
                            AND (a) ``ask_user_question`` was called, or
                            (b) the response contains a question /
                            imperative-clarification phrase. Matches the
                            system prompt's "open-ended stays prose" rule.

Existing pytest is not touched; this is the benchmark's own set.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Callable, Optional

from claritymed.stores.manifest_store import ManifestStore
from claritymed.stores.profile import ProfileStore

from tests.benchmarks.tool_invoke.base import USER_ID

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
    expected_behavior: str
    # one of: "call_tool" | "call_tools" | "decline" | "ask_tool" |
    # "ask_tool_or_text" | "ask_then_call_tool"
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


def _p_substance_peanut(args: dict) -> tuple[bool, str]:
    s = _str(args, "substance").lower()
    if "peanut" in s or "花生" in s:
        return True, f"substance={s!r}"
    return False, f"substance does not look like peanut: {s!r}"


def _p_medication_any(args: dict) -> tuple[bool, str]:
    """Catch-all — any non-empty name is valid for ask_then_save_medication."""
    n = _str(args, "name").lower()
    if n:
        return True, f"name={n!r}"
    return False, "name is empty"


def _p_birth_year_from_age(stated_age: int) -> Callable[[dict], tuple[bool, str]]:
    """Return a predicate that checks update_profile_field maps stated age → birth_year.

    Accepts ±1 year to handle the ambiguity of whether the user's birthday
    has already passed in the current calendar year.
    """
    current_year = datetime.date.today().year
    expected_year = current_year - stated_age

    def _check(args: dict) -> tuple[bool, str]:
        if args.get("field") != "birth_date":
            return False, f"field={args.get('field')!r}, expected birth_date"
        val = _str(args, "value")
        if not val:
            return False, "value empty"
        try:
            year = int(val[:4])
        except ValueError:
            return False, f"cannot parse year from {val!r}"
        if abs(year - expected_year) <= 1:
            return True, f"birth_date={val!r} (year={year} ≈ {expected_year})"
        return (
            False,
            f"year={year} not within ±1 of {expected_year} for age {stated_age}",
        )

    return _check


# --- seeds -----------------------------------------------------------


def _seed_profile_weight() -> dict:
    """Seed weight_kg=72.5 so the profile already has this value."""
    store = ProfileStore(USER_ID)
    store.update_profile_field("weight_kg", 72.5, owner_user_id=USER_ID)
    return {}


def _seed_profile_sex_female() -> dict:
    """Seed sex='female' so repeating 'I'm a woman' should be a no-op."""
    store = ProfileStore(USER_ID)
    store.update_profile_field("sex", "female", owner_user_id=USER_ID)
    return {}


def _seed_profile_occupation() -> dict:
    """Seed current_occupation='nurse' for the repeat-occupation FP case."""
    store = ProfileStore(USER_ID)
    store.update_profile_field("current_occupation", "nurse", owner_user_id=USER_ID)
    return {}


def _seed_allergy_penicillin() -> dict:
    """Seed an active penicillin allergy so repeating it should be a no-op."""
    from claritymed.core.schemas.patient import Allergy
    from claritymed.stores.profile import ProfileStore as _PS

    _PS(USER_ID).add_allergy(
        Allergy(substance="penicillin", severity="severe", source="self_report"),
        owner_user_id=USER_ID,
    )
    return {}


def _seed_condition_hypertension() -> dict:
    """Seed an active hypertension condition."""
    from claritymed.core.schemas.patient import Condition
    from claritymed.stores.profile import ProfileStore as _PS

    _PS(USER_ID).add_condition(
        Condition(display="hypertension"),
        owner_user_id=USER_ID,
    )
    return {}


def _seed_medication_metformin() -> dict:
    """Seed an active metformin medication."""
    from claritymed.core.schemas.patient import Medication
    from claritymed.stores.profile import ProfileStore as _PS

    _PS(USER_ID).add_medication(
        Medication(display="metformin", dose="500 mg", frequency="twice daily"),
        owner_user_id=USER_ID,
    )
    return {}


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
        expected_behavior="ask_tool",
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
        expected_behavior="ask_tool",
        args_predicate=_p_args_present,
        prompts={
            "en": "Save the result from yesterday into my records.",
            "zh": "把昨天的那份结果保存到我的记录里。",
        },
    ),
    Case(
        # No content at all — model must ask "what would you like to save?"
        # The wrong move is hallucinating a save_record with invented args.
        # Open-ended (could be anything), so a plain-text follow-up is
        # spec-compliant per ask.yaml v6: enumerable → tool, open-ended → text.
        name="ask_empty_save",
        tier="hard",
        expected_behavior="ask_tool_or_text",
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
        expected_behavior="ask_tool",
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
        expected_behavior="ask_tool",
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
        # The answer space is open-ended (any chronic condition), so a
        # plain-text "which conditions?" follow-up is spec-compliant per
        # ask.yaml v6 (enumerable → tool, open-ended → text).
        name="ask_vague_conditions",
        tier="hard",
        expected_behavior="ask_tool_or_text",
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
    # Pure knowledge — names a drug but never first-person, never
    # personal data. Misfiring save_medication here means the model
    # latched on the keyword instead of the disclosure pattern.
    Case(
        name="fp_pure_knowledge",
        tier="fp",
        expected_behavior="decline",
        args_predicate=_p_args_present,
        prompts={
            "en": "How does penicillin actually kill bacteria?",
            "zh": "青霉素到底是怎么杀死细菌的？",
        },
    ),
    # Third-party subject — first-person possessive ("my cat") but the
    # subject the medical fact applies to is NOT the user. Saving here
    # would pollute the user's own record with someone else's data.
    Case(
        name="fp_third_party_subject",
        tier="fp",
        expected_behavior="decline",
        args_predicate=_p_args_present,
        prompts={
            "en": "Is amoxicillin safe for my cat?",
            "zh": "阿莫西林给猫吃安全吗？",
        },
    ),
    # Hypothetical — first-person grammar but conditional ("if I were").
    # No actual disclosure has happened. Saving an allergy that the user
    # only floated as a what-if is a destructive false positive.
    Case(
        name="fp_hypothetical",
        tier="fp",
        expected_behavior="decline",
        args_predicate=_p_args_present,
        prompts={
            "en": (
                "If I were allergic to sulfa drugs, what antibiotics should I avoid?"
            ),
            "zh": "假如我对磺胺类过敏，有哪些抗生素需要避开？",
        },
    ),
    # ---- HARD: age → birth_year inference ---------------------------
    Case(
        # User states age only — model must infer birth_year = current_year − age
        # and call update_profile_field(field="birth_date", value="YYYY-01-01").
        name="age_to_birth_year",
        tier="hard",
        expected_behavior="call_tool",
        expected_tool="update_profile_field",
        args_predicate=_p_birth_year_from_age(35),
        prompts={
            "en": "By the way, I'm 35 years old.",
            "zh": "对了，我今年 35 岁。",
        },
    ),
    Case(
        # Conversational phrasing with extra context — model must still
        # extract the age and map it to birth_date, ignoring the surrounding
        # chitchat about birthdays.
        name="age_to_birth_year_casual",
        tier="hard",
        expected_behavior="call_tool",
        expected_tool="update_profile_field",
        args_predicate=_p_birth_year_from_age(35),
        prompts={
            "en": (
                "I just turned 35 last month. Crazy how fast time flies. "
                "Anyway, I don't think you have my age on file."
            ),
            "zh": (
                "我上个月刚满 35 岁，时间过得真快。顺便说一下，"
                "我觉得你档案里还没有我的年龄。"
            ),
        },
    ),
    # ---- FP: already-in-profile — seeded data should not be re-saved --
    # Each case pre-seeds one or more profile facts, then repeats that
    # same information in natural language. The expected behavior is
    # "decline" — the model should recognise the profile already has the
    # data (via [Patient profile] block) and skip re-invoking the tool.
    Case(
        name="fp_repeat_weight",
        tier="fp",
        expected_behavior="decline",
        args_predicate=_p_args_present,
        seed=_seed_profile_weight,
        prompts={
            "en": "My weight is still around 72-73 kg, hasn't changed.",
            "zh": "我的体重还是 72-73 公斤左右，没什么变化。",
        },
    ),
    Case(
        name="fp_repeat_sex",
        tier="fp",
        expected_behavior="decline",
        args_predicate=_p_args_present,
        seed=_seed_profile_sex_female,
        prompts={
            "en": "Just so you know, I'm a woman.",
            "zh": "顺便说一下，我是女性。",
        },
    ),
    Case(
        name="fp_repeat_occupation",
        tier="fp",
        expected_behavior="decline",
        args_predicate=_p_args_present,
        seed=_seed_profile_occupation,
        prompts={
            "en": "I work as a nurse, in case that's helpful context.",
            "zh": "我是一名护士，提供一下背景信息。",
        },
    ),
    Case(
        name="fp_repeat_allergy",
        tier="fp",
        expected_behavior="decline",
        args_predicate=_p_args_present,
        seed=_seed_allergy_penicillin,
        prompts={
            "en": "Just a reminder — I'm still allergic to penicillin.",
            "zh": "提醒一下，我对青霉素还是过敏的。",
        },
    ),
    Case(
        name="fp_repeat_condition",
        tier="fp",
        expected_behavior="decline",
        args_predicate=_p_args_present,
        seed=_seed_condition_hypertension,
        prompts={
            "en": "I have high blood pressure, as you probably already know.",
            "zh": "我有高血压，你应该已经知道了。",
        },
    ),
    Case(
        name="fp_repeat_medication",
        tier="fp",
        expected_behavior="decline",
        args_predicate=_p_args_present,
        seed=_seed_medication_metformin,
        prompts={
            "en": "I'm still taking metformin 500 mg twice a day for my diabetes.",
            "zh": "我还在吃二甲双胍 500 mg 一天两次，控制糖尿病。",
        },
    ),
    # ---- HARD: sequential ask → tool --------------------------------
    # These cases use _AutoAnswerFirstOptionChannel: the model asks a
    # clarifying question, receives option[0] as the answer, then must
    # proceed to call the ingest tool with that answer.
    # Predicate grading checks:
    #   - asked at least once   (sequential flow occurred)
    #   - expected tool called  (model acted on the answer)
    # If the model skips the ask and guesses directly, outcome is
    # "called_without_asking" (still predicate_pass=True — the save
    # happened, just without the interactive step).
    Case(
        name="ask_then_save_allergy",
        tier="hard",
        expected_behavior="ask_then_call_tool",
        expected_tool="save_allergy",
        args_predicate=_p_substance_peanut,
        prompts={
            "en": (
                "I'm allergic to peanuts — I always get a reaction. "
                "Please add it to my profile."
            ),
            "zh": "我对花生过敏，每次接触都有反应。请帮我加到档案里。",
        },
    ),
    Case(
        name="ask_then_save_medication",
        tier="hard",
        expected_behavior="ask_then_call_tool",
        expected_tool="save_medication",
        args_predicate=_p_medication_any,
        prompts={
            "en": (
                "I take a beta blocker for my blood pressure. "
                "Please add it to my chart."
            ),
            "zh": "我在吃 beta blocker（β 受体阻滞剂）控制血压。请帮我加到病历里。",
        },
    ),
]
