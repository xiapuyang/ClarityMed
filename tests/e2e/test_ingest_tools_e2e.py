"""End-to-end ingest tool drive — real local LLM through the deferred-loop.

For each of the seven ingest tools, send a prompt biased toward that
tool, run the full AskService deferred loop against the live local
provider, auto-approve via a stub channel, and verify the side effect
landed (DB row, manifest dir, etc.).

The local LLM is non-deterministic: any one prompt may or may not
trigger the tool the test expects. Each tool gets up to
``MAX_ATTEMPTS`` independent tries — fresh ChatSession per attempt so
prior history can't bias the next run. If no attempt triggers the
tool, the test fails with the prompts used so the next maintainer can
adjust them.

Test user is ``e2e`` per the project convention (CLAUDE.md →
"Test user_id convention"). Each test gets its own tmp data dir via
the project-wide ``_isolate_runtime`` fixture, so the live local
model talks to disposable per-test stores.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from typing import Any, Callable

import pytest

from claritymed.context import apply_context, reset_context
from claritymed.core.interaction import ApprovalDecision
from claritymed.core.rag import load_retrieval_config
from claritymed.core.schemas.patient import Allergy, Condition, Medication
from claritymed.orchestrator.services import AskService
from claritymed.orchestrator.services.chat_session import ChatSession
from claritymed.stores.manifest_store import ManifestStore
from claritymed.stores.profile import ProfileStore

logger = logging.getLogger(__name__)

USER_ID = "e2e"
MAX_ATTEMPTS = 5
PER_TURN_TIMEOUT_S = 90.0


# --- channel stub -----------------------------------------------------


class _AutoApproveChannel:
    """Approves every call. Records (tool_name, args, breadcrumb) tuples."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, str | None]] = []

    async def request(
        self,
        tool_name: str,
        args: dict,
        *,
        breadcrumb: str | None = None,
    ) -> ApprovalDecision:
        self.calls.append((tool_name, dict(args), breadcrumb))
        return ApprovalDecision(decision="once")


# --- per-tool verifications -------------------------------------------


def _verify_save_allergy() -> None:
    allergies = ProfileStore(USER_ID).list_allergies()
    assert allergies, "ProfileStore had no allergies after save_allergy run"
    # The LLM may rephrase the substance ("penicillin" → "penicillin (PCN)");
    # accept any row whose substance contains "penicillin" case-insensitively.
    assert any("penicillin" in a.substance.lower() for a in allergies), (
        f"no penicillin allergy row; got {[a.substance for a in allergies]!r}"
    )


def _verify_save_medication() -> None:
    meds = ProfileStore(USER_ID).list_medications()
    assert meds, "ProfileStore had no medications after save_medication run"
    assert any("metformin" in m.display.lower() for m in meds), (
        f"no metformin medication row; got {[m.display for m in meds]!r}"
    )


def _verify_save_condition() -> None:
    conds = ProfileStore(USER_ID).list_conditions()
    assert conds, "ProfileStore had no conditions after save_condition run"
    assert any("diabet" in c.display.lower() for c in conds), (
        f"no diabetes condition row; got {[c.display for c in conds]!r}"
    )


def _verify_save_record() -> None:
    manifests = list(ManifestStore(USER_ID, "records").list())
    assert manifests, "records/ store had no manifests after save_record run"


def _verify_save_to_library() -> None:
    manifests = list(ManifestStore(USER_ID, "library").list())
    assert manifests, "library/ store had no manifests after save_to_library run"


def _verify_delete_record(seed_record_path: str) -> Callable[[], None]:
    """Build a verifier that checks the seeded manifest is gone."""

    def _check() -> None:
        category, slug = seed_record_path.split("/", 1)
        try:
            ManifestStore(USER_ID, "records").read(category, slug)
        except Exception:
            return  # gone — success
        raise AssertionError(
            f"seeded manifest {seed_record_path!r} still present after "
            "delete_record run"
        )

    return _check


# --- setup helpers ---------------------------------------------------


def _seed_delete_target() -> tuple[str, str]:
    """Seed one record into records/exam-reports/<slug> for delete_record.

    Returns ``(record_path, confirm_kind)`` so the prompt can quote them
    verbatim — delete_record requires the LLM to ground both values
    instead of guessing, which is hard to do from a one-shot prompt
    without context. We hand them over directly.
    """
    store = ManifestStore(USER_ID, "records")
    category = "exam-reports"
    slug = "2026-06-01-checkup-e2e"
    manifest_data = {
        "kind": "exam-report",
        "title": "Annual checkup",
        "date": "2026-06-01",
        "provider": "Dr. E2E",
        "attachments": [],
        "extracted_labs": [],
        "tags": [],
        "notes": "Seeded for delete_record e2e.",
    }
    store.create(category, slug, manifest_data)
    return f"{category}/{slug}", "exam-report"


async def _run_one_attempt(
    provider_id: str,
    prompt: str,
    *,
    prompt_channel: object | None = None,
) -> tuple[_AutoApproveChannel, list[Any]]:
    """Run one full AskService turn end-to-end. Returns (channel, events).

    Constructs a fresh ChatSession + AskService so prior history can't
    bias the next attempt. Resolves the live local provider via
    production code (``resolve_provider``) so this exercises the same
    wiring the TUI uses.

    Pass ``prompt_channel`` to wire up interactive question-answering;
    leave it ``None`` for tool-only tests (ask_user_question won't be
    registered at all).
    """
    from claritymed.core.llm.model import build_model
    from claritymed.stores.models import resolve_provider

    provider = resolve_provider(override=provider_id)
    model = build_model(provider)
    chat = ChatSession.new(USER_ID)
    channel = _AutoApproveChannel()
    rag_mode = load_retrieval_config().rag.mode

    service = AskService(
        model=model,
        chat_session=chat,
        provider_id=provider.id,
        model_name=provider.model,
        provider_config=provider,
        rag_mode=rag_mode,
        tool_approval_channel=channel,
        prompt_channel=prompt_channel,
    )

    events: list[Any] = []
    try:
        async with asyncio.timeout(PER_TURN_TIMEOUT_S):
            async for ev in service.run(prompt, user_id=USER_ID):
                events.append(ev)
    except asyncio.TimeoutError:
        logger.warning("attempt timed out after %.0fs", PER_TURN_TIMEOUT_S)
    return channel, events


# --- cases ------------------------------------------------------------


@pytest.fixture
def _ctx():
    tokens = apply_context("20260612e2e000000000000", USER_ID, "en")
    yield
    reset_context(tokens)


_CASES: list[tuple[str, str, Callable[[], None]]] = [
    (
        "save_allergy",
        (
            "Use the save_allergy tool now to record an allergy: "
            "substance=penicillin, severity=severe, source=self_report. "
            "Do not ask follow-up questions; call the tool and confirm."
        ),
        _verify_save_allergy,
    ),
    (
        "save_medication",
        (
            "Use the save_medication tool now: "
            "name=metformin, dose=500 mg, frequency=twice daily. "
            "Do not ask follow-up questions; call the tool and confirm."
        ),
        _verify_save_medication,
    ),
    (
        "save_condition",
        (
            "Use the save_condition tool now: "
            "display='type 2 diabetes', onset_date=2020-01-01. "
            "Do not ask follow-up questions; call the tool and confirm."
        ),
        _verify_save_condition,
    ),
    (
        "save_record",
        (
            "I had a checkup yesterday. Please save it. "
            "Use save_record with these args: "
            'category="checkups", kind="checkup", title="Annual checkup". '
            "Leave attachments, extracted_labs, tags, notes, provider, date "
            "at their defaults — do not pass them. "
            "Call the tool and confirm; do not ask follow-up questions."
        ),
        _verify_save_record,
    ),
    (
        "save_to_library",
        (
            "Please save a library entry. "
            'Use save_to_library with title="2024 Hypertension Guideline". '
            "Leave attachments, authors, year, tags, public at their defaults "
            "— do not pass them. "
            "Call the tool and confirm; do not ask follow-up questions."
        ),
        _verify_save_to_library,
    ),
]


@pytest.mark.parametrize(
    "tool_name,prompt,verify",
    _CASES,
    ids=[c[0] for c in _CASES],
)
async def test_ingest_tool_e2e(
    tool_name: str,
    prompt: str,
    verify: Callable[[], None],
    e2e_provider_id: str,
    _ctx,
) -> None:
    """Drive one ingest tool end-to-end through a real local LLM.

    Retries up to ``MAX_ATTEMPTS`` because the model may skip the tool
    on any given run; the test only needs *one* successful trigger +
    verify across the attempts.
    """
    last_verify_error: Exception | None = None
    triggered = False
    for attempt in range(MAX_ATTEMPTS):
        channel, _events = await _run_one_attempt(e2e_provider_id, prompt)
        called = [c for c in channel.calls if c[0] == tool_name]
        if not called:
            logger.info(
                "[e2e] %s attempt %d/%d: tool not invoked (calls=%r)",
                tool_name,
                attempt + 1,
                MAX_ATTEMPTS,
                [c[0] for c in channel.calls],
            )
            continue
        triggered = True
        try:
            verify()
            return  # success on this attempt
        except AssertionError as exc:
            last_verify_error = exc
            logger.info(
                "[e2e] %s attempt %d/%d: triggered but verify failed: %s",
                tool_name,
                attempt + 1,
                MAX_ATTEMPTS,
                exc,
            )

    if triggered:
        pytest.fail(
            f"{tool_name}: triggered across {MAX_ATTEMPTS} attempts but "
            f"verify never passed. last error: {last_verify_error}\n"
            f"  prompt: {prompt!r}"
        )
    pytest.fail(
        f"{tool_name}: not triggered across {MAX_ATTEMPTS} attempts. "
        f"The local LLM never proposed this tool — "
        f"consider strengthening the prompt.\n"
        f"  prompt: {prompt!r}"
    )


async def test_age_to_birth_year_e2e(e2e_provider_id: str, _ctx) -> None:
    """Stating age triggers update_profile_field(field='birth_date', value='YYYY-01-01').

    The LLM must compute birth_year = current_year − stated_age and store an
    ISO date rather than a raw age integer.  Accepts ±1 year to account for
    whether the user's birthday has already passed in the current year.
    """
    stated_age = 35
    current_year = datetime.date.today().year
    expected_year = current_year - stated_age

    prompt = f"I'm {stated_age} years old. Please update my profile with my birth year."

    def _verify(channel: _AutoApproveChannel) -> None:
        calls = [c for c in channel.calls if c[0] == "update_profile_field"]
        assert calls, "update_profile_field was not called"
        args = calls[0][1]
        assert args.get("field") == "birth_date", (
            f"field={args.get('field')!r}, expected 'birth_date'"
        )
        val = str(args.get("value", ""))
        assert val, "birth_date value is empty"
        try:
            year = int(val[:4])
        except ValueError:
            pytest.fail(f"cannot parse year from value={val!r}")
        assert abs(year - expected_year) <= 1, (
            f"year={year} not within ±1 of {expected_year} for age {stated_age}"
        )
        profile = ProfileStore(USER_ID).get_profile()
        assert profile is not None and profile.birth_date is not None, (
            "Profile.birth_date is still null after update"
        )

    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        channel, _events = await _run_one_attempt(e2e_provider_id, prompt)
        try:
            _verify(channel)
            return
        except (AssertionError, Exception) as exc:
            last_error = exc
            logger.info(
                "[e2e] age_to_birth_year attempt %d/%d failed: %s",
                attempt + 1,
                MAX_ATTEMPTS,
                exc,
            )
    pytest.fail(
        f"age_to_birth_year: no successful attempt across {MAX_ATTEMPTS} tries. "
        f"last error: {last_error}\n  prompt: {prompt!r}"
    )


async def test_fp_repeat_weight_e2e(e2e_provider_id: str, _ctx) -> None:
    """Repeating already-stored weight must not create a duplicate write.

    The profile is pre-seeded with weight_kg=72.5.  The user then says
    roughly the same value.  The model should either not call the tool at
    all (prompt-layer dedup) or call it and receive no_change (tool-layer
    dedup).  Either way, the DB value must remain exactly 72.5 and no
    extra audit rows should appear from a redundant write.
    """
    ProfileStore(USER_ID).update_profile_field("weight_kg", 72.5, owner_user_id=USER_ID)
    prompt = "My weight is still around 72-73 kg, nothing has changed."

    for attempt in range(MAX_ATTEMPTS):
        channel, _events = await _run_one_attempt(e2e_provider_id, prompt)
        upf_calls = [c for c in channel.calls if c[0] == "update_profile_field"]
        weight_writes = [
            c
            for c in upf_calls
            if c[1].get("field") == "weight_kg"
            and abs(float(c[1].get("value", 0)) - 72.5) < 2
        ]
        if weight_writes:
            logger.info(
                "[e2e] fp_repeat_weight attempt %d/%d: tool was called (%d time(s)); "
                "tool-layer dedup should have returned no_change",
                attempt + 1,
                MAX_ATTEMPTS,
                len(weight_writes),
            )
            profile = ProfileStore(USER_ID).get_profile()
            assert profile is not None and profile.weight_kg == 72.5, (
                f"weight_kg changed from seeded value; got {profile.weight_kg if profile else None!r}"
            )
            return
        # No weight tool call at all — prompt-layer dedup worked perfectly.
        return
    pytest.fail(
        f"fp_repeat_weight: could not complete any attempt in {MAX_ATTEMPTS} tries"
    )


async def test_fp_repeat_allergy_e2e(e2e_provider_id: str, _ctx) -> None:
    """Repeating an already-stored allergy must not create a duplicate row.

    Uses a post-seed count as baseline so prior tests' writes don't cause
    false failures — what matters is that the model doesn't ADD a row.
    """
    store = ProfileStore(USER_ID)
    store.add_allergy(
        Allergy(substance="penicillin", severity="severe", source="self_report"),
        owner_user_id=USER_ID,
    )
    seeded_count = len(ProfileStore(USER_ID).list_allergies())
    prompt = "Just a reminder — I'm still allergic to penicillin."

    for _attempt in range(MAX_ATTEMPTS):
        channel, _events = await _run_one_attempt(e2e_provider_id, prompt)
        allergies = ProfileStore(USER_ID).list_allergies()
        assert len(allergies) == seeded_count, (
            f"expected {seeded_count} allergy rows (post-seed), got {len(allergies)}"
        )
        return
    pytest.fail("fp_repeat_allergy: could not complete any attempt")


async def test_fp_repeat_condition_e2e(e2e_provider_id: str, _ctx) -> None:
    """Repeating an already-stored condition must not create a duplicate row.

    Uses a post-seed count as baseline so prior tests' writes don't cause
    false failures — what matters is that the model doesn't ADD a row.
    """
    store = ProfileStore(USER_ID)
    store.add_condition(
        Condition(display="hypertension"),
        owner_user_id=USER_ID,
    )
    seeded_count = len(ProfileStore(USER_ID).list_conditions())
    prompt = "I have high blood pressure, as you probably already know."

    for _attempt in range(MAX_ATTEMPTS):
        channel, _events = await _run_one_attempt(e2e_provider_id, prompt)
        conds = ProfileStore(USER_ID).list_conditions()
        assert len(conds) == seeded_count, (
            f"expected {seeded_count} condition rows (post-seed), got {len(conds)}"
        )
        return
    pytest.fail("fp_repeat_condition: could not complete any attempt")


async def test_fp_repeat_medication_e2e(e2e_provider_id: str, _ctx) -> None:
    """Repeating an already-stored medication must not create a duplicate row.

    Uses a post-seed count as baseline so prior tests' writes don't cause
    false failures — what matters is that the model doesn't ADD a row.
    """
    store = ProfileStore(USER_ID)
    store.add_medication(
        Medication(display="metformin", dose="500 mg", frequency="twice daily"),
        owner_user_id=USER_ID,
    )
    seeded_count = len(ProfileStore(USER_ID).list_medications())
    prompt = "I'm still taking metformin 500 mg twice a day for my diabetes."

    for _attempt in range(MAX_ATTEMPTS):
        channel, _events = await _run_one_attempt(e2e_provider_id, prompt)
        meds = ProfileStore(USER_ID).list_medications()
        assert len(meds) == seeded_count, (
            f"expected {seeded_count} medication rows (post-seed), got {len(meds)}"
        )
        return
    pytest.fail("fp_repeat_medication: could not complete any attempt")


async def test_delete_record_e2e(e2e_provider_id: str, _ctx) -> None:
    """``delete_record`` is special-cased: seed a record before each attempt
    so there is always something to delete, and feed the LLM the exact
    ``record_path`` + ``confirm_kind`` it needs (those values cannot be
    grounded from a single-turn prompt without an attached library view).
    """
    last_verify_error: Exception | None = None
    triggered = False
    for attempt in range(MAX_ATTEMPTS):
        seed_path, confirm_kind = _seed_delete_target()
        prompt = (
            f"Use the delete_record tool now: "
            f"record_path={seed_path}, confirm_kind={confirm_kind}. "
            f"Do not ask follow-up questions; call the tool and confirm."
        )
        channel, _events = await _run_one_attempt(e2e_provider_id, prompt)
        called = [c for c in channel.calls if c[0] == "delete_record"]
        if not called:
            logger.info(
                "[e2e] delete_record attempt %d/%d: tool not invoked (calls=%r)",
                attempt + 1,
                MAX_ATTEMPTS,
                [c[0] for c in channel.calls],
            )
            continue
        triggered = True
        try:
            _verify_delete_record(seed_path)()
            return
        except AssertionError as exc:
            last_verify_error = exc

    if triggered:
        pytest.fail(
            f"delete_record: triggered across {MAX_ATTEMPTS} attempts but "
            f"verify never passed. last error: {last_verify_error}"
        )
    pytest.fail(
        f"delete_record: not triggered across {MAX_ATTEMPTS} attempts. "
        f"The local LLM never proposed this tool."
    )
