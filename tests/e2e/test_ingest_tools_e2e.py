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
import logging
from typing import Any, Callable

import pytest

from claritymed.context import apply_context, reset_context
from claritymed.core.interaction import ApprovalDecision
from claritymed.core.rag import load_retrieval_config
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


def _verify_update_profile_weight() -> None:
    profile = ProfileStore(USER_ID).get_profile()
    assert profile is not None and profile.weight_kg is not None, (
        "Profile.weight_kg is still null after update_profile_field run"
    )
    # The exact value depends on what the LLM emitted; we just check that
    # *something* in the right range got written.
    assert 30 <= profile.weight_kg <= 250, (
        f"weight_kg out of plausible range: {profile.weight_kg!r}"
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
) -> tuple[_AutoApproveChannel, list[Any]]:
    """Run one full AskService turn end-to-end. Returns (channel, events).

    Constructs a fresh ChatSession + AskService so prior history can't
    bias the next attempt. Resolves the live local provider via
    production code (``resolve_provider``) so this exercises the same
    wiring the TUI uses.
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
        "update_profile_field",
        (
            "My weight is now 72.5 kg. Please record that update. "
            "Use the update_profile_field tool — do not ask follow-up "
            "questions, call it and confirm."
        ),
        _verify_update_profile_weight,
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
