"""Vision tool-trigger benchmark runner.

Measures whether the LLM invokes ``detect_disease_from_image`` for
image+text turns that warrant it and correctly refrains on FP-tier
inputs. Mirrors ``tests/benchmarks/tool_invoke/symptoms/run.py``
line-for-line; differences are bounded:

* Detection signal is the vision plugin's confirm-modal count rather
  than the symptoms multi-modal Q&A. ``MODAL_THRESHOLD = 1`` (vision
  fires one confirm per tool call vs symptoms' ≥3).
* Pre-flight checks both vision-server (:8085) and
  medical-clip-server (:8086). Promoted BUSI artifact required.
* Each trial seeds an image attachment into the bench user blob
  store with synthetic ``ocr.json`` so the LLM-decision axis is
  isolated from medical-clip classifier accuracy (which has its own
  Unit 2 metric).

Pre-flight:
* ``uv run claritymed-vision-server`` (port 8085)
* ``uv run claritymed-medical-clip-server`` (port 8086)
* ``configs/vision.yaml`` diseases[0].enabled = true + manifest sha set
* Promoted BUSI artifact at
  ``~/.claritymed/models/vision/breast_cancer_ultrasound/breast_busi_unet_v1/``

Example::

    uv run python -m tests.benchmarks.tool_invoke.vision.run \\
        --models omlx,deepseek-v4-pro \\
        --user-langs en,zh --tiers base,hard,fp --trials 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from claritymed.config import load_env_file, load_vision_config
from claritymed.core.rag.schemas import load_retrieval_config
from claritymed.context import apply_context, new_request_id, reset_context
from claritymed.core.interaction.schemas import (
    AskUserQuestionInput,
    AskUserQuestionResult,
)
from claritymed.core.vision.registry import VisionRegistry
from claritymed.orchestrator.features.vision_plugin import VisionFeature
from claritymed.orchestrator.services import AskService
from claritymed.orchestrator.services.chat_session import ChatSession
from claritymed.stores.blob_store import BlobStore
from claritymed.stores.session_attachments import SessionAttachments

from tests.benchmarks.tool_invoke import base
from tests.benchmarks.tool_invoke.vision.cases import (
    CASES,
    DISAMBIG_MIN_OPTIONS,
    MODAL_THRESHOLD,
    Case,
)

logger = logging.getLogger(__name__)

PER_TURN_TIMEOUT_S = 240.0
_DEFAULT_VISION_URL = "http://127.0.0.1:8085"
_DEFAULT_MEDICAL_CLIP_URL = "http://127.0.0.1:8086"

CORRECT_OUTCOMES: frozenset[str] = frozenset({"correct"})

_FIXTURES_ROOT = Path(__file__).resolve().parents[3] / "fixtures" / "vision"


# --- channels ---------------------------------------------------------------


class _AutoAnswerChannel:
    """Auto-answer the confirm modal with the first label (always 'yes').

    Vision's disambig askuserquestion modals also flow through here when
    the LLM picks them (rule 4/5 in the tool description). Picking the
    first option is deterministic and keeps the bench focused on the
    decision boundary, not on the modal answer.
    """

    def __init__(self) -> None:
        self.calls: list[AskUserQuestionInput] = []

    async def ask(self, payload: AskUserQuestionInput) -> AskUserQuestionResult:
        self.calls.append(payload)
        answers: dict[str, str] = {}
        numeric_values: dict[str, float] = {}
        for q in payload.questions:
            if q.numeric is not None:
                numeric_values[q.question] = float(q.numeric.min)
                continue
            if not q.options:
                answers[q.question] = "Yes"
                continue
            answers[q.question] = q.options[0].label
        return AskUserQuestionResult(answers=answers, numeric_values=numeric_values)


class _NullApprovalChannel:
    """Auto-approve any ingest tool that legitimately fires on an FP case."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def request(
        self,
        tool_name: str,
        args: dict,
        *,
        breadcrumb: str | None = None,
    ) -> Any:
        from claritymed.core.interaction import ApprovalDecision

        self.calls.append({"tool_name": tool_name, "args": dict(args)})
        return ApprovalDecision(decision="once")


# --- factory ----------------------------------------------------------------


def _make_vision_factory(registry: VisionRegistry, get_session_id):
    config = registry._config  # type: ignore[attr-defined]

    def _factory() -> VisionFeature:
        return VisionFeature(
            config=config,
            registry=registry,
            get_session_id=get_session_id,
        )

    return _factory


# --- server pre-flight ------------------------------------------------------


def _service_ready(url: str, *, require_models: bool = False) -> bool:
    try:
        resp = httpx.get(f"{url}/health", timeout=3.0)
    except httpx.HTTPError:
        return False
    if resp.status_code != 200:
        return False
    if not require_models:
        return True
    body = resp.json()
    return bool(body.get("models_loaded"))


# --- fixture seeding --------------------------------------------------------


def _pick_fixture_bytes(subdir: str) -> bytes | None:
    """Return the first non-placeholder file in ``tests/fixtures/vision/<subdir>``."""
    fixture_dir = _FIXTURES_ROOT / subdir
    for path in sorted(fixture_dir.glob("*")):
        if (
            path.is_file()
            and not path.name.startswith("_")
            and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ):
            return path.read_bytes()
    return None


def _seed_attachment(*, user_id: str, session_id: str, seed_dict: dict) -> str | None:
    """Drop fixture bytes into the blob store + write sentinel; return sha.

    Seeds support two shapes:

    * Tagged (the common case) — ``modality`` / ``is_medical`` /
      ``ocr_has_report`` are set, status defaults to ``"done"`` with
      synthetic text whose presence is gated on ``ocr_has_report``.
    * Bare (Rule 6 path) — ``status="empty"`` and the vision-tag fields
      are ``None``. ``BlobStore.write_ocr_result`` drops ``None`` kwargs
      from the payload, so the rendered ``<image>`` tag carries no
      modality / is_medical / ocr_has_report attribute and the LLM sees
      the genuinely-bare tag the Rule 6 prompt covers.
    """
    image_bytes = _pick_fixture_bytes(seed_dict["fixture_subdir"])
    if image_bytes is None:
        return None
    blob_store = BlobStore(user_id)
    sha = blob_store.store(image_bytes, "png")
    SessionAttachments(user_id, session_id).add(
        sha256=sha,
        filename=f"{seed_dict['fixture_subdir']}.png",
        mime="image/png",
        size=len(image_bytes),
    )
    status = seed_dict.get("status", "done")
    # ``status="empty"`` cases have no OCR text by definition. For the
    # ``done`` path we still gate the synthetic text on
    # ``ocr_has_report`` so Rule 2 cases see a clinician-report-shaped
    # body and Rule 3 cases see an empty extraction (the modality tag
    # alone is enough to fire Rule 3).
    if status == "empty":
        text = ""
    else:
        text = (
            "FINDINGS: synthetic report text. IMPRESSION: bench seed."
            if seed_dict.get("ocr_has_report")
            else ""
        )
    # When modality / is_medical / ocr_has_report are present in the
    # seed we pass them through; absence (.get returns None) is handled
    # by write_ocr_result's None-drops-kwarg semantics. The bench-seed
    # confidence is a stand-in for the live classifier and is meaningless
    # for the bare-tag path (omitted alongside modality).
    modality = seed_dict.get("modality")
    blob_store.write_ocr_result(
        sha,
        status=status,
        kind="ocr",
        ext="png",
        provider="bench-seed",
        chain_tried=["bench-seed"],
        reason=None,
        text=text,
        original_filename=f"{seed_dict['fixture_subdir']}.png",
        modality=modality,
        modality_confidence=0.9 if modality is not None else None,
        is_medical=seed_dict.get("is_medical"),
        ocr_has_report=seed_dict.get("ocr_has_report"),
    )
    return sha


# --- trial dataclass --------------------------------------------------------


@dataclass
class TrialRecord:
    request_id: str
    timestamp_utc: str
    model: str
    lang: str
    case_name: str
    tier: str
    expected_behavior: str
    expected_tool: str | None
    user_prompt: str
    # Detection: vision confirm-modal calls (≥1 = tool invoked).
    modal_call_count: int
    # Option count on the first modal's first question. Used to
    # distinguish the plugin's confirm modal (always exactly 2 options:
    # yes / no) from an LLM-issued disambig modal (one option per enabled
    # disease — ≥``DISAMBIG_MIN_OPTIONS`` in practice). 0 when no modal
    # fired or when the first question was numeric / unoptioned.
    first_modal_options: int
    # Ingest-tool approval calls — should be empty unless an FP case
    # legitimately reroutes to ingest. Captured for inspection.
    ingest_calls: list[dict]
    final_response_text: str
    outcome: str
    predicate_pass: bool
    predicate_reason: str
    tool_invoked: bool
    latency_ms: float
    had_error: bool
    error_msg: str | None


# --- single trial -----------------------------------------------------------


async def _run_one_trial(
    provider_id: str,
    user_lang: str,
    case: Case,
    vision_url: str,
) -> TrialRecord:
    from claritymed.core.llm.model import build_model
    from claritymed.stores.models import resolve_provider

    base.wipe_bench_user_dir()
    base.reload_runtime()

    request_id = new_request_id()
    tokens = apply_context(request_id, base.USER_ID, user_lang)

    channel = _AutoAnswerChannel()
    approval = _NullApprovalChannel()
    final_chunks: list[str] = []
    error_msg: str | None = None
    t0 = time.perf_counter()
    sha: str | None = None
    prompt = ""

    try:
        provider = resolve_provider(override=provider_id)
        model = build_model(provider)
        chat = ChatSession.new(base.USER_ID)
        rag_mode = load_retrieval_config().rag.mode

        if case.seed is not None:
            sha = _seed_attachment(
                user_id=base.USER_ID,
                session_id=chat.session_id,
                seed_dict=case.seed(),
            )
            if sha is None:
                error_msg = (
                    f"no fixture in tests/fixtures/vision/"
                    f"{case.seed()['fixture_subdir']}; populate or skip"
                )

        # Substitute {sha8} into the prompt; cases without a seed leave
        # the placeholder untouched.
        prompt = case.prompts[user_lang]
        if sha is not None:
            prompt = prompt.replace("{sha8}", sha[:8])

        if error_msg is None:
            vision_config = load_vision_config()
            registry = VisionRegistry(vision_config)
            try:
                await registry.bootstrap()
                service = AskService(
                    model=model,
                    chat_session=chat,
                    provider_id=provider.id,
                    model_name=provider.model,
                    provider_config=provider,
                    rag_mode=rag_mode,
                    prompt_channel=channel,
                    tool_approval_channel=approval,
                    vision_factory=_make_vision_factory(
                        registry, lambda: chat.session_id
                    ),
                )
                try:
                    async with asyncio.timeout(PER_TURN_TIMEOUT_S):
                        async for ev in service.run(prompt, user_id=base.USER_ID):
                            if getattr(ev, "type", None) == "error":
                                error_msg = (
                                    f"{getattr(ev, 'error_type', 'unknown')}: "
                                    f"{getattr(ev, 'message', '')}"
                                )
                                continue
                            text = getattr(ev, "text", None)
                            if isinstance(text, str):
                                final_chunks.append(text)
                except asyncio.TimeoutError:
                    error_msg = f"timeout after {PER_TURN_TIMEOUT_S:.0f}s"
            finally:
                await registry.aclose()
    except Exception as exc:  # noqa: BLE001
        error_msg = f"{type(exc).__name__}: {exc}"
    finally:
        reset_context(tokens)

    latency_ms = (time.perf_counter() - t0) * 1000
    modal_count = len(channel.calls)
    # Inspect the FIRST modal call: confirm has exactly 2 yes/no options,
    # disambig has one option per disease. Subsequent modals (if any) are
    # follow-ups and don't change the decision boundary the bench measures.
    first_modal_options = 0
    if channel.calls:
        first = channel.calls[0]
        if first.questions:
            first_modal_options = len(first.questions[0].options)
    final_text = "".join(final_chunks).strip()

    outcome_info = _classify(case, modal_count, first_modal_options, error_msg)

    return TrialRecord(
        request_id=request_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=provider_id,
        lang=user_lang,
        case_name=case.name,
        tier=case.tier,
        expected_behavior=case.expected_behavior,
        expected_tool=case.expected_tool,
        user_prompt=prompt,
        modal_call_count=modal_count,
        first_modal_options=first_modal_options,
        ingest_calls=list(approval.calls),
        final_response_text=final_text,
        outcome=outcome_info["outcome"],
        predicate_pass=outcome_info["predicate_pass"],
        predicate_reason=outcome_info["predicate_reason"],
        tool_invoked=outcome_info["tool_invoked"],
        latency_ms=round(latency_ms, 1),
        had_error=error_msg is not None,
        error_msg=error_msg,
    )


# --- classify ---------------------------------------------------------------


def _classify(
    case: Case,
    modal_count: int,
    first_modal_options: int,
    error_msg: str | None,
) -> dict[str, Any]:
    # "tool_invoked" reads the confirm-modal signal: the plugin builds its
    # confirm modal in vision_plugin._confirm_question with exactly 2
    # options (the localized yes / no labels). An LLM-issued disambig
    # modal has one option per disease (≥ DISAMBIG_MIN_OPTIONS), so the
    # exact-2 check cleanly separates "the tool was called" from "the LLM
    # asked the user to disambig".
    confirm_fired = modal_count >= MODAL_THRESHOLD and first_modal_options == 2
    tool_invoked = confirm_fired

    if error_msg is not None:
        return dict(
            outcome="errored",
            predicate_pass=False,
            predicate_reason=f"trial raised: {error_msg}",
            tool_invoked=tool_invoked,
        )

    if case.expected_behavior == "call_tool":
        pred_ok, pred_reason = case.args_predicate(modal_count)
        outcome = "correct" if pred_ok else "no_tool"
        return dict(
            outcome=outcome,
            predicate_pass=pred_ok,
            predicate_reason=pred_reason,
            tool_invoked=tool_invoked,
        )

    if case.expected_behavior == "ask_clarification":
        pred_ok, pred_reason = case.args_predicate(modal_count, first_modal_options)
        if pred_ok:
            outcome = "correct"
        elif modal_count == 0:
            outcome = "no_clarification"
        else:
            # A modal fired but it looks like the confirm modal — the LLM
            # jumped to the tool instead of asking. Distinct outcome so
            # the bench report can separate "didn't ask" from "asked but
            # via the wrong modal shape".
            outcome = "called_tool_instead"
        return dict(
            outcome=outcome,
            predicate_pass=pred_ok,
            predicate_reason=pred_reason,
            tool_invoked=tool_invoked,
        )

    if case.expected_behavior == "decline":
        # ANY modal — confirm or disambig — is a false positive for a
        # decline case. The user-visible UX of decline is "answer in plain
        # text without prompting the user", so even an over-eager disambig
        # counts against it.
        if modal_count >= MODAL_THRESHOLD:
            return dict(
                outcome="false_positive",
                predicate_pass=False,
                predicate_reason=(
                    f"modal fired (modal_calls={modal_count}, "
                    f"first_options={first_modal_options}); "
                    f"{'tool' if confirm_fired else 'disambig'} path"
                ),
                tool_invoked=tool_invoked,
            )
        return dict(
            outcome="correct",
            predicate_pass=True,
            predicate_reason=f"no modal fired (modal_calls={modal_count})",
            tool_invoked=False,
        )

    raise ValueError(f"unknown expected_behavior: {case.expected_behavior!r}")


# --- aggregate --------------------------------------------------------------


def _summary_rows(trials: list[TrialRecord]) -> list[dict]:
    by_cell: dict[tuple[str, str, str], list[TrialRecord]] = {}
    for t in trials:
        key = (t.model, t.lang, t.case_name)
        by_cell.setdefault(key, []).append(t)

    rows: list[dict] = []
    for (model, lang, case_name), cell in sorted(by_cell.items()):
        latencies = [t.latency_ms for t in cell if not t.had_error]
        n = len(cell)
        n_correct = sum(1 for t in cell if t.outcome in CORRECT_OUTCOMES)
        n_err = sum(1 for t in cell if t.had_error)
        n_invoked = sum(1 for t in cell if t.tool_invoked)
        # An LLM-issued disambig fires a modal but does NOT mark
        # tool_invoked (option count > 2). Reported as a separate rate
        # so the bench can show ``ask_clarification`` cells passing via
        # the disambig path vs ``call_tool`` cells regressing into one.
        n_disambig = sum(
            1
            for t in cell
            if t.modal_call_count >= MODAL_THRESHOLD
            and t.first_modal_options >= DISAMBIG_MIN_OPTIONS
        )
        rows.append(
            {
                "model": model,
                "user_lang": lang,
                "case": case_name,
                "tier": cell[0].tier,
                "expected_behavior": cell[0].expected_behavior,
                "n_trials": n,
                "correct_rate": round(n_correct / n, 3) if n else 0.0,
                "disambig_rate": round(n_disambig / n, 3) if n else 0.0,
                "tool_invoked_rate": round(n_invoked / n, 3) if n else 0.0,
                "error_count": n_err,
                "mean_latency_ms": round(statistics.mean(latencies), 1)
                if latencies
                else "",
                "p50_latency_ms": round(statistics.median(latencies), 1)
                if latencies
                else "",
                "p95_latency_ms": base.p95(latencies),
            }
        )
    return rows


def _outcomes_rows(trials: list[TrialRecord]) -> list[dict]:
    by_cell: dict[tuple[str, str, str, str, str, str], int] = {}
    for t in trials:
        key = (t.model, t.lang, t.case_name, t.tier, t.expected_behavior, t.outcome)
        by_cell[key] = by_cell.get(key, 0) + 1

    rows: list[dict] = []
    for key, count in sorted(by_cell.items()):
        model, lang, case_name, tier, eb, outcome = key
        rows.append(
            {
                "model": model,
                "user_lang": lang,
                "case": case_name,
                "tier": tier,
                "expected_behavior": eb,
                "outcome": outcome,
                "count": count,
            }
        )
    return rows


# --- CLI --------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    base.add_common_args(p)
    p.add_argument(
        "--vision-url",
        default=_DEFAULT_VISION_URL,
        help=f"vision server base URL (default: {_DEFAULT_VISION_URL})",
    )
    p.add_argument(
        "--medical-clip-url",
        default=_DEFAULT_MEDICAL_CLIP_URL,
        help=f"medical-clip server base URL (default: {_DEFAULT_MEDICAL_CLIP_URL})",
    )
    return p.parse_args()


async def _main_async(args: argparse.Namespace) -> int:
    load_env_file()

    if not _service_ready(args.vision_url, require_models=True):
        print(
            f"vision-server not ready (no models loaded) at {args.vision_url}\n"
            "  Start it with: uv run claritymed-vision-server\n"
            "  Verify configs/vision.yaml diseases[0].enabled = true",
            file=sys.stderr,
        )
        return 1
    if not _service_ready(args.medical_clip_url):
        print(
            f"medical-clip-server not ready at {args.medical_clip_url}\n"
            "  Start it with: uv run claritymed-medical-clip-server",
            file=sys.stderr,
        )
        return 1

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    user_langs = [s.strip() for s in args.user_langs.split(",") if s.strip()]
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    names = (
        [n.strip() for n in args.cases.split(",") if n.strip()] if args.cases else None
    )
    selected = base.select_cases(CASES, tiers, names)
    if not selected:
        print("no cases selected", file=sys.stderr)
        return 2

    _now = datetime.now()
    ts = _now.strftime("%Y%m%d_%H%M%S_") + f"{_now.microsecond // 1000:03d}"
    out_dir = Path(args.out) if args.out else Path("data/bench/vision") / ts
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "trials.jsonl"
    csv_path = out_dir / "summary.csv"
    outcomes_csv_path = out_dir / "outcomes.csv"

    total = len(models) * len(user_langs) * len(selected) * args.trials
    print(
        f"models={models} user_langs={user_langs} cases={len(selected)} "
        f"trials={args.trials} → total={total} trials; out={out_dir}"
    )

    trials: list[TrialRecord] = []
    counter = 0
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for model in models:
            for ulang in user_langs:
                for case in selected:
                    for k in range(args.trials):
                        counter += 1
                        t0 = time.perf_counter()
                        rec = await _run_one_trial(model, ulang, case, args.vision_url)
                        trials.append(rec)
                        fh.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
                        fh.flush()
                        elapsed_ms = (time.perf_counter() - t0) * 1000
                        print(
                            f"[{counter}/{total}] {model:<22} u={ulang} "
                            f"{case.name:<40} {k + 1}/{args.trials} "
                            f"{case.expected_behavior:<12} -> "
                            f"{rec.outcome:<16} modal={rec.modal_call_count} "
                            f"{elapsed_ms:>6.0f}ms"
                        )

    base.write_csv(csv_path, _summary_rows(trials))
    base.write_csv(outcomes_csv_path, _outcomes_rows(trials))

    print(f"\nwrote {len(trials)} trials → {jsonl_path}")
    print(
        f"render HTML: uv run python -m tests.benchmarks.tool_invoke.report "
        f"--run {out_dir}"
    )
    return 0


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    try:
        sys.exit(asyncio.run(_main_async(args)))
    except KeyboardInterrupt:
        os._exit(130)


if __name__ == "__main__":
    main()
