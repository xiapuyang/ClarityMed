"""Dump the exact LLM prompt for one symptom bench trial.

Reproduces the tool-invoke symptoms bench setup for one case + provider
combination, wraps the model with a capturing shim, and writes the raw
messages + tool schemas sent to the LLM to disk. Also reports whether
the symptoms tool was invoked and prints a response preview.

The point of this script is prompt archaeology: once you can see exactly
what the model sees on a failing turn, you can A/B minimal edits to the
``ask.yaml`` / ``predict_disease_from_symptoms_tool.yaml`` prompts and
re-run this script to check whether a tool call now fires — without
paying the full bench cost of N cases x M trials.

Pre-flight (same as the bench):
* ``scripts/run.sh symptoms`` (the symptoms server on :8084)
* ``configs/symptoms.yaml`` datasets[0].enabled = true

Example::

    uv run python scripts/dump_symptom_prompt.py \\
        --case flu_triad_3d_bare_report \\
        --provider omlx

Outputs (under ``data/prompt_lab/<timestamp>/``):
    messages.txt         human-readable dump (kind + text per part)
    messages.json        raw ModelMessage list (model_dump)
    tool_schemas.json    tool defs sent to the LLM this turn
    response.txt         final response text + tool_invoked flag
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator

import dataclasses

import httpx
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelResponse,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings

from claritymed.config import load_env_file
from claritymed.context import apply_context, new_request_id, reset_context
from claritymed.core.llm.model import build_model
from claritymed.core.rag import load_retrieval_config
from claritymed.core.symptoms.client import SymptomsServerClient
from claritymed.orchestrator.services import AskService, build_rag_strategy
from claritymed.orchestrator.services.chat_session import ChatSession
from claritymed.stores.models import resolve_provider

from tests.benchmarks.tool_invoke import base
from tests.benchmarks.tool_invoke.symptoms.cases import CASES, MODAL_THRESHOLD
from tests.benchmarks.tool_invoke.symptoms.run import (
    _AutoAnswerChannel,
    _NullApprovalChannel,
    _make_symptoms_factory,
)

logger = logging.getLogger(__name__)

DEFAULT_SYMPTOMS_URL = "http://127.0.0.1:8084"
DEFAULT_CASE = "flu_triad_3d_bare_report"
DEFAULT_PROVIDER = "omlx"
DEFAULT_LANG = "en"
PER_TURN_TIMEOUT_S = 240.0
CASE_MAP = {c.name: c for c in CASES}


class CapturingModel(WrapperModel):
    """Wrapper that records every ``request`` / ``request_stream`` call.

    Captured fields per call: messages, ModelSettings, ModelRequestParameters
    (contains ``function_tools`` — the tool schemas exposed to the LLM this
    turn). Forwards to the wrapped model unchanged so the real LLM
    response still drives the agent loop.
    """

    def __init__(self, wrapped: Any) -> None:
        super().__init__(wrapped)
        self.captured: list[dict[str, Any]] = []

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self._record(messages, model_settings, model_request_parameters)
        return await self.wrapped.request(
            messages, model_settings, model_request_parameters
        )

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: Any | None = None,
    ) -> AsyncIterator[Any]:
        self._record(messages, model_settings, model_request_parameters)
        async with self.wrapped.request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as stream:
            yield stream

    def _record(
        self,
        messages: list[ModelMessage],
        settings: ModelSettings | None,
        params: ModelRequestParameters,
    ) -> None:
        self.captured.append(
            {
                "messages": ModelMessagesTypeAdapter.dump_python(
                    list(messages), mode="json"
                ),
                "settings": dict(settings) if settings else None,
                "tool_defs": [
                    dataclasses.asdict(t) for t in (params.function_tools or [])
                ],
                "allow_text_output": params.allow_text_output,
                "output_tools": [
                    dataclasses.asdict(t) for t in (params.output_tools or [])
                ],
            }
        )


def _symptoms_server_ready(base_url: str) -> bool:
    """Return True when the symptoms server has at least one dataset loaded."""
    try:
        resp = httpx.get(f"{base_url}/health", timeout=3.0)
    except httpx.HTTPError:
        return False
    if resp.status_code != 200:
        return False
    return bool(resp.json().get("datasets_loaded"))


def _install_ask_override(extra_text: str) -> Any:
    """Patch ``PromptRegistry.get`` to append ``extra_text`` to the ``ask`` prompt.

    Returns a restore callable — call it in ``finally`` to undo the patch.
    Surgical: only ``ask`` is modified; other prompts pass through.
    """
    from claritymed.core.prompts.registry import PromptRegistry

    orig = PromptRegistry.get

    def patched(self, name, *args, **kwargs):
        value = orig(self, name, *args, **kwargs)
        if name == "ask" and isinstance(value, str):
            return value + "\n\n" + extra_text
        return value

    PromptRegistry.get = patched  # type: ignore[method-assign]

    def restore() -> None:
        PromptRegistry.get = orig  # type: ignore[method-assign]

    return restore


async def run_one(
    case_name: str,
    provider_id: str,
    lang: str,
    symptoms_url: str,
    extra_system: str | None = None,
) -> tuple[CapturingModel, str, int, float, str | None]:
    """Run a single bench-shaped trial with the model wrapped for capture.

    Returns: (capturing_model, final_response_text, modal_call_count,
    latency_ms, error_msg).
    """
    base.wipe_bench_user_dir()
    base.reload_runtime()

    case = CASE_MAP[case_name]
    prompt = case.prompts[lang]
    request_id = new_request_id()
    tokens = apply_context(request_id, base.USER_ID, lang)

    channel = _AutoAnswerChannel()
    approval = _NullApprovalChannel()
    chunks: list[str] = []
    error_msg: str | None = None
    t0 = time.perf_counter()

    provider = resolve_provider(override=provider_id)
    inner = build_model(provider)
    capturing = CapturingModel(inner)

    restore_override = _install_ask_override(extra_system) if extra_system else None

    try:
        chat = ChatSession.new(base.USER_ID)
        rag_mode = load_retrieval_config().rag.mode
        strategy = build_rag_strategy(model=capturing)

        async with SymptomsServerClient(symptoms_url) as client:
            service = AskService(
                model=capturing,
                chat_session=chat,
                provider_id=provider.id,
                model_name=provider.model,
                provider_config=provider,
                strategy=strategy,
                rag_mode=rag_mode,
                prompt_channel=channel,
                tool_approval_channel=approval,
                symptoms_factory=_make_symptoms_factory(client),
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
                            chunks.append(text)
            except asyncio.TimeoutError:
                error_msg = f"timeout after {PER_TURN_TIMEOUT_S:.0f}s"
    except Exception as exc:  # noqa: BLE001
        error_msg = f"{type(exc).__name__}: {exc}"
    finally:
        reset_context(tokens)
        if restore_override is not None:
            restore_override()

    latency_ms = (time.perf_counter() - t0) * 1000
    return capturing, "".join(chunks).strip(), len(channel.calls), latency_ms, error_msg


def _render_messages_txt(captured: list[dict[str, Any]]) -> str:
    """Render captured requests as a human-readable multi-section blob."""
    lines: list[str] = []
    for i, call in enumerate(captured):
        lines.append(f"{'=' * 80}\nREQUEST #{i + 1}\n{'=' * 80}")
        lines.append(f"tool_defs: {len(call['tool_defs'])} tools exposed")
        for t in call["tool_defs"]:
            desc = (t.get("description") or "")[:120].replace("\n", " ")
            lines.append(f"  • {t.get('name')}: {desc}...")
        lines.append(f"allow_text_output: {call['allow_text_output']}")
        lines.append("")
        for j, msg in enumerate(call["messages"]):
            kind = msg.get("kind", "?")
            lines.append(f"--- msg[{j}] kind={kind} ---")
            for part in msg.get("parts") or []:
                pkind = part.get("part_kind") or part.get("kind") or "?"
                if pkind == "text" or "content" in part:
                    content = part.get("content") or part.get("text") or ""
                    lines.append(f"  [{pkind}] {content}")
                elif pkind in ("tool-call", "tool_call"):
                    lines.append(
                        f"  [{pkind}] name={part.get('tool_name')} "
                        f"args={json.dumps(part.get('args'), ensure_ascii=False)[:200]}"
                    )
                elif pkind in ("tool-return", "tool_return"):
                    ret = part.get("content") or ""
                    lines.append(
                        f"  [{pkind}] name={part.get('tool_name')} "
                        f"content={str(ret)[:300]}"
                    )
                else:
                    lines.append(
                        f"  [{pkind}] {json.dumps(part, ensure_ascii=False)[:400]}"
                    )
            lines.append("")
    return "\n".join(lines)


def _tool_invoked(modal_count: int) -> bool:
    return modal_count >= MODAL_THRESHOLD


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default=DEFAULT_CASE, choices=sorted(CASE_MAP))
    p.add_argument("--provider", default=DEFAULT_PROVIDER)
    p.add_argument("--lang", default=DEFAULT_LANG, choices=["en", "zh"])
    p.add_argument("--symptoms-url", default=DEFAULT_SYMPTOMS_URL)
    p.add_argument(
        "--out",
        default=None,
        help="output dir (default: data/prompt_lab/<timestamp>/)",
    )
    p.add_argument(
        "--extra-system-file",
        default=None,
        help=(
            "optional path to a text file whose contents are APPENDED to "
            "the ``ask`` system prompt (surgical monkey-patch of the "
            "PromptRegistry). Use to A/B test additive rules without "
            "editing store/ask.yaml."
        ),
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="repeat the trial N times (default 1) to smooth stochastic outputs",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


async def _main_async(args: argparse.Namespace) -> int:
    load_env_file()

    if not _symptoms_server_ready(args.symptoms_url):
        print(
            f"symptoms server not ready at {args.symptoms_url}\n"
            "  Start it with: scripts/run.sh symptoms",
            file=sys.stderr,
        )
        return 1

    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S_") + f"{now.microsecond // 1000:03d}"
    out_dir = Path(args.out) if args.out else Path("data/prompt_lab") / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    extra_system: str | None = None
    if args.extra_system_file:
        extra_system = Path(args.extra_system_file).read_text()
        (out_dir / "extra_system.txt").write_text(extra_system)

    print(
        f"case={args.case}  provider={args.provider}  lang={args.lang}  "
        f"repeat={args.repeat}  out={out_dir}"
    )
    print(f"user prompt: {CASE_MAP[args.case].prompts[args.lang]!r}")
    if extra_system:
        print(f"extra_system: {len(extra_system)} chars from {args.extra_system_file}")
    print()

    results: list[dict[str, Any]] = []
    last_capturing: CapturingModel | None = None
    for i in range(args.repeat):
        capturing, response_text, modal_count, latency_ms, err = await run_one(
            args.case, args.provider, args.lang, args.symptoms_url, extra_system
        )
        last_capturing = capturing
        invoked = _tool_invoked(modal_count)
        results.append(
            {
                "trial": i + 1,
                "tool_invoked": invoked,
                "modal_calls": modal_count,
                "latency_ms": round(latency_ms, 0),
                "error": err,
                "response_preview": response_text[:300],
                "response_full": response_text,
            }
        )
        print(
            f"[trial {i + 1}/{args.repeat}] tool_invoked={invoked}  "
            f"modals={modal_count}  {latency_ms:.0f}ms  "
            f"err={err or 'none'}"
        )

    assert last_capturing is not None
    (out_dir / "messages.json").write_text(
        json.dumps(last_capturing.captured, ensure_ascii=False, indent=2)
    )
    (out_dir / "messages.txt").write_text(_render_messages_txt(last_capturing.captured))
    tool_defs = (
        last_capturing.captured[0]["tool_defs"] if last_capturing.captured else []
    )
    (out_dir / "tool_schemas.json").write_text(
        json.dumps(tool_defs, ensure_ascii=False, indent=2)
    )
    (out_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2)
    )

    n_ok = sum(1 for r in results if r["tool_invoked"])
    print()
    print(f"tool_invoked_rate: {n_ok}/{len(results)} = {n_ok / len(results):.0%}")
    print(f"tool_defs_exposed: {len(tool_defs)}")
    print(f"dumped → {out_dir}")
    return 0


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    sys.exit(asyncio.run(_main_async(args)))


if __name__ == "__main__":
    main()
