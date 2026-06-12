"""Probe every provider in ``configs/models.yaml`` with a real request.

For each catalog entry, send one tiny ``"Say OK."`` prompt and report whether
the round trip works. Local providers without a running server show up as
errors; cloud providers without env keys show up as ``skip (no creds)`` and
incur no network call. Useful before kicking off a long benchmark run --
catches a missing key or a down MLX server in seconds.

Usage:
    uv run python -m tests.benchmarks.check_providers
    uv run python -m tests.benchmarks.check_providers --only omlx,ollama
    uv run python -m tests.benchmarks.check_providers --timeout 10
"""

from __future__ import annotations

import argparse
import asyncio
import time
import uuid
from dataclasses import dataclass

from pydantic_ai import Agent

from claritymed.config import CLARITYMED_HOME, load_env_file
from claritymed.context import apply_context, reset_context
from claritymed.core.llm.model import build_model
from claritymed.core.schemas import ProviderConfig
from claritymed.stores.models import is_provider_available, load_models

_PROMPT = "Reply with exactly the two letters OK and nothing else."


@dataclass
class _Result:
    provider_id: str
    kind: str
    has_creds: bool
    ok: bool
    latency_ms: float | None
    response: str | None
    error: str | None


async def _probe(provider: ProviderConfig, timeout_s: float) -> _Result:
    has_creds = is_provider_available(provider)
    if not has_creds:
        return _Result(
            provider_id=provider.id,
            kind=provider.kind,
            has_creds=False,
            ok=False,
            latency_ms=None,
            response=None,
            error="skip (no creds)",
        )

    # PHI scrub / tracing pipelines emit audit events that require
    # request_id / user_id / language to be set. The probe never runs
    # through the orchestrator that normally sets them, so install a
    # synthetic context per probe to avoid noisy "audit failed" logs.
    tokens = apply_context(
        request_id=f"probe-{uuid.uuid4().hex[:8]}",
        user_id="probe",
        language="en",
    )
    t0 = time.perf_counter()
    try:
        agent: Agent = Agent(build_model(provider), output_type=str)
        result = await asyncio.wait_for(agent.run(_PROMPT), timeout=timeout_s)
        elapsed = (time.perf_counter() - t0) * 1000
        text = (result.output or "").strip()
        return _Result(
            provider_id=provider.id,
            kind=provider.kind,
            has_creds=True,
            ok=True,
            latency_ms=elapsed,
            response=text[:60],
            error=None,
        )
    except asyncio.TimeoutError:
        elapsed = (time.perf_counter() - t0) * 1000
        return _Result(
            provider_id=provider.id,
            kind=provider.kind,
            has_creds=True,
            ok=False,
            latency_ms=elapsed,
            response=None,
            error=f"timeout after {timeout_s}s",
        )
    except Exception as exc:  # noqa: BLE001 -- probe; surface any failure
        elapsed = (time.perf_counter() - t0) * 1000
        msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
        return _Result(
            provider_id=provider.id,
            kind=provider.kind,
            has_creds=True,
            ok=False,
            latency_ms=elapsed,
            response=None,
            error=f"{type(exc).__name__}: {msg[:120]}",
        )
    finally:
        reset_context(tokens)


def _format_row(r: _Result) -> str:
    status = "ok " if r.ok else ("SKIP" if not r.has_creds else "FAIL")
    latency = f"{r.latency_ms:>6.0f}ms" if r.latency_ms is not None else "      -"
    tail = r.response if r.ok else (r.error or "")
    return f"  {status}  {r.provider_id:<22} {r.kind:<6} {latency}  {tail}"


async def _main_async(only: list[str] | None, timeout_s: float) -> int:
    providers = load_models().providers
    if only:
        providers = [p for p in providers if p.id in only]
        missing = set(only) - {p.id for p in providers}
        if missing:
            print(f"unknown provider ids: {sorted(missing)}")
            return 2

    # Local first, then cloud, alpha within each — same order as the report.
    providers = sorted(providers, key=lambda p: (p.kind != "local", p.id))
    print(
        f"probing {len(providers)} provider(s) "
        f"(timeout={timeout_s}s, prompt={_PROMPT!r})\n"
    )

    # Serial on purpose: parallel runs with 9+ concurrent HTTP calls choked
    # both local servers and cloud auth endpoints, turning every probe into
    # a timeout. This script is a one-shot diagnostic, not throughput.
    results: list[_Result] = []
    for p in providers:
        r = await _probe(p, timeout_s)
        results.append(r)
        print(_format_row(r), flush=True)

    ok_count = sum(1 for r in results if r.ok)
    skip_count = sum(1 for r in results if not r.has_creds)
    fail_count = len(results) - ok_count - skip_count
    print(
        f"\n  {ok_count} ok · {fail_count} failed · {skip_count} skipped "
        f"(of {len(results)})"
    )
    return 0 if fail_count == 0 else 1


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--only",
        default="",
        help="comma-separated provider ids to probe (default: all)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="per-provider request timeout in seconds (default: 20)",
    )
    return p.parse_args()


def main() -> int:
    # Reuse the same loader the Typer root callback runs at CLI startup so the
    # probe sees identical credentials to a real ``claritymed`` invocation.
    # Reads ``CLARITYMED_HOME/.env`` (default ``~/.claritymed/.env``); shell
    # vars win over file values.
    applied = load_env_file()
    if applied:
        print(f"loaded {len(applied)} key(s) from {CLARITYMED_HOME / '.env'}")

    args = _parse_args()
    only = [s.strip() for s in args.only.split(",") if s.strip()] or None
    return asyncio.run(_main_async(only, args.timeout))


if __name__ == "__main__":
    raise SystemExit(main())
