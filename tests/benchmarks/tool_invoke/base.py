"""Shared benchmark infrastructure for tool_invoke runners.

Each tool submodule (ingest/, symptoms/, …) imports from here to avoid
duplicating env management, arg parsing, and output helpers.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import shutil
from pathlib import Path
from typing import Any, TypeVar

from claritymed import config as _cfg

USER_ID = "bench"
BENCH_HOME = Path.home() / ".claritymed"
BENCH_USER_DIR = BENCH_HOME / "data" / "users" / USER_ID


def wipe_bench_user_dir() -> None:
    if BENCH_USER_DIR.exists():
        shutil.rmtree(BENCH_USER_DIR)


def reload_runtime() -> None:
    """Re-resolve config and clear per-user caches for a clean trial slate."""
    importlib.reload(_cfg)
    _cfg.reload_configs()
    from claritymed.stores import profile as _profile

    _profile._ENGINES.clear()
    from claritymed.stores.account import reset_account_cache

    reset_account_cache()


def p95(xs: list[float]) -> float | str:
    if not xs:
        return ""
    s = sorted(xs)
    idx = max(0, int(round(0.95 * (len(s) - 1))))
    return round(s[idx], 1)


_C = TypeVar("_C")


def select_cases(
    cases: list[_C], tiers: list[str], names: list[str] | None
) -> list[_C]:
    out = [c for c in cases if c.tier in tiers]  # type: ignore[attr-defined]
    if names:
        wanted = set(names)
        out = [c for c in out if c.name in wanted]  # type: ignore[attr-defined]
    return out


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Add the standard benchmark CLI arguments shared across all tool runners."""
    p.add_argument(
        "--models",
        required=True,
        help="comma-separated provider ids (e.g. omlx,deepseek-v4-pro)",
    )
    p.add_argument(
        "--user-langs",
        "--langs",
        dest="user_langs",
        default="en,zh",
        help="comma-separated user-input languages (default: en,zh)",
    )
    p.add_argument(
        "--trials",
        type=int,
        default=3,
        help="trials per (model, user-lang, case) cell (default: 3)",
    )
    p.add_argument(
        "--tiers",
        default="base,hard,fp",
        help="case tiers to include (default: base,hard,fp)",
    )
    p.add_argument(
        "--cases",
        default=None,
        help="optional comma-separated case names to filter within tiers",
    )
    p.add_argument(
        "--out",
        default=None,
        help="output directory (tool runners supply a sensible default)",
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--no-phoenix-upload",
        action="store_true",
        help=(
            "Skip the post-run Phoenix Experiments upload even when a "
            "tracing endpoint is configured. Local data/bench/ files are "
            "always written regardless of this flag."
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
