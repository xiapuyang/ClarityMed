"""Per-run cases snapshot — the content layer that makes Phoenix Dataset
versions a real case-history archive.

Why a separate file (not just ``trials.jsonl``)
-----------------------------------------------

Trial rows record the *formatted* user prompt (with seeds like
``{age}`` substituted) — useful for inspecting "what did the LLM see?"
but lossy as a case definition: the templates and the predicate
*source* aren't in there. Without those, lookup of an old
``save_allergy@v2`` would tell you what one instance asked, not what
the case *was*.

The snapshot captures the full content for every case selected in this
run, computed at run start when ``cases.py`` is the source of truth.
The Phoenix uploader prefers this file over re-deriving from trials so
each dataset example carries (prompts dict, predicate source, seed
source, expected_*). Phoenix versions the dataset by content — if a
case changes across runs, the new version retains the old version's
examples too, so ``case_id="save_allergy@v2"`` is recoverable forever
even after v3 exists.

Why not in ``manifest.json``
----------------------------

Manifest is a small metadata layer (counts, hashes, prompt versions).
Snapshot is potentially large (per-case prompts × langs + predicate
source = ~1-5 KB per case × ~30-50 cases = 50-250 KB). Keeping them
separate lets ``manifest.json`` stay glanceable and human-readable
without paging through case bodies.

Limit: this snapshot only covers cases *selected* for the current
run (after ``--tiers`` / ``--cases`` filters). Cases that exist in
``cases.py`` but weren't selected are not archived from this run.
This is intentional — the goal is "for each Phoenix experiment run,
recover the exact case content used" not "mirror the entire cases.py".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

from tests.benchmarks.tool_invoke.cases_baseline import (
    content_sha256,
    safe_source,
)

SCHEMA_VERSION = 1

SNAPSHOT_FILENAME = "cases_snapshot.json"


class CaseSnapshotEntry(BaseModel):
    """Full content for one case as it existed at run time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str  # "<name>@v<revision>" — stable cross-run lookup key
    name: str
    revision: int
    content_sha256: str
    tier: str
    expected_behavior: str
    expected_tool: str | None = None
    expected_tools: list[str] = Field(default_factory=list)
    prompts: dict[str, str] = Field(default_factory=dict)
    args_predicate_src: str
    seed_src: str | None = None


class CasesSnapshot(BaseModel):
    """Wrapper carrying schema + runner + the case list."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = SCHEMA_VERSION
    runner: str
    bench_ts: str
    cases: list[CaseSnapshotEntry] = Field(default_factory=list)


# --- builders --------------------------------------------------------


def build_snapshot(
    *,
    runner: str,
    bench_ts: str,
    cases: Iterable[Any],
) -> CasesSnapshot:
    """Snapshot every case in ``cases`` as ``CasesSnapshot``.

    Hashes via :func:`cases_baseline.content_sha256` so the snapshot's
    ``content_sha256`` always matches the pre-commit gate's view — if
    they ever diverge, the gate is checking something the snapshot
    didn't actually archive (a serious-but-loud failure mode).
    """
    entries: list[CaseSnapshotEntry] = []
    for case in cases:
        revision = int(getattr(case, "revision", 1))
        entries.append(
            CaseSnapshotEntry(
                case_id=f"{case.name}@v{revision}",
                name=case.name,
                revision=revision,
                content_sha256=content_sha256(case),
                tier=case.tier,
                expected_behavior=case.expected_behavior,
                expected_tool=getattr(case, "expected_tool", None),
                expected_tools=list(getattr(case, "expected_tools", []) or []),
                prompts=dict(case.prompts),
                args_predicate_src=safe_source(case.args_predicate),
                seed_src=(
                    safe_source(case.seed) if getattr(case, "seed", None) else None
                ),
            )
        )
    return CasesSnapshot(runner=runner, bench_ts=bench_ts, cases=entries)


# --- IO --------------------------------------------------------------


def write_snapshot(out_dir: Path, snapshot: CasesSnapshot) -> Path:
    """Persist ``cases_snapshot.json`` next to ``trials.jsonl``."""
    path = out_dir / SNAPSHOT_FILENAME
    path.write_text(
        json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def read_snapshot(run_dir: Path) -> CasesSnapshot | None:
    """Load a run's snapshot. Returns ``None`` when the file is absent
    (legacy run dir from before this layer existed) so callers can
    seamlessly fall back to other content sources.

    Fails loudly on an unknown ``schema_version`` so a future shape
    change doesn't silently look "uploadable" while losing fields.
    """
    path = run_dir / SNAPSHOT_FILENAME
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        msg = (
            f"cases snapshot at {path} schema_version={version}; "
            f"reader expects {SCHEMA_VERSION}."
        )
        raise ValueError(msg)
    return CasesSnapshot.model_validate(raw)


__all__ = [
    "SCHEMA_VERSION",
    "SNAPSHOT_FILENAME",
    "CaseSnapshotEntry",
    "CasesSnapshot",
    "build_snapshot",
    "read_snapshot",
    "write_snapshot",
]
