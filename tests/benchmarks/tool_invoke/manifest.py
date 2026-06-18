"""Per-bench-run manifest: the metadata layer above ``trials.jsonl``.

Each run directory under ``data/bench/<runner>/<ts>/`` carries a
``manifest.json`` with enough context to (a) make cross-run comparisons
sound (case-content hash + prompt versions + commit sha) and (b) drive
the Phoenix Experiments upload without re-deriving anything from the
trial rows.

Schema is pydantic-validated; bumping ``SCHEMA_VERSION`` is the signal to
the reader (``phoenix_upload.read_manifest``) that field shapes changed.

Why a separate file (not stuffed into ``trials.jsonl``): trial rows are
per-cell data, the manifest is per-run metadata. Keeping them in two
files lets the JSONL stay a flat homogeneous table that ``judge.py`` and
``report.py`` already grep with no schema branching.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field

# Bump when the on-disk shape changes incompatibly. Readers fail loudly
# on an unknown version rather than silently parsing partial data.
SCHEMA_VERSION = 1

MANIFEST_FILENAME = "manifest.json"


class CasesSection(BaseModel):
    """Description of the case set this run iterated over."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    count: int
    tiers: list[str]
    # SHA-256 over the (name, revision, expected_behavior, expected_tool,
    # expected_tools) tuples of every selected case, sorted by name. Two
    # runs with identical content_sha256 are guaranteed to have iterated
    # the same cases at the same revisions — even if cases.py grew new
    # *unselected* cases in between.
    content_sha256: str


class ConfigSection(BaseModel):
    """Runner-supplied config knobs that meaningfully affect results."""

    model_config = ConfigDict(frozen=True, extra="allow")

    models: list[str]
    user_langs: list[str]
    # symptoms/vision runners may omit tool_prompt_langs (axis only
    # exists for ingest). Extra runner-specific keys land under
    # ``extra="allow"``.
    tool_prompt_langs: list[str] | None = None
    trials: int


class RunManifest(BaseModel):
    """One ``manifest.json`` payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = SCHEMA_VERSION
    runner: str  # "ingest" | "symptoms" | "vision"
    bench_ts: str  # the directory's timestamp suffix
    started_at: str  # ISO-8601
    finished_at: str  # ISO-8601
    commit_sha: str | None
    cases: CasesSection
    # Prompt name → version string (e.g. {"tool_proposal": "v4"}).
    # Captured from PromptRegistry at run start so a later Phoenix upload
    # ties results to the exact prompt text that drove them.
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    config: ConfigSection


# --- builders --------------------------------------------------------


def cases_content_sha256(cases: Iterable[Any]) -> str:
    """Hash the case identity tuples so two runs with the same case
    universe produce the same hash regardless of insertion order.

    Includes only fields that affect what "the same case" means for a
    comparison: ``name``, ``revision``, ``expected_behavior``,
    ``expected_tool``, ``expected_tools``. The predicate Callable can't
    be hashed but a behaviour change *should* bump ``revision`` (the
    Case docstring spells this out).
    """
    rows = []
    for c in cases:
        rows.append(
            (
                c.name,
                int(getattr(c, "revision", 1)),
                c.expected_behavior,
                getattr(c, "expected_tool", None),
                list(getattr(c, "expected_tools", []) or []),
            )
        )
    rows.sort(key=lambda r: r[0])
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def collect_prompt_versions(prompt_names: Iterable[str]) -> dict[str, str]:
    """Resolve current ``latest`` version for each named prompt.

    Returns a ``{name: version}`` dict. Missing prompts are silently
    skipped — the manifest records what the registry actually has at run
    time, not what callers expected.
    """
    from claritymed.core.prompts.registry import get_default_registry

    registry = get_default_registry()
    known = set(registry.list())
    out: dict[str, str] = {}
    for name in prompt_names:
        if name not in known:
            continue
        prompt = registry._prompts[name]
        latest = max(prompt.versions, key=lambda v: (v.created_at, v.version))
        out[name] = latest.version
    return out


def current_commit_sha() -> str | None:
    """Short git SHA of HEAD, or ``None`` outside a git tree."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    sha = out.stdout.strip()
    return sha or None


def now_iso() -> str:
    """ISO-8601 UTC timestamp suitable for manifest start/finish fields."""
    return datetime.now(timezone.utc).isoformat()


# --- IO --------------------------------------------------------------


def write_manifest(out_dir: Path, manifest: RunManifest) -> Path:
    """Persist ``manifest.json`` under the run dir. Returns the path."""
    path = out_dir / MANIFEST_FILENAME
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def read_manifest(run_dir: Path) -> RunManifest:
    """Load the manifest from a run dir. Fails loudly on unknown
    ``schema_version`` so the reader code doesn't silently misinterpret
    a future shape."""
    path = run_dir / MANIFEST_FILENAME
    raw = json.loads(path.read_text(encoding="utf-8"))
    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        msg = (
            f"manifest schema_version={version} in {path} but reader "
            f"expects {SCHEMA_VERSION}. Re-run the bench or update the "
            f"reader before upload."
        )
        raise ValueError(msg)
    return RunManifest.model_validate(raw)


__all__ = [
    "MANIFEST_FILENAME",
    "SCHEMA_VERSION",
    "CasesSection",
    "ConfigSection",
    "RunManifest",
    "cases_content_sha256",
    "collect_prompt_versions",
    "current_commit_sha",
    "now_iso",
    "read_manifest",
    "write_manifest",
]
