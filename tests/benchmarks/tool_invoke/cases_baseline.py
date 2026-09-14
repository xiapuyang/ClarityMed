"""Snapshot-test machinery for case-revision discipline.

The rule we want to enforce: bumping ``Case.revision`` (in
``{ingest,symptoms,vision}/cases.py``) is the *only* legitimate way to
change ``prompts`` or ``args_predicate`` semantics for an existing
``name``. New cases (new names) are always OK; renaming or deleting a
case is OK too. Silent edits to an existing name without a revision
bump invalidate every prior bench run that referenced that case ID, so
we want them blocked at ``git commit`` time, not on the next bench
diff investigation.

How this module is used
-----------------------

``test_cases_baseline.py`` calls :func:`current_state` (live introspect
of ``cases.py``) and :func:`load_baseline` (the committed JSON next to
it). :func:`compare` returns a structured report; the test fails when
``silent_changes`` is non-empty.

``--update-cases-baseline`` (the pytest CLI flag added in this dir's
``conftest.py``) rewrites the baseline JSON from the current state.
Run it after intentionally bumping a revision; the JSON change lands
in the same PR as the case edit, making the audit trail "what changed
and when" trivially git-loggable.

Why hash source code instead of the bytecode
--------------------------------------------

``inspect.getsource`` returns the file text for the function — stable
across Python versions, human-readable in PR diffs, survives
refactoring that doesn't touch the function body. ``co_code`` is
shorter but flips on every minor interpreter bump even when behavior
is unchanged, generating spurious "silent change" failures.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

SCHEMA_VERSION = 1

# Baseline files live alongside this module so they're trivial to find
# in a PR diff and natural to add to the pre-commit ``files:`` filter.
_BASELINE_DIR = Path(__file__).resolve().parent

# Runners whose cases are tracked. Adding a new runner here + creating
# its ``cases_baseline_<runner>.json`` is the entire onboarding step.
RUNNERS: tuple[str, ...] = ("ingest", "symptoms", "vision")


# --- model ----------------------------------------------------------


@dataclass(frozen=True)
class CaseRecord:
    """One case's identity at a point in time."""

    name: str
    revision: int
    content_sha256: str


@dataclass
class ComparisonReport:
    """What changed between the current cases.py state and the baseline.

    ``silent_changes`` is the only field that should ever fail a build.
    Everything else is informational — adds / deletes / intentional
    bumps are normal PR activity, the test prints them so reviewers
    don't need to diff baseline JSON to see what's new.
    """

    runner: str
    additions: list[CaseRecord] = field(default_factory=list)
    deletions: list[CaseRecord] = field(default_factory=list)
    silent_changes: list[tuple[CaseRecord, CaseRecord]] = field(default_factory=list)
    intentional_bumps: list[tuple[CaseRecord, CaseRecord]] = field(default_factory=list)
    revision_downgrades: list[tuple[CaseRecord, CaseRecord]] = field(
        default_factory=list
    )
    unchanged: list[CaseRecord] = field(default_factory=list)

    @property
    def has_blocking_issues(self) -> bool:
        return bool(self.silent_changes) or bool(self.revision_downgrades)


# --- hashing --------------------------------------------------------


def content_sha256(case: Any) -> str:
    """Hash everything about a case that defines "what's the test asking?".

    Excluded on purpose: ``revision`` itself (so we can detect a
    revision bump as "sha differs *and* revision bumped" — including
    revision in the hash would make every bump look like a silent
    change too). Comparison code reads them as two separate fields.

    Fields included:
      - ``name`` / ``tier`` / ``expected_behavior`` (identity + cohort)
      - ``expected_tool`` / ``expected_tools`` (what success looks like)
      - ``prompts`` dict (the en/zh strings the model sees)
      - ``args_predicate`` source (the grader; semantic changes here
        are *the* thing we want to gate on)
      - ``seed`` source if present (pre-trial DB seeding can shift
        what's considered a "duplicate" e.g. for save_allergy v2's
        dedup check)
    """
    parts = {
        "name": case.name,
        "tier": case.tier,
        "expected_behavior": case.expected_behavior,
        "expected_tool": getattr(case, "expected_tool", None),
        "expected_tools": list(getattr(case, "expected_tools", []) or []),
        "prompts": dict(case.prompts),
        "args_predicate_src": safe_source(case.args_predicate),
        "seed_src": safe_source(case.seed) if getattr(case, "seed", None) else None,
    }
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def safe_source(fn: Callable | None) -> str:
    """``inspect.getsource`` that doesn't blow up on weird callables.

    Closures created at module load (e.g. ``_p_birth_year_from_age``
    returns a nested function) lose their own source line through
    ``getsource`` — Python reports the *enclosing* function's source,
    which still flips when the closure body changes, so it gives us the
    signal we want. C-extensions and lambdas defined inside lists fall
    through to a qualname-based fallback so the hash is still stable
    across runs, just less precise about body changes.
    """
    try:
        return inspect.getsource(fn).strip()
    except (TypeError, OSError):
        qualname = getattr(fn, "__qualname__", repr(fn))
        module = getattr(fn, "__module__", "?")
        return f"<unhashable:{module}:{qualname}>"


# --- introspection: live cases ---------------------------------------


def current_state(runner: str) -> dict[str, CaseRecord]:
    """Import ``cases.py`` for ``runner`` and snapshot every Case in CASES.

    Returns ``{name: CaseRecord}``. Importing is intentional rather than
    AST-parsing: the test must observe what the bench runner would
    actually see at runtime (after any module-level constants / case
    builders execute).
    """
    cases_module = _import_cases(runner)
    cases = getattr(cases_module, "CASES")
    out: dict[str, CaseRecord] = {}
    for case in cases:
        record = CaseRecord(
            name=case.name,
            revision=int(getattr(case, "revision", 1)),
            content_sha256=content_sha256(case),
        )
        if record.name in out:
            msg = (
                f"duplicate case name {record.name!r} in "
                f"{runner}/cases.py — names must be unique"
            )
            raise ValueError(msg)
        out[record.name] = record
    return out


def _import_cases(runner: str):
    """Defer the import so a failed cases.py module doesn't break the
    whole test session collection."""
    if runner == "ingest":
        from tests.benchmarks.tool_invoke.ingest import cases as mod
    elif runner == "symptoms":
        from tests.benchmarks.tool_invoke.symptoms import cases as mod
    elif runner == "vision":
        from tests.benchmarks.tool_invoke.vision import cases as mod
    else:
        msg = f"unknown runner: {runner!r}"
        raise ValueError(msg)
    return mod


# --- baseline IO ----------------------------------------------------


def baseline_path(runner: str) -> Path:
    return _BASELINE_DIR / f"cases_baseline_{runner}.json"


def load_baseline(runner: str) -> dict[str, CaseRecord]:
    """Read the committed baseline. Missing file → empty dict so
    first-time setup is a normal "additions" report."""
    path = baseline_path(runner)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        msg = (
            f"cases baseline at {path} has schema_version={version}; "
            f"expected {SCHEMA_VERSION}. Update reader or regenerate."
        )
        raise ValueError(msg)
    out: dict[str, CaseRecord] = {}
    for name, body in raw.get("cases", {}).items():
        out[name] = CaseRecord(
            name=name,
            revision=int(body["revision"]),
            content_sha256=str(body["content_sha256"]),
        )
    return out


def save_baseline(runner: str, state: dict[str, CaseRecord]) -> Path:
    """Write ``state`` to ``cases_baseline_<runner>.json``. Sorted by
    name so the file diffs cleanly when cases are added or revised."""
    path = baseline_path(runner)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "runner": runner,
        "cases": {
            name: {"revision": rec.revision, "content_sha256": rec.content_sha256}
            for name, rec in sorted(state.items())
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


# --- comparison -----------------------------------------------------


def compare(
    runner: str,
    current: dict[str, CaseRecord],
    baseline: dict[str, CaseRecord],
) -> ComparisonReport:
    """Diff ``current`` against ``baseline``, classifying each case.

    Decision table for a name present in both sides::

        sha same                          → unchanged
        sha differs, revision same        → silent_change  ❌
        sha differs, revision incremented → intentional_bump ✅
        sha same,    revision incremented → intentional_bump ✅
                                            (cosmetic refactor + bump;
                                             rare but legal — the bump
                                             alone signals "fresh start")
        revision decreased                → revision_downgrade ❌
    """
    report = ComparisonReport(runner=runner)
    current_names = set(current)
    baseline_names = set(baseline)

    for name in sorted(current_names - baseline_names):
        report.additions.append(current[name])

    for name in sorted(baseline_names - current_names):
        report.deletions.append(baseline[name])

    for name in sorted(current_names & baseline_names):
        cur = current[name]
        base = baseline[name]
        if cur.revision < base.revision:
            report.revision_downgrades.append((base, cur))
            continue
        if cur.revision == base.revision:
            if cur.content_sha256 == base.content_sha256:
                report.unchanged.append(cur)
            else:
                report.silent_changes.append((base, cur))
            continue
        # revision incremented
        report.intentional_bumps.append((base, cur))
    return report


# --- formatting -----------------------------------------------------


def format_report(report: ComparisonReport) -> str:
    """Human-readable report used in pytest failure messages.

    Each section starts with a count and ends with one bullet per case
    so reviewers can scan straight to the offending names.
    """
    lines: list[str] = [f"runner={report.runner}"]
    if report.silent_changes:
        lines.append(
            f"silent_changes ({len(report.silent_changes)}) — "
            "case body changed but revision did NOT bump:"
        )
        for base, cur in report.silent_changes:
            lines.append(
                f"  - {cur.name}: revision={cur.revision} sha "
                f"{base.content_sha256[:8]}→{cur.content_sha256[:8]} "
                f"(bump Case.revision to {cur.revision + 1} OR revert "
                "this edit)"
            )
    if report.revision_downgrades:
        lines.append(
            f"revision_downgrades ({len(report.revision_downgrades)}) — "
            "revision went *down*, which is never legal:"
        )
        for base, cur in report.revision_downgrades:
            lines.append(f"  - {cur.name}: {base.revision}→{cur.revision}")
    if report.intentional_bumps:
        lines.append(
            f"intentional_bumps ({len(report.intentional_bumps)}) "
            "— revision incremented, will rewrite baseline on next "
            "--update-cases-baseline:"
        )
        for base, cur in report.intentional_bumps:
            lines.append(f"  - {cur.name}: revision {base.revision}→{cur.revision}")
    if report.additions:
        lines.append(
            f"additions ({len(report.additions)}) "
            "— new case names, will be added to baseline:"
        )
        for cur in report.additions:
            lines.append(f"  - {cur.name} (revision={cur.revision})")
    if report.deletions:
        lines.append(
            f"deletions ({len(report.deletions)}) "
            "— removed from cases.py, will drop from baseline:"
        )
        for base in report.deletions:
            lines.append(f"  - {base.name} (was revision={base.revision})")
    return "\n".join(lines)


def needs_baseline_update(report: ComparisonReport) -> bool:
    """True when --update-cases-baseline would change the on-disk file.

    Used by the test to suggest the right next command: even
    non-blocking changes (a pure addition, an intentional bump) want
    the baseline rewritten so the snapshot stays current.
    """
    return any(
        [
            report.additions,
            report.deletions,
            report.intentional_bumps,
            report.silent_changes,  # so the test mentions update path when failing
        ]
    )


__all__ = [
    "RUNNERS",
    "SCHEMA_VERSION",
    "CaseRecord",
    "ComparisonReport",
    "baseline_path",
    "compare",
    "content_sha256",
    "current_state",
    "format_report",
    "load_baseline",
    "needs_baseline_update",
    "safe_source",
    "save_baseline",
]
