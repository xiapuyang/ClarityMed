"""Enforce case-revision discipline across all bench runners.

One parametrized test per runner. Failure modes that block the build:

* **silent_changes** — an existing ``name`` was edited (prompts /
  args_predicate / etc.) without bumping ``Case.revision``. Past
  bench runs that referenced this case ID are now incomparable to
  any future run. Fix: bump ``revision`` for the changed case AND
  re-run with ``--update-cases-baseline``.
* **revision_downgrades** — a ``revision`` field went *down*.
  Physically can't be right; usually means a revert went wrong.

Soft (non-blocking) outcomes that auto-update the baseline JSON when
the flag is passed:

* **additions** — new ``name``, baseline absorbs it.
* **deletions** — case removed from ``cases.py``, baseline drops it.
* **intentional_bumps** — sha differs *and* revision bumped, baseline
  records the new sha so future commits start clean.

How the snapshot file works
---------------------------

Each runner has ``cases_baseline_<runner>.json`` committed to git
beside ``cases.py``. It stores ``{name: {revision, content_sha256}}``
for every case. ``--update-cases-baseline`` rewrites it from current
state. The test asserts current ↔ baseline are consistent without
that flag.

Pre-commit hook
---------------

``.pre-commit-config.yaml`` runs *only this test* on changes to either
``cases.py`` or the baseline JSON files, so forgetting to bump a
revision fails ``git commit`` in <1s rather than at PR-review time.
"""

from __future__ import annotations

import pytest

from tests.benchmarks.tool_invoke.cases_baseline import (
    RUNNERS,
    baseline_path,
    compare,
    current_state,
    format_report,
    load_baseline,
    needs_baseline_update,
    save_baseline,
)


@pytest.mark.parametrize("runner", RUNNERS)
def test_cases_baseline_in_sync(runner: str, request: pytest.FixtureRequest) -> None:
    """Verify ``cases.py`` and ``cases_baseline_<runner>.json`` agree.

    Update-mode short-circuits before any assertion so an intentional
    revision bump in the same edit session can land with one command.
    """
    current = current_state(runner)
    baseline = load_baseline(runner)
    report = compare(runner, current, baseline)

    update_mode = bool(request.config.getoption("--update-cases-baseline"))
    if update_mode:
        # Bypass blocking assertions in update mode — the user is
        # explicitly telling us "rewrite the snapshot". We still refuse
        # to write through a revision downgrade because that's never
        # legitimate (would let a revert silently undo a published
        # bump).
        if report.revision_downgrades:
            pytest.fail(format_report(report))
        save_baseline(runner, current)
        pytest.skip(
            f"baseline rewritten: {baseline_path(runner).name} ({len(current)} cases)"
        )

    if report.has_blocking_issues:
        suggestion = (
            "\n\nFix:\n"
            "  1. For silent_changes — bump Case.revision in cases.py "
            "(no other rule says you must, but every revision-aware "
            "tool downstream — bench compare, Phoenix Experiments — "
            "depends on it).\n"
            "  2. Run: uv run pytest "
            "tests/benchmarks/tool_invoke/test_cases_baseline.py "
            "--update-cases-baseline\n"
            "  3. Commit cases.py + cases_baseline_<runner>.json "
            "together."
        )
        pytest.fail(format_report(report) + suggestion)

    if needs_baseline_update(report):
        # Non-blocking: surface as a skip message so the user sees
        # that an --update-cases-baseline run is needed.
        pytest.skip(
            "baseline drift detected (additions / deletions / "
            "intentional bumps). Run with --update-cases-baseline and "
            f"commit the JSON.\n\n{format_report(report)}"
        )
