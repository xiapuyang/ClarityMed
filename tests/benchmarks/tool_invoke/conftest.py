"""Pytest plumbing scoped to ``tests/benchmarks/tool_invoke/``.

Currently adds one CLI option, ``--update-cases-baseline``, used by
``test_cases_baseline.py`` to rewrite the per-runner snapshot JSON
files when a case revision was intentionally bumped.

Keeping this conftest local (rather than in ``tests/conftest.py``)
matters because the option name has no meaning outside the bench
suite — adding it at the session root would pollute every pytest
``--help`` output and risk collisions with other suites.
"""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Wire the snapshot-rewrite flag.

    Usage::

        uv run pytest tests/benchmarks/tool_invoke/test_cases_baseline.py \\
            --update-cases-baseline

    The flag has no effect on any test other than ``test_cases_baseline``.
    """
    parser.addoption(
        "--update-cases-baseline",
        action="store_true",
        default=False,
        help=(
            "Rewrite cases_baseline_<runner>.json from the live "
            "cases.py state. Use after intentionally bumping "
            "Case.revision or adding/removing cases."
        ),
    )
