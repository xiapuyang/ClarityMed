"""Reporting layer for eval runs.

Each module turns one or more ``LmEvalRunner`` JSONL outputs into a
copy-pasteable report (markdown table to stdout + persisted file under
``data/evals/results/``).
"""

from claritymed.evals.reporting.delta import (
    DeltaReport,
    DeltaReportError,
    build_delta_report,
    find_latest_pair,
    render_delta_markdown,
    write_delta_markdown,
)

__all__ = [
    "DeltaReport",
    "DeltaReportError",
    "build_delta_report",
    "find_latest_pair",
    "render_delta_markdown",
    "write_delta_markdown",
]
