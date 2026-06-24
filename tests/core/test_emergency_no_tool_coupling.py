"""Static guard for KTD-E5: tools never import the emergency gate.

The plan's strictest architectural rule is one-way coupling — the
emergency gate calls down into the agent loop, but the agent's tools
(``symptoms_plugin``, ``vision_plugin``, ``ingest_tools_plugin``, and
every future tool) never reach up to import ``EmergencyTriage`` or
write safety signals that belong to the gate. The reverse direction was
explicitly considered and rejected (plan KTD-E5) because every tool
becoming gate-aware would compound coupling as new tools land.

This test fails loud if any future change adds such an import.
"""

from __future__ import annotations

from pathlib import Path

# Forbidden symbols: anything that would let a tool reach into the
# gate's decision layer or write to the audit-trail surface the gate
# owns. ``EmergencyAssessment`` and ``MatchedRule`` are the data
# shapes; ``EmergencyTriage`` is the facade. If a tool legitimately
# needs to *read* a triage decision in the future, the right move is
# to add the field to ``AskDeps`` so the gate stays the sole producer
# — not to import from this module list directly.
_FORBIDDEN_IMPORTS = (
    "from claritymed.core.emergency",
    "import claritymed.core.emergency",
)


def _features_dir() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "src"
        / "claritymed"
        / "orchestrator"
        / "features"
    )


def _tool_python_files() -> list[Path]:
    """Every tool / plugin file under orchestrator/features/."""
    return sorted(p for p in _features_dir().rglob("*.py") if p.name != "__init__.py")


def test_orchestrator_features_dir_exists():
    """Sanity check the discovery path before claiming the guard holds."""
    files = _tool_python_files()
    assert files, (
        f"Expected tool plugin files under {_features_dir()} but found none. "
        "Either the path moved (update the test) or every plugin disappeared "
        "(probably wrong)."
    )


def test_no_tool_imports_emergency_gate():
    """Every tool file is free of imports from ``claritymed.core.emergency``.

    Plan KTD-E5: tools may not import gate types or write safety signals
    that belong to the gate. If a tool needs to know the current triage
    level, add the field to :class:`AskDeps` so the gate stays the sole
    producer — do not reach across modules.
    """
    offenders: list[tuple[Path, str]] = []
    for path in _tool_python_files():
        text = path.read_text(encoding="utf-8")
        for needle in _FORBIDDEN_IMPORTS:
            if needle in text:
                offenders.append((path, needle))
    assert not offenders, (
        "KTD-E5 violation — tool plugin imports from claritymed.core.emergency:\n"
        + "\n".join(f"  {p}: {needle!r}" for p, needle in offenders)
        + "\n\nMove the dependency into AskDeps or AskService instead."
    )
