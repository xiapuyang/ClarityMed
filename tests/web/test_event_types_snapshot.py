"""Cross-repo event-types contract — written to ``event_types.json``.

Why this exists: the frontend (``claritymed-ui``) ships a hand-written
``events.ts`` discriminated union that has to stay in step with the
Python ``Event`` union here. Both sides cannot be sync-tested in one
repo (the frontend lives in a sibling repo) so we drop a committed
snapshot of the discriminator literals on the backend side. The
frontend tests assert ``events.ts`` matches a copied-over version of
the same file.

Test modes:

* ``CI=1`` → assert-only. Any drift between ``event_types.json`` on
  disk and the live ``Event`` union fails the build. Used in CI.
* Default (local dev) → write-and-assert. Regenerates the snapshot
  from the live union; the developer can ``git diff`` to see what
  changed and commit the new snapshot.

The snapshot file is committed to the backend repo so anyone can diff
it against the live union without running tests.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import get_args

import pytest

from claritymed.core.events import Event

# Snapshot lives inside the web package so it ships with installs. The
# frontend's manual `cp` step is documented in the plan's Operational
# Notes; updating the snapshot is a one-line copy in the FE repo.
SNAPSHOT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "claritymed"
    / "web"
    / "event_types.json"
)

SNAPSHOT_VERSION = 1


def _live_event_types() -> list[str]:
    """Return the discriminator literals from the live Event union.

    ``Event`` is ``Union[TokenChunk, Done, …]`` — ``get_args(Event)``
    gives the member classes. Each class declares
    ``type: Literal["..."]`` which lives in its pydantic fields as the
    default value.
    """
    types: list[str] = []
    for cls in get_args(Event):
        type_field = cls.model_fields.get("type")
        assert type_field is not None, f"{cls.__name__} missing `type` field"
        value = type_field.default
        assert isinstance(value, str), (
            f"{cls.__name__}.type default must be a str literal, got {value!r}"
        )
        types.append(value)
    return sorted(types)


def _snapshot_payload() -> dict[str, object]:
    return {"version": SNAPSHOT_VERSION, "types": _live_event_types()}


def _read_snapshot() -> dict[str, object] | None:
    if not SNAPSHOT_PATH.exists():
        return None
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def test_event_types_snapshot_in_sync():
    """Assert the on-disk snapshot matches the live Event union.

    CI mode (``CI=1`` in env) is assert-only — drift fails the build.
    Local mode regenerates the snapshot first so the developer can
    review and commit. Either way, the assertion at the end must hold.
    """
    live = _snapshot_payload()
    is_ci = bool(os.environ.get("CI", "").strip())

    if not is_ci:
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(
            json.dumps(live, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    on_disk = _read_snapshot()
    assert on_disk is not None, (
        f"snapshot missing at {SNAPSHOT_PATH}; run the test locally "
        "(without CI=1) to regenerate."
    )
    assert on_disk == live, (
        "Event union drifted from snapshot. Run this test locally "
        "(without CI=1) to regenerate, then commit "
        f"{SNAPSHOT_PATH.relative_to(SNAPSHOT_PATH.parents[3])} "
        f"and copy the new file to "
        "claritymed-ui/references/event_types.json. "
        f"Symmetric difference: live={set(live['types']) - set(on_disk['types'])} "
        f"snapshot={set(on_disk['types']) - set(live['types'])}"
    )


def test_event_types_payload_shape():
    """The committed snapshot must be ``{"version": N, "types": [...]}``."""
    payload = _snapshot_payload()
    assert payload["version"] == SNAPSHOT_VERSION
    assert isinstance(payload["types"], list)
    # No duplicates — discriminator literals must be unique across the union.
    types = payload["types"]
    assert len(types) == len(set(types)), f"duplicate type literals: {types}"
    # Sanity floor — the current union has 12 types; if it drops below
    # 8 something obviously broke.
    assert len(types) >= 8


def test_ci_mode_skips_regeneration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """With ``CI=1``, a stale snapshot fails rather than being rewritten."""
    monkeypatch.setenv("CI", "1")
    fake = tmp_path / "event_types.json"
    fake.write_text(
        json.dumps({"version": SNAPSHOT_VERSION, "types": ["wrong"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr("tests.web.test_event_types_snapshot.SNAPSHOT_PATH", fake)
    with pytest.raises(AssertionError):
        test_event_types_snapshot_in_sync()
