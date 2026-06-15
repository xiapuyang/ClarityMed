"""ProfileContextFeature — injects a compact patient-profile block into turns.

In ``deterministic`` mode the block is spliced directly into the user prompt
via ``pre_invoke``; the LLM always sees the data without needing a tool call.
In ``tool`` mode the agent may call ``retrieve_profile()`` on demand — useful
when profile data is large and the context window is tight.

Both modes produce an identical formatted string; the distinction is only in
*when* the string reaches the model.

Format (only non-empty lines are emitted)::

    [Patient profile]
    sex: male  birth_date: 1989-01-01  weight_kg: 72.0  height_cm: 178.0
    residence: Shanghai  birthplace: Beijing  marital_status: married  has_children: true
    current_occupation: engineer
    Allergies (1): penicillin (severe/documented)
    Active conditions (2): type 2 diabetes (since 2020-01-01), hypertension
    Current medications (1): metformin 500mg twice daily
    Records (12 total): labs/2026-05-10-lipid (lab_report · 2026-05-10), ...

"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable

import yaml

from claritymed.core.features.base import FeatureMode, TurnContext
from claritymed.stores.manifest_store import ManifestStore
from claritymed.stores.profile import ProfileStore

if TYPE_CHECKING:
    from pydantic_ai.toolsets import AbstractToolset

logger = logging.getLogger(__name__)

# Maximum number of recent records shown in the summary line.
_MAX_RECORDS_SHOWN = 3


def _format_profile_block(user_id: str) -> str:
    """Read all profile data synchronously and return the formatted block.

    Called inside ``asyncio.to_thread`` by the async entry points so the
    SQLite + file I/O never blocks the event loop.

    Returns:
        Formatted ``[Patient profile]`` block, or ``""`` when the user has
        no profile data at all.
    """
    store = ProfileStore(user_id)

    profile = store.get_profile()
    allergies = store.list_allergies()
    conditions = store.list_conditions()
    medications = store.list_medications()

    # Collect manifest paths for the records summary.
    try:
        record_paths = list(ManifestStore(user_id, "records").list())
    except Exception:  # noqa: BLE001 — no records dir yet
        record_paths = []

    lines: list[str] = []

    # ------------------------------------------------------------------ #
    # Biometric / biographical summary line(s)                            #
    # ------------------------------------------------------------------ #
    if profile is not None:
        bio_parts: list[str] = []
        if profile.sex is not None:
            bio_parts.append(f"sex: {profile.sex}")
        if profile.birth_date is not None:
            bio_parts.append(f"birth_date: {profile.birth_date}")
        if profile.weight_kg is not None:
            bio_parts.append(f"weight_kg: {profile.weight_kg}")
        if profile.height_cm is not None:
            bio_parts.append(f"height_cm: {profile.height_cm}")
        if bio_parts:
            lines.append("  ".join(bio_parts))

        geo_parts: list[str] = []
        if profile.residence is not None:
            geo_parts.append(f"residence: {profile.residence}")
        if profile.birthplace is not None:
            geo_parts.append(f"birthplace: {profile.birthplace}")
        if profile.marital_status is not None:
            geo_parts.append(f"marital_status: {profile.marital_status}")
        if profile.has_children is not None:
            geo_parts.append(f"has_children: {str(profile.has_children).lower()}")
        if geo_parts:
            lines.append("  ".join(geo_parts))

        if profile.current_occupation is not None:
            lines.append(f"current_occupation: {profile.current_occupation}")

    # ------------------------------------------------------------------ #
    # Allergies — active only (end_date is None)                          #
    # ------------------------------------------------------------------ #
    active_allergies = [a for a in allergies if a.end_date is None]
    if active_allergies:
        parts = [f"{a.substance} ({a.severity}/{a.source})" for a in active_allergies]
        lines.append(f"Allergies ({len(active_allergies)}): {', '.join(parts)}")

    # ------------------------------------------------------------------ #
    # Conditions — show active + resolved (with suffix)                   #
    # ------------------------------------------------------------------ #
    if conditions:
        active = [c for c in conditions if c.end_date is None]
        resolved = [c for c in conditions if c.end_date is not None]

        cond_parts: list[str] = []
        for c in active:
            entry = c.display
            if c.onset_date:
                entry += f" (since {c.onset_date})"
            cond_parts.append(entry)
        for c in resolved:
            entry = c.display + " (resolved)"
            cond_parts.append(entry)

        lines.append(f"Active conditions ({len(conditions)}): {', '.join(cond_parts)}")

    # ------------------------------------------------------------------ #
    # Medications — active only (end_date is None)                        #
    # ------------------------------------------------------------------ #
    active_meds = [m for m in medications if m.end_date is None]
    if active_meds:
        med_parts: list[str] = []
        for m in active_meds:
            entry = m.display
            if m.dose:
                entry += f" {m.dose}"
            if m.frequency:
                entry += f" {m.frequency}"
            med_parts.append(entry)
        lines.append(
            f"Current medications ({len(active_meds)}): {', '.join(med_parts)}"
        )

    # ------------------------------------------------------------------ #
    # Records — count + up to 3 most recent (sorted by path descending)  #
    # ------------------------------------------------------------------ #
    if record_paths:
        total = len(record_paths)
        # Sort descending so the most recent slug (date prefix) comes first.
        sorted_paths = sorted(record_paths, key=lambda p: str(p), reverse=True)
        recent = sorted_paths[:_MAX_RECORDS_SHOWN]

        record_snippets: list[str] = []
        for path in recent:
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                category = raw.get("category", "")
                slug = raw.get("slug", "")
                kind = raw.get("kind", "")
                event_date = raw.get("date", "")
                label = f"{category}/{slug}"
                meta_parts = [
                    p for p in [kind, str(event_date) if event_date else ""] if p
                ]
                if meta_parts:
                    label += f" ({' · '.join(meta_parts)})"
                record_snippets.append(label)
            except Exception:  # noqa: BLE001 — skip corrupt manifests silently
                continue

        if total > _MAX_RECORDS_SHOWN and record_snippets:
            record_snippets.append("...")

        if record_snippets:
            lines.append(f"Records ({total} total): {', '.join(record_snippets)}")
        else:
            lines.append(f"Records ({total} total)")

    if not lines:
        return ""

    return "[Patient profile]\n" + "\n".join(lines)


class ProfileContextFeature:
    """Inject a compact patient-profile block into each ask turn.

    Supports two modes:

    * ``deterministic`` — ``pre_invoke`` returns the block; the LLM always
      sees it, with no tool call required.
    * ``tool`` — ``pre_invoke`` returns ``""``, and ``as_tool`` returns a
      zero-argument ``retrieve_profile`` callable the model invokes when
      it decides the profile is relevant.

    All blocking I/O (SQLite queries, YAML reads) runs in a thread via
    ``asyncio.to_thread`` to avoid stalling the event loop.
    """

    name = "profile_context"
    mode: FeatureMode

    def __init__(self, *, mode: str = "deterministic") -> None:
        if mode not in ("deterministic", "tool"):
            raise ValueError(
                f"ProfileContextFeature mode must be 'deterministic' or 'tool', "
                f"got {mode!r}"
            )
        self.mode = mode  # type: ignore[assignment]

    async def pre_invoke(self, ctx: TurnContext) -> str:
        """Return the profile block in deterministic mode, ``""`` in tool mode."""
        if self.mode != "deterministic":
            return ""
        user_id = ctx.deps.user_id
        return await asyncio.to_thread(_format_profile_block, user_id)

    def as_tool(self) -> Callable | None:
        """Return ``retrieve_profile`` callable in tool mode, ``None`` otherwise."""
        if self.mode != "tool":
            return None
        return _make_retrieve_profile_tool()

    def as_toolset(self) -> "AbstractToolset[Any] | None":
        """No pre-built toolset — single callable exposed via ``as_tool``."""
        return None


def _make_retrieve_profile_tool() -> Callable:
    """Build a closure that the pydantic-ai agent can call as a tool.

    The closure captures nothing at construction time; ``user_id`` is read
    from the pydantic-ai ``RunContext`` deps at call time so each agent run
    gets the correct user.
    """
    from pydantic_ai import RunContext

    async def retrieve_profile(ctx: RunContext) -> str:
        """Return a compact summary of the patient's profile, allergies, conditions, medications, and recent records."""
        user_id = ctx.deps.user_id
        return await asyncio.to_thread(_format_profile_block, user_id)

    return retrieve_profile
