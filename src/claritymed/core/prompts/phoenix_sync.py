"""Two-way sync between the local YAML prompt store and Phoenix.

The YAML store under ``core/prompts/store/`` stays the **runtime source of
truth**. Phoenix is where the team edits, evaluates, and experiments with
prompts via its UI. ``push`` ships YAML changes to Phoenix; ``pull``
brings Phoenix-side edits back into YAML for a code-review-able diff.

Naming convention: each ``(name, language)`` pair maps to one Phoenix
prompt called ``claritymed_<name>_<language>`` (e.g. ``claritymed_ask_en``).
Phoenix's own version history lives inside that prompt; the tag
``production`` marks the version that should round-trip with the YAML
``versions[latest]`` entry.

Why one Phoenix prompt per language instead of multi-message versions:
Phoenix prompt UI is built around a single message list per version.
Encoding two languages in one prompt would force per-render branching
that the UI doesn't surface well. One prompt per language is the lowest-
friction representation for editors and round-trips cleanly.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field

from claritymed.config import PROMPTS_STORE
from claritymed.core.prompts.registry import Prompt, PromptRegistry, PromptVersion

if TYPE_CHECKING:
    from phoenix.client import Client

logger = logging.getLogger(__name__)

Language = Literal["en", "zh"]
LANGUAGES: tuple[Language, ...] = ("en", "zh")

PRODUCTION_TAG = "production"
PHOENIX_MODEL_NAME = "claritymed"  # placeholder — real model lives in models.yaml
PHOENIX_TEMPLATE_FORMAT = "NONE"  # prompts are final strings, not templates
SyncAction = Literal["pushed", "pulled", "skipped", "missing", "error"]


class SyncEntry(BaseModel):
    """One name/language outcome from a sync run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_name: str
    language: Language
    phoenix_name: str
    action: SyncAction
    detail: str = ""


class SyncReport(BaseModel):
    """Aggregate result of ``push`` or ``pull``. Empty ``errors`` == success."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    direction: Literal["push", "pull"]
    dry_run: bool
    entries: list[SyncEntry] = Field(default_factory=list)

    @property
    def errors(self) -> list[SyncEntry]:
        return [e for e in self.entries if e.action == "error"]

    @property
    def changed(self) -> list[SyncEntry]:
        return [e for e in self.entries if e.action in ("pushed", "pulled")]


# --- public API ---------------------------------------------------------


def phoenix_prompt_name(prompt_name: str, language: Language) -> str:
    """Map ``(name, language)`` to the Phoenix prompt identifier."""
    return f"claritymed_{prompt_name}_{language}"


def push(
    name: Optional[str] = None,
    *,
    client: Optional["Client"] = None,
    store_dir: Path = PROMPTS_STORE,
    dry_run: bool = False,
) -> SyncReport:
    """Push YAML latest-version content to Phoenix (per language).

    ``name`` filters to one prompt. ``dry_run`` returns a report describing
    what *would* change without contacting Phoenix's write side.
    """
    client = client or _default_client()
    registry = PromptRegistry(store_dir=store_dir)
    targets = _select_targets(registry, name)
    entries: list[SyncEntry] = []
    for prompt_name in targets:
        latest = max(
            registry._prompts[prompt_name].versions, key=lambda v: v.created_at
        )
        for lang in LANGUAGES:
            entries.append(
                _push_one(client, prompt_name, lang, latest.languages[lang], dry_run)
            )
    return SyncReport(direction="push", dry_run=dry_run, entries=entries)


def pull(
    name: Optional[str] = None,
    *,
    client: Optional["Client"] = None,
    store_dir: Path = PROMPTS_STORE,
    dry_run: bool = False,
    into_new_version: bool = False,
    new_version_name: Optional[str] = None,
) -> SyncReport:
    """Pull Phoenix production-tagged content back into YAML.

    Default behaviour is **in-place**: each diverging language replaces the
    latest YAML version's content and bumps ``created_at`` to today. The
    version string and notes line are preserved so the audit trail in git
    stays compact — ``git diff`` shows exactly the prompt-text change.

    ``into_new_version=True`` switches to **append**: a new version block
    is added to the YAML file, leaving older versions intact for inline
    history / rollback. ``new_version_name`` overrides the default
    ``vN+1`` autoincrement so callers can tag explicit semver bumps
    (``v1.1``, ``v2.0-eval``, etc.) — providing it implies
    ``into_new_version=True``.
    """
    client = client or _default_client()
    registry = PromptRegistry(store_dir=store_dir)
    targets = _select_targets(registry, name)
    if new_version_name is not None:
        into_new_version = True
    entries: list[SyncEntry] = []
    pulled_by_name: dict[str, dict[Language, str]] = {}
    for prompt_name in targets:
        latest = max(
            registry._prompts[prompt_name].versions, key=lambda v: v.created_at
        )
        for lang in LANGUAGES:
            entry, new_text = _pull_one(
                client, prompt_name, lang, latest.languages[lang]
            )
            entries.append(entry)
            if entry.action == "pulled" and new_text is not None:
                pulled_by_name.setdefault(prompt_name, {})[lang] = new_text
    if not dry_run:
        for prompt_name, lang_text in pulled_by_name.items():
            yaml_path = store_dir / f"{prompt_name}.yaml"
            if into_new_version:
                _append_yaml_version(yaml_path, lang_text, registry, new_version_name)
            else:
                _overwrite_latest_yaml_version(yaml_path, lang_text, registry)
    return SyncReport(direction="pull", dry_run=dry_run, entries=entries)


# --- internals ----------------------------------------------------------


def _default_client() -> "Client":
    from phoenix.client import Client

    return Client()


def _select_targets(registry: PromptRegistry, name: Optional[str]) -> list[str]:
    if name is None:
        return registry.list()
    if name not in registry.list():
        msg = f"prompt {name!r} not found in {registry.store_dir}"
        raise ValueError(msg)
    return [name]


def _push_one(
    client: "Client",
    prompt_name: str,
    lang: Language,
    yaml_content: str,
    dry_run: bool,
) -> SyncEntry:
    phoenix_name = phoenix_prompt_name(prompt_name, lang)
    try:
        existing_text = _try_get_production(client, phoenix_name)
    except Exception as exc:  # noqa: BLE001
        return SyncEntry(
            prompt_name=prompt_name,
            language=lang,
            phoenix_name=phoenix_name,
            action="error",
            detail=f"phoenix fetch failed: {exc}",
        )
    if existing_text is not None and existing_text == yaml_content:
        return SyncEntry(
            prompt_name=prompt_name,
            language=lang,
            phoenix_name=phoenix_name,
            action="skipped",
            detail="phoenix already matches YAML",
        )
    if dry_run:
        detail = "would create" if existing_text is None else "would update"
        return SyncEntry(
            prompt_name=prompt_name,
            language=lang,
            phoenix_name=phoenix_name,
            action="pushed",
            detail=detail,
        )
    try:
        _create_phoenix_version(client, phoenix_name, prompt_name, lang, yaml_content)
    except Exception as exc:  # noqa: BLE001
        return SyncEntry(
            prompt_name=prompt_name,
            language=lang,
            phoenix_name=phoenix_name,
            action="error",
            detail=f"phoenix create failed: {exc}",
        )
    return SyncEntry(
        prompt_name=prompt_name,
        language=lang,
        phoenix_name=phoenix_name,
        action="pushed",
        detail="created" if existing_text is None else "updated",
    )


def _pull_one(
    client: "Client",
    prompt_name: str,
    lang: Language,
    yaml_content: str,
) -> tuple[SyncEntry, Optional[str]]:
    phoenix_name = phoenix_prompt_name(prompt_name, lang)
    try:
        existing_text = _try_get_production(client, phoenix_name)
    except Exception as exc:  # noqa: BLE001
        return (
            SyncEntry(
                prompt_name=prompt_name,
                language=lang,
                phoenix_name=phoenix_name,
                action="error",
                detail=f"phoenix fetch failed: {exc}",
            ),
            None,
        )
    if existing_text is None:
        return (
            SyncEntry(
                prompt_name=prompt_name,
                language=lang,
                phoenix_name=phoenix_name,
                action="missing",
                detail="no production tag on Phoenix",
            ),
            None,
        )
    if existing_text == yaml_content:
        return (
            SyncEntry(
                prompt_name=prompt_name,
                language=lang,
                phoenix_name=phoenix_name,
                action="skipped",
                detail="YAML already matches Phoenix",
            ),
            None,
        )
    return (
        SyncEntry(
            prompt_name=prompt_name,
            language=lang,
            phoenix_name=phoenix_name,
            action="pulled",
            detail="YAML will receive a new version",
        ),
        existing_text,
    )


def _try_get_production(client: "Client", phoenix_name: str) -> Optional[str]:
    """Return the text of the production-tagged version, or ``None`` if no
    such prompt or no production tag exists. Other errors propagate so the
    caller surfaces them in the report."""
    try:
        version = client.prompts.get(prompt_identifier=phoenix_name, tag=PRODUCTION_TAG)
    except Exception as exc:  # noqa: BLE001
        text = str(exc).lower()
        if "not found" in text or "404" in text:
            return None
        raise
    return _extract_text(version)


def _extract_text(version) -> str:
    """Read the assistant-facing string out of a Phoenix ``PromptVersion``.

    Our prompts are stored as a single system message with template
    format ``NONE``; ``.format()`` returns an OpenAI-style envelope from
    which we pluck the content.
    """
    formatted = version.format()
    messages = getattr(formatted, "messages", None) or []
    if not messages:
        return ""
    content = messages[0].get("content")
    if isinstance(content, str):
        return content
    # Fallback for structured content blocks
    if isinstance(content, list) and content and isinstance(content[0], dict):
        return content[0].get("text", "")
    return ""


def _create_phoenix_version(
    client: "Client",
    phoenix_name: str,
    prompt_name: str,
    lang: Language,
    text: str,
) -> None:
    from phoenix.client.types import PromptVersion as PhoenixPromptVersion

    version = PhoenixPromptVersion(
        [{"role": "system", "content": text}],
        model_name=PHOENIX_MODEL_NAME,
        template_format=PHOENIX_TEMPLATE_FORMAT,
        description=f"{prompt_name} ({lang})",
    )
    created = client.prompts.create(
        version=version,
        name=phoenix_name,
        prompt_description=f"ClarityMed system prompt — {prompt_name} ({lang})",
    )
    if created.id is None:
        msg = f"phoenix.prompts.create returned a version without an id for {phoenix_name}"
        raise RuntimeError(msg)
    client.prompts.tags.create(
        prompt_version_id=created.id,
        name=PRODUCTION_TAG,
        description=f"Pushed from YAML on {date.today().isoformat()}.",
    )


def _append_yaml_version(
    yaml_path: Path,
    lang_text: dict[Language, str],
    registry: PromptRegistry,
    explicit_version_name: Optional[str] = None,
) -> None:
    """Append a new version block to ``<name>.yaml`` carrying the pulled
    text. The new version's ``created_at`` is today so it becomes
    ``latest`` automatically.

    Both languages must always be present in a YAML version (the registry
    validator enforces this), so we fall back to the existing YAML text
    for whichever language did not change. ``explicit_version_name``
    overrides the auto-generated ``vN+1`` label.
    """
    prompt_name = yaml_path.stem
    current = registry._prompts[prompt_name]
    today = date.today()
    next_version = explicit_version_name or _next_version_name(current.versions)
    new_version = PromptVersion(
        version=next_version,
        created_at=today,
        notes=f"Pulled from Phoenix on {today.isoformat()}.",
        languages=_merged_languages(current, lang_text),
    )
    updated = Prompt(
        name=current.name,
        description=current.description,
        versions=[*current.versions, new_version],
    )
    _write_yaml(yaml_path, updated)


def _overwrite_latest_yaml_version(
    yaml_path: Path,
    lang_text: dict[Language, str],
    registry: PromptRegistry,
) -> None:
    """Replace the **latest** version's languages with the pulled text and
    bump ``created_at`` to today. Keeps the same version string and notes
    so git diff stays focused on the prompt-text change.

    Use ``_append_yaml_version`` (callers pass ``into_new_version=True``)
    when an explicit history entry is desired.
    """
    prompt_name = yaml_path.stem
    current = registry._prompts[prompt_name]
    latest = max(current.versions, key=lambda v: v.created_at)
    today = date.today()
    refreshed = PromptVersion(
        version=latest.version,
        created_at=today,
        notes=latest.notes,
        languages=_merged_languages(current, lang_text),
    )
    versions = [
        v if v.version != latest.version else refreshed for v in current.versions
    ]
    updated = Prompt(
        name=current.name,
        description=current.description,
        versions=versions,
    )
    _write_yaml(yaml_path, updated)


def _merged_languages(
    current: Prompt, lang_text: dict[Language, str]
) -> dict[Language, str]:
    """Compose the new per-language block, falling back to the current
    YAML latest version for any language that did not change on Phoenix."""
    latest = max(current.versions, key=lambda v: v.created_at)
    return {
        "en": lang_text.get("en", latest.languages["en"]),
        "zh": lang_text.get("zh", latest.languages["zh"]),
    }


def _write_yaml(yaml_path: Path, prompt: Prompt) -> None:
    payload = prompt.model_dump(mode="json")
    with yaml_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, allow_unicode=True, sort_keys=False)


def _next_version_name(versions: list[PromptVersion]) -> str:
    """Return ``vN+1`` where ``N`` is the highest ``vN`` already in use,
    or fall back to a date-stamped tag when names don't follow the
    ``v<int>`` convention."""
    highest = 0
    saw_v = False
    for v in versions:
        s = v.version
        if s.startswith("v") and s[1:].isdigit():
            saw_v = True
            highest = max(highest, int(s[1:]))
    if saw_v:
        return f"v{highest + 1}"
    return f"phoenix-{date.today().isoformat()}"


__all__ = [
    "LANGUAGES",
    "PRODUCTION_TAG",
    "SyncEntry",
    "SyncReport",
    "phoenix_prompt_name",
    "pull",
    "push",
]
