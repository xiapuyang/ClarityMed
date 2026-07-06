"""Open-source hygiene tests for the ``import-medical-record`` skill.

CLAUDE.md "Open-Source Hygiene" requires that committed skill content
never hardcodes personal identifiers, paths to a developer's home, or
non-English content. These tests run in CI so a maintainer can't
accidentally paste local context during skill development.

If you're seeing a failure here:

* Chinese characters / kana → translate to English. Reason: the skill
  is generic across projects; mixed-language content is a maintenance
  smell and risks accidental cultural assumptions.
* ``/Users/`` paths → use ``~/.claude/skills/...`` or relative
  ``skills/...`` paths. Reason: absolute home paths leak the
  developer's username.
* Personal names → use generic ``"user"``/``"alice"``. Reason: the
  repo will eventually be public.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

SKILL_DIR = Path(__file__).parent.parent.parent / "skills" / "import-medical-record"
SKILL_MD = SKILL_DIR / "SKILL.md"


@pytest.fixture
def skill_md() -> str:
    if not SKILL_MD.exists():
        pytest.fail(f"SKILL.md not found at {SKILL_MD}")
    return SKILL_MD.read_text(encoding="utf-8")


# --- frontmatter -------------------------------------------------------


def test_skill_md_has_yaml_frontmatter(skill_md: str):
    """Claude Code skills require a ``name`` + ``description`` frontmatter
    block at the top of SKILL.md."""
    assert skill_md.startswith("---\n"), "SKILL.md must open with YAML frontmatter"
    parts = skill_md.split("---\n", 2)
    assert len(parts) >= 3, "SKILL.md frontmatter must close with `---`"
    fm = yaml.safe_load(parts[1])
    assert isinstance(fm, dict)
    assert "name" in fm
    assert "description" in fm
    assert fm["name"] == "import-medical-record"


def test_skill_description_lists_trigger_phrases(skill_md: str):
    """CLAUDE.md skill rule: descriptions should be slightly pushy to
    combat undertriggering. Names both what the skill does AND the
    exact phrases that should activate it."""
    parts = skill_md.split("---\n", 2)
    fm = yaml.safe_load(parts[1])
    description = fm["description"].lower()
    # At least three of these obvious user phrases should appear in the
    # description. We don't require them all (skill author may reword),
    # but a description with none of them is undertriggering.
    triggers = (
        "import",
        "ingest",
        "load",
        "record",
        "medical",
        "notion",
        "apple health",
    )
    hits = sum(1 for t in triggers if t in description)
    assert hits >= 3, (
        f"description trigger phrases too thin (hits={hits}): {description!r}"
    )


# --- content hygiene --------------------------------------------------


def test_skill_md_contains_no_chinese_characters(skill_md: str):
    """The skill body must be English-only. ``[一-鿿]`` covers
    CJK Unified Ideographs U+4E00–U+9FFF."""
    chinese = re.findall(r"[一-鿿]", skill_md)
    assert not chinese, (
        f"SKILL.md contains Chinese characters: {''.join(chinese[:20])!r}. "
        "Translate to English."
    )


def test_skill_md_contains_no_kana(skill_md: str):
    """Same English-only constraint covers Japanese kana
    (Hiragana U+3040–U+309F, Katakana U+30A0–U+30FF)."""
    kana = re.findall(r"[぀-ヿ]", skill_md)
    assert not kana, f"SKILL.md contains kana: {''.join(kana[:20])!r}"


def test_skill_md_contains_no_user_absolute_paths(skill_md: str):
    """`/Users/<name>/...` leaks the developer's home directory. Use
    ``~/.claude/skills/...`` or repo-relative paths."""
    matches = re.findall(r"/Users/[A-Za-z0-9_-]+", skill_md)
    assert not matches, f"SKILL.md contains absolute /Users/ paths: {matches[:5]}"


def test_skill_md_contains_no_personal_name_denylist(skill_md: str):
    """A pragmatic, not-exhaustive denylist. Real protection comes from
    the absolute-path test above; this catches obvious slip-throughs
    like quoting a real chat message."""
    lower = skill_md.lower()
    denied = ("sharp", "xia", "xiapuyang")
    found = [w for w in denied if w in lower]
    assert not found, (
        f"SKILL.md contains personal-name denylist hits: {found}. "
        "Replace with generic 'user' / 'alice'."
    )


# --- structural ------------------------------------------------------


def test_skill_md_under_500_lines():
    """CLAUDE.md skill writing rule: SKILL.md under 500 lines (move
    reference material to ``references/``)."""
    lines = SKILL_MD.read_text(encoding="utf-8").splitlines()
    assert len(lines) < 500, (
        f"SKILL.md is {len(lines)} lines — move detail to references/"
    )


def test_validate_draft_script_present_and_executable():
    script = SKILL_DIR / "scripts" / "validate_draft.py"
    assert script.exists(), "scripts/validate_draft.py missing"
    head = script.read_text(encoding="utf-8").splitlines()[0]
    assert head.startswith("#!"), (
        f"validate_draft.py needs a shebang on line 1 (got {head!r})"
    )


def test_session_plan_script_present_and_executable():
    script = SKILL_DIR / "scripts" / "session_plan.py"
    assert script.exists(), "scripts/session_plan.py missing"
    head = script.read_text(encoding="utf-8").splitlines()[0]
    assert head.startswith("#!"), (
        f"session_plan.py needs a shebang on line 1 (got {head!r})"
    )


def test_references_present():
    refs_dir = SKILL_DIR / "references"
    expected = {
        "template-format.md",
        "multi-session-import.md",
        "content-hygiene.md",
    }
    actual = {p.name for p in refs_dir.glob("*.md")}
    assert expected.issubset(actual), f"missing reference files: {expected - actual}"
