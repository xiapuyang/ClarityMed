# Content Hygiene Rules for Maintainers

This skill ships in a repo that will eventually be public. Treat all
files under `skills/import-medical-record/` as if they were already on
the open internet.

## What NEVER goes in skill content

* **Personal names** of the developer or family. Use generic `alice`,
  `bob`, `user` for examples.
* **Absolute paths under `/Users/`** or any other home directory.
  Use `~/.claude/...`, repo-relative paths, or `$TMPDIR/...`.
* **Non-English content.** SKILL.md and references are English-only.
  The skill itself can run in any language (the user picks the
  conversation language), but the source content stays English.
* **Real medical content** in examples. Use clearly fake values:
  `BP 120/80` (textbook), `metformin 500mg` (generic), never a real
  family member's diagnosis.
* **API keys, tokens, webhook URLs** — even "expired" ones. Pretend
  every string in this directory will end up on a public website.

## What does belong in the skill

* The conversational flow that drives the import.
* References to the CLI's documented surface
  (`claritymed record …`).
* Pointers to project files (relative paths from the repo root).
* Example template snippets using neutral identifiers
  (`alice`, `exam-reports`, `notion-abcdef12`).

## CI guard

`tests/skills/test_import_skill_hygiene.py` runs in CI and catches:

* Chinese characters or kana in SKILL.md.
* `/Users/` absolute paths in SKILL.md.
* Hits against the personal-name denylist.
* Missing frontmatter / missing trigger phrases / SKILL.md >= 500
  lines.
* Missing or non-executable scripts.
* Missing reference files.

When a check fails, the test message describes the remediation
(translate to English, switch to `~/.claude/...`, etc.).

## Why the runtime/design split matters

CLAUDE.md "Skill Writing Rules" pushes us to keep SKILL.md as
*runtime instructions* — what Claude does at skill-execution time.
This document is *maintainer guidance* — what humans editing the
skill should keep in mind. Mixing the two would inflate SKILL.md
past the 500-line guideline and load maintainer prose into the
model's context at runtime for no benefit.

If you find yourself writing "the rationale is X" or "we built it
this way because Y" — that belongs here, not in SKILL.md.

## Why this skill is generic, not ClarityMed-specific

SKILL.md says "the user said 'import these records'" — generic. It
names the CLI (`claritymed record ...`) because that's the contract;
it doesn't name the project's product theme, mission, or copy. A
skill that quotes a marketing tagline is one fewer thing the next
maintainer can refactor without breaking the skill.
