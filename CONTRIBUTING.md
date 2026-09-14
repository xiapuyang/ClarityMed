# Contributing to ClarityMed

Thanks for your interest — contributions are welcome.

## Local setup

```bash
uv sync                                     # base + dev + runtime-ocr
uv run pre-commit install                   # install commit-time hooks
uv run pre-commit install --hook-type pre-push   # install push-time hooks (unit + e2e)
```

## Before opening a PR

1. **Run the full pre-commit sweep**:
   ```bash
   uv run pre-commit run --all-files
   ```
   This runs gitleaks (secret scan), ruff (lint + format), the AI-bypass
   pattern check, and the commit-message length check.

2. **Run the unit tests with coverage gate**:
   ```bash
   uv run pytest
   ```
   The `--cov-fail-under=80` gate is enforced. If your change lowers
   coverage, add tests — do not lower the threshold (this is enforced in
   [CLAUDE.md](CLAUDE.md#test-coverage)).

3. **If you touched an LLM call path, run e2e**:
   ```bash
   uv sync --extra e2e
   uv run pytest tests/e2e -v --no-cov
   ```

## Commit messages

Follow the conventional commit style already in the log:

```
type(scope): short subject in imperative mood
```

Where `type` is one of `feat` / `fix` / `refactor` / `test` / `docs` /
`chore` / `perf` and `scope` is a short module hint (e.g.
`emergency`, `symptoms`, `rag`, `cli`).

Subject lines are hard-capped at **72 characters** by a pre-commit hook.
`wip` / `WIP` prefixes are blocked.

## Design constraints

Read [`CLAUDE.md`](CLAUDE.md) before making non-trivial changes.
The **Key design constraints** section lists non-negotiable invariants
that must hold across the codebase:

- PHI never leaves the machine unless the provider is `local`.
- `EmergencyTriage` runs only as a pre-step in `AskService.handle` —
  never called from tool plugins.
- Prompts live in `core/prompts/store/*.yaml` — no hardcoded strings.
- No JOINs / no FKs — cross-table references use integer `*_id`.
- `core/` may not reverse-depend on `core/orchestrator/`.

## Sensitive data

**Do not commit** any of the following (`.gitignore` already blocks
them; keep them out of `git add` even if pre-commit doesn't catch a
new pattern):

- `data/` — per-user runtime state, PHI-adjacent
- `logs/` — request/response logs
- `.env`, `*.env.local` — API keys, secrets
- `credentials.json`, `service-account.json`, `*.pem`, `*.key` —
  private credentials
- Real names, emails, phone numbers, addresses, IDs — not in code,
  not in seed data, not in test fixtures. See the **Open-Source
  Hygiene** section in `CLAUDE.md` for the full list.

## Test user IDs

Tests share two fixed user IDs to avoid polluting real dev directories
and to allow one-shot cleanup:

- Unit tests → `user_id="test"`
- E2E tests (`tests/e2e/`) → `user_id="e2e"`

Do not invent new UIDs (`alice` / `bob` / `u1`) — pick one of these two.

## Workflow skills (optional)

If you use Claude Code, the [`compound-engineering`](CLAUDE.md#development-workflow)
skills are wired up:

| Skill | When to invoke |
|---|---|
| `/ce:brainstorm` | Vague requirements, need divergent options |
| `/ce:plan` | Multi-step implementation ahead |
| `/ce:review` | Feature done, before opening PR |
| `/ce:compound` | Solved a non-trivial problem, capture the learning |

## Reporting security issues

Do **not** open a public issue for security vulnerabilities. Instead,
email `215984777+xiapuyang@users.noreply.github.com` with a
description and, if possible, a minimal reproduction.

## License

By contributing you agree that your contributions will be licensed
under the [MIT License](LICENSE).
