---
name: import-medical-record
description: |
  Import arbitrary medical records (Notion exports, Apple Health bundles,
  PDFs + photos + text from clipboard) into ClarityMed's per-event store.
  Use when the user says "import these records", "ingest this Notion
  export", "load my Apple Health bundle", "import medical history from a
  folder", "save these PDFs as records", "add records from this
  directory", "bulk import medical history", "we have a folder of lab
  reports", or describes pre-existing medical content they want into the
  agent's memory. This skill is the universal adapter — it inspects the
  source, builds a per-user-keyed YAML template, asks the user to confirm
  cases and facts, then invokes the `claritymed record import-from-template`
  CLI which mechanically applies the template. No format-specific
  knowledge is in the CLI; everything format-aware lives here.
---

# Import Medical Record

This skill drives a conversational flow for importing arbitrary medical
records into ClarityMed. It is the *universal adapter* — the only thing
the underlying CLI knows about is a template directory shape. New
source formats are a smarter conversation here, not new code.

## What gets imported

Two kinds of content land via this flow:

1. **Cases** — discrete events (a checkup, lab panel, prescription).
   Become `Manifest` rows on disk with attachments stored in the
   user's CAS blob pool.
2. **Facts** — durable profile rows (sex, weight, allergies, conditions,
   medications). Become rows in the user's `profile.db`.

Both go through explicit user confirmation before any write happens.

## Workflow overview

The flow has 12 steps. Steps 3-5 cover the **multi-session split** —
this is critical for large sources (50+ cases / multiple users) where
a single session would blow Claude's context window. See
`references/multi-session-import.md`.

```
1. Preflight — health check
2. Identify the source + look up prior plan
3. (first session only) Discovery + recommend a split strategy
4. (first session only) Write the per-source plan file
5. Pick this session's scope
6. Build the template directory contract for this scope
7. Conversational organization of cases + facts within the scope
8. Per-attachment OCR via `claritymed record ocr-extract`
9. Confirmation gates (user mapping, per-user creation, template)
10. Run `claritymed record import-from-template`
11. Update the session plan
12. Report + prompt next-step
```

## Step 1 — Preflight

Run `claritymed record health --json` BEFORE reading any user content.
Parse the JSON. If `ok: false`, surface the failed checks verbatim and
refuse to proceed. Specifically reject when:

* `phi_policy_not_local_only` — cost/cloud guard.
* `ocr_provider_build_failed` — extraction would crash mid-session.
* `no_user_account` — `claritymed init-user` must run first.

If the user opts out of fact extraction from attachments (a documented
limited-mode workflow), the skill MAY proceed without OCR healthy but
MUST warn that no facts will be extracted from any attachment.

## Step 2 — Identify the source + look up prior plan

Ask the user for a *source identifier* (a path or short hint like
`notion-2024-h1`). Compute `source_id = sha8(canonicalized identifier)`.
Then call:

```
uv run python skills/import-medical-record/scripts/session_plan.py read <source_id>
```

If a plan exists, show the history table (`ts, scope, import_id,
case_count, fact_count, status`) plus the `remaining_scopes` list.
Honor the prior plan unless the user explicitly says "start over."

If no plan exists, continue to step 3.

If a plan exists with non-empty `remaining_scopes`, skip to step 5.

## Step 3 — First-session discovery + split recommendation

Do a *lightweight* scan: count files, detect candidate user tags,
sample the first few entries. Do NOT read-and-organize the full
source yet — that's step 7 against the chosen scope.

Surface:

* Source size estimate (files, estimated cases, candidate user_ids).
* Recommended split strategy:
  * **by_user** (default for 2+ users with >10 cases each)
  * **by_folder** (when source has natural folder structure)
  * **single_shot** (only when ≤30 cases AND single user)

Ask the user to confirm.

## Step 4 — Write the initial plan

Call:

```
uv run python skills/import-medical-record/scripts/session_plan.py init <source_id> \
    --description "..." --split <strategy> \
    --users <uid:role,...> --remaining-scopes <s1,s2,...>
```

This creates `data/_skill_sessions/<source_id>/plan.yaml` mode 0600.
The plan is non-PHI (only labels, counts, scope status).

## Step 5 — Pick this session's scope

From `remaining_scopes`, the user picks one (default: first). The rest
of the workflow operates ONLY on that scope's content.

## Step 6 — The template directory contract

The CLI's `import-from-template` consumes a directory shaped like:

```
template/
  _meta.yaml         # schema_version, default_category, user_ids
  uid1.yaml          # cases + facts for uid1
  uid2.yaml          # cases + facts for uid2
```

Build it under `$TMPDIR/claritymed-import-draft-<uuid>/template/`.
Validate before invoking the CLI:

```
uv run python skills/import-medical-record/scripts/validate_draft.py <tempdir>
```

See `references/template-format.md` for the full contract.

## Step 7 — Organize cases + facts

For each candidate case in the scope:

* Pick a `kind` from the existing vocabulary (`exam-report`,
  `lab-report`, `prescription`, `note`, …). Singular — `category` (the
  on-disk folder) is plural. **kind is required**; the loader rejects
  cases without it.
* Pick a `category` (or rely on `_meta.default_category`).
* Choose attachments — paths must be readable from the skill's
  invocation environment.

For facts (R15 conservative-only):

* Only propose facts when the source unambiguously supports them.
* Each fact's `source:` annotation records what was looked at
  (attachment sha or body_md).

## Step 8 — OCR for fact extraction

Per attachment that the skill plans to extract facts from, call:

```
uv run claritymed record ocr-extract <path> --user <uid> --json
```

Cases by `status`:

* `ok` → run conservative fact extraction over `text`. The fact's
  `source:` annotation records the attachment sha.
* `empty` → annotate the case `_ocr_warnings: ["attachment <filename>:
  no text detected — likely a pure image"]`. Do NOT extract facts
  from this attachment.
* `failed` → annotate `_ocr_warnings: ["attachment <filename>: OCR
  failed (<reason>) — facts may be missing"]`.

`_ocr_warnings` is metadata-only and stripped from the template before
invoking the CLI (the CLI rejects unknown keys per `extra="forbid"`).

**Privacy note:** `ocr-extract` writes the file to the user's local
BlobStore at `data/users/<uid>/blobs/` and caches the OCR result there
BEFORE the user confirms the import. If the user drops the candidate
case, the blob + cache remain (they don't count as "records" — no
manifest references them — but they persist). Re-feeding the same file
later is a cache hit (no re-OCR). This trade-off makes the
single-OCR-across-the-pipeline guarantee possible.

**Latency optimization:** while organizing in step 7, spawn
`ocr-extract` per attachment in the background via `Bash(run_in_background=true)`.
By the time step 8 needs OCR text, most attachments are done; only
late-stage cases might still be running.

## Step 9 — Confirmation gates

Three explicit gates in order:

1. **User mapping** — confirm "this source's `user-a` label maps to
   user_id `alice` in ClarityMed."
2. **Per-user creation** — for any user_id in the template that
   doesn't yet exist in `data/users/`, ask whether to run
   `claritymed init-user <uid>`.
3. **Template confirmation** — show per-user summary including any
   `_ocr_warnings`. Example:

   > `user-a`: 12 cases, 1 allergy, **2 attachments had no
   > extractable text (cases #3, #7) — review manually before
   > confirming**.

Facts are a separate sub-gate inside (3) so the user has to actively
say "yes, add these facts" — wrong allergies/medications influence
the agent indefinitely.

## Step 10 — Run the import

```
uv run claritymed record import-from-template <tempdir> --user <uid> --json
```

Parse the streamed JSON events line by line and surface progress to
the user. Capture the `import_id` from the final `summary` event.

## Step 11 — Update the session plan

```
uv run python skills/import-medical-record/scripts/session_plan.py complete <source_id> \
    --scope <s> --import-id <id> --case-count <n> --fact-count <n>
```

## Step 12 — Report + next-step prompt

Summarize this session's counts + `import_id`. If `remaining_scopes`
is non-empty, prompt to run the skill again on the same source. If
empty, congratulate.

## Error handling

* **Health check failure** → refuse to start; surface the failed
  checks; instruct the user to fix and re-run.
* **Template validation failure** → CLI exits non-zero with
  `{"event":"fatal"}`; show the error; offer to edit the draft.
* **Mid-flight row error** → CLI keeps `template/` alive; instruct
  the user to fix the source issue and run `claritymed record
  import-resume <import-id>`. Do NOT mark this scope complete in the
  session plan until resume finishes cleanly.
* **SIGINT during import** → user can resume via `import-resume`;
  the CLI's WAL pattern ensures the in-flight row is recorded as
  `error(interrupted)`.

## Why splits are safe

The three dedup layers (slug determinism, BlobStore CAS, fact
equivalence) make re-feeding the same content across sessions a
no-op:

* Same case re-fed → `skipped(already_imported)`.
* Same attachment re-fed → BlobStore CAS dedupe.
* Same fact re-fed → equivalence skip; no `update_time` bump.

A user who accidentally picks the same scope twice gets a second
`import_id` with every row reporting `skipped`. No data harm.

## Installation

```
ln -s "$(pwd)/skills/import-medical-record" ~/.claude/skills/import-medical-record
```

The CLI commands the skill invokes are part of the project's `uv`
environment — run skill-spawned bash through `uv run` so the project
Python is on PATH.
