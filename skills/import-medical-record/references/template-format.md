# Template Format — Canonical Contract

The CLI `claritymed record import-from-template` accepts a directory
shaped like:

```
template/
  _meta.yaml         # schema_version, default_category, user_ids
  uid1.yaml          # cases + facts for uid1
  uid2.yaml          # cases + facts for uid2
```

Anything else in the directory (subdirs, `.DS_Store`, hidden files,
symlinks, README files) is a loader error.

## `_meta.yaml`

```yaml
schema_version: 1
created_at: "2026-06-25T12:34:56Z"
source_hint: "notion-export-2024-h1"   # free text; PHI-free; for the audit trail
default_category: exam-reports          # optional; backfills case.category if unset
user_ids:
  - alice
  - bob
```

Every entry in `user_ids` must have a corresponding `<user_id>.yaml`
on disk. Conversely, every `<user_id>.yaml` must appear in `user_ids`.

## `<user_id>.yaml`

```yaml
cases:
  - case_id: notion-page-abcdef12   # globally unique across all files
    event_date: "2024-01-15"
    title: Annual checkup
    kind: exam-report               # REQUIRED — singular vocabulary
    category: exam-reports          # optional if _meta.default_category set
    body_md: |
      Free-text clinical note. Lands in Manifest.body.
    attachments:
      - path: /tmp/draft-xxx/file.pdf
        original_filename: file.pdf
        mime: application/pdf
    tags:
      - routine
facts:
  profile:                          # dict, NOT a Profile model
    sex: female                     # only set explicitly-provided keys
    weight_kg: 60.0                 # None values are skipped (no overwrite)
  allergies:
    - substance: penicillin
      severity: severe
      source: clinical_record
  conditions:
    - display: Type 2 diabetes
      onset_date: "2020-05-01"
  medications:
    - display: metformin
      dose: 500mg
      onset_date: "2024-01-20"
```

## Invariants the loader enforces

* `case_id` is globally unique across all per-user files. Two files
  with the same `case_id` → loader error mentioning both filenames.
* `kind` is **required** per case (no defaulting from `category`).
* `category` is either set on the case or available via
  `_meta.default_category`.
* Attachment `path` is an existing file (not a symlink).
* User_id stems match `^[a-zA-Z0-9_-]{1,32}$`.
* `case_id` matches `^[a-zA-Z0-9_][a-zA-Z0-9_\-]{0,127}$`.
* No `_meta.created_at` drift moves the import_id (the hash strips
  it before canonicalizing).
* No `user_id:` field inside a per-user YAML body (the user_id is
  filename-authoritative — the loader rejects extras).

## Pydantic-level guarantees

The template loader uses `extra="forbid"` everywhere, so a maintainer
who adds an undocumented key during skill development gets a load-time
rejection rather than silent drift. If you want to extend the contract,
update both:

* `src/claritymed/ingest/records/template_schema.py` (the pydantic
  models), and
* this document (the canonical contract).

If you can't update both, you're not allowed to add the key.

## Why `body` lives on `Manifest.body`, not `Manifest.notes`

`body_md` is the case's primary content (potentially 2000+ words).
`Manifest.notes` is reserved for one-line tooling annotations
appended after the title; cramming a multi-paragraph body into it
silently shifts embed boundaries and ranking quality. `Manifest.body`
is the additive field added for this purpose.

## What is NOT in the template

* No `user_id` field inside per-user YAMLs. The loader derives
  user_id from the filename and rejects body-level overrides.
* No `_ocr_warnings` field at write time. The skill uses this in its
  draft to surface "this attachment had no extractable text" at the
  confirmation gate, but **strips it before invoking the CLI** —
  unknown keys are rejected.
* No top-level `embed_status` or `revision`. The case writer sets
  `embed_status="pending_retry"` so the reconcile worker picks the
  manifest up post-import; `revision` is `1` (the writer's choice).
