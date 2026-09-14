# Multi-Session Import — When and How to Split

A single Claude Code session can't realistically read + organize +
iterate on a 100+-case multi-user source. The source-side enumeration
alone burns context, before any draft writes. Rather than try to
engineer the skill into being context-frugal under all loads, this
skill **coaches the user into a per-session split** and treats each
session as a fully independent invocation of the CLI.

## Split strategies

| Strategy      | When                                                        |
|---------------|-------------------------------------------------------------|
| `by_user`     | 2+ users with >10 cases each (recommended default)           |
| `by_folder`   | Source has natural folder structure with disjoint content   |
| `single_shot` | ≤30 cases AND single user                                    |

## When NOT to split by_folder

Only when folders are disjoint. A source where a single case is spread
across multiple folders will be split across sessions and the dedup
layers will mark some as duplicates — usually not what the user wants.

## Cross-session continuity

`data/_skill_sessions/<source_id>/plan.yaml` is the user-visible thread
that survives across sessions. It records:

* `source.path_or_hint` snapshot for sanity-checking the source
  didn't drift between sessions.
* `total_estimated_cases` snapshot from session 1.
* `users` mapping table.
* `split_strategy`.
* `sessions[]` — one entry per completed scope with import_id and
  counts.
* `remaining_scopes[]`.

The plan is NON-PHI: only labels, counts, user_id handles the user
themselves chose. PHI never goes here.

## Why splits are safe

The three dedup layers guarantee that re-feeding the same content
across sessions is a no-op:

1. **Slug determinism** — `<event_date>-tmpl-<sha10(case_id)>`. Re-fed
   case → `ManifestStore.create()` raises `FileExistsError` →
   `skipped(already_imported)`.
2. **BlobStore CAS** — `store(content, ext)` is content-addressed.
   Re-fed attachment bytes → same sha → idempotent no-op.
3. **R12 fact equivalence** — applied at write time against the live
   `profile.db`. Re-fed allergy → existing row found → skipped.

A user who accidentally picks the same scope twice gets a second
`import_id` with every row reporting `skipped`. No data harm — only
one wasted session.

## Source change detection

If session 2's `total_estimated_cases` differs significantly from
session 1's snapshot, surface a warning: "source appears to have
changed since session 1 — verify the split strategy still makes
sense." Do NOT auto-invalidate the plan; let the user decide whether
to "start over" or continue.

## OCR cache semantics

`claritymed record ocr-extract` writes the file to the user's
BlobStore at `data/users/<uid>/blobs/` and caches the OCR result
**before** the user confirms the import. Practical consequences:

* **Re-feeding the same file is a cache hit.** No re-OCR cost when
  the skill re-runs in a later session.
* **Dropped candidate cases leave orphan blobs.** They don't count
  as "records" (no manifest references them) but they persist in
  the user's blob pool. Manually purge with:

  ```bash
  rm -rf data/users/<uid>/blobs/<sha[:2]>/<sha>/
  ```

* **Failed OCR results are sticky.** Matching existing project
  semantics: the failure sentinel says "don't auto-retry." To force
  a retry, delete the blob's `ocr.json` sentinel and re-run the
  skill.

This trade-off is what makes the single-OCR-across-the-pipeline
guarantee possible. The skill's OCR cache and the post-import lazy
`OcrWorker` share one sentinel per blob — each attachment is OCR'd
at most once across the entire pipeline.

## What to do if the user re-discovers a source mid-stream

Show the prior plan history. Ask:

* "Continue from session N+1 against the remaining scopes" → step 5
  of the workflow.
* "Start over (delete the plan file and re-discover)" → `rm -rf
  data/_skill_sessions/<source_id>/`; restart from step 3.

The skill never silently re-discovers; it always honors the prior
plan unless the user explicitly opts to start over.
