# ClarityMed CLI Reference

```
uv run claritymed [COMMAND] [SUBCOMMAND] [OPTIONS]
```

---

## Global options

These flags appear on every command.

| Flag | Short | Description |
|------|-------|-------------|
| `--user` | `-u` | User ID (required for most commands unless a default is configured) |
| `--lang` | `-l` | Language code: `en` or `zh` |
| `--provider` | `-p` | Override LLM provider ID from `configs/models.yaml` |

---

## Commands

### `init-user` — create a user account

```
claritymed init-user USER_ID [--name NAME]
```

| Argument | Description |
|----------|-------------|
| `USER_ID` | Alphanumeric, hyphens, underscores. First user created becomes admin. Idempotent. |

| Option | Description |
|--------|-------------|
| `--name / -n` | Display name (defaults to `USER_ID`) |

**Examples**

```bash
# Create the first (admin) user
uv run claritymed init-user alice

# Create a user with a display name
uv run claritymed init-user bob --name "Bob Smith"
```

---

### `tui` — interactive terminal UI

```
claritymed tui [--user USER] [--lang LANG] [--provider PROVIDER]
```

**Examples**

```bash
# Launch with defaults
uv run claritymed tui

# Launch as a specific user in English
uv run claritymed tui -u alice -l en

# Force a specific LLM provider
uv run claritymed tui -u alice -p claude
```

---

### `ask` — single headless question

```
claritymed ask QUESTION [--user USER] [--lang LANG] [--provider PROVIDER]
```

Streams a grounded answer to stdout. Mirrors the TUI ask flow without the interface.

**Examples**

```bash
# Basic question
uv run claritymed ask "What are the side effects of metformin?" -u alice

# Chinese reply
uv run claritymed ask "二甲双胍的副作用有哪些？" -u alice -l zh

# Use a specific provider
uv run claritymed ask "Explain my last lab result" -u alice -p gpt4o
```

---

### `ingest profile` — save a profile field

```
claritymed ingest profile FIELD [--user USER] [--lang LANG]
```

`FIELD` is a `key=value` pair. Appends or overwrites a single key in the user's profile YAML.

**Examples**

```bash
# Record an allergy
uv run claritymed ingest profile allergy=penicillin -u alice

# Record a diagnosis
uv run claritymed ingest profile diagnosis=hypertension -u alice

# Set preferred language
uv run claritymed ingest profile lang=zh -u alice
```

---

### `modes` — list interaction modes

```
claritymed modes
```

Prints all modes configured in `configs/modes.yaml` (e.g. `ask`, `rag`, `translate`).

```bash
uv run claritymed modes
```

---

### `rag` — manage personal RAG documents

#### `rag add` — index a document

```
claritymed rag add PATH [--user USER] [--lang LANG] [--public]
```

| Option | Description |
|--------|-------------|
| `--public` | Mark as public reference: skips PHI scrub, allows cloud providers |

**Examples**

```bash
# Add a personal medical note (PHI-scrubbed, local-only)
uv run claritymed rag add ./discharge_summary.pdf -u alice

# Add a public clinical guideline (no scrub, cloud-safe)
uv run claritymed rag add ./aha_heart_failure_2024.pdf -u alice --public

# Add a plain-text note
uv run claritymed rag add ./my_symptoms.txt -u alice
```

#### `rag list` — show indexed documents

```
claritymed rag list [--user USER]
```

```bash
uv run claritymed rag list -u alice
```

#### `rag show` — inspect chunks of a document

```
claritymed rag show DOC_ID [--user USER]
```

`DOC_ID` comes from `rag list`.

```bash
uv run claritymed rag show abc123 -u alice
```

#### `rag rm` — delete a document

```
claritymed rag rm DOC_ID [--user USER]
```

```bash
uv run claritymed rag rm abc123 -u alice
```

#### `rag corpora list` — list system corpora

```
claritymed rag corpora list
```

```bash
uv run claritymed rag corpora list
```

#### `rag corpora ingest` — ingest a system corpus (admin)

```
claritymed rag corpora ingest NAME [--user USER] [--limit N] [--dry-run] [--raw-dir PATH]
```

| Option | Description |
|--------|-------------|
| `--limit` | Cap the number of documents ingested |
| `--dry-run` | Parse and chunk without writing to Qdrant |
| `--raw-dir` | Override the default raw corpus path |

**Examples**

```bash
# Ingest the statpearls corpus
uv run claritymed rag corpora ingest statpearls -u alice

# Dry-run to verify parsing
uv run claritymed rag corpora ingest statpearls --dry-run

# Ingest only the first 100 docs
uv run claritymed rag corpora ingest statpearls --limit 100

# Use a custom raw data path
uv run claritymed rag corpora ingest statpearls --raw-dir /data/statpearls
```

#### `rag corpora migrate-payload` — patch Qdrant metadata (admin)

```
claritymed rag corpora migrate-payload NAME [--user USER]
```

Updates metadata fields on existing Qdrant points without re-embedding. Use this when an ingest left stale payload values.

```bash
uv run claritymed rag corpora migrate-payload statpearls -u alice
```

---

### `audit grep` — search the audit log

```
claritymed audit grep [--trace-id ID] [--request-id ID] [--user-id ID]
                      [--kind KIND] [--since ISO] [--until ISO]
                      [--limit N] [--json]
```

Reads all `audit.log*` files under `CLARITYMED_LOG_DIR` and prints matching lines.

| Option | Description |
|--------|-------------|
| `--trace-id` | Filter by OTel trace ID |
| `--request-id` | Filter by request ID |
| `--user-id` | Filter by user ID |
| `--kind` | Filter by audit event kind (e.g. `mode.ask`, `ocr.extract`) |
| `--since` | ISO 8601 prefix lower bound, e.g. `2026-06-08T10` |
| `--until` | ISO 8601 prefix upper bound |
| `--limit` | Stop after N matches |
| `--json` | Emit raw JSONL (no formatter prefix) |

**Audit kind values**

| Kind | When emitted |
|------|-------------|
| `request_start` / `request_end` | CLI entry / exit |
| `mode.ask` | Ask pipeline started |
| `mode.ingest` | Profile ingest |
| `mode.rag` | RAG retrieval started |
| `rag.retrieval` | Chunk retrieval result |
| `llm.call.start` | LLM call boundary |
| `ocr.extract` | OCR extraction (status, provider, latency) |
| `phi_guard_allow` / `phi_guard_block` | PHI guard decision |
| `account_created` | New user init |

**Examples**

```bash
# All events for a user today
uv run claritymed audit grep --user-id alice --since 2026-06-08

# All OCR events (any status)
uv run claritymed audit grep --kind ocr.extract

# Failed OCR events in the last hour
uv run claritymed audit grep --kind ocr.extract --since 2026-06-08T09 --json \
  | jq 'select(.payload.status == "error")'

# Trace a single request end-to-end
uv run claritymed audit grep --request-id <request_id>

# Last 20 ask events
uv run claritymed audit grep --kind mode.ask --limit 20

# Emit JSONL for downstream processing
uv run claritymed audit grep --user-id alice --json | jq '.payload.duration_ms'
```

---

### `audit ocr-overrides` — list blobs with OCR-carried clinician reports

```
claritymed audit ocr-overrides [--user-id USER] [--limit N] [--with-text] [--json]
```

Walks every `data/users/<id>/blobs/<sha[:2]>/<sha>/ocr.json` and prints
matches where `ocr_has_report: true` (the KTD-V6 short-circuit set —
images whose OCR text already carries a clinician report, so the vision
tool stays out and the LLM answers from the report text).

| Option | Description |
|--------|-------------|
| `--user-id / -u` | Restrict to one user; default walks every user under `data/users/` |
| `--limit` | Stop after N matches |
| `--with-text` | Inline `ocr.md` contents under `ocr_text` (JSON only); useful for benchmarking answer quality on the OCR-override branch |
| `--json` | Emit JSONL — one record per blob, pipeable into the vision benchmark as real-world OCR-override seeds |

**Examples**

```bash
# Quick visual scan of every user's OCR-override blobs
uv run claritymed audit ocr-overrides

# Just for one user
uv run claritymed audit ocr-overrides -u alice

# Pipe into a benchmark seed file (one JSON per line, includes OCR text)
uv run claritymed audit ocr-overrides --json --with-text \
  > data/bench/ocr_override_seeds.jsonl
```

Exit code: `0` on at least one match, `1` if no override sentinels were found.

---

### `prompts` — sync prompts with Phoenix

#### `prompts push` — upload YAML → Phoenix

```
claritymed prompts push [NAME] [--dry-run]
```

Tags the pushed version as `production` in Phoenix.

```bash
# Push all prompts
uv run claritymed prompts push

# Push a single prompt
uv run claritymed prompts push ask

# Preview without writing
uv run claritymed prompts push --dry-run
```

#### `prompts pull` — download Phoenix → YAML

```
claritymed prompts pull [NAME] [--dry-run] [--into-new-version] [--version-name LABEL]
```

| Option | Description |
|--------|-------------|
| `--dry-run` | Show diff without writing YAML |
| `--into-new-version` | Append a new version block instead of overwriting |
| `--version-name` | Explicit version label, implies `--into-new-version` |

```bash
# Overwrite latest version in-place
uv run claritymed prompts pull

# Preview diff only
uv run claritymed prompts pull --dry-run

# Append as a new version (auto-labelled v2, v3, …)
uv run claritymed prompts pull --into-new-version

# Append with explicit label
uv run claritymed prompts pull --version-name v1.1

# Pull a single prompt
uv run claritymed prompts pull ocr --dry-run
```

#### `prompts diff` — compare YAML vs Phoenix

```
claritymed prompts diff [NAME] [--color / --no-color]
```

Exit code `0` = in sync, `1` = drift detected (usable in CI).

```bash
uv run claritymed prompts diff

# Single prompt
uv run claritymed prompts diff ask

# CI gate (no color)
uv run claritymed prompts diff --no-color
```

---

### `finetune preprocess` — prepare fine-tune data

```
claritymed finetune preprocess NAME --input DIR --output DIR [--limit N]
```

Cleans a raw corpus into Alpaca-style JSONL train/val/test splits under `data/finetune/<name>/`. **Not indexed in Qdrant.**

```bash
# Preprocess the MedDialog Chinese corpus
uv run claritymed finetune preprocess meddialog_cn \
  --input ./raw/meddialog \
  --output ./data/finetune/meddialog_cn

# Cap to 500 dialogs for a quick test
uv run claritymed finetune preprocess meddialog_cn \
  --input ./raw/meddialog \
  --output ./data/finetune/meddialog_cn \
  --limit 500
```

---

## Common workflows

### First-time setup

```bash
# 1. Create the admin user
uv run claritymed init-user alice --name "Alice"

# 2. Ingest a system corpus (admin)
uv run claritymed rag corpora ingest statpearls -u alice

# 3. Launch the TUI
uv run claritymed tui -u alice
```

### Add personal documents, then ask

```bash
# Index a discharge summary (PHI — stays local)
uv run claritymed rag add ./discharge_may2026.pdf -u alice

# Index a public guideline (cloud-safe)
uv run claritymed rag add ./esc_hf_guidelines.pdf -u alice --public

# Ask a grounded question
uv run claritymed ask "Summarise my heart failure treatment plan" -u alice
```

### Debug a slow or failing OCR extraction

```bash
# Find recent OCR events
uv run claritymed audit grep --kind ocr.extract --since 2026-06-08 --json \
  | jq '{file: .payload.file, status: .payload.status, ms: .payload.duration_ms, provider: .payload.provider}'

# Find only errors
uv run claritymed audit grep --kind ocr.extract --json \
  | jq 'select(.payload.status == "error") | {file: .payload.file, error: .payload.error}'
```

### Prompt iteration cycle

```bash
# 1. Push current YAML to Phoenix
uv run claritymed prompts push

# 2. Edit in Phoenix UI, run evals

# 3. Check drift
uv run claritymed prompts diff

# 4. Pull back (safe preview first)
uv run claritymed prompts pull --dry-run
uv run claritymed prompts pull
```
