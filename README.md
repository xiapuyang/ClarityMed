# ClarityMed

**Local-first medical AI assistant.** ClarityMed keeps PHI-sensitive
reasoning on-device, runs a deterministic emergency-triage gate before any
LLM composition, and orchestrates multi-modal diagnostics (symptoms,
imaging, RAG over patient notes) through a single pydantic-ai agent loop.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](pyproject.toml)
[![Coverage ≥80%](https://img.shields.io/badge/coverage-%E2%89%A580%25-brightgreen)](pyproject.toml)
[![Powered by pydantic-ai](https://img.shields.io/badge/agent-pydantic--ai%201.0-8a2be2)](https://ai.pydantic.dev/)

[中文文档 →](README.zh.md)

---

## Highlights

Every number below comes from an evaluation checked into the repo — no
marketing figures.

| Metric | Result | Source |
|---|---|---|
| Symptoms 3-class Influenza F1 | **0.904** | `typed_basd_pne_flu_other_v1` checkpoint |
| Symptoms 3-class Macro F1 | **0.735** | same |
| Symptoms Brier R² (vs XGBoost baseline) | **+0.52** (vs +0.17) | calibration improvement |
| Emergency Triage Critical Recall | **0.958** (target ≥0.95) | `balanced` profile, hold-out set |
| Emergency Triage Adversarial FPR | **0.014** (target <0.15) | same |
| Emergency Triage F-β=2 | **0.965** | same |
| Session-start latency (p50) | **81 ms** | `ddxplus_pneumonia_flu` bench, local MLX |
| Per-turn latency (p50) | **130 ms** | same |
| Local throughput (MLX Qwen3-35B-A3B-Q4) | **40–60 tok/s** | M-series Apple Silicon, ~18 GB RAM |
| Flu F1 vs XGBoost baseline | **0.904 vs 0.526** (+72%) | typed_basd vs xgb_v5_recallig |
| Pneumonia precision | **1.00** | new 3-class model, zero false positives |

## Technical Innovations

**1. PHI-safe cloud routing** — a single-field gate
(`ProviderConfig.kind`) ensures no PHI leaves the machine unless the
provider is explicitly `local`. Cloud calls pass through `phi_guard`
plus an assembled-prompt double-scrub; there is no per-user opt-in
flag to forget to set. The gate is the invariant — every LLM call
goes through it, or it doesn't run.

**2. Deterministic emergency triage as a pre-agent safety gate** —
critical conditions (STEMI, anaphylaxis, ectopic pregnancy, active
suicidal ideation) short-circuit the agent loop entirely. A local-LLM
extractor + rule engine composes an i18n action reply in <200 ms
instead of the 3–5 s an agent-composed message would take. Rule
authors keep veto power via `minimum_sensitivity_floor` — no
operator profile can weaken anaphylaxis / active-SI detection.

**3. Multi-provider abstraction via pydantic-ai 1.0** — 8+ providers
(OpenAI, Anthropic, DeepSeek, Moonshot, Alibaba, OpenRouter, Ollama,
MLX / llama.cpp / LM Studio) unified through YAML config. Reasoning
effort passes through as a single `thinking` field
(`minimal`/`low`/`medium`/`high`/`xhigh`); pydantic-ai translates it
per-vendor. Add a new local backend with one YAML entry, zero Python.

**4. Prompts-as-YAML with Phoenix sync** —
`core/prompts/store/*.yaml` is the runtime single source of truth.
Phoenix (arize-phoenix) is the *editing UI + eval platform*, never
in the runtime path. `claritymed prompts push/pull` is the only
bridge; version history is immutable, dual-language (en/zh) is
validator-enforced.

**5. Multi-modal per-disease vision registry** — YAML-driven
hot-swap for BUSI ultrasound, ChestX-ray14, RSNA Pneumonia
detection, chest-CT classification. Add a new disease/model by
appending a YAML entry; per-disease training pipelines (forge / yolo_forge)
share a common MLflow experiment view.

**6. Bilingual RAG at scale** — single English knowledge base +
cross-language embedding search (bge-m3) with a fallback OCR chain
(PyMuPDF → marker-pdf → RapidOCR / MinerU) and LOINC normalization.
Chinese medical questions retrieve English guidelines without needing
a parallel corpus.

## Tech Stack

| Layer | Technology |
|---|---|
| Agent runtime | [pydantic-ai](https://ai.pydantic.dev/) 1.0 (Agent + tools + FallbackModel) |
| Local inference | MLX (Apple Silicon), Ollama, llama.cpp, LM Studio |
| Cloud providers | OpenAI, Anthropic, DeepSeek, Moonshot, Alibaba, OpenRouter |
| RAG | Qdrant + BGE-M3 embedder + bge-reranker-v2-m3, LlamaIndex core |
| Symptoms model | typed-BASD (Bayesian Attention Sensor Decoder) + XGBoost baseline on DDXPlus |
| Vision | BiomedCLIP (modality gate), U-Net (segmentation-models-pytorch), Ultralytics YOLO (detection) |
| Storage | SQLite (audit + settings mirror) + YAML (user settings source of truth) |
| Observability | Phoenix (arize-phoenix) + OpenTelemetry OTLP |
| UI | Textual (TUI), FastAPI + JWT cookie auth (Web) |
| PHI scrubbing | OpenAI Privacy Filter (ONNX) + regex pass |
| OCR chain | PyMuPDF → marker-pdf → RapidOCR / MinerU → pypandoc-binary |
| Package management | uv (default-groups: `dev`, `runtime-ocr`) |
| Testing | pytest with `--cov-fail-under=80` + pre-push gates + deepeval e2e |

## Architecture

Data flow at a glance:

```
User input
   │
   ▼
inject_context()  ── ContextVars: user_id, session_id, provider_id
   │
   ▼
Orchestrator (AskService)
   │
   ├─▶ EmergencyTriage pre-gate ──[critical]──▶ Short-circuit reply
   │
   ├─▶ Retrieval (RAG, deterministic)
   │
   ▼
phi_guard  ── if ProviderConfig.kind == "cloud": scrub PHI
   │                                             else: pass through
   ▼
LLMClient.chat() ── pydantic-ai Agent + tools
   │        (symptoms_plugin, vision_plugin, rag_plugin, ...)
   ▼
GroundedAnswer (structured output)
   │
   ▼
Audit log (SQLite) + Phoenix trace
```

Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full
module-level breakdown, and [`CLAUDE.md`](CLAUDE.md) for the
non-negotiable design constraints (PHI invariants, EmergencyTriage
placement, provider resolution order, prompt YAML rules).

## Quickstart

```bash
# 1. Clone and sync
git clone <repo-url> && cd ClarityMed
uv sync                                    # base install (CLI + TUI + runtime OCR)

# 2. Configure environment
cp .env.example .env                       # fill in provider API keys as needed

# 3. Launch TUI
uv run claritymed tui                      # → http://localhost or Textual TUI

# 4. Optional: single-shot ask (no session)
uv run claritymed ask "chest pain for 20 min radiating to left arm"
```

For local-only inference (MLX or Ollama), no API keys are needed —
just point `configs/models.yaml` at your local server.

## Feature-specific extras

Each pipeline that needs heavy ML deps is an optional extra:

```bash
uv sync --extra rag-server           # BGE-M3 + reranker (FlagEmbedding, ~2 GB torch)
uv sync --extra symptoms-server      # typed-BASD + XGBoost training/inference
uv sync --extra medical-clip-server  # BiomedCLIP modality gate
uv sync --extra vision-server        # BUSI / ChestX-ray / RSNA disease detection
uv sync --extra yolo-forge           # Ultralytics detection pipeline
uv sync --extra web                  # FastAPI + JWT cookie auth
uv sync --extra e2e                  # deepeval E2E tests (needs live services)
```

## Development

```bash
# Sync all dev + runtime-ocr groups
uv sync

# Run unit tests with coverage gate (80%)
uv run pytest

# Full pre-commit sweep (gitleaks / ruff / AI bypass / commit-msg length)
uv run pre-commit run --all-files

# Install the pre-push hook (runs full unit + e2e suites before every push)
uv run pre-commit install --hook-type pre-push

# Phoenix prompts sync (push YAML → Phoenix; pull Phoenix → YAML)
uv run claritymed prompts push [NAME] [--dry-run]
uv run claritymed prompts pull [NAME] [--into-new-version]
```

Multi-provider e2e comparison:

```bash
CLARITYMED_E2E_PROVIDERS=omlx,deepseek uv run pytest tests/e2e -v --no-cov
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full contributor
workflow and [`docs/tracing.md`](docs/tracing.md) for Phoenix
observability configuration.

## Project Status

ClarityMed is a research / educational project — not a certified
medical device, not intended for clinical use. All benchmark numbers
are on public datasets (DDXPlus, BUSI, RSNA Pneumonia, ChestX-ray14).
The emergency-triage gate is a *safety net* on top of the LLM path,
not a substitute for professional medical care.

## License

[MIT](LICENSE) — see the LICENSE file for the full text.
