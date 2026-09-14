# Changelog

All notable changes to this project are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Nothing yet.

## [0.1.0] — 2026-09-14

First open-source release. Consolidates ~360 commits of feature work
onto `main` from the development branch.

### Added

- **PHI-safe orchestrator** — single-field cloud/local gate
  (`ProviderConfig.kind`), `phi_guard` pre-scrub on cloud-bound
  prompts, and audit-log correlation with Phoenix traces.
- **Deterministic emergency triage gate** — local-LLM extractor +
  rule-based composer runs as a pre-step in `AskService.handle`.
  Critical conditions (STEMI, anaphylaxis, ectopic pregnancy,
  active SI) short-circuit the agent loop with sub-200ms i18n
  replies. `minimum_sensitivity_floor` gives rule authors veto
  power against operator profile downgrades.
- **Symptoms plugin (typed-BASD)** — three-class differential
  (pneumonia / influenza / other) trained on DDXPlus. Influenza
  F1 = 0.904, Macro F1 = 0.735, Brier R² = +0.52 vs XGBoost baseline
  (+0.17). Companion XGBoost pipeline for baseline comparisons.
- **Vision plugin + per-disease registry** — YAML-driven hot-swap
  for BUSI ultrasound, ChestX-ray14 classification, RSNA Pneumonia
  detection, chest-CT. Shared MLflow experiment view across
  classification `forge` and detection `yolo_forge` pipelines.
- **Bilingual RAG** — Qdrant + BGE-M3 embedder + bge-reranker-v2-m3.
  OCR fallback chain (PyMuPDF → marker-pdf → RapidOCR / MinerU) +
  LOINC normalization. Cross-language retrieval (Chinese question →
  English guideline).
- **Multi-provider abstraction via pydantic-ai 1.0** — 8+ providers
  through YAML config, unified `thinking` reasoning-effort field.
- **Prompts-as-YAML with Phoenix sync** — `core/prompts/store/*.yaml`
  as runtime source of truth; `claritymed prompts push/pull` for
  Phoenix round-trip.
- **Skills for Claude Code** — `import-medical-record` conversational
  driver over the `claritymed record` CLI (bulk import, OCR, resume).
- **Web interface** — FastAPI + JWT cookie auth + admin UI (Vite SPA).
- **Textual TUI** — multi-card symptom render, session attachments,
  multilingual (en/zh) profile-aware output.

### Documentation

- English + Chinese README with quantitative highlights and
  architecture flow.
- `docs/ARCHITECTURE.md` module map.
- `CLAUDE.md` with non-negotiable design constraints (PHI invariants,
  EmergencyTriage placement, provider resolution, prompt rules).
- `docs/tracing.md`, `docs/emergency-gate-eval.md`,
  `docs/hardware-sizing.md`.

### Infrastructure

- Pre-commit hooks: gitleaks, ruff, AI-bypass pattern detector,
  WIP-commit blocker, 72-char subject enforcement, framework-file
  pytest gate, bench-cases revision gate.
- Pre-push hooks: full unit test + e2e test suites.
- GitHub Actions CI: `uv sync` → pre-commit → pytest with 80%
  coverage floor + diff-coverage ≥80% on PR changed lines.

[Unreleased]: https://github.com/xiapuyang/ClarityMed/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/xiapuyang/ClarityMed/releases/tag/v0.1.0
