# ClarityMed

**本地优先的医疗 AI 助手。** ClarityMed 把涉及 PHI 的推理留在本地，
在任何 LLM 组装回复前先跑一道确定性的紧急分诊 gate，通过一个统一的
pydantic-ai agent loop 编排多模态诊断（症状预测、影像识别、患者笔记
RAG）。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](pyproject.toml)
[![Coverage ≥80%](https://img.shields.io/badge/coverage-%E2%89%A580%25-brightgreen)](pyproject.toml)
[![Powered by pydantic-ai](https://img.shields.io/badge/agent-pydantic--ai%201.0-8a2be2)](https://ai.pydantic.dev/)

[English →](README.md)

---

## 主要指标

以下每个数字都来自 repo 里 check-in 的评测结果——不是营销数据。

| 指标 | 结果 | 来源 |
|---|---|---|
| Symptoms 3-class 流感 F1 | **0.904** | `typed_basd_pne_flu_other_v1` checkpoint |
| Symptoms 3-class Macro F1 | **0.735** | 同上 |
| Symptoms Brier R²（vs XGBoost baseline）| **+0.52**（vs +0.17）| 校准度显著提升 |
| Emergency Triage Critical Recall | **0.958**（目标 ≥0.95）| `balanced` profile, hold-out set |
| Emergency Triage 对抗 FPR | **0.014**（目标 <0.15）| 同上 |
| Emergency Triage F-β=2 | **0.965** | 同上 |
| Session-start 延迟 (p50) | **81 ms** | `ddxplus_pneumonia_flu` bench，本地 MLX |
| Per-turn 延迟 (p50) | **130 ms** | 同上 |
| 本地吞吐（MLX Qwen3-35B-A3B-Q4）| **40–60 tok/s** | Apple Silicon M 系列，~18 GB 内存 |
| 流感 F1 vs XGBoost 基线 | **0.904 vs 0.526**（+72%）| typed_basd vs xgb_v5_recallig |
| 肺炎 precision | **1.00** | 新 3-class 模型，零误报 |

## 技术创新

**1. PHI-safe 云端路由** —— 通过单一字段（`ProviderConfig.kind`）
决定 PHI 是否出域。只有 `local` 的 provider 才能看到 PHI；`cloud`
调用必须先过 `phi_guard` + 组装后的 prompt 二次 scrub。**没有** per-user
"云端 opt-in" 标志需要记得设置——gate 就是不变量，每次 LLM 调用要么
走 gate，要么根本跑不通。

**2. 确定性紧急分诊作为 pre-agent 安全门** —— STEMI、过敏性休克、
异位妊娠、活跃性自杀意念等 critical 情况直接短路 agent loop：本地
LLM extractor + 规则引擎在 <200 ms 内组合出 i18n 的 action reply，
而不是让 agent 花 3–5 s 组装消息。规则作者通过
`minimum_sensitivity_floor` 保留 veto 权 —— 任何运营者 profile 都
不能弱化 anaphylaxis / active-SI 的检测。

**3. 基于 pydantic-ai 1.0 的多 provider 抽象** —— 8+ 个 provider
（OpenAI、Anthropic、DeepSeek、Moonshot、Alibaba、OpenRouter、Ollama、
MLX / llama.cpp / LM Studio）通过 YAML 配置统一。推理档位通过单一
`thinking` 字段传递（`minimal`/`low`/`medium`/`high`/`xhigh`），
pydantic-ai 按 vendor 自己翻译。新加一个本地后端只需一条 YAML，
零 Python 改动。

**4. Prompts-as-YAML + Phoenix 同步** —— `core/prompts/store/*.yaml`
是运行时唯一真相；Phoenix (arize-phoenix) 只是**编辑 UI + eval 平台**，
永远不在 runtime 路径上。`claritymed prompts push/pull` 是唯一
桥梁；版本历史不可变，双语（en/zh）在启动期 fail-fast 校验。

**5. 多模态 per-disease 视觉注册表** —— YAML 驱动的 hot-swap：
BUSI 乳腺超声、ChestX-ray14 胸片、RSNA Pneumonia 检测、chest-CT
分类。加一个新病种/模型只需 append 一条 YAML；per-disease 训练
pipeline（forge / yolo_forge）共享一个 MLflow experiment 视图。

**6. 规模化双语 RAG** —— 单一英文知识库 + 跨语言 embedding 检索
（bge-m3）+ fallback OCR 链（PyMuPDF → marker-pdf → RapidOCR / MinerU）
+ LOINC 归一。中文医疗问题能检索英文指南，无需平行语料。

## 技术栈

| 层 | 技术选型 |
|---|---|
| Agent 运行时 | [pydantic-ai](https://ai.pydantic.dev/) 1.0（Agent + tools + FallbackModel）|
| 本地推理 | MLX（Apple Silicon）、Ollama、llama.cpp、LM Studio |
| 云端 provider | OpenAI、Anthropic、DeepSeek、Moonshot、Alibaba、OpenRouter |
| RAG | Qdrant + BGE-M3 embedder + bge-reranker-v2-m3、LlamaIndex core |
| 症状模型 | typed-BASD（Bayesian Attention Sensor Decoder）+ XGBoost baseline，DDXPlus 语料 |
| 视觉 | BiomedCLIP（模态识别）、U-Net（segmentation-models-pytorch）、Ultralytics YOLO（检测）|
| 存储 | SQLite（audit + settings mirror）+ YAML（user settings 真相源）|
| 观测 | Phoenix (arize-phoenix) + OpenTelemetry OTLP |
| UI | Textual（TUI）、FastAPI + JWT cookie 认证（Web）|
| PHI scrubbing | OpenAI Privacy Filter (ONNX) + regex 兜底 |
| OCR 链 | PyMuPDF → marker-pdf → RapidOCR / MinerU → pypandoc-binary |
| 依赖管理 | uv（default-groups: `dev`, `runtime-ocr`）|
| 测试 | pytest `--cov-fail-under=80` + pre-push gate + deepeval e2e |

## 架构

数据流一览：

```
User input
   │
   ▼
inject_context()  ── ContextVars: user_id, session_id, provider_id
   │
   ▼
Orchestrator (AskService)
   │
   ├─▶ EmergencyTriage pre-gate ──[critical]──▶ 短路回复
   │
   ├─▶ Retrieval（RAG，确定性）
   │
   ▼
phi_guard  ── ProviderConfig.kind == "cloud" ? scrub PHI : 直通
   │
   ▼
LLMClient.chat() ── pydantic-ai Agent + tools
   │        （symptoms_plugin、vision_plugin、rag_plugin、...）
   ▼
GroundedAnswer（结构化输出）
   │
   ▼
Audit log (SQLite) + Phoenix trace
```

模块级细节见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)；不可协商的
设计约束（PHI 不变量、EmergencyTriage 摆放位置、provider 解析顺序、
prompt YAML 规则）在 [`CLAUDE.md`](CLAUDE.md)。

## 快速开始

```bash
# 1. 克隆并同步
git clone <repo-url> && cd ClarityMed
uv sync                                    # 基础安装（CLI + TUI + runtime OCR）

# 2. 配置环境
cp .env.example .env                       # 按需填入 provider API keys

# 3. 启动 TUI
uv run claritymed tui

# 4. 可选：单次问答（无 session）
uv run claritymed ask "胸痛 20 分钟放射至左臂"
```

如果只用本地推理（MLX 或 Ollama），不需要任何 API key ——
把 `configs/models.yaml` 指向本地服务即可。

## 功能相关 extras

需要重量级 ML 依赖的 pipeline 都是 optional extra：

```bash
uv sync --extra rag-server           # BGE-M3 + reranker（FlagEmbedding, ~2 GB torch）
uv sync --extra symptoms-server      # typed-BASD + XGBoost 训练/推理
uv sync --extra medical-clip-server  # BiomedCLIP 模态识别
uv sync --extra vision-server        # BUSI / ChestX-ray / RSNA 病种检测
uv sync --extra yolo-forge           # Ultralytics 检测 pipeline
uv sync --extra web                  # FastAPI + JWT cookie 认证
uv sync --extra e2e                  # deepeval E2E 测试（需要 live services）
```

## 开发

```bash
# 同步所有 dev + runtime-ocr 组
uv sync

# 跑单测 + 覆盖率门槛（80%）
uv run pytest

# 全量 pre-commit（gitleaks / ruff / AI bypass / commit-msg 长度）
uv run pre-commit run --all-files

# 安装 pre-push hook（每次 push 前跑全量 unit + e2e）
uv run pre-commit install --hook-type pre-push

# Phoenix prompts 双向同步（YAML → Phoenix / Phoenix → YAML）
uv run claritymed prompts push [NAME] [--dry-run]
uv run claritymed prompts pull [NAME] [--into-new-version]
```

多 provider e2e 对比：

```bash
CLARITYMED_E2E_PROVIDERS=omlx,deepseek uv run pytest tests/e2e -v --no-cov
```

完整 contributor 流程见 [`CONTRIBUTING.md`](CONTRIBUTING.md)，
Phoenix 可观测性配置见 [`docs/tracing.md`](docs/tracing.md)。

## 项目状态

ClarityMed 是研究 / 教学项目 —— **不是** 已认证的医疗器械，
**不用于临床**。所有 benchmark 数字都基于公开数据集（DDXPlus、BUSI、
RSNA Pneumonia、ChestX-ray14）。紧急分诊 gate 是 LLM 路径**之上的
安全网**，不能替代专业医疗建议。

## 许可证

[MIT](LICENSE) —— 完整条款见 LICENSE 文件。
