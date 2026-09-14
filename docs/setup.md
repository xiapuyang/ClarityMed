# Setup — clone 到 TUI 跑起来

按层叠加：跑 `tui` 只需 §1–§3；要 RAG / symptoms / vision 等高级功能再按需启对应 sidecar。

---

## 1. 依赖安装

```bash
# 基础环境（含 dev group）
uv sync

# 想用 symptoms / vision / RAG / medical-clip 任一 sidecar，挑对应 extra：
uv sync --extra rag-server          # bge-m3 embedder + bge-reranker-v2-m3（含 FlagEmbedding ~2GB）
uv sync --extra symptoms-server     # typed-BASD 推理 server
uv sync --extra vision-server       # PyTorch + smp（vision 检测）
uv sync --extra medical-clip-server # BiomedCLIP（图像 modality 分类，vision/OCR 的前置门）
uv sync --extra ocr-scanned         # 扫描件 OCR（pikepdf + pdfplumber）
uv sync --extra ocr-office          # docx/pptx/xlsx → markdown
uv sync --extra ocr-image           # 图片 OCR
```

`pre-push` 钩子（一次性安装）：

```bash
uv run pre-commit install --hook-type pre-push
```

---

## 2. 环境变量

`.env` 实际读的是 `~/.claritymed/.env`（`config.py::load_env_file` 从 `CLARITYMED_HOME/.env` 加载），**不是** repo 根。这样 OS clone 出去的代码永远不带 secrets。

```bash
mkdir -p ~/.claritymed
cp .env.example ~/.claritymed/.env
# 按需填写 cloud provider key；只用本地 omlx/ollama 可全部留空
```

最少要填的两类：

| 想用什么 | 必须的 env |
|---|---|
| 本地 omlx（默认 provider） | 无（如果服务端有 bearer auth 才填 `OMLX_API_KEY`） |
| 任何 `cloud` provider | `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY` / `GEMINI_API_KEY` / … 按 `configs/models.yaml` 里启用的项目挑 |
| Phoenix 远程 tracing | `PHOENIX_API_KEY`（本地 `localhost:6006` 不需要） |
| MineRU 解析 PDF/Office | `MINERU_API_TOKEN` |

`configs/models.yaml` 的 `default_provider` 当前是 `omlx`。如果不打算跑本地 MLX server，记得改成你能跑的 provider id。

---

## 3. 初始化 data 目录 + 建用户

```bash
# 建第一个用户（自动提权为 admin），同时创建 data/users/<id>/ 目录骨架
uv run claritymed init-user <your-id> --name "<Display Name>"

# 启 TUI（首次启动前 medical-clip / symptoms / vision sidecar 都不需要——会优雅降级）
uv run claritymed tui -u <your-id>
```

仅跑到这里就能：用本地或 cloud LLM 做单轮对话、写 profile、用 ingest 工具。RAG / 症状预测 / 影像检测都需要继续 §4–§6。

---

## 4. 模型预下载

### 4.1 从 HuggingFace Hub 拉的模型

首次用到时会自动下载到 `~/.cache/huggingface/`（或 `$HF_HOME`）。网差 / 想预热的话先手动拉一次。`huggingface-cli` 随 `huggingface_hub` 一起装，`uv sync` 后就有：

```bash
# RAG embedder（~2 GB）— rag-server extra 必需
uv run huggingface-cli download BAAI/bge-m3

# RAG reranker（~2 GB）— rag-server extra 必需
uv run huggingface-cli download BAAI/bge-reranker-v2-m3

# Medical-CLIP（~400 MB）— vision/OCR modality 前置门
uv run huggingface-cli download microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224

# Symptoms 主诉匹配（~440 MB）— symptoms-server 启动加载
uv run huggingface-cli download cambridgeltl/SapBERT-from-PubMedBERT-fulltext
```

想换缓存盘：`export HF_HOME=/path/to/cache` 后再跑。CN 网络可加 `HF_ENDPOINT=https://hf-mirror.com`。

### 4.2 本地 LLM（按需）

按 `configs/models.yaml` 里启用的 provider 准备：

```bash
# Ollama（默认 ollama 条目里写的是 qwen3:14b）
ollama pull qwen3:14b

# MLX（omlx 条目，默认 Qwen3.6-35B-A3B-oQ4-mtp）—— 用 mlx_lm 或 MLX 自家工具拉
# 自托管 OpenAI-compatible server 自己跑起来即可，repo 不管 serving
```

不打算跑本地 LLM 就改 `configs/models.yaml` 的 `default_provider` 到对应 cloud id（`openai` / `claude` / `deepseek-v4-pro` 等），并在 `~/.claritymed/.env` 填好对应 key。

### 4.3 项目自训练 weights（disease 模型）

放在 `~/.claritymed/models/` 下，由 train pipeline 产出。没训练的 disease 在 `configs/vision.yaml` 里保持 `enabled: false`；强行翻 `true` 但 manifest sha 对不上 → vision-server 启动时 fail-loud。

本节是「装完先用 `--quick` 验证 pipeline」的 onboarding 路径——真数据 + 真 torch，但预算压成 `trials=3 / search_epochs=1 / max_epochs=2`，几分钟出结果。**`--quick` 产出的 artifact 不能上线**（精度未收敛），只用来确认本机 pipeline 没坏。生产训练（`--max-epochs 100 --patience 15`、HPO、promote）走 [`docs/vision-model-workflow.md`](vision-model-workflow.md)。

vision 数据集走 Kaggle，前置：

```bash
uv sync --extra yolo-forge           # kaggle CLI + ultralytics + pydicom
# 放 ~/.kaggle/kaggle.json（chmod 600）或导 KAGGLE_USERNAME / KAGGLE_KEY
# RSNA 还需在 https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/rules 接受规则
```

```bash
# Symptoms（DDXPlus typed-BASD）
#   → ~/.claritymed/models/symptoms/ddxplus/typed_basd_v2/
uv run claritymed-symptoms-prepare-ddxplus
uv run claritymed-symptoms-train-ddxplus

# Vision — 每个 dataset 都先跑一次 --quick 确认 pipeline 通
uv run claritymed-vision-download-busi
uv run claritymed-vision-forge train \
  --model claritymed.ingest.vision.busi.models.unet_resnet50:UNET_RESNET50 \
  --quick
#   → ~/.claritymed/models/vision/breast_cancer_ultrasound/run/<model_id>_<ts>/

uv run claritymed-vision-download-chest-ct
uv run claritymed-vision-forge train \
  --model claritymed.ingest.vision.chest_ct.models.resnet50_v1:RESNET50_V1 \
  --quick
#   → ~/.claritymed/models/vision/lung_cancer_chest_ct/run/<model_id>_<ts>/

uv run python -m claritymed.ingest.vision.chest_xray_pneumonia.download
uv run claritymed-vision-forge train \
  --model claritymed.ingest.vision.chest_xray_pneumonia.models.resnet50_v1:RESNET50_V1 \
  --quick
#   → ~/.claritymed/models/vision/chest_xray_pneumonia/run/<model_id>_<ts>/

uv run python -m claritymed.ingest.vision.rsna_pneumonia.download
uv run claritymed-vision-yolo-forge train \
  --model claritymed.ingest.vision.rsna_pneumonia_yolo.models.yolov8n_v1:RSNA_YOLOV8N_V1 \
  --quick --skip-search
#   → ~/.claritymed/models/vision/rsna_pneumonia_detection/run/<model_id>_<ts>/
```

去掉 `--quick`、配上 `--max-epochs / --patience` 才是真正训练——产出的 `manifest_sha256` 由 forge 写回 `configs/symptoms.yaml` / `configs/vision.yaml`。完整流程（数据准备、调参、promote）见 [`docs/symptoms-model-workflow.md`](symptoms-model-workflow.md) 和 [`docs/vision-model-workflow.md`](vision-model-workflow.md)。

---

## 5. Sidecar 启动（按需）

所有 sidecar 由 `scripts/run.sh` 统一管，loopback-only，日志在 `~/.claritymed/logs/`，pidfile 在 `~/.claritymed/run/`，幂等。

```bash
# 全开
scripts/run.sh

# 按需单开
scripts/run.sh medical-clip   # :8086 — vision/OCR modality 前置门
scripts/run.sh symptoms       # :8084 — 症状预测
scripts/run.sh vision         # :8085 — 影像检测（要 medical-clip 在跑）
scripts/run.sh embedder       # :8082 — RAG（需 §6 Qdrant）
scripts/run.sh reranker       # :8083 — RAG
scripts/run.sh both           # embedder + reranker
```

| 想用的功能 | 必须启的 sidecar |
|---|---|
| 纯对话 / profile / ingest | 无 |
| 上传医学图片让 LLM 看 | `medical-clip` |
| 影像疾病检测工具 | `medical-clip` + `vision` |
| 症状 → 鉴别诊断 | `symptoms` |
| 文献检索 RAG | `embedder` + `reranker` + Qdrant（§6） |

停止 / 重启：`scripts/stop.sh [scope]` / `scripts/restart.sh [scope]`，scope 取值同上。

---

## 6. RAG 数据栈（Qdrant + 语料 ingest）

只在要用 `rag.enabled: true` 时才需要，完整流程详见 [`docs/rag-setup.md`](rag-setup.md)。最小步骤：

```bash
# 1. Qdrant Docker（数据持久化到 ~/.claritymed/shared/qdrant）
docker run -d --name claritymed-qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v ~/.claritymed/shared/qdrant:/qdrant/storage \
  qdrant/qdrant

# 2. Embedder + Reranker（先确认 §1 装了 rag-server extra）
scripts/run.sh both

# 3. 翻 configs/retrieval.yaml 里 rag.enabled: true

# 4. Ingest 一个系统语料
uv run claritymed rag corpora ingest statpearls -u <admin-id>

# 5. 可选：自己的文档
uv run claritymed rag add ./my-note.pdf -u <admin-id>
```

Term service 用本地 UMLS/CMeKG 时还要跑：

```bash
uv run python scripts/init_terminology.py --seed
```

---

## 7. 可选：Phoenix tracing

`configs/app.yaml` 默认 `tracing.enabled: true` 指向 `http://localhost:6006`。本地起 Phoenix：

```bash
uvx arize-phoenix serve   # 或 docker 跑
```

不想要 tracing 直接把 `tracing.enabled` 改 `false`，CI / e2e 也用 false。详见 [`docs/tracing.md`](tracing.md)。

---

## 验证清单

```bash
# 依赖装好
uv run claritymed --help

# 用户建好
ls data/users/<your-id>/

# Sidecar 健康
curl -fsS http://127.0.0.1:8086/health   # medical-clip
curl -fsS http://127.0.0.1:8084/health   # symptoms
curl -fsS http://127.0.0.1:8085/health   # vision
curl -fsS http://127.0.0.1:8082/health   # embedder
curl -fsS http://127.0.0.1:8083/health   # reranker
curl -fsS http://localhost:6333/collections   # qdrant

# 跑通最小路径
uv run claritymed ask "hello" -u <your-id>
uv run claritymed tui -u <your-id>
```
