# 硬件性能要求评估

面向 **单机本地部署**（macOS Apple Silicon 或 Linux/NVIDIA）。所有推理服务默认 loopback (`127.0.0.1`) 绑定，PHI 出域只由 `ProviderConfig.kind` 门控。

## TL;DR — 三档配置

| 档位 | CPU | 统一/独立内存 | 磁盘 | 覆盖 |
|---|---|---|---|---|
| **Baseline (最低)** | 8 核 | 16 GB | 60 GB | 关掉 vision + medical-clip + 用 `qwen3:14b` Q2 |
| **Recommended (当前默认)** | 10–12 核 | **32 GB** VRAM/统一内存 | 100 GB | Qwen3.6-35B-A3B Q4 + 全部 vision/symptoms/embedder 常驻 |
| **Comfort** | 16 核 | 64 GB | 200 GB | 上述 + Q6/Q8 权重、多个 vision 模型热常驻、Phoenix 本地全量 trace |

**关键判断**：当前默认栈 (`omlx: Qwen3.6-35B-A3B-oQ4-mtp` + BGE-M3 + 5 个 vision 模型 + BiomedCLIP + SapBERT + XGBoost) 在 M-series 32 GB 统一内存机器上稳定跑得动，前提是**按需加载**——vision 模型不常驻。要进一步压 → 走 [2-bit / 4-bit 降级方案](#2-bit-与-4-bit-降级)。

---

## 常驻服务清单

app 启动后可能出现的进程（每个都是独立端口 + 独立进程，方便按需关停）：

| 服务 | 端口 | 依赖 | 是否可选 |
|---|---|---|---|
| MLX / omlx LLM | 8000 | `mlx_lm.server` (外部) | ✅ 默认主 provider |
| Ollama LLM | 11434 | `ollama serve` (外部) | ⚪ 备选/fallback |
| Embedder (BGE-M3) | 8082 | `--extra rag-server` | ✅ RAG 必需 |
| Reranker (BGE-reranker-v2-m3) | 8083 | `--extra rag-server` | ✅ RAG 必需 |
| Symptoms (XGBoost + SapBERT) | 8084 | `--extra symptoms-server` | 🟡 症状交互功能需要 |
| Vision (ResNet50 / U-Net ×5) | 8085 | `--extra vision-server` | 🟡 影像检测需要 |
| Medical-CLIP (BiomedCLIP) | 8086 | `--extra medical-clip-server` | 🟡 vision 前置模态门控 |
| Qdrant | 6333 | Docker (system) + `qdrant-client.local` (per-user) | ✅ RAG 必需 |
| Phoenix | 6006 | `arize-phoenix` (Docker 或 `uvx`) | ⚪ 可选，`configs/app.yaml` `tracing.enabled=false` 即完全跳过 |
| Textual TUI | — | in-process | ✅ 入口 |

`configs/models.yaml:44-125` 是 provider 目录；`default_provider: omlx`。EmergencyTriage 强制 `kind: local`（`core/emergency/_provider.py`），从 `models.yaml` 里第一个 `kind: local` provider 抓凭据——即用 `omlx`。

---

## 逐服务资源画像

### 1. LLM (最大头)

| provider | 模型 | 量化 | 权重体积 | 加载时 VRAM | 备注 |
|---|---|---|---|---|---|
| `omlx` (默认) | `Qwen3.6-35B-A3B-oQ4-mtp` | Q4 | ~18 GB | ~20 GB | 35B 总参 / 3B 激活 (MoE)，MLX 后端 |
| `ollama` (备选) | `qwen3:14b` | Q4_K_M | ~8 GB | ~10 GB | 稠密 14B，可 Q2 压到 ~5 GB |

**吞吐**（M4 Pro 参考量级，未在本仓库正式 bench）：
- Qwen3.6-35B-A3B-Q4 因为 MoE 激活只 3B，实际每 token 计算量接近 3B 稠密模型，40-60 tok/s
- 稠密 14B-Q4 大约 20-30 tok/s

**降级路径**：
- MLX 服务端可用 `mlx_lm.convert --hf-path <repo> -q --q-bits 2` 生成 Q2 变体；35B-Q2 磁盘 ~10 GB / VRAM ~12 GB，精度损失明显但能跑
- Ollama：`ollama pull qwen3:14b-q2_K` 或 `qwen2.5:7b-q4_K_M` 更小
- 关键前提：**EmergencyTriage 的 extractor+composer 都要走这个 provider**，太差的模型会漏红旗——不建议 EmergencyTriage 用 <7B 或低于 Q4 的权重

### 2. Embedder — BGE-M3

- 磁盘：~2.2 GB (`~/.claritymed/models/bge-m3`)
- 内存：FP32 加载约 **1.5-2 GB**（配置里 `batch_size: 64`，`dense_dim: 1024`）
- 设备：`configs/retrieval.yaml:85-94`，默认 auto (CPU/MPS/CUDA)
- **常驻**：启动后一直在 8082 端口挂着
- 降级：`sentence-transformers/all-MiniLM-L6-v2` (~80 MB / 300 MB RAM)，但**会掉召回**——BGE-M3 的 hybrid dense+sparse 是当前 retrieval 质量的基线，不建议动

### 3. Reranker — BGE-reranker-v2-m3

- 磁盘：~2.3 GB
- 内存：~3× embedder（cross-encoder 每对 query-doc 独立算），`batch_size: 32`
- 常驻在 8083
- 降级：`BAAI/bge-reranker-base` (~275 MB) 精度略降但显著省内存

### 4. Vision Server (ResNet50 / U-Net ×5)

`configs/vision.yaml:31-207` 声明了 6 个 disease，5 个 enabled：

| disease | 模型 | 磁盘 | 加载时 RAM |
|---|---|---|---|
| breast_cancer_ultrasound | `breast_busi_unet_v1` | ~100 MB | ~500 MB |
| lung_cancer_chest_ct | `lung_chest_ct_resnet50_v1` | ~100 MB | ~500 MB |
| skin_cancer_dermoscopy | `skin_isic_resnet50_v1` | ~100 MB | ~500 MB |
| lung_cancer_histopathology | `lung_histopath_resnet50_v1` | ~100 MB | ~500 MB |
| colon_cancer_histopathology | `colon_histopath_resnet50_v1` | ~100 MB | ~500 MB |

- **按需加载**：只有当 medical-clip 判为对应模态才加载，进程内 LRU
- 图像限制：10 KB–20 MB，224–4096 px（`configs/app.yaml:114-122`）
- 无量化空间：ResNet50 已经很小，Q4 收益不明显

### 5. Medical-CLIP — BiomedCLIP

- 模型：`microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`
- 磁盘：~400 MB (FP16)
- RAM：~600 MB 常驻
- 端口 8086，`device: auto`（`configs/medical_clip.yaml:16-18`）
- 零可选量化——上游只发 FP16

### 6. Symptoms Server — XGBoost + SapBERT

- **两部分**：
  - SapBERT (`cambridgeltl/SapBERT-from-PubMedBERT-fulltext`)：~440 MB 磁盘，~500 MB CPU RAM，只做 16 症状初筛
  - XGBoost `xgb_pne_inf_v5_recallig`：~30 MB，纯 CPU
- 端口 8084
- Session TTL 30 min，最多 18 轮问答

### 7. Qdrant Vector DB

- **两个部署面**：
  - **System collections**（statpearls / textbooks / ATS-IDSA guidelines）→ Docker `http://localhost:6333`，或远程 `CLARITYMED_QDRANT_URL`
  - **Per-user collections** → `qdrant-client.local`，落在 `~/.claritymed/data/users/<uid>/rag/qdrant/`
- 磁盘增长：每用户 ~10-50 MB（1000 chunks 量级）
- 内存：进程内 ~100-300 MB，远端 Docker 独立 500 MB-1 GB

### 8. Phoenix Observability

- 完全可选，`configs/app.yaml` `tracing.enabled` 手动切换
- 关掉后 no-op（trace 直接丢），**不影响任何功能**
- 开启：本地 Docker 或 `uvx arize-phoenix serve`，进程 ~500 MB RAM
- CI / 离线 session 一律关

### 9. TUI + In-process 模型

- Textual app + Privacy Filter (ONNX, ~3 MB) + Magika (~1 MB) → **~250 MB**
- 无 LLM 权重

---

## 组合内存估算（RSS 峰值近似）

假设"用户开着 TUI，正在问一个带 CT 图片的问题"这个典型场景：

```
LLM (omlx MLX)         20 GB   ← MoE 激活期间峰值
BGE-M3 embedder         1.8 GB
BGE reranker            0.8 GB  ← 首次问答后加载
BiomedCLIP              0.6 GB
Vision (1 个模型热)     0.5 GB
Symptoms (未触发)         0 GB  ← 走 symptoms plugin 才加载
Qdrant (local client)   0.3 GB
Textual + Python        0.3 GB
Phoenix (若开)          0.5 GB
─────────────────────────────
合计                  ~24-25 GB
```

**结论**：M4 Pro 32 GB / M3 Max 36 GB 是当前默认栈的**甜点位**。24 GB 机器要么关 vision 要么把 LLM 换到 14B/Q4。

---

## 磁盘占用

```
~/.claritymed/models/
├── mlx-community/Qwen3.6-35B-A3B-oQ4-mtp   ~18 GB
├── bge-m3                                    ~2.2 GB
├── bge-reranker-v2-m3                        ~2.3 GB
├── BiomedCLIP-.../                           ~0.4 GB
├── SapBERT-.../                              ~0.44 GB
└── vision/{5 diseases}/                      ~0.5 GB
                                              ─────
                                              ~24 GB
~/.claritymed/data/users/<uid>/               ~10-50 MB / 用户
Qdrant Docker volume (system RAG)             ~2-5 GB
logs/                                         增长快，audit + trace，建议 rotate
```

用户备份、Ollama 的备选权重、`marker-pdf` 之类的可选 extra 会再吃 5-15 GB。100 GB SSD 是舒服区间。

---

## GPU / 加速器

| 平台 | 推荐规格 | 说明 |
|---|---|---|
| macOS | **M3 Pro / M4 Pro 32 GB 起**，M-series Max 更佳 | 统一内存架构，MLX 直接吃系统内存；Q4 35B 需要 ≥20 GB 可寻址 |
| Linux + NVIDIA | RTX 4090 24 GB / RTX 5090 32 GB | Ollama + CUDA；35B-Q4 刚好塞下 24 GB，vision/embedder 挤主显存 → 建议 32 GB |
| Linux CPU-only | 不推荐 | 14B-Q4 都能跑但延迟 >5 s / token，交互不可用 |

**注意**：medical-clip / vision / embedder 都 `device: auto`——自动检测 CUDA / MPS / CPU。M-series 上会自动走 MPS。

---

## 2-bit 与 4-bit 降级

用户明确问了这条。现状与选项：

**当前状态**：主 LLM 已经是 **Q4**（`Qwen3.6-35B-A3B-oQ4-mtp` 的 `oQ4` 后缀），embedder/reranker/vision/CLIP 都是 FP16 或 FP32（这些模型本身参数少，量化收益小、精度损失大，**不建议**再压）。

**LLM 进一步降级路径（若 32 GB 不够）**：

1. **换更小的 Q4 稠密模型**（推荐）
   - Ollama: `qwen3:14b`（默认已经在 models.yaml 里）→ ~8 GB VRAM
   - Ollama: `qwen2.5:7b-instruct-q4_K_M` → ~5 GB VRAM
   - 优点：稠密模型无 MoE 路由开销，小机器上更稳
   - 缺点：EmergencyTriage 的 extractor 需要靠模型理解长对话历史里的红旗症状，7B 模型漏报率会升——**先在 `tests/e2e/` 里跑 emergency e2e 再切**

2. **把当前 35B 压到 Q2**（不推荐）
   ```bash
   mlx_lm.convert --hf-path <original-repo> \
     --mlx-path ~/.claritymed/models/Qwen3.6-35B-A3B-Q2 \
     -q --q-bits 2
   ```
   - 磁盘 ~10 GB / VRAM ~12 GB
   - 精度损失显著（Perplexity 掉 15-30%），medical 场景下 hallucination 上升，**只在硬件极度受限时考虑**
   - 用完记得在 `models.yaml` 里加个新 provider entry，别覆盖 omlx

3. **切换 provider 到 cloud**（PHI 会走 phi_guard scrub）
   - 把 `default_provider` 换成 `deepseek-flash` / `gemini-flash-lite`
   - 本地 0 GB VRAM，但 EmergencyTriage 仍然强制 local——最低还是要一个能跑 7B 的本地 provider

**降级检查清单**：
- [ ] `uv run pytest tests/e2e/test_emergency_gate.py` 全绿
- [ ] 在 `docs/spikes/` 记一份 before/after 的 latency + accuracy 对比
- [ ] `models.yaml` 里新 entry 而不是覆盖旧 entry（`CLAUDE.md` 里 "Format / Schema Upgrades" 那条规则）

---

## 可关停项 (省资源清单)

按"用户不会立刻察觉"的顺序：

1. **Phoenix** → `configs/app.yaml` `tracing.enabled: false`，省 ~500 MB
2. **Reranker** → 只用 dense retrieval，`configs/retrieval.yaml` 里关掉 reranker，省 ~800 MB（召回质量下降）
3. **Vision + medical-clip** → 不装 `vision-server` / `medical-clip-server` extra，省 ~1.5 GB + 磁盘 900 MB（丢掉影像检测）
4. **Symptoms** → 不装 `symptoms-server` extra，省 ~500 MB（丢掉症状交互）
5. **Marker/RapidOCR** → 不装 `ocr-scanned` / `ocr-image` extra，省磁盘几个 GB（扫描件走 LLM 兜底，慢但可用）

极简 baseline：只保留 LLM (7B-Q4) + BGE-M3 + Qdrant → **~9 GB RAM，能做 chat + RAG**，其他全砍。

---

## 下一步（如果要精确 profiling）

本仓库目前**没有一份实测 memory/CPU/latency 报告**（`docs/benchmarks/` 只有 `tool_invoke.md` 和 `cross_dataset_drift/`）。要落到"哪台机器精确 XX 秒 / XX GB"这个层级，需要：

```bash
# LLM: MLX
python -c "import mlx.core as mx; ..."  # 或用 mlx_lm.server 的 --log-metrics

# 各 server 内存
ps -o rss,command -p <pid>  # 或 psutil.Process().memory_info()

# 端到端 latency
uv run claritymed bench ...  # scripts/ 下有若干 bench 脚本
```

建议把结果落到 `docs/benchmarks/hardware-profile-<machine>.md`，跟本文档配套。
