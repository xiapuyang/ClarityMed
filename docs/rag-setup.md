# RAG 子系统启动手册 — Qdrant / Embedder / Reranker

把 `rag.enabled: true` 翻上来之前需要三个外部依赖跑起来：

1. **Qdrant** — 向量库（dense + sparse 混检 + RRF 融合）。**System collections（StatPearls 等公开知识）走 Docker server**（§1）；**user_rag（per-user PHI）走 local 模式**，每个 user 独立 SQLite 在 `data/users/<id>/qdrant/`（§1.6）。这样公开数据享受并发访问，PHI 在 OS file-level 隔离。
2. **Embedder** — `bge-m3` HTTP server，提供 `/embed` + `/embed_sparse`。
3. **Reranker** — `bge-reranker-v2-m3` HTTP server，提供 `/rerank`。

**Canonical 路径：** Embedder / Reranker 跑自带的两个 FastAPI 进程
（`claritymed-embedder` + `claritymed-reranker` console scripts，源码在
`src/claritymed/servers/`，包 `FlagEmbedding`）。零适配、原生 ARM、不需要 Docker。
两个进程通过 `scripts/rag/{run,stop,restart}.sh` 一键管理（详见 §4）。
TEI Docker 是 Linux x86_64 上的备选，详见 §2 / §3。

三者都跑起来后，把 `configs/retrieval.yaml` 的 `rag.enabled` 改成 `true`，CLI / TUI 启动时会构造 `HybridRetriever` + `NaiveHybridStrategy` 并注入到 `AskService`。任何依赖缺失立刻 fail-loud（`EmbedderUnreachableError` / `RerankerUnreachableError` / qdrant 异常），**不会**静默回落到 LLM-only。

---

## 0. 前置：硬件 / 软件清单

| 项 | 要求 |
|---|---|
| 平台 | macOS (Apple Silicon) / Linux x86_64 / Linux ARM64 |
| Python | 3.11+（项目本身要求） |
| Docker | **必需** — Qdrant server 跑容器（§1）；Embedder / Reranker canonical 路径仍然是原生 Python，不要 Docker |
| 内存 | bge-m3 + reranker 加载到内存各约 2.5 GB；同时跑两个 server 留 8 GB 余量 |
| 磁盘 | Qdrant 数据约 0.5 GB / 万 chunks；模型权重共 ~4.5 GB |
| 网络 | 首次下载模型权重需通外网 |

Qdrant 必须以独立 server 进程跑（容器 / 二进制 / Cloud）。代码端通过 `build_qdrant_client(url=...)` 连上去，没有本地文件锁。

---

## 1. Qdrant — server-only

### 1.1 起 Docker server

```bash
# 1. 启动容器（数据落到 ~/.claritymed/shared/qdrant，重启不丢）
docker run -d --name claritymed-qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v ~/.claritymed/shared/qdrant:/qdrant/storage \
  qdrant/qdrant

# 2. 验证 server up
curl -fsS http://localhost:6333/collections | python -m json.tool
# 期望: {"result":{"collections":[...]},"status":"ok",...}
```

**默认连接：** `configs/retrieval.yaml` 的 `qdrant.url` 默认就是 `http://localhost:6333`，无需任何额外配置。指向远程 / 不同端口时三选一：

```bash
# A. 临时
export CLARITYMED_QDRANT_URL=http://qdrant.staging:6333

# B. 永久（项目 loader 会自动读 ~/.claritymed/.env）
echo 'CLARITYMED_QDRANT_URL=http://qdrant.staging:6333' >> ~/.claritymed/.env

# C. 改 configs/retrieval.yaml 的 qdrant.url
```

**容器管理：**

```bash
docker stop  claritymed-qdrant    # 数据保留
docker start claritymed-qdrant
docker logs  claritymed-qdrant -f
docker rm -f claritymed-qdrant    # 删容器（数据仍在挂载目录）
```

### 1.2 Qdrant Cloud（带鉴权）

```yaml
# configs/retrieval.yaml
qdrant:
  url: "https://xxx.aws.cloud.qdrant.io"
  api_key_env: "QDRANT_CLOUD_KEY"   # 必须设对应 env，否则 MissingApiKeyError 启动就抛
```

```bash
export QDRANT_CLOUD_KEY=xxx
```

> **PHI 数据严禁上 SaaS endpoint** —— 本地 Docker 才能跑 PHI。

### 1.3 Ingest 进 server

```bash
# 前置：embedder 在跑（§2）+ Docker 容器在跑
curl -fsS http://127.0.0.1:8082/health   # {"status":"ok"}
curl -fsS http://localhost:6333/collections | python -m json.tool

# 跑 ingest（支持 resume — 中断后再跑会跳过已写入的 doc_id）
uv run claritymed rag corpora ingest statpearls --user admin

# 跑完验证
curl -fsS http://localhost:6333/collections/statpearls_en | python -m json.tool
# 期望: points_count > 0
```

### 1.4 从旧的 local-mode 数据迁移过来？

如果项目早期跑过 local 模式（`~/.claritymed/shared/qdrant/collection/<name>/storage.sqlite`），那份 sqlite 在 server 模式下读不出来。两个选项：

| 选项 | 命令 | 时间 |
|---|---|---|
| **迁移已算好的 vectors** | `uv run python scripts/migrate_qdrant_local_to_server.py` | 几分钟（跳过 embedder） |
| **重 ingest** | `uv run claritymed rag corpora ingest statpearls --user admin` | 30-60 分钟（重算 BGE-M3） |

迁移完成 + 验证 ingest 流程后回收 local 残留：

```bash
rm -rf ~/.claritymed/shared/qdrant/collection
rm  -f ~/.claritymed/shared/qdrant/.lock
rm -rf ~/.claritymed/shared/qdrant/aliases
```

### 1.5 注意事项

- **端口冲突**：`6333` (HTTP) / `6334` (gRPC) 已被占就改 `-p 7333:6333` 之类，对应 `CLARITYMED_QDRANT_URL=http://localhost:7333`。
- **server 必须先起来 CLI 才能用 RAG**。容器没跑、yaml `rag.enabled: true` 的话，第一次 ask 会在 retrieval 那步抛 connection refused。要么先 `docker start claritymed-qdrant`，要么 `rag.enabled: false` 临时跳过 RAG。

### 1.6 user_rag — local-only

跟 system collections **不一样**：user_rag 走 `qdrant-client` 的 local 文件模式，**不**进 Docker server。

```
data/users/<user_id>/qdrant/storage.sqlite   ← 每 user 一个独立 SQLite
data/users/<user_id>/parent_docstore.json    ← 每 user 一个 parent 文本
```

**为什么 split：**
- **PHI 隔离**：user 上传的文档可能包含 PHI（病历、化验单等），物理上不进 server 命名空间，OS file-mode bits 直接守住跨 user 边界。System 是公开知识，可以共享。
- **代码 bug 防护**：路径从 `user_id` 派生（`user_root(user_id) / "qdrant"`），写错了根本拿不到其他 user 的 client。
- **故障域**：server 崩了不影响 user_rag 上传/查询；user 的 SQLite 坏了只影响那一个 user。

**唯一代价（已知接受）：** 同一 user 不能并发 ——
- 同 user 同时跑 `claritymed rag add` + TUI → 文件锁冲突，CLI 那边拿不到锁报错
- 跨 user 没事（不同路径不同锁）
- 实际场景几乎不会触发：上传文档时一般也没在问问题

**目录结构由代码自动建**，user 第一次 `rag add` 时创建。没建之前 retrieval 自动 fall-back 到 system-only，不报错。

```bash
# 添加一个 user 文档（local-mode 写入，server 不参与）
uv run claritymed rag add ./my-medical-note.md --user alice

# 验证
ls ~/.claritymed/data/users/alice/qdrant/    # 应该看到 collection/ aliases/ .lock
```

### 1.7 日常工作流

环境装好（§1.1 + §2 + §3）之后日常就这四步：

```bash
# 1. 启 system Docker server（开机一次，之后 docker 自启）
docker start claritymed-qdrant   # 首次用 docker run ... 创建（见 §1.1）

# 2. Ingest 公开知识（写 server；通常只跑一次 + 偶尔追加）
uv run claritymed rag corpora ingest statpearls --user admin

# 3. 上传 user 私人文档（写 local；按需）—— 跟 server 完全独立
uv run claritymed rag add ./my-blood-test.pdf --user alice

# 4. 问问题（同时查 server + alice 的 local）
uv run claritymed tui --user alice
```

**并发是否 OK：**

| 场景 | 结论 |
|---|---|
| System ingest 跑着 + alice 用 TUI 问问题 | ✅ 不冲突（server 多 client 并发 OK） |
| Alice 和 bob 同时用各自 TUI | ✅ 不冲突（不同 user_rag 文件路径不同锁） |
| Alice 跑 `rag add` 同时她自己开着 TUI | ❌ 文件锁冲突（user_rag SQLite 锁是 exclusive；rare 场景，§1.6 已说明）|
| Alice 同时开两个 TUI | ❌ 同上 — 一个会拿不到锁。先关一个再开 |

---

## 2. Embedder — bge-m3 HTTP server

代码端只认两个 endpoint：

| 方法 | 路径 | 入参 | 出参 |
|---|---|---|---|
| `POST` | `/embed` | `{"inputs": ["...", ...]}` | `[[float, ...], ...]` (1024-dim) |
| `POST` | `/embed_sparse` | `{"inputs": ["...", ...]}` | `[{"<token_id>": weight, ...}, ...]` 或 `[{"indices": [...], "values": [...]}]` |

推荐**原生 Python server**（源码在 `src/claritymed/servers/embedder.py`）—
这是 canonical 路径。HuggingFace **text-embeddings-inference** (TEI) 是
Linux x86_64 上的备选，但 Apple Silicon 上同时有三个已知阻塞
（manifest 缺 ARM / hf-hub URL bug / 缺 ONNX），不推荐踩。

### 启动 — 原生 Python server（canonical 路径）

为什么不用 TEI / Infinity / vLLM？我们的 `BgeM3HttpEmbedder` 写死了 TEI 风格的
`/embed` + `/embed_sparse` wire format —— 第三方 server 大概率要适配。**自己起一
个 FastAPI 包 FlagEmbedding** 反而是最省事的（零适配，原生 ARM，无 Docker / ONNX
/ Rosetta 链路）。Server 代码在 `src/claritymed/servers/embedder.py`，wire format
跟客户端解析器一对一对齐，通过 `claritymed-embedder` console script 启动。

```bash
# 1. 装 server 端可选 deps（约 2 GB：torch + FlagEmbedding）
uv sync --extra rag-server

# 2. 宿主端预下载模型权重到 ~/.claritymed/models/bge-m3（约 2.3 GB）
mkdir -p ~/.claritymed/models
uv run hf download BAAI/bge-m3 --local-dir ~/.claritymed/models/bge-m3

# 3. 起 server（默认绑 127.0.0.1:8082；加载模型约 30 秒）
uv run --extra rag-server claritymed-embedder
# ↑ 前台跑；想后台 nohup 或 systemd / launchd 都行，自己定

# 4. 健康检查（另开终端）
curl -fsS http://127.0.0.1:8082/health
# {"status":"ok"}
```

启动顺序：FastAPI lifespan hook 在收第一个请求前同步加载 `BGEM3FlagModel`，
所以 `/health` 返回 `ok` 时模型一定 ready。环境变量覆盖：
`BGE_M3_MODEL_PATH` / `BGE_M3_PORT` / `BGE_M3_DEVICE`（默认自动探测：Apple Silicon
→ `mps`，NVIDIA → `cuda`，否则 `cpu`。FlagEmbedding 某些版本 MPS 上对 XLMRoberta
有 kernel 不稳，碰到崩可以显式 `BGE_M3_DEVICE=cpu` 回退）。

### 备选 — TEI Docker (仅 Linux x86_64)

Linux x86_64 服务器上 TEI 性能更好（Rust + ONNX Runtime），但要先把 BGE-M3
转 ONNX（`optimum-cli export onnx --model BAAI/bge-m3 --task feature-extraction`），
而且 **sparse head 不在标准 ONNX 导出里** —— sparse 路径要么自己改 export 脚本，
要么直接用 §2 的 Python server。Apple Silicon 上**不要走这条**（三层 bug：
manifest / hf-hub / ONNX），见 §8 故障表。

### 烟测

```bash
curl -fsS -X POST http://127.0.0.1:8082/embed \
  -H 'content-type: application/json' \
  -d '{"inputs": ["hello world"]}' | python -c 'import sys,json; v=json.load(sys.stdin); print(len(v), len(v[0]))'
# 期望输出: 1 1024

curl -fsS -X POST http://127.0.0.1:8082/embed_sparse \
  -H 'content-type: application/json' \
  -d '{"inputs": ["hello world"]}' | head -c 200
```

### 需要鉴权？

把环境变量名写到 yaml：

```yaml
embedders:
  catalog:
    - id: bge_m3_http
      base_url: "https://embed.example.com"
      api_key_env: "CLARITYMED_EMBED_KEY"
```

env 缺失 → `MissingApiKeyError` 启动即抛（**不会**用 placeholder 静默通过）。

---

## 3. Reranker — bge-reranker-v2-m3 HTTP server

Endpoint 单一：

```
POST /rerank
{
  "query": "...",
  "texts": ["d1", "d2", ...],
  "raw_scores": false,
  "return_text": false
}
→ [{"index": 1, "score": 0.93}, {"index": 0, "score": 0.21}, ...]
```

### 启动 — 原生 Python server（canonical 路径）

跟 §2 同结构。Server 代码在 `src/claritymed/servers/reranker.py`，包装
`FlagEmbedding.FlagReranker`，吐 TEI `/rerank` 形状，通过 `claritymed-reranker`
console script 启动。

```bash
# 1. server deps（如果 §2 已跑过就跳过）
uv sync --extra rag-server

# 2. 预下载模型（约 2.2 GB —— bge-reranker-v2-m3 比 bge-m3 还大一点）
uv run hf download BAAI/bge-reranker-v2-m3 \
  --local-dir ~/.claritymed/models/bge-reranker-v2-m3

# 3. 起 server（默认 127.0.0.1:8083）
uv run --extra rag-server claritymed-reranker

# 4. 健康检查
curl -fsS http://127.0.0.1:8083/health
# {"status":"ok"}
```

环境变量：`BGE_RERANKER_MODEL_PATH` / `BGE_RERANKER_PORT` / `BGE_RERANKER_DEVICE`。

### 烟测

```bash
curl -fsS http://127.0.0.1:8083/health && echo OK

curl -fsS -X POST http://127.0.0.1:8083/rerank \
  -H 'content-type: application/json' \
  -d '{"query":"chest pain","texts":["myocardial infarction","banana bread recipe"]}'
# 期望: 第一个 index=0 score > 第二个 index=1 score
```

### 失败语义（重要）

* 4xx / 5xx / timeout / 形状错 → `RerankerUnreachableError`（在 `bge_v2_m3.py`）。
* **`HybridRetriever` 把这视为 fail-soft**：retrieval 仍然出结果（按 RRF 顺序，不重排），并在 `RetrievalTrace.rerank_fallback=True` + audit `rag.rerank.fallback` 记一笔。
* 也就是说 reranker 挂了，ask 仍能跑，只是质量下降。embedder 挂了就 hard fail。

---

## 4. 一键启停脚本

§2 / §3 教的是手敲 `claritymed-embedder` / `claritymed-reranker` 把单个 server
跑起来——平时不用那么干，用这三个脚本管两个 server：

```bash
scripts/rag/run.sh        # 启动两个 server（已在跑的会跳过）
scripts/rag/stop.sh       # SIGTERM 优雅停；10 秒不退就 SIGKILL
scripts/rag/restart.sh    # = stop + run

# 也可以只操作一个
scripts/rag/run.sh embedder
scripts/rag/restart.sh reranker
```

行为细节：
- 日志落到 `${CLARITYMED_LOG_DIR:-~/.claritymed/logs}/embedder.log` /
  `reranker.log`，append 模式（保留历史方便事后查崩溃栈）
- pidfile 在 `~/.claritymed/run/{embedder,reranker}.pid`
- `run.sh` 启动后 poll `/health` 最多 180 秒（`RAG_HEALTH_TIMEOUT_S` 可覆盖）；
  模型挂了或超时会把日志尾巴 20 行打到 stderr 并 exit 非零
- 三个脚本都接受 `embedder` / `reranker` / `both`（默认）三种参数

---

## 5. 配置对齐

默认 `configs/retrieval.yaml` 已经指向 `127.0.0.1:8082` / `127.0.0.1:8083`：

```yaml
embedders:
  active: bge_m3_http
  catalog:
    - id: bge_m3_http
      kind: http
      base_url: "http://127.0.0.1:8082"
      dense_dim: 1024            # ← 必须跟模型对齐；BGE-M3 是 1024
      batch_size: 32
      timeout_s: 30
      api_key_env: null

rerankers:
  active: bge_v2_m3_http
  catalog:
    - id: bge_v2_m3_http
      kind: http
      base_url: "http://127.0.0.1:8083"
      batch_size: 32
      timeout_s: 30
      api_key_env: null
```

**dense_dim 改不得**：BGE-M3 是 1024-dim，已经写入 Qdrant collection 的向量改维度会让所有 cosine 距离失效。要换模型必须 drop collection 重建。

---

## 6. 翻总开关

```yaml
# configs/retrieval.yaml
rag:
  enabled: true
  max_evidence: 5
```

下一次 `claritymed ask` 或 `claritymed tui` 启动时，CLI / TUI 会调用 `build_hybrid_retriever()` → `build_strategy()` → 注入 `AskService(strategy=...)`。

启动顺序：

1. `load_retrieval_config()` 读 yaml；活动 id 错 → `Unknown<X>Error`
2. `build_embedder(cfg.embedders)` 构造 `BgeM3HttpEmbedder`（**这步不打网**，只构造 client）
3. `build_reranker(cfg.rerankers)` 同上
4. `build_term_service(cfg.term_service)` —— **可能**读 `data/terminology/concepts.jsonl`（见 §7）
5. `build_router(...)` 装载 `system_rag.collections` catalog
6. `AsyncQdrantClient(path=...)` 在本地目录里建文件锁
7. 首次 `await strategy.retrieve(...)` 时才真正 dial embedder / reranker

也就是说：**embedder / reranker 没启动，启动 CLI 本身不会爆**；第一次 ask 才爆。

---

## 7. 可选 — TermService 数据

`configs/retrieval.yaml` 默认 `term_service.active: umls_cmekg_local`，需要：

```
data/terminology/concepts.jsonl   # UMLS + CMeKG 概念表，每行一个 JSON 对象
```

文件不存在 → `FileNotFoundError` 启动即抛。两种应对：

* **有数据：** 准备好 jsonl（schema 参考 `core/rag/terms/umls_cmekg.py`）
* **没数据：** 把 active 切到 `none`：

  ```yaml
  term_service:
    active: none
  ```

  Term expansion 变成 no-op，retrieval 不带跨语言 / 同义扩展，质量小幅下降但完全可用。

UMLS / CMeKG 受版权限制，**不分发**；自行从 NLM / CMeKG 官方申请。

---

## 8. 烟测全链路

```bash
# 1. 三个服务确认 up
curl -fsS http://127.0.0.1:8082/health   # embedder
curl -fsS http://127.0.0.1:8083/health   # reranker
ls ~/.claritymed/data/qdrant             # qdrant dir

# 2. Bootstrap 一个 admin —— ingest 命令带 require_admin guard
uv run python -c "from claritymed.stores import init_user; init_user('admin')"

# 3. 下载 StatPearls + 顺手 normalize 到 ingest 期望的 schema
#    （首跑要拉 ~300 MB tar + clone MedRAG repo 来 chunk + 重写 9638 个 jsonl，
#    几分钟到十几分钟。MedRAG 的 {id,content,title} 会被改写成 {doc_id,text,title}
#    并按 article 聚合，由我们自己的 parent_child chunker 重切。）
uv run python scripts/fetch_corpora.py

# 4. 把 normalized 目录链到 ingest 期望的位置
#    注意指向的是 normalized 子目录，不是 chunk/ —— rglob 不会双数
mkdir -p ~/.claritymed/shared/knowledge/raw
ln -s "$PWD/data/download/medrag/statpearls/normalized" \
      ~/.claritymed/shared/knowledge/raw/statpearls

# 5. 先 dry-run 验 chunk 流程（不调 embedder、不写 qdrant）
#    Qdrant flock 是 exclusive 的，所以 TUI 必须先关掉，否则 RuntimeError
uv run claritymed rag corpora ingest statpearls --dry-run --limit 10 --user admin

# 6. 真 ingest（写 qdrant collection + parent docstore；需要 embedder 在跑）
uv run claritymed rag corpora ingest statpearls --limit 50 --user admin

# 7. 用一个 user 跑一发 ask
uv run claritymed ask "what causes pleural effusion?" --user alice
```

期望在 audit log 看到：

```
rag.retrieval  user_id=alice strategy=naive_hybrid active_collections=[statpearls_en] num_chunks=5 filtered_phi=0 fallback_triggered=false
mode.ask       user_id=alice answer_len=... model=... provider_id=... latency_ms=...
```

排错：

* 只看到 `mode.ask` 没 `rag.retrieval` → `strategy=None`，回头查 `rag.enabled`
  是否真为 `true`，`_maybe_build_strategy` 是否构造出来
* `rag.retrieval` 有但 `num_chunks=0` 且 `fallback_triggered=false` → qdrant
  collection 是空的，§8 第 5 步的真 ingest 没跑或没写成功；
  `cat ~/.claritymed/shared/qdrant/meta.json` 看是否有 `statpearls_en` 条目

---

## 9. 常见故障

| 症状 | 根因 | 处置 |
|---|---|---|
| `EmbedderUnreachableError` 启动后第一次 ask 就抛 | embedder server 没起 / 端口不对 | `curl /health`；确认 `claritymed-embedder` 在跑、`base_url` 匹配 |
| `RerankerUnreachableError` 出现在 audit，但 ask 仍返回结果 | reranker 挂了 / 模型未加载完 | fail-soft 正常；要恢复重排质量重启 `claritymed-reranker` |
| `MissingApiKeyError` | yaml 写了 `api_key_env` 但 env 没设 | `export <NAME>=...` 或把 `api_key_env: null` |
| `Storage folder ... already accessed by another instance of Qdrant client` | 旧 local 模式残留（不该再出现）；如果跑到这条说明哪里还在调 `AsyncQdrantClient(path=...)` | grep 仓库里有没有 `AsyncQdrantClient(path=` 的残留，应该只剩 `:memory:` 测试 fixture |
| `Connection refused` / `Failed to connect to ...:6333` 启动后第一次 ask 抛 | Qdrant server 没起 | `docker ps` 看容器；`docker start claritymed-qdrant`；或检查 `CLARITYMED_QDRANT_URL` 指对没 |
| `MissingApiKeyError: qdrant.api_key_env=...` | 配了 `api_key_env` 但 env 没设 | `export <NAME>=...`；或把 yaml `api_key_env: null` 去掉鉴权 |
| `BGE-M3 model dir not found` 启动 server 就报 | `~/.claritymed/models/bge-m3` 还没下载 | `uv run hf download BAAI/bge-m3 --local-dir ~/.claritymed/models/bge-m3` |
| `ImportError: FlagEmbedding` 启动 server 报 | 没装可选 deps | `uv sync --extra rag-server` |
| (备选路径) TEI 容器 `no matching manifest for linux/arm64/v8` | TEI 镜像只发了 amd64 | 用 §2 原生 Python server，不要走 TEI |
| (备选路径) TEI 容器 Exited(1) + `relative URL without a base` | TEI 1.5 hf-hub URL bug | 同上 — Apple Silicon 走 §2 原生 Python |
| (备选路径) TEI 容器 `File "/model/onnx/model.onnx" does not exist` | TEI CPU 后端要 ONNX，BAAI 不发 | 同上 — 或用 `optimum-cli` 手动转 ONNX（不含 sparse） |
| `rag.retrieval` 出现但 `num_chunks=0` 且 `fallback_triggered=false` | qdrant collection 空 / 没 ingest 过 | 跑 §8 第 2-5 步 fetch + symlink + ingest；`cat ~/.claritymed/shared/qdrant/meta.json` 验有 collection |
| Qdrant collection 找不到 | 没 ingest 过 / collection 名错 | `claritymed rag corpora list` 看 catalog；`claritymed rag corpora ingest statpearls` |
| dense_dim 不匹配 / 检索结果全 0 | yaml 里 `dense_dim` 跟模型实际维度对不上 | 改 yaml；如果已经写过数据要 drop collection 重建 |
| `FileNotFoundError: concepts.jsonl` | term_service 数据缺 | 切到 `term_service.active: none` 或准备数据 |

---

## 10. 关闭 / 回滚

直接把 `rag.enabled` 切回 `false` 即可。`AskService` 拿不到 strategy 时退化为 LLM-only 路径，embedder / reranker / qdrant 全部不会被 touch —— 可以放心把 server 进程停掉而不影响 `ask`。

```bash
scripts/rag/stop.sh
```

Qdrant 本地目录里的数据保留；下次再 `enabled: true` + 重启两个 server 直接继续用。
模型权重也留在 `~/.claritymed/models/`，不用重下。
