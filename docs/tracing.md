# Phoenix Tracing

## Quick start

```yaml
# configs/app.yaml
tracing:
  enabled: true
  endpoint: http://localhost:6006   # 默认，可改成远端 Phoenix
  api_key_env: null                 # 远端鉴权时填 env var 名，如 PHOENIX_API_KEY
  phi_kind: null                    # null=自动检测；local/cloud 可手动覆盖
  service_name: claritymed
```

```bash
# 本地起 Phoenix（任选其一）
docker run -p 6006:6006 arizephoenix/phoenix:latest
uvx arize-phoenix serve

uv run claritymed tui
```

打开 `http://localhost:6006` 查看每次 `agent.run` / `model_call` 的
prompt / response / token usage（含 `cache_read_tokens` / `cache_write_tokens`）/
latency 分层（`total_ms` / `ttft_ms` / `completion_ms`）/ 多步 trace。

## Audit ↔ Trace 关联

每条 `audit.log` 行带 `trace_id` + `span_id`；每个 OTel span 带三个 baggage attribute：

- `claritymed.request_id` — 跟 audit 行的 `request_id` 一一对应
- `claritymed.user_id`
- `claritymed.session_id` — `AskService` 在调 LLM 时 attach，可按对话粒度聚合

Phoenix 里搜某个 `trace_id` 能定位到对应 audit 行；反过来 grep audit.log 拿到
`claritymed.session_id` 也能在 Phoenix 里 group by 对话。

### Feature-scoped baggage

部分 plugin 会在工具体执行期间附加额外 baggage，用来按 feature 维度聚合 trace：

- **症状工具**：`claritymed.symptoms.dataset_id` / `.model_id` / `.session_id` ——
  `SymptomsFeature._attach_symptoms_baggage` 在 server start session 成功后 attach、
  loop 结束 detach。Phoenix 里 group by `.session_id` 可拿到一次完整的 5-12 问诊
  trace。
- **影像工具**：`claritymed.vision.disease_id` / `.model_id` / `.server_id` ——
  `VisionFeature` 在 `/v1/detect` 拿到 `RawDetection` 后 attach，audit
  payload 写完立刻 detach。Phoenix 里 group by `.disease_id` 可对比不同疾病的
  尾延迟分布。

## PHI scrubbing

`phi_kind: null`（默认）时：

- localhost endpoint (`localhost` / `127.0.0.1` / `::1`) → 不 scrub，PHI 留在本机
- 远端 endpoint → 自动启用 PHI scrubbing，OI-written span attributes 在导出前被清洗

**不要**把 `endpoint` 指向第三方 SaaS 且同时把 `phi_kind` 设成 `local`——PHI 会出域。
远端 Phoenix 的 `phi_kind` 留 `null`，auto-detect 会兜底。

## Protected config

`configs/app.yaml` 里的 `tracing.enabled` 由**开发者手动管理**。
本地开发通常设为 `true`；CI 和离线 session 设为 `false`。

code review / sweep 工具**不得**修改这个字段——它是运维开关，不是代码质量问题。
