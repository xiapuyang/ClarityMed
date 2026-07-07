# CLAUDE.md

## Commands

```bash
# 同步依赖（含 dev group）
uv sync

# 跑测试 + 覆盖率门槛（pyproject.toml 设了 --cov-fail-under=80）
uv run pytest

# 跑全部 pre-commit hook（gitleaks / ruff / ruff-format / AI bypass 检测）
uv run pre-commit run --all-files

# 安装 pre-push hook（一次性；安装后 git push 前自动跑全量单测 + e2e）
uv run pre-commit install --hook-type pre-push

# 启动 TUI（headless 镜像走 ask/rag 子命令）
uv run claritymed tui [--user <id>] [--lang en|zh] [--provider <id>]

# 单次问答 / 数据落盘
uv run claritymed ask "..." [--user <id>] [--provider <id>]
uv run claritymed rag add ./path/to/note.md --user <id>

# Phoenix prompts 双向同步
uv run claritymed prompts push [NAME] [--dry-run]
uv run claritymed prompts pull [NAME] [--dry-run] [--into-new-version] [--version-name v1.1]

# E2E 多 provider 对比（逗号分隔；未设置则 pick_reachable_provider）
# CLARITYMED_E2E_PROVIDERS=omlx,deepseek uv run pytest tests/e2e/test_ingest_tools_e2e.py

# Tool prompt 语言与用户语言解耦（A/B tool description 语言对工具调用准确率的影响）
# CLARITYMED_TOOL_PROMPT_LANG=en|zh uv run claritymed tui
```

## Observability — Phoenix tracing

`configs/app.yaml` 里 `tracing.enabled` 由**开发者手动管理**，code review / sweep 不得修改。
本地开发通常设为 `true`；CI 和离线 session 设为 `false` 保持 no-op。

配置示例、audit ↔ trace 关联、PHI scrubbing 行为详见 [`docs/tracing.md`](docs/tracing.md)。

## Prompts workflow

Prompts 以 `core/prompts/store/*.yaml` 为**运行时唯一真相**。Phoenix 只是编辑 UI + eval 平台，runtime 不调 Phoenix。

命名约定：`(name, language)` → Phoenix prompt `claritymed_<name>_<lang>`（如 `claritymed_ask_zh`），`production` tag 标识当前同步版本。

流程：`prompts push` → 在 Phoenix UI 改 prompt → `prompts pull`（加 `--into-new-version` 保留旧版本）。

## Architecture

详细模块说明见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

数据流：`inject_context()` 装 ContextVars → orchestrator 取 prompt → `phi_guard` 判 cloud/local → `LLMClient.chat()` → audit log。

## Key design constraints

- **PHI 出域只看一个字段**：`ProviderConfig.kind`。`local` 可以带 PHI；`cloud`
  必须先过 `core.orchestrator.phi_guard`。任何绕过 orchestrator 直接调
  `LLMClient` 的代码都破坏这条不变量——不要这么做。
- **Cloud 调用是二层 AND**：env var 存在 ∧ `ProviderConfig.kind == "cloud"`。
  少任意一层就跑不通。`cloud_provider_opt_in` per-user 标志曾作为第三层存在，
  被判定为过度设计删掉；cloud-bound PHI 兜底由 `phi_guard` + assembled-prompt
  二次 scrub 承担，不再依赖每用户开关。
- **每用户偏好以文件为准**：`data/<user_id>/settings.yaml` 是单一真相，
  SQLite 表只是查询用的镜像；冲突时 YAML 赢。
- **Provider resolution 失败即响**：`settings.yaml` 里把 `provider_id`
  打错字会抛 `UnknownProviderError`，**不会**静默回落到 default —— 默认回落
  会让指向 cloud 的用户在不知情下被切到别的后端。
- **解析顺序**：`override`（CLI `--provider`） > `Account.provider_id` >
  `ModelsConfig.default_provider`。这个顺序在 `stores/models.py:resolve_provider`
  里硬编码，改的话连带改文档。
- **`model` 字段两种形态**，由 `base_url` 决定：
  - `base_url` 缺省 → 必须 `"<prefix>:<model>"`（pydantic-ai 的
    `KnownModelName` 形式，如 `"openai:gpt-4o"`）。pydantic-ai 用 prefix
    选 Model/Provider 并读官方 env var。
  - `base_url` 有值 → 裸 model 名（如 `"qwen3:14b"`、
    `"mlx-community/Llama-3.2-3B"`），原样转发给服务端。
  混着写（`"openai:gpt-4o"` + `base_url`）会被 validator 拒。
- **`api_key_env` 只在 `base_url` 有值时生效**。stock cloud entry 里写它
  会被 validator 拒（避免和 pydantic-ai 自己读的 env 出现两层来源）。
  自托管端点声明了 `api_key_env` 但 env 没设 → `MissingApiKeyError`
  立刻抛，而不是悄悄回落到 `OllamaProvider` 的 placeholder。
- **本地 OpenAI-compatible 服务器都走同一条路**：Ollama / MLX / llama.cpp /
  LM Studio 在 `build_model` 里全部映射到 `OllamaProvider(base_url, api_key)`
  —— 它是 pydantic-ai 里唯一不强求 API key 的 Provider，恰好能承载"要不要
  auth"两种情况。要加新本地后端，只在 YAML 加条目就行，不用动 Python。
- **Reasoning 走一个统一字段**：`ProviderConfig.thinking` 接受
  `true / false / "minimal" / "low" / "medium" / "high" / "xhigh"`，由
  `build_model_settings` 透传到 pydantic-ai 的 `ModelSettings.thinking`。
  pydantic-ai 自己把它翻成 `anthropic_thinking` / `openai_reasoning_effort` /
  `google_thinking_config`，所以我们这边**不要**按 vendor 分支翻译——
  那是它的活。模型不支持推理时静默忽略，可以放心地把同一档位挂在 cloud
  和 local 条目上。需要按问题复杂度临时上调？orchestrator 把 catalog
  默认值和 per-request 值合并后再传给 `Agent`，schema 不用动。
- **No JOINs / no FKs**（全局规则的项目化复述）—— 跨表关联用 app 代码拼，
  引用永远用 `*_id` 整数而不是 name 字符串。
- **`core/` 不许反向依赖 `core/orchestrator/`**。依赖方向是单向的：
  `orchestrator` 站在 `llm` / `prompts` / `schemas` / `observability` / `i18n`
  之上去编排 pipeline；底层 primitives 永远不 import orchestrator 里的东西
  （包括 `FeaturePlugin`、`phi_guard`、`AskService` 等）。要把"某个能力"
  下沉到 core 时，先把它抽成不依赖 orchestrator 的纯接口，再让 orchestrator
  去 wire；不要让 `core/llm/*.py` 反过来 `from claritymed.core.orchestrator …`。
  破坏这条会让 plugin 模型瓦解，编排层无处可下手。
- **EmergencyTriage 只能从 AskService.handle 的 pre-step 调用**（KTD-E5）。
  这是个确定性的安全门：先于 agent loop 跑，结果一律放进
  `deps.triage` + `GroundedAnswer.red_flags[]`。**任何 tool plugin（包括
  `symptoms_plugin` / `vision_plugin` / 未来的 plugin）禁止 import
  `EmergencyTriage` / `EmergencyAssessment`，也禁止往 `red_flags[]` 写
  东西**。Tool 自己的安全机制（如 `symptoms_plugin.severity_override` +
  `symptoms.safety_keywords` 审计）保持独立——这是有意为之的两条并行
  安全路径，不是缺口。让 gate 反向 callback 进 tool 会迫使每个 tool 都
  变成 gate-aware，把松耦合彻底破坏，永远不要走这条回头路。
- **EmergencyTriage 的 LLM 必须用 local provider**。extractor 看的是原始
  对话历史（含 PHI），composer 看 matched_rules + 结构化症状（不含原文
  PHI 但仍是医疗判断面）；都必须留在本地。`core/emergency/_provider.py`
  通过 `models.yaml` 找第一个 `kind: local` provider，没有就让 gate 静默
  停在 noop 模式（不阻塞用户答题，但日志会响），**不要**回退到 cloud。
- **`critical` 永远短路 agent loop**（KTD-E3）。`triage.level=="critical"`
  时 AskService 直接走 `_stream_critical_short_circuit`：i18n 行动文案
  + `emergency_reply.yaml` 组合输出，不跑 retrieval / tool dispatch /
  agent.run。给 STEMI 用户省下 3-5 秒是这个分支存在的全部理由；不要
  为了"丰富回复"在这里加任何 hop。
- **Rule 的 `minimum_sensitivity_floor` 是规则作者的 veto 权**（KTD-E10）。
  `floor=lenient` 意味着任何 profile（包括 `lenient`）都不能改这条规则的
  `min_qualifier_matches` —— anaphylaxis / active_SI / ectopic_pregnancy
  是 canonical 的"漏报代价无限大"案例。`load_validated_emergency_config`
  在启动期 fail-loud 拒绝违反 floor 的 profile override；新增规则时把
  floor 想清楚再写。
- **`off` 灵敏度剩四条 safeguard**（原有 `default_sensitivity: off` 加载期
  拒收已按 operator 请求下线；`gate_disabled` footer 也按 operator 请求
  置空，i18n 查找路径保留但当前跑出来是空串——见 `configs/i18n/*/emergency.yaml`
  注释）：env override (`CLARITYMED_FORCE_EMERGENCY_GATE`) 默认 on 会把
  用户 off 降级到 lenient；写入 per-user `settings.yaml` 仍需要
  `off_acknowledged_at`（app-wide default 走 `emergency.yaml`，不受这条
  约束）；off-path 每次都发 `redflag.gate_disabled` 审计；CLI
  `--emergency-sensitivity off` 也走完同样的 safeguard 链。运营者如果
  想让 `default_sensitivity: off` 真正生效，需要同时把
  `CLARITYMED_FORCE_EMERGENCY_GATE` 设为 `0/false`——env override 默认
  on，会自动把 off 升回 lenient。想恢复"每条回复末尾都有免责声明"这条
  可见 safeguard，只需在 `emergency.yaml` 里把 `gate_disabled` 填回文案，
  `_emergency_footer_text` 下一次请求自动读到（i18n 走 mtime 缓存）。

## Use pydantic-ai's built-ins before writing your own

依赖里有 `pydantic-ai-slim[anthropic,openai]>=1.0`。在自己写 provider 分发、
消息类型、env 读取、重试、token 统计**之前**，先确认 pydantic-ai 是不是已
经给了。

**不要重造的轮子：**

- **按字符串选 model**：`Agent('openai:gpt-4o')` / `infer_model('anthropic:claude-sonnet-4-5')`
  已经把 `'<provider>:<model>'` 解析成正确的 `Model + Provider`。不要再手写
  `if api == "openai": ... elif api == "anthropic": ...`。
- **env → api_key**：`OpenAIProvider`/`AnthropicProvider`/`DeepSeekProvider`/
  `OpenRouterProvider`/`MoonshotAIProvider`/`AlibabaProvider`/`GoogleGLAProvider`/
  `OllamaProvider` 构造时自动读对应 env var；不要再自己 `os.environ.get(...)`
  + raise `MissingApiKeyError`。
- **消息 / 响应类型**：`pydantic_ai.messages.ModelRequest` / `ModelResponse`、
  `pydantic_ai.usage.RunUsage` / `RequestUsage`、`AgentRunResult` 已经覆盖
  text、finish_reason、model name、token usage。不要再自定义 `ChatMessage` /
  `Usage` 的子集。
- **多模型回退**：`pydantic_ai.models.fallback.FallbackModel(primary, secondary)`，
  不要自己写 try/except 链。
- **结构化输出 + 工具循环**：`Agent(..., output_type=GroundedAnswer)` +
  `@agent.tool` 自带 JSON schema 重试。不要再写 parse-retry-validate。
- **采样参数**：`pydantic_ai.settings.ModelSettings` 覆盖 max_tokens /
  temperature / top_p / 等。

**只有这几种情况才包一层：**

- 包装在执行项目特有策略（PHI guard 读 `ProviderConfig.kind`——pydantic-ai
  不知道这个概念）。
- 需要 fail-loud 语义而 pydantic-ai 给的是 silent fallback。例：自托管端点
  的 `api_key_env` 声明了但 env 没设——`OllamaProvider` 会静默用
  placeholder，我们要立刻 `MissingApiKeyError`。这种"加一层就为了让它响"
  是可以的，但要在新代码里**明确写明**"为什么不让框架自己处理"。
- 真的需要在 `Agent` 之下的层（用 `pydantic_ai.direct.model_request`，那是
  官方支持的 public API）。
- 某个 vendor / endpoint 确实不在 `pydantic_ai.providers` 里（罕见，**先 grep
  再说**）。

**写之前自查：** 在 `pydantic_ai` 包里 grep 想写的类型或函数名。如果你打算
写 50 行 adapter 干 pydantic-ai 用 10 行就能干的事，回头看是不是漏了这一步。

## Prompts — No Hardcoded Strings

所有 system prompt 和 user-facing prompt 必须放在 `core/prompts/store/<name>.yaml`，
通过 `PromptRegistry().get(name)` 读取。**不允许**在 Python 源码里硬编码提示词字符串。

规则：
- 新增 LLM 调用必须先建对应的 YAML，再在 Python 里引用 `_PROMPT_NAME = "..."` 常量。
- 已有 YAML 不满足需求时，追加新 `version` 而不是修改现有 version（版本不可变）。
- 中英双语均为必填（validator 强制校验），缺任一语言会在启动时 fail-fast。
- Phoenix 同步走 `prompts push/pull`，不要手动编辑 Phoenix 侧再回写 YAML。

## Test user_id convention

测试里写 `data/users/<uid>/` 时**统一**用两个固定 uid，避免污染开发者真实用户目录、也方便 fixture 一把清掉：

- 单元测试 → `user_id="test"`
- e2e 测试（`tests/e2e/`） → `user_id="e2e"`

任何 `apply_context(...)`、`SettingsStore(...)`、`BlobStore(...)`、`ProfileStore(...)` 等需要 user_id 的入口都必须挑这两个之一。新加测试不要再发明 `alice`/`bob`/`u1` 之类——历史代码里残留的也鼓励顺手替换掉。临时手工调用 (e.g. dev 跑 CLI 时) 用别的就好。

## Development Workflow

```
需求模糊  →  /ce:brainstorm  →  docs/brainstorms/
                ↓
            /ce:plan        →  docs/plans/
                ↓
           实现代码
                ↓
            /ce:review      →  修复问题
                ↓
        遇到坑/解决问题  →  /ce:compound  →  docs/solutions/
                ↓
             提 PR
```

| Skill | 触发时机 |
|---|---|
| `/ce:brainstorm` | 需求不清晰，需要发散 |
| `/ce:plan` | 开始实现前，需要多步方案 |
| `/ce:review` | 功能完成后、提 PR 前 |
| `/ce:compound` | 解决了一个非平凡问题后 |


# Karpathy Guidelines

Behavioral guidelines to reduce common LLM coding mistakes, derived from [Andrej Karpathy's observations](https://x.com/karpathy/status/2015883857489522876) on LLM coding pitfalls.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## Open-Source Hygiene

**这个项目可能某天开源。** 写代码时按"已经在 GitHub 上"的标准来 —— 后期翻历史改 personalization 的成本远大于一开始就避开。

**永远不在源码里硬编码：**

- 真名、用户名、个人邮箱、电话号码（包括 seed 数据、greeting 文案、test fixture、注释）
- 个人 inbox / webhook URL / API endpoint（包括 AgentMail、Slack hook、Telegram chat_id）
- 真实地理位置 / 时区 / 生日 / 出生年（用 `UTC` 作中性默认，让前端读浏览器 locale）
- 真实金额、账户余额、持仓数据（即使是"示例"也用明显假的数字，如 `1234.56`）
- 任何形式的密钥、token、password、bearer —— 即使是"临时占位"也不行

**正确做法：**

- 密钥类 → 环境变量（`os.environ.get("X")`），并在 `.env.example` 里列出全部变量名 + 一行说明
- 用户可个性化的值（display name、timezone、notify email、watchlist）→ 读 `data/settings.json` 或同等的运行时配置文件，gitignored
- DB seed 用中性占位（`"User"`、`"admin@example.local"`）
- UI greeting 类做成 `settings.display_name || ""` fallback 到通用问候语

**Commit 之前自问：**

- 这个字符串如果出现在 public GitHub 搜索结果里我介意吗？介意就别提交。
- 这个 `data/` 文件是 sane default（如通用 watchlist），还是我个人数据？后者必须 gitignore。
- 我的 `git config user.email` 是 personal Gmail 还是 `<id>+<name>@users.noreply.github.com`？前者会永久嵌入 commit metadata，建议切换。

**配套文件（项目初始化时就该存在）：**

- `.env.example` —— 列出所有环境变量
- `README.md` —— 至少一段话说项目是什么 + 如何启动
- `LICENSE` —— 真要开源前再选（默认 MIT）；不急但别忘
- `.gitignore` —— `data/`、`logs/`、`.env`、`*.tgz`、个人导入数据全屏蔽

## Log Level Rules

**级别选择：**

| 情形 | 级别 |
|------|------|
| 影响用户的未预期异常（`had_error=True`、请求失败、数据写入失败） | `ERROR` + `exc_info=True` |
| 预期失败路径但调用方仍会感知（modal 崩溃、channel unavailable、规则未命中） | `WARNING` |
| 正常流程里的关键节点（请求开始/结束、工具调用结果） | `INFO` |
| 内部状态追踪、性能计时、调试细节 | `DEBUG` |

**规则：**
- `except` 块里的 `logger.debug/info` 只允许出现在"已知且良性的异常路径"（如 graceful teardown）。凡是会导致 `had_error=True`、发出 `Error` 事件、或让用户看到错误消息的 except 块，必须用 `logger.error(..., exc_info=True)`。
- 重新 `raise` 前的日志可以降一级（`warning`），因为上层还会处理；但不能降到 `debug`。

## Pydantic model_copy vs model_validate

**`model_copy(update={field: value})` 不跑 validators**，直接把 `value` 原值写进字段。如果 `value` 是字符串而字段类型是 `date`/`datetime`/`Decimal` 等，Pydantic 不会 coerce，SQLAlchemy 写库时就会报 `TypeError`。

**需要 coerce 时必须用 `model_validate`：**

```python
# 错误 — model_copy 不 coerce
updated = current.model_copy(update={field: value})

# 正确 — model_validate 跑完整 validator 链
data = current.model_dump(mode="python")
data[field] = value
updated = MyModel.model_validate(data)
```

适用场景：任何从外部来的值（LLM 输出、用户输入、API 请求）写入 Pydantic 模型的 date/datetime/Decimal/Enum 字段时，都走 `model_validate`，不走 `model_copy(update=...)`。

## Multi-Terminal Worktree Isolation

这个 repo 经常被多终端并发编辑——一个 session 跑 bench / e2e，另一个改代码。两个 session 看到的是**同一个 working tree**：终端 B 的 skill 跑一句 `git stash`，终端 A 没提交的活会被一锅端走（已经发生过两次：`stash@{0}` 和 `stash@{1}`）。

防御策略：怀疑有并行 session 时，**主动开 worktree 隔离**。

项目级 `.claude/settings.json` 已经设了 `worktree.baseRef: head`，所以 `EnterWorktree` 默认从**当前 HEAD** 切新分支，不会回到 origin/main——继承当前 feature 分支的全部 commits。

调用规范：

```
EnterWorktree(name="<branch-shortname>-<purpose>")
# 例：tui-agent-bench / tui-agent-fix-allergy / tui-agent-yaml-v2
```

`name` 务必有语义。让 `git worktree list` 一眼读得懂"这是干啥的"——不要让自动生成的随机后缀堆满 `.claude/worktrees/` 目录。

End-of-session 清理（和 stash 类似容易被忘，但**不会丢数据**——只是堆 orphan）：

| 情况 | 动作 |
|---|---|
| 工作已 merge 回主 feature 分支 | `ExitWorktree(action="remove")` —— 删目录 + 分支。误删的分支可在 90 天内通过 `git reflog` 找到最后一次 HEAD 的 SHA，再 `git branch <name> <sha>` 恢复 |
| 工作 parked、明天继续 | `ExitWorktree(action="keep")` 保留目录和分支。预计拖几天的话，顺手在 `docs/parked-worktrees.md` 记一笔（路径 + 分支 + 一句话用途），免得回头看到 `tui-agent-bench-2` 想不起来当初干啥 |
| 直接关终端 | 数据不丢，worktree 目录和分支会留在 `.claude/worktrees/` 堆着。下次 session 启动时 SessionStart hook 会列出来提醒处理 |

每次 session 启动时 global SessionStart hook 会自动 `git worktree list`，多于 1 个会提示——这是兜底，不是替代清理。

什么时候**不**用 worktree：单终端工作 + 没并发风险时。worktree 给你隔离的代价是每次都要 EnterWorktree / 合并 / Exit，单线工作时纯属负担。
