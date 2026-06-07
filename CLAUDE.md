# CLAUDE.md

## Commands

```bash
# TODO: 填写启动命令
# uv run python serve.py

# 同步依赖
uv sync
```

## Architecture

TODO: 描述目录结构和数据流

```
myapp/
  # TODO: 填写模块说明
```

## Key design constraints

- TODO: 写反直觉的约定和决策（颜色规范、特殊数据格式等），而不是显而易见的东西

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
