# Textual Dev & Web 部署指南

## 一、textual-dev（开发工具）

### 安装

仅需开发环境，加到 dev group：

```bash
uv add --dev textual-dev
```

### `textual run` — 运行应用

```
textual run [OPTIONS] FILE or FILE:APP
```

| Flag | 默认值 | 说明 |
|------|--------|------|
| `--dev` | off | 开发模式：连接调试 console，CSS 文件改动无需重启 |
| `--host HOST` | 127.0.0.1 | console 服务监听地址 |
| `--port PORT` | 8081 | console 服务端口 |
| `-c, --command` | — | 把参数当 shell 命令执行，而非 .py 文件 |
| `--press TEXT` | — | 模拟按键序列（逗号分隔），用于自动化截图 |
| `--screenshot DELAY` | — | 启动后等 N 秒截图 |
| `--screenshot-path PATH` | — | 截图保存目录 |
| `--screenshot-filename NAME` | — | 截图文件名 |
| `-r, --show-return` | off | 退出时打印 App.return_value |

**ClarityMed 常用命令：**

```bash
# 开发模式：CSS 热重载 + 接收 console 输出
uv run textual run --dev -c "claritymed tui --user demo"

# 截图（CI / 文档用）
uv run textual run --screenshot 2 --screenshot-path docs/ -c "claritymed tui --user demo"
```

### `textual console` — 调试终端

TUI 全屏运行时 stdout 不可见，console 是独立进程接收日志输出。

```
textual console [OPTIONS]
```

| Flag | 说明 |
|------|------|
| `--port PORT` | 监听端口，默认 8081，需与 `textual run --port` 一致 |
| `-v` | 显示详细日志（含通常被过滤的事件） |
| `-x GROUP` | 排除某类消息，可多次使用 |

可排除的 GROUP：`EVENT` `DEBUG` `INFO` `WARNING` `ERROR` `PRINT` `SYSTEM` `LOGGING` `WORKER`

**典型用法：**

```bash
# 终端1：先起 console
uv run textual console -x SYSTEM -x EVENT

# 终端2：再跑应用（--dev 让它连上 console）
uv run textual run --dev -c "claritymed tui --user demo"

uv run textual run --dev -c "claritymed tui --provider omlx"
```

App 里用 `self.log(...)` 或 `from textual.logging import TextualHandler` 输出，console 里可见。

### `textual serve` — 本地 Web 预览

把 TUI 变成浏览器可访问的 Web 应用，适合演示和远程调试。

```
textual serve [OPTIONS] FILE or FILE:APP
```

| Flag | 说明 |
|------|------|
| `-p, --port INTEGER` | 监听端口，默认 8000 |
| `-h, --host TEXT` | 绑定地址，默认 localhost；局域网访问加 `-h 0.0.0.0` |
| `-t, --title TEXT` | 页面标题 |
| `-u, --url TEXT` | 公开 URL（配合反向代理使用） |
| `--dev` | 同时启用 devtools |
| `-c, --command` | 把参数当命令执行 |

```bash
# 本机访问
uv run textual serve -p 9000 -c "claritymed tui --user demo"

# 局域网访问（注意：demo 数据，别用真实 PHI）
uv run textual serve -h 0.0.0.0 -p 9000 -c "claritymed tui --user demo"
```

> **PHI 警告**：`textual serve` 不带任何认证。演示时只用脱敏/mock 数据。

### 其他诊断命令

```bash
uv run textual diagnose   # 打印环境信息（Python 版本、Textual 版本、终端能力）
uv run textual keys       # 实时显示按键事件名，用于调试键绑定
uv run textual colors     # 查看设计系统色板
uv run textual borders    # 查看所有边框样式
uv run textual easing     # 查看动画缓动函数
```

---

## 二、工作原理

核心思路是把终端渲染层替换成 WebSocket 传输层，分三层：

### 1. Textual 的渲染架构是可插拔的

Textual 内部把"计算 UI 应该长什么样"和"把结果画到哪里"分开了。正常跑时输出到终端（ANSI escape codes），`textual serve` 时换成另一个 Driver，把同样的渲染结果通过 WebSocket 推给浏览器。

### 2. 浏览器端是一个终端模拟器

`textual serve` 启动后，浏览器打开的页面里内嵌了 [xterm.js](https://xtermjs.org/)（一个 JS 写的终端模拟器）。服务端把 ANSI 序列通过 WebSocket 实时推过来，xterm.js 解析渲染，视觉上跟真实终端一模一样。

### 3. 输入反向传回

键盘/鼠标事件在浏览器里被捕获，序列化后通过同一条 WebSocket 发回 Python 进程，Textual 的事件系统正常处理，跟本地跑没区别。

```
浏览器 xterm.js
    ↕ WebSocket (ANSI序列 / 键鼠事件)
textual serve (aiohttp/asyncio)
    ↕
你的 Textual App 进程（完全不知道自己在被 serve）
```

**应用代码零改动**——App 只管产生 UI 事件，不关心渲染目标是终端还是 WebSocket。这个抽象边界是 Textual 设计时就留好的。

---

## 三、textual-web（生产部署）

`textual serve` 是本地开发工具，生产环境用 `textual-web`：生成公网可访问的持久 URL，底层走 Textualize 托管的 WebSocket relay。

### 安装

```bash
pipx install textual-web
```

### 快速体验（无账号，随机 URL）

```bash
textual-web
```

生成随机公网 URL，每次运行都会变。适合临时演示。

### 配置文件

创建 `serve.toml`（不要提交到 git，含 API key）：

```toml
[account]
api_key = "YOUR_API_KEY"   # 注册后生成，见下方

[app.ClarityMed]
command = "uv run claritymed tui --user demo"
slug = "claritymed"        # URL 中的路径段，固定不变
```

```bash
textual-web --config serve.toml
# → 访问 https://ganglion.io/<account-slug>/claritymed
```

### 注册账号（获取持久 URL）

```bash
textual-web --signup
```

终端弹出注册 TUI，填写后生成 `ganglion.toml`，里面包含：

```toml
[account]
api_key = "JSKK234LLNWEDSSD"
```

把 `api_key` 放入你的 `serve.toml`，或者直接在 `ganglion.toml` 追加 `[app.*]` 配置。

### 多应用配置

```toml
[account]
api_key = "..."

[app.ClarityMed-EN]
command = "uv run claritymed tui --lang en --user demo"
slug = "claritymed-en"

[app.ClarityMed-ZH]
command = "uv run claritymed tui --lang zh --user demo"
slug = "claritymed-zh"
```

### 调试

```bash
DEBUG=1 textual-web --config serve.toml
```

### 注意事项

| 项目 | 说明 |
|------|------|
| 认证 | textual-web 本身不带用户认证，公网 URL 任何人可访问 |
| PHI | 生产演示必须用脱敏数据；绝不把真实用户数据暴露到 textual-web |
| 平台 | macOS 和 Linux；Windows 尚未支持 |
| 状态 | beta 阶段，不建议用于真实生产流量 |
| 持久性 | 注册账号后 slug 固定；未注册每次 URL 不同 |

### `.gitignore` 追加

```
serve.toml
ganglion.toml
```
