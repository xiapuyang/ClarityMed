# ClarityMed

TODO: 一段话说这个项目是什么 / 解决什么问题。

## Quickstart

```bash
git clone <repo-url>
cd ClarityMed
cp .env.example .env  # 填入需要的环境变量
# TODO: 启动命令
```

## Skills

### `import-medical-record`

Conversational driver for bulk-importing arbitrary medical records
(Notion exports, Apple Health bundles, clipboard pastes) into
ClarityMed's per-event store. Install:

```bash
ln -s "$(pwd)/skills/import-medical-record" \
    ~/.claude/skills/import-medical-record
```

Once installed, ask Claude Code: "import these records from
`<source>`" and follow the prompts. The skill calls the
`claritymed record …` CLI (`import-from-template`, `import-status`,
`import-resume`, `ocr-extract`, `health`).

## License

[MIT](LICENSE)
