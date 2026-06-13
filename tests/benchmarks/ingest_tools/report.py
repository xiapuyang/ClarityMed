"""Render an interactive HTML report from a benchmark run.

Reads ``trials.jsonl`` (and any ``judge_*.jsonl`` files) under
``--run <dir>`` and writes ``report.html`` next to them. Open the file
in a browser — no server needed, no external CSS/JS — and use the
filters at the top to slice by model / language / tier / behavior /
case. Below each filter set the page shows:

* an aggregated summary card for the current selection
* a group-by-able table (model / case / tier / etc.) with correct rate,
  outcome distribution, latency, and judge score if available
* a per-trial detail table you can drill down into

Pure HTML + vanilla JS. Data is embedded in a ``<script>`` block, so
sharing the report = sending one file.

Usage::

    uv run python -m tests.benchmarks.ingest_tools.report \\
        --run data/bench/<ts>/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _load_judge_index(run_dir: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    for path in sorted(run_dir.glob("judge_*.jsonl")):
        for rec in _load_jsonl(path):
            index[rec["request_id"]] = rec
    return index


# Trial fields kept in the embedded JSON — drops the verbose payloads
# (full prompt, final text) so the HTML stays small. Inspector script
# is the place to read full content.
_TRIAL_FIELDS_LIGHT = [
    "request_id",
    "model",
    "lang",
    "tool_prompt_lang",
    "case_name",
    "tier",
    "expected_behavior",
    "outcome",
    "predicate_pass",
    "no_tool",
    "ask_user_q_count",
    "latency_ms",
    "error",
]


def _light_trial(trial: dict, judge: dict | None) -> dict:
    out = {k: trial.get(k) for k in _TRIAL_FIELDS_LIGHT}
    out["tool_calls"] = [c["tool_name"] for c in trial.get("tool_calls", [])]
    if judge:
        out["judge_score"] = judge.get("score")
        out["judge_provider"] = judge.get("judge_provider")
    else:
        out["judge_score"] = None
        out["judge_provider"] = None
    return out


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Ingest-tool benchmark — {run_name}</title>
<style>
  body {{ font: 14px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         margin: 18px; color: #1a1a1a; }}
  h1 {{ font-size: 18px; margin: 0 0 4px; }}
  .sub {{ color: #666; font-size: 12px; margin-bottom: 14px; }}
  .filters {{ background: #f5f5f7; padding: 12px 16px; border-radius: 8px;
              margin-bottom: 14px;
              display: grid; grid-template-columns: 160px 1fr;
              row-gap: 8px; column-gap: 14px; align-items: center; }}
  .filters > label {{ font-weight: 600; font-size: 12px; color: #555;
                      text-align: right; }}
  .filters .chips {{ display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }}
  .filters .pill {{ background: #fff; border: 1px solid #ccc; padding: 2px 10px;
                    border-radius: 12px; cursor: pointer; font-size: 12px; }}
  .filters .pill.on {{ background: #007aff; color: #fff; border-color: #007aff; }}
  .filters .pair {{ display: flex; gap: 10px; align-items: center; }}
  .filters .pair > label {{ font-weight: 600; font-size: 12px; color: #555; }}
  .filters select {{ font: inherit; padding: 3px 6px; }}
  .card-row {{ display: flex; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }}
  .card {{ background: #fff; border: 1px solid #e0e0e3; border-radius: 8px;
           padding: 10px 14px; min-width: 130px; }}
  .card .big {{ font-size: 22px; font-weight: 700; }}
  .card .lbl {{ font-size: 11px; color: #666; text-transform: uppercase;
                letter-spacing: 0.5px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13px;
           margin-bottom: 24px; border: 1px solid #d8d8dc;
           border-radius: 6px; overflow: hidden; }}
  th, td {{ padding: 6px 10px; text-align: left;
            border-right: 1px solid #ececef;
            border-bottom: 1px solid #ececef; }}
  th:last-child, td:last-child {{ border-right: 0; }}
  tbody tr:last-child td {{ border-bottom: 0; }}
  th {{ background: #f5f5f7; font-weight: 600; cursor: pointer;
        position: sticky; top: 0; }}
  th.num, td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  tr:hover {{ background: #f8f8fa; }}
  .bar {{ display: inline-block; height: 8px; background: #007aff; vertical-align: middle;
          border-radius: 2px; }}
  .bar-bg {{ display: inline-block; background: #eee; width: 80px; height: 8px;
             border-radius: 2px; margin-right: 6px; vertical-align: middle; }}
  details {{ margin-top: 8px; }}
  summary {{ cursor: pointer; user-select: none; font-weight: 600; }}
  .outcome-correct {{ color: #1a8a32; }}
  .outcome-fail {{ color: #c1351a; }}
  .outcome-other {{ color: #a36b00; }}
</style>
</head>
<body>
<h1>Ingest-tool benchmark — <code>{run_name}</code></h1>
<div class="sub">{n_trials} trials · {n_cells} cells · generated from
<code>{trials_path}</code></div>

<div class="filters">
  <label>Model:</label> <span class="chips" id="f-model"></span>
  <label>User lang:</label> <span class="chips" id="f-lang"></span>
  <label>Tool prompt lang:</label> <span class="chips" id="f-tpl"></span>
  <label>Tier:</label> <span class="chips" id="f-tier"></span>
  <label>Expected behavior:</label> <span class="chips" id="f-eb"></span>
  <label>Group by:</label>
  <div class="pair">
    <select id="groupby">
      <option value="model">model</option>
      <option value="case_name">case</option>
      <option value="tier">tier</option>
      <option value="expected_behavior">expected_behavior</option>
      <option value="lang">user_lang</option>
      <option value="tool_prompt_lang">tool_prompt_lang</option>
      <option value="outcome">outcome</option>
    </select>
    <label>Then by:</label>
    <select id="groupby2">
      <option value="">(none)</option>
      <option value="model">model</option>
      <option value="case_name">case</option>
      <option value="tier">tier</option>
      <option value="expected_behavior">expected_behavior</option>
      <option value="lang">user_lang</option>
      <option value="tool_prompt_lang">tool_prompt_lang</option>
      <option value="outcome">outcome</option>
    </select>
  </div>
</div>

<div class="card-row" id="cards"></div>

<h2 style="font-size: 14px; margin: 18px 0 8px;">Group summary</h2>
<div id="group-table"></div>

<details>
  <summary>Outcome distribution (filtered)</summary>
  <div id="outcome-table" style="margin-top: 10px;"></div>
</details>

<details>
  <summary>Per-trial detail (filtered, capped at 500 rows)</summary>
  <div id="trial-table" style="margin-top: 10px;"></div>
</details>

<script>
const TRIALS = {trials_json};
const CORRECT_OUTCOMES = new Set(["correct","correct_with_extra","asked_with_call","correct_text_ask","called_without_asking"]);
const FACETS = ["model","lang","tool_prompt_lang","tier","expected_behavior"];
const FACET_LABELS = {{model:"f-model", lang:"f-lang", tool_prompt_lang:"f-tpl",
                       tier:"f-tier", expected_behavior:"f-eb"}};

// per-facet active set; empty = all
const active = {{}};
FACETS.forEach(f => active[f] = new Set());

function uniq(arr) {{ return [...new Set(arr)].sort(); }}

function buildPills() {{
  FACETS.forEach(f => {{
    const host = document.getElementById(FACET_LABELS[f]);
    host.innerHTML = "";
    uniq(TRIALS.map(t => t[f])).forEach(v => {{
      const pill = document.createElement("span");
      pill.className = "pill on";
      pill.textContent = v ?? "(null)";
      pill.dataset.value = v ?? "";
      active[f].add(v);
      pill.onclick = () => {{
        if (active[f].has(v)) {{
          active[f].delete(v);
          pill.classList.remove("on");
        }} else {{
          active[f].add(v);
          pill.classList.add("on");
        }}
        render();
      }};
      host.appendChild(pill);
    }});
  }});
}}

function filtered() {{
  return TRIALS.filter(t => FACETS.every(f => active[f].has(t[f])));
}}

function fmt(x, d=3) {{
  if (x == null || isNaN(x)) return "—";
  return Number(x).toFixed(d);
}}

function renderCards(rows) {{
  const n = rows.length;
  const nCorrect = rows.filter(t => CORRECT_OUTCOMES.has(t.outcome)).length;
  const nNoTool = rows.filter(t => t.no_tool).length;
  const nErr = rows.filter(t => t.error).length;
  const nAsk = rows.reduce((a,t) => a + (t.ask_user_q_count||0), 0);
  const lats = rows.filter(t => !t.error).map(t => t.latency_ms);
  const meanLat = lats.length ? lats.reduce((a,b)=>a+b,0) / lats.length : 0;
  const judge = rows.filter(t => t.judge_score != null).map(t => t.judge_score);
  const meanJudge = judge.length ? judge.reduce((a,b)=>a+b,0) / judge.length : null;
  const cards = [
    ["TRIALS", n],
    ["CORRECT", `${{nCorrect}} (${{n ? (nCorrect/n*100).toFixed(0):0}}%)`],
    ["NO TOOL", `${{nNoTool}} (${{n ? (nNoTool/n*100).toFixed(0):0}}%)`],
    ["ASK CALLS", nAsk],
    ["ERRORS", nErr],
    ["MEAN LATENCY", `${{meanLat.toFixed(0)}}ms`],
  ];
  if (meanJudge != null) cards.push(["JUDGE MEAN", meanJudge.toFixed(2)]);
  const host = document.getElementById("cards");
  host.innerHTML = cards.map(([lbl,big]) =>
    `<div class="card"><div class="lbl">${{lbl}}</div><div class="big">${{big}}</div></div>`
  ).join("");
}}

function bar(rate) {{
  const pct = Math.round(rate * 100);
  const cls = rate >= 0.8 ? "outcome-correct" : (rate >= 0.4 ? "outcome-other" : "outcome-fail");
  return `<span class="bar-bg"><span class="bar" style="width:${{pct}}%"></span></span><span class="${{cls}}">${{pct}}%</span>`;
}}

function renderGroup(rows) {{
  const g1 = document.getElementById("groupby").value;
  const g2 = document.getElementById("groupby2").value;
  const groups = {{}};
  rows.forEach(t => {{
    const k = g2 ? `${{t[g1]}}||${{t[g2]}}` : String(t[g1]);
    (groups[k] ||= []).push(t);
  }});
  const keys = Object.keys(groups).sort();
  const headers = g2 ? [g1, g2] : [g1];
  let html = `<table><thead><tr>`;
  headers.forEach(h => html += `<th>${{h}}</th>`);
  html += `<th class="num">n</th><th>correct_rate</th><th class="num">no_tool%</th><th class="num">ask</th><th class="num">errors</th><th class="num">mean_ms</th><th class="num">judge_mean</th></tr></thead><tbody>`;
  keys.forEach(k => {{
    const cell = groups[k];
    const n = cell.length;
    const nCorrect = cell.filter(t => CORRECT_OUTCOMES.has(t.outcome)).length;
    const nNoTool = cell.filter(t => t.no_tool).length;
    const nAsk = cell.reduce((a,t)=>a+(t.ask_user_q_count||0),0);
    const nErr = cell.filter(t=>t.error).length;
    const lats = cell.filter(t=>!t.error).map(t=>t.latency_ms);
    const meanLat = lats.length ? lats.reduce((a,b)=>a+b,0)/lats.length : 0;
    const judge = cell.filter(t=>t.judge_score!=null).map(t=>t.judge_score);
    const meanJudge = judge.length ? judge.reduce((a,b)=>a+b,0)/judge.length : null;
    const parts = k.split("||");
    let row = `<tr>`;
    parts.forEach(p => row += `<td>${{p}}</td>`);
    row += `<td class="num">${{n}}</td>`;
    row += `<td>${{bar(nCorrect/n)}}</td>`;
    row += `<td class="num">${{(nNoTool/n*100).toFixed(0)}}%</td>`;
    row += `<td class="num">${{nAsk}}</td>`;
    row += `<td class="num">${{nErr}}</td>`;
    row += `<td class="num">${{meanLat.toFixed(0)}}</td>`;
    row += `<td class="num">${{meanJudge != null ? meanJudge.toFixed(2) : "—"}}</td>`;
    row += `</tr>`;
    html += row;
  }});
  html += `</tbody></table>`;
  document.getElementById("group-table").innerHTML = html;
}}

function renderOutcomeDist(rows) {{
  const counts = {{}};
  rows.forEach(t => counts[t.outcome] = (counts[t.outcome]||0)+1);
  const keys = Object.keys(counts).sort((a,b)=>counts[b]-counts[a]);
  const total = rows.length;
  let html = `<table><thead><tr><th>outcome</th><th class="num">n</th><th>share</th></tr></thead><tbody>`;
  keys.forEach(k => {{
    const c = counts[k];
    const cls = CORRECT_OUTCOMES.has(k) ? "outcome-correct" : "outcome-fail";
    html += `<tr><td class="${{cls}}">${{k}}</td><td class="num">${{c}}</td><td>${{bar(c/total)}}</td></tr>`;
  }});
  html += `</tbody></table>`;
  document.getElementById("outcome-table").innerHTML = html;
}}

function renderTrials(rows) {{
  const capped = rows.slice(0, 500);
  let html = `<table><thead><tr><th>request_id</th><th>model</th><th>u/t</th><th>case</th><th>outcome</th><th>tools</th><th class="num">ask</th><th class="num">ms</th><th class="num">judge</th></tr></thead><tbody>`;
  capped.forEach(t => {{
    const cls = CORRECT_OUTCOMES.has(t.outcome) ? "outcome-correct" : "outcome-fail";
    html += `<tr><td><code>${{t.request_id}}</code></td><td>${{t.model}}</td><td>${{t.lang}}/${{t.tool_prompt_lang}}</td><td>${{t.case_name}}</td><td class="${{cls}}">${{t.outcome}}</td><td>${{(t.tool_calls||[]).join(", ") || "—"}}</td><td class="num">${{t.ask_user_q_count}}</td><td class="num">${{t.latency_ms.toFixed(0)}}</td><td class="num">${{t.judge_score != null ? t.judge_score : "—"}}</td></tr>`;
  }});
  html += `</tbody></table>`;
  if (rows.length > capped.length) html += `<div class="sub">… ${{rows.length-capped.length}} more rows hidden</div>`;
  document.getElementById("trial-table").innerHTML = html;
}}

function render() {{
  const rows = filtered();
  renderCards(rows);
  renderGroup(rows);
  renderOutcomeDist(rows);
  renderTrials(rows);
}}

buildPills();
document.getElementById("groupby").onchange = render;
document.getElementById("groupby2").onchange = render;
render();
</script>
</body></html>
"""


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run",
        required=True,
        help="run directory containing trials.jsonl (and optional judge_*.jsonl)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="output HTML path (default: <run>/report.html)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run)
    trials_path = run_dir / "trials.jsonl"
    if not trials_path.exists():
        print(f"trials.jsonl not found: {trials_path}", file=sys.stderr)
        return 2

    trials = _load_jsonl(trials_path)
    judge_index = _load_judge_index(run_dir)
    light = [_light_trial(t, judge_index.get(t["request_id"])) for t in trials]

    n_cells = len(
        {(t["model"], t["lang"], t["tool_prompt_lang"], t["case_name"]) for t in trials}
    )

    html = _HTML_TEMPLATE.format(
        run_name=run_dir.name,
        n_trials=len(trials),
        n_cells=n_cells,
        trials_path=trials_path,
        trials_json=json.dumps(light, ensure_ascii=False),
    )

    out_path = Path(args.out) if args.out else run_dir / "report.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path} ({len(light)} trials embedded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
