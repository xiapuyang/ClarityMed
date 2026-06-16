#!/usr/bin/env bash
# Start the BGE-M3 embedder + bge-reranker-v2-m3 reranker (RAG), the
# typed-BASD symptoms differential-diagnosis server, and/or the BUSI
# vision inference server as background processes.
# Logs go to $CLARITYMED_LOG_DIR (default ~/.claritymed/logs/),
# pidfiles to $CLARITYMED_HOME/run/.
#
# Idempotent: a server whose pidfile points to a live process is left
# alone. After spawning, polls /health for up to RAG_HEALTH_TIMEOUT_S
# seconds (default 180); if the process dies during boot it tails the
# last log lines and exits non-zero.
#
# Why 180 s default: cold reload of bge-reranker-v2-m3 under CPU
# contention with a just-started embedder regularly crosses 90 s on
# Apple Silicon. The symptoms server is much smaller (~9 MB typed-BASD
# weights) but reuses the same timeout for simplicity. Override per-call:
#   RAG_HEALTH_TIMEOUT_S=300 scripts/run.sh
#
# Symptoms server notes:
#   - port 8084; extra `symptoms-server`; console script
#     `claritymed-symptoms-server`.
#   - Lifespan reads configs/symptoms.yaml and loads every dataset whose
#     `enabled: true`. `enabled: false` (the shipped default) boots a
#     valid /health but /v1/datasets/* will 404 until you flip it.
#   - CLARITYMED_SYMPTOMS_SKIP_LOAD=1 boots without touching weights —
#     useful when you only want the FastAPI surface up for plugin tests.
#
# Vision server notes:
#   - port 8085; extra `vision-server`; console script
#     `claritymed-vision-server`.
#   - Requires `uv sync --extra vision-server` (torch/torchvision/smp).
#   - Lifespan reads configs/vision.yaml and loads the model whose
#     weights_subpath is listed there. Needs a deployed checkpoint
#     (run the pipeline first).
#   - CLARITYMED_VISION_SKIP_LOAD=1 boots without loading weights —
#     useful when you only want the FastAPI surface up for plugin tests.
#
# Medical-CLIP server notes:
#   - port 8086; extra `medical-clip-server`; console script
#     `claritymed-medical-clip-server`.
#   - Requires `uv sync --extra medical-clip-server` (open_clip_torch).
#   - Tags images with DICOM modality — consumed by the vision plugin
#     before sending to vision-server. Required for vision e2e tests.
#
# Usage:
#   scripts/run.sh                # start all five servers
#   scripts/run.sh embedder       # only the embedder
#   scripts/run.sh reranker
#   scripts/run.sh symptoms       # only the symptoms server
#   scripts/run.sh vision         # only the vision server
#   scripts/run.sh medical-clip   # only the medical-clip server
#   scripts/run.sh both           # RAG only (embedder + reranker), legacy
#   scripts/run.sh all            # explicit form of the no-arg default

set -euo pipefail

HOME_DIR="${CLARITYMED_HOME:-$HOME/.claritymed}"
LOG_DIR="${CLARITYMED_LOG_DIR:-$HOME_DIR/logs}"
RUN_DIR="$HOME_DIR/run"
HEALTH_TIMEOUT_S="${RAG_HEALTH_TIMEOUT_S:-180}"
mkdir -p "$LOG_DIR" "$RUN_DIR"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

start_one() {
  local name="$1"   # embedder | reranker | symptoms — drives pidfile / log / labels
  local port="$2"
  local extra="$3"  # uv extra: rag-server | symptoms-server
  local bin="$4"    # console script: claritymed-embedder / -reranker / -symptoms-server
  local pidfile="$RUN_DIR/$name.pid"
  local logfile="$LOG_DIR/$name.log"

  if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "[$name] already running (pid $(cat "$pidfile"))"
    return 0
  fi

  echo "[$name] starting → $logfile"
  # nohup so the process survives this shell; uv handles venv + extras.
  # Append (>>) so prior crash logs are preserved for postmortem.
  nohup uv run --extra "$extra" "$bin" >> "$logfile" 2>&1 &
  echo $! > "$pidfile"

  # Health poll. Cold reload of the cross-encoder on CPU + contention
  # with an embedder that just woke up regularly takes >90 s; the
  # default 180 s leaves 3x headroom for typical Apple Silicon CPU.
  local pid
  pid="$(cat "$pidfile")"
  local i
  for i in $(seq 1 "$HEALTH_TIMEOUT_S"); do
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "[$name] ready on :$port (pid $pid)"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[$name] process died during startup — last 20 log lines:" >&2
      tail -20 "$logfile" >&2
      rm -f "$pidfile"
      return 1
    fi
    sleep 1
  done

  # Timeout but process still alive — most likely still loading. We
  # leave the process + pidfile so the user can `tail -f $logfile`,
  # then re-run this script (will hit the "already running" path and
  # confirm /health) when they're ready.
  echo "[$name] timed out after ${HEALTH_TIMEOUT_S}s waiting for /health" >&2
  echo "[$name] process is still alive (pid $pid) — likely still loading" >&2
  echo "[$name] tail -f $logfile  # watch progress" >&2
  echo "[$name] $0 $name           # re-run when curl /health works" >&2
  return 1
}

case "${1:-all}" in
  embedder)     start_one embedder     8082 rag-server           claritymed-embedder ;;
  reranker)     start_one reranker     8083 rag-server           claritymed-reranker ;;
  symptoms)     start_one symptoms     8084 symptoms-server      claritymed-symptoms-server ;;
  vision)       start_one vision       8085 vision-server        claritymed-vision-server ;;
  medical-clip) start_one medical-clip 8086 medical-clip-server  claritymed-medical-clip-server ;;
  both)
    start_one embedder 8082 rag-server claritymed-embedder
    start_one reranker 8083 rag-server claritymed-reranker
    ;;
  all)
    start_one embedder     8082 rag-server          claritymed-embedder
    start_one reranker     8083 rag-server          claritymed-reranker
    start_one symptoms     8084 symptoms-server     claritymed-symptoms-server
    start_one vision       8085 vision-server       claritymed-vision-server
    start_one medical-clip 8086 medical-clip-server claritymed-medical-clip-server
    ;;
  *)
    echo "Usage: $0 [embedder|reranker|symptoms|vision|medical-clip|both|all]" >&2
    exit 2
    ;;
esac
