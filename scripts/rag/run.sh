#!/usr/bin/env bash
# Start the BGE-M3 embedder + bge-reranker-v2-m3 reranker as background
# processes. Logs go to $CLARITYMED_LOG_DIR (default ~/.claritymed/logs/),
# pidfiles to $CLARITYMED_HOME/run/.
#
# Idempotent: a server whose pidfile points to a live process is left
# alone. After spawning, polls /health for up to RAG_HEALTH_TIMEOUT_S
# seconds (default 180); if the process dies during boot it tails the
# last log lines and exits non-zero.
#
# Why 180 s default: cold reload of bge-reranker-v2-m3 under CPU
# contention with a just-started embedder regularly crosses 90 s on
# Apple Silicon. Override per-call when needed:
#   RAG_HEALTH_TIMEOUT_S=300 scripts/rag/run.sh
#
# Usage:
#   scripts/rag/run.sh                # start both
#   scripts/rag/run.sh embedder       # only the embedder
#   scripts/rag/run.sh reranker

set -euo pipefail

HOME_DIR="${CLARITYMED_HOME:-$HOME/.claritymed}"
LOG_DIR="${CLARITYMED_LOG_DIR:-$HOME_DIR/logs}"
RUN_DIR="$HOME_DIR/run"
HEALTH_TIMEOUT_S="${RAG_HEALTH_TIMEOUT_S:-180}"
mkdir -p "$LOG_DIR" "$RUN_DIR"

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo_root"

start_one() {
  local name="$1"   # embedder | reranker
  local port="$2"
  local pidfile="$RUN_DIR/$name.pid"
  local logfile="$LOG_DIR/$name.log"

  if [ -f "$pidfile" ] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then
    echo "[$name] already running (pid $(cat "$pidfile"))"
    return 0
  fi

  echo "[$name] starting → $logfile"
  # nohup so the process survives this shell; uv handles venv + extras.
  # Append (>>) so prior crash logs are preserved for postmortem.
  nohup uv run --extra rag-server "claritymed-$name" >> "$logfile" 2>&1 &
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

case "${1:-both}" in
  embedder) start_one embedder 8082 ;;
  reranker) start_one reranker 8083 ;;
  both)
    start_one embedder 8082
    start_one reranker 8083
    ;;
  *)
    echo "Usage: $0 [embedder|reranker|both]" >&2
    exit 2
    ;;
esac
