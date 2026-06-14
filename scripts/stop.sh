#!/usr/bin/env bash
# Stop the RAG servers (embedder, reranker) and/or the symptoms server
# via pidfile. SIGTERM first, then SIGKILL after 10 s of grace. Falls
# back to ``pkill -f <pattern>`` if the pidfile is missing or stale
# (covers servers started outside this script).
#
# Symptoms uses the full console-script name (`claritymed-symptoms-server`)
# as its pkill pattern so it does NOT match the unrelated training CLI
# `claritymed-symptoms-train-ddxplus`.
#
# Usage:
#   scripts/stop.sh                # stop all three (embedder + reranker + symptoms)
#   scripts/stop.sh embedder
#   scripts/stop.sh reranker
#   scripts/stop.sh symptoms       # only the symptoms server
#   scripts/stop.sh both           # RAG only (embedder + reranker), legacy
#   scripts/stop.sh all            # explicit form of the no-arg default

set -euo pipefail

HOME_DIR="${CLARITYMED_HOME:-$HOME/.claritymed}"
RUN_DIR="$HOME_DIR/run"

stop_one() {
  local name="$1"
  local pkill_pattern="$2"  # exact console-script name for the pkill fallback
  local pidfile="$RUN_DIR/$name.pid"

  if [ -f "$pidfile" ]; then
    local pid
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null; then
      echo "[$name] stopping pid $pid"
      kill "$pid"
      # Graceful: poll up to 10 s.
      local i
      for i in $(seq 1 10); do
        if ! kill -0 "$pid" 2>/dev/null; then
          break
        fi
        sleep 1
      done
      if kill -0 "$pid" 2>/dev/null; then
        echo "[$name] still alive after 10 s — sending SIGKILL"
        kill -9 "$pid" 2>/dev/null || true
      fi
    else
      echo "[$name] pidfile stale (pid $pid not alive)"
    fi
    rm -f "$pidfile"
  else
    echo "[$name] no pidfile — trying pkill"
    if pkill -f "$pkill_pattern" 2>/dev/null; then
      echo "[$name] killed via pkill"
    else
      echo "[$name] nothing to stop"
    fi
  fi
}

case "${1:-all}" in
  embedder) stop_one embedder claritymed-embedder ;;
  reranker) stop_one reranker claritymed-reranker ;;
  symptoms) stop_one symptoms claritymed-symptoms-server ;;
  both)
    stop_one embedder claritymed-embedder
    stop_one reranker claritymed-reranker
    ;;
  all)
    stop_one embedder claritymed-embedder
    stop_one reranker claritymed-reranker
    stop_one symptoms claritymed-symptoms-server
    ;;
  *)
    echo "Usage: $0 [embedder|reranker|symptoms|both|all]" >&2
    exit 2
    ;;
esac
