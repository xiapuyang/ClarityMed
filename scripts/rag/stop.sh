#!/usr/bin/env bash
# Stop the two RAG servers via pidfile. SIGTERM first, then SIGKILL
# after 10 s of grace. Falls back to ``pkill -f claritymed-<name>`` if
# the pidfile is missing or stale (covers servers started outside this
# script).
#
# Usage:
#   scripts/rag/stop.sh                # stop both
#   scripts/rag/stop.sh embedder
#   scripts/rag/stop.sh reranker

set -euo pipefail

HOME_DIR="${CLARITYMED_HOME:-$HOME/.claritymed}"
RUN_DIR="$HOME_DIR/run"

stop_one() {
  local name="$1"
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
    if pkill -f "claritymed-$name" 2>/dev/null; then
      echo "[$name] killed via pkill"
    else
      echo "[$name] nothing to stop"
    fi
  fi
}

case "${1:-both}" in
  embedder) stop_one embedder ;;
  reranker) stop_one reranker ;;
  both)
    stop_one embedder
    stop_one reranker
    ;;
  *)
    echo "Usage: $0 [embedder|reranker|both]" >&2
    exit 2
    ;;
esac
