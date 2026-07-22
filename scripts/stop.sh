#!/usr/bin/env bash
# Stop any combination of the five ClarityMed inference servers via pidfile.
# SIGTERM first, then SIGKILL after 10 s of grace. Also runs
# ``pkill -f <pattern>`` as an unconditional post-sweep — catches:
#   * pidfile missing entirely (server started outside this script);
#   * pidfile PID died but the orphan child still holds the port
#     (e.g. shell that ran run.sh exited, `uv run` wrapper adopted by
#     init, python server child kept listening). Without the sweep,
#     the next restart hits Errno 48 "address already in use" because
#     stop.sh silently left the orphan alive.
#
# Symptoms uses the full console-script name (`claritymed-symptoms-server`)
# as its pkill pattern so it does NOT match the unrelated training CLI
# `claritymed-symptoms-train-ddxplus`. Same precision applied to vision and
# medical-clip to avoid collateral kills.
#
# Usage:
#   scripts/stop.sh                # stop all five servers
#   scripts/stop.sh embedder
#   scripts/stop.sh reranker
#   scripts/stop.sh symptoms       # only the symptoms server
#   scripts/stop.sh vision         # only the vision server (:8085)
#   scripts/stop.sh medical-clip   # only the medical-clip server (:8086)
#   scripts/stop.sh both           # RAG only (embedder + reranker), legacy
#   scripts/stop.sh all            # explicit form of the no-arg default

set -euo pipefail

HOME_DIR="${CLARITYMED_HOME:-$HOME/.claritymed}"
RUN_DIR="$HOME_DIR/run"

stop_one() {
  local name="$1"
  local pkill_pattern="$2"  # exact console-script name for the pkill sweep
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
    echo "[$name] no pidfile"
  fi

  # Unconditional post-sweep. Two failure modes this catches:
  #   1. No pidfile at all — server started outside this script.
  #   2. Pidfile PID is dead but its orphan child is still listening on
  #      the port. Real-world path: parent shell dies, ``uv run``
  #      wrapper reparents to init (ppid=1), the python server child
  #      keeps the socket. Without this sweep the next run.sh hits
  #      Errno 48 and restart silently fails.
  if pkill -f "$pkill_pattern" 2>/dev/null; then
    echo "[$name] killed leftover(s) via pkill (pattern: $pkill_pattern)"
  fi
}

case "${1:-all}" in
  embedder)     stop_one embedder     claritymed-embedder ;;
  reranker)     stop_one reranker     claritymed-reranker ;;
  symptoms)     stop_one symptoms     claritymed-symptoms-server ;;
  vision)       stop_one vision       claritymed-vision-server ;;
  medical-clip) stop_one medical-clip claritymed-medical-clip-server ;;
  both)
    stop_one embedder claritymed-embedder
    stop_one reranker claritymed-reranker
    ;;
  all)
    stop_one embedder     claritymed-embedder
    stop_one reranker     claritymed-reranker
    stop_one symptoms     claritymed-symptoms-server
    stop_one vision       claritymed-vision-server
    stop_one medical-clip claritymed-medical-clip-server
    ;;
  *)
    echo "Usage: $0 [embedder|reranker|symptoms|vision|medical-clip|both|all]" >&2
    exit 2
    ;;
esac
