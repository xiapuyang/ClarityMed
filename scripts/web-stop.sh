#!/usr/bin/env bash
# Stop the user-facing web stack: claritymed-web, admin_ui,
# claritymed-ui. Three layers, applied in order until the target port
# is free:
#
#   1. SIGTERM the pidfile-tracked process; 10s grace; SIGKILL.
#   2. Kill anyone still LISTENing on the target port (catches
#      instances started manually, outside this script).
#
# Layer 2 is what makes ``web-restart.sh`` reliable — a bare
# ``CLARITYMED_DEV=1 uv run claritymed-web`` in a terminal leaves no
# pidfile, but it owns the LISTEN socket, so the port probe finds it.
#
# Delegates ``claritymed_ui`` to that project's own ``stop.sh``
# (default location ``../claritymed-ui``, overridable via
# ``CLARITYMED_UI_DIR``).
#
# Usage:
#   scripts/web-stop.sh                       # all three
#   scripts/web-stop.sh web
#   scripts/web-stop.sh admin_ui
#   scripts/web-stop.sh claritymed_ui

set -uo pipefail

scope="${1:-all}"
case "$scope" in
  web|admin_ui|claritymed_ui|all) ;;
  *)
    echo "Usage: $0 [web|admin_ui|claritymed_ui|all]" >&2
    exit 2
    ;;
esac

HOME_DIR="${CLARITYMED_HOME:-$HOME/.claritymed}"
RUN_DIR="$HOME_DIR/run"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
ui_dir="${CLARITYMED_UI_DIR:-}"
if [ -z "$ui_dir" ]; then
  if [ -d "$repo_root/../claritymed-ui" ]; then
    ui_dir="$(cd "$repo_root/../claritymed-ui" && pwd)"
  fi
fi

# Graceful stop via pidfile. Returns 0 either way — we still want
# layer 2 (port-kill) to run as the ground-truth check.
stop_pidfile() {
  local name="$1"
  local pidfile="$RUN_DIR/$name.pid"
  [ -f "$pidfile" ] || return 0
  local pid
  pid="$(cat "$pidfile" 2>/dev/null || true)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    echo "[$name] stopping pid $pid"
    kill "$pid" 2>/dev/null || true
    local i
    for i in $(seq 1 10); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      echo "[$name] still alive after 10s — SIGKILL"
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi
  rm -f "$pidfile"
}

# Catch-all: free the port no matter who's holding it. SIGTERM with
# 2s grace, then SIGKILL.
free_port() {
  local name="$1"
  local port="$2"
  local pids
  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  [ -n "$pids" ] || return 0
  # shellcheck disable=SC2086
  echo "[$name] port $port held by pid(s) $(echo $pids | tr '\n' ' ')— terminating"
  kill $pids 2>/dev/null || true
  sleep 2
  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "[$name] port $port still held — SIGKILL"
    kill -9 $pids 2>/dev/null || true
    sleep 1
  fi
}

stop_one() {
  local name="$1"
  local port="$2"
  stop_pidfile "$name"
  free_port    "$name" "$port"
  # If both layers found nothing, say so.
  if ! lsof -tiTCP:"$port" -sTCP:LISTEN >/dev/null 2>&1 && [ ! -f "$RUN_DIR/$name.pid" ]; then
    [ "$port" != "0" ] && echo "[$name] port $port free"
  fi
}

stop_web()      { stop_one web      8120 ; }
stop_admin_ui() { stop_one admin_ui 5174 ; }
stop_claritymed_ui() {
  if [ -n "$ui_dir" ] && [ -x "$ui_dir/stop.sh" ]; then
    "$ui_dir/stop.sh"
  else
    echo "[claritymed_ui] $ui_dir/stop.sh missing — local fallback" >&2
    stop_one claritymed_ui 5173
  fi
}

case "$scope" in
  web)            stop_web ;;
  admin_ui)       stop_admin_ui ;;
  claritymed_ui)  stop_claritymed_ui ;;
  all)
    stop_web
    stop_admin_ui
    stop_claritymed_ui
    ;;
esac
