#!/usr/bin/env bash
# Start the user-facing web stack: claritymed-web, admin_ui, and
# claritymed-ui. **Idempotent start, not restart**: if a pidfile says
# the service is alive OR the target port is already held, this
# script aborts non-zero with the suggestion to use ``web-restart.sh``.
# Killing live services without an explicit instruction belongs in
# ``web-stop.sh`` / ``web-restart.sh``.
#
# claritymed-ui lives in its own repo. The script delegates to that
# project's ``run.sh`` so each repo owns its own startup contract.
# Default location is ``../claritymed-ui`` sibling of this repo;
# override with ``CLARITYMED_UI_DIR``.
#
# Modes:
#   dev   ``CLARITYMED_DEV=1`` for claritymed-web; both Vite UIs run
#         ``npm run dev`` (HMR + watch).
#   prod  No DEV flag; both UIs run ``npm install && npm run build``
#         once, then ``npm run preview`` on the same ports (5174 and
#         5173) as dev — keeps the admin servers graph honest in both
#         modes.
#
# Ports:
#   claritymed-web   8120 (override CLARITYMED_WEB_PORT)
#   admin_ui         5174
#   claritymed-ui    5173
#
# Pidfiles: ``$CLARITYMED_HOME/run/{web,admin_ui,claritymed_ui}.pid``
# Logs:    ``$CLARITYMED_LOG_DIR/{web,admin_ui,claritymed_ui}.log``
# Boot timeout: ``WEB_HEALTH_TIMEOUT_S`` (default 60).
#
# Usage:
#   scripts/web-run.sh                       # dev, all three
#   scripts/web-run.sh prod                  # prod, all three
#   scripts/web-run.sh dev web               # only claritymed-web
#   scripts/web-run.sh dev admin_ui          # only admin_ui
#   scripts/web-run.sh prod claritymed_ui    # only claritymed-ui (prod)

set -uo pipefail

mode="${1:-dev}"
scope="${2:-all}"

if [[ "$mode" != "dev" && "$mode" != "prod" ]]; then
  echo "Usage: $0 [dev|prod] [web|admin_ui|claritymed_ui|all]" >&2
  exit 2
fi
case "$scope" in
  web|admin_ui|claritymed_ui|all) ;;
  *)
    echo "Usage: $0 [dev|prod] [web|admin_ui|claritymed_ui|all]" >&2
    exit 2
    ;;
esac

HOME_DIR="${CLARITYMED_HOME:-$HOME/.claritymed}"
LOG_DIR="${CLARITYMED_LOG_DIR:-$HOME_DIR/logs}"
RUN_DIR="$HOME_DIR/run"
HEALTH_TIMEOUT_S="${WEB_HEALTH_TIMEOUT_S:-60}"
mkdir -p "$LOG_DIR" "$RUN_DIR"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
admin_ui_dir="$repo_root/src/claritymed/web/admin_ui"
ui_dir="${CLARITYMED_UI_DIR:-}"
if [ -z "$ui_dir" ]; then
  if [ -d "$repo_root/../claritymed-ui" ]; then
    ui_dir="$(cd "$repo_root/../claritymed-ui" && pwd)"
  fi
fi

cd "$repo_root"

# Refuse to spawn when an instance is already alive — pidfile-tracked
# or just holding the port (covers things started manually like
# ``uv run claritymed-web`` in a terminal or ``npm run dev``).
# Returns non-zero so the caller aborts; suggests ``web-restart.sh``.
check_free() {
  local name="$1"
  local port="$2"
  local pidfile="$RUN_DIR/$name.pid"
  if [ -f "$pidfile" ]; then
    local pid
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "[$name] already running (pid $pid) — use scripts/web-restart.sh to restart" >&2
      return 1
    fi
    # Stale pidfile — drop it so the new instance can write a fresh one.
    rm -f "$pidfile"
  fi
  local pids
  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    # shellcheck disable=SC2086
    echo "[$name] port $port held by pid(s) $(echo $pids | tr '\n' ' ')— use scripts/web-restart.sh or scripts/web-stop.sh $name" >&2
    return 1
  fi
  return 0
}

# Spawn ``"$@"`` from ``$cwd`` as a background process, write its pid
# to ``$RUN_DIR/$name.pid``, redirect stdout+stderr to the log, and
# poll ``http://127.0.0.1:$port$hpath`` until it returns 200 or the
# child dies. Returns non-zero on either timeout or child death; in
# both cases the last 30 log lines go to stderr so the operator can
# triage without grepping.
start_bg() {
  local name="$1"; shift
  local port="$1"; shift
  local hpath="$1"; shift
  local cwd="$1"; shift
  local logfile="$LOG_DIR/$name.log"
  local pidfile="$RUN_DIR/$name.pid"

  echo "[$name] starting → $logfile"
  (
    cd "$cwd"
    nohup "$@" >> "$logfile" 2>&1 &
    echo $! > "$pidfile"
  )
  local pid
  pid="$(cat "$pidfile")"

  local i
  for i in $(seq 1 "$HEALTH_TIMEOUT_S"); do
    if curl --noproxy '127.0.0.1,localhost' -fsS "http://127.0.0.1:$port$hpath" >/dev/null 2>&1; then
      echo "[$name] ready on :$port (pid $pid)"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[$name] process died during startup — last 30 log lines:" >&2
      tail -30 "$logfile" >&2
      rm -f "$pidfile"
      return 1
    fi
    sleep 1
  done
  echo "[$name] timed out after ${HEALTH_TIMEOUT_S}s waiting for $hpath" >&2
  echo "[$name] process still alive (pid $pid) — tail -f $logfile" >&2
  return 1
}

# Install npm deps lazily: first run only, when node_modules is
# missing. Skips the multi-second lockfile check on warm reruns.
ensure_npm_deps() {
  local name="$1"
  local dir="$2"
  if [ ! -d "$dir/node_modules" ]; then
    echo "[$name] npm install (first run) in $dir"
    (cd "$dir" && npm install --no-audit --no-fund)
  fi
}

start_web() {
  check_free web 8120 || return $?
  if [ "$mode" = "dev" ]; then
    start_bg web 8120 /health "$repo_root" \
      env CLARITYMED_DEV=1 uv run claritymed-web
  else
    start_bg web 8120 /health "$repo_root" \
      uv run claritymed-web
  fi
}

start_admin_ui() {
  check_free admin_ui 5174 || return $?
  ensure_npm_deps admin_ui "$admin_ui_dir"
  if [ "$mode" = "dev" ]; then
    start_bg admin_ui 5174 /admin/ "$admin_ui_dir" \
      npm run dev
  else
    echo "[admin_ui] building dist..."
    (cd "$admin_ui_dir" && npm run build)
    start_bg admin_ui 5174 /admin/ "$admin_ui_dir" \
      npm run preview -- --port 5174 --strictPort
  fi
}

start_claritymed_ui() {
  if [ -z "$ui_dir" ] || [ ! -d "$ui_dir" ]; then
    echo "[claritymed_ui] skip — set CLARITYMED_UI_DIR or place" >&2
    echo "  the project at ../claritymed-ui sibling of this repo." >&2
    return 1
  fi
  if [ ! -x "$ui_dir/run.sh" ]; then
    echo "[claritymed_ui] $ui_dir/run.sh missing or not executable" >&2
    return 1
  fi
  "$ui_dir/run.sh" "$mode"
}

rc=0
case "$scope" in
  web)            start_web            || rc=$? ;;
  admin_ui)       start_admin_ui       || rc=$? ;;
  claritymed_ui)  start_claritymed_ui  || rc=$? ;;
  all)
    # Run claritymed-web first so the UIs have an upstream to proxy
    # /api and /auth to. UI failures don't abort web — they're
    # independent enough to triage separately.
    start_web           || rc=$?
    start_admin_ui      || rc=$?
    start_claritymed_ui || rc=$?
    ;;
esac
exit "$rc"
