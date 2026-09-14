#!/usr/bin/env bash
# Stop + start the user-facing web stack. Same scope arg shape as
# ``web-run.sh`` / ``web-stop.sh``: the first arg is the mode (dev or
# prod, default dev) and the optional second arg picks a single
# service. Other args forward verbatim.
#
# Drops ``set -e`` deliberately: if one service's lifecycle hops
# returns non-zero (slow boot, transient pidfile race), we still want
# to attempt the others rather than block the rest of the stack on
# one flaky hop. ``web-run.sh`` / ``web-stop.sh`` each report their
# own pass/fail to stderr.
#
# Usage:
#   scripts/web-restart.sh                       # dev, all three
#   scripts/web-restart.sh prod                  # prod, all three
#   scripts/web-restart.sh dev web               # only claritymed-web
#   scripts/web-restart.sh prod claritymed_ui

set -uo pipefail

mode="${1:-dev}"
scope="${2:-all}"

if [[ "$mode" != "dev" && "$mode" != "prod" ]]; then
  echo "Usage: $0 [dev|prod] [web|admin_ui|claritymed_ui|all]" >&2
  exit 2
fi

here="$(cd "$(dirname "$0")" && pwd)"
"$here/web-stop.sh" "$scope"
"$here/web-run.sh"  "$mode" "$scope"
