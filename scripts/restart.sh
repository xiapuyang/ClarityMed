#!/usr/bin/env bash
# Stop + start any combination of the five ClarityMed inference servers.
# Same scope argument shape as run/stop — `"$@"` is forwarded verbatim,
# so any arg they accept works here.
#
# Usage:
#   scripts/restart.sh             # restart all five servers
#   scripts/restart.sh embedder
#   scripts/restart.sh reranker
#   scripts/restart.sh symptoms    # only the symptoms server
#   scripts/restart.sh vision      # only the vision server (:8085)
#   scripts/restart.sh medical-clip  # only the medical-clip server (:8086)
#   scripts/restart.sh both        # RAG only (embedder + reranker), legacy
#   scripts/restart.sh all         # explicit form of the no-arg default
#
# Drops `set -e` deliberately. If one of the server lifecycle hops
# returns non-zero (e.g. embedder /health was slow but the process is
# fine), we still want to attempt the others; otherwise a transient
# embedder hiccup would block the reranker/symptoms restart and leave
# their ports down. run.sh / stop.sh each report their own pass/fail
# to stderr.

set -uo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
"$here/stop.sh" "$@"
"$here/run.sh" "$@"
