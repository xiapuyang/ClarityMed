#!/usr/bin/env bash
# Stop + start the RAG servers and/or the symptoms server. Same scope
# argument shape as run/stop — `"$@"` is forwarded verbatim, so any arg
# they accept (embedder | reranker | symptoms | both | all) works here.
#
# Usage:
#   scripts/restart.sh             # restart all three (embedder + reranker + symptoms)
#   scripts/restart.sh embedder
#   scripts/restart.sh reranker
#   scripts/restart.sh symptoms    # only the symptoms server
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
