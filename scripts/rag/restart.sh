#!/usr/bin/env bash
# Stop + start the RAG servers. Same scope argument shape as run/stop.
#
# Usage:
#   scripts/rag/restart.sh             # restart both
#   scripts/rag/restart.sh embedder
#   scripts/rag/restart.sh reranker
#
# Drops `set -e` deliberately. If one of the two server lifecycle hops
# returns non-zero (e.g. embedder /health was slow but the process is
# fine), we still want to attempt the other; otherwise a transient
# embedder hiccup would block the reranker restart and leave 8083 down.
# run.sh / stop.sh each report their own pass/fail to stderr.

set -uo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
"$here/stop.sh" "$@"
"$here/run.sh" "$@"
