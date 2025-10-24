#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="/Volumes/DatenAP/Code/api-ai"
BUILD_COMMAND="echo '✅ Python API - no build needed'"

usage() {
  cat <<'USAGE'
Usage: build-local.sh [--clean]

Runs the configured build command inside the repository root to verify the
release build. Optionally deletes the dist/ directory first when --clean is
provided.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

clean_flag=false
if [[ "${1:-}" == "--clean" ]]; then
  clean_flag=true
fi

cd "$REPO_ROOT"

if [[ "$clean_flag" == true && -d dist ]]; then
  rm -rf dist
fi

# shellcheck disable=SC2086
$BUILD_COMMAND

echo "✅ Local build finished. Output: ${REPO_ROOT}/dist"
