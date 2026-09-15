#!/usr/bin/env bash
# Source this file from the repository root in Ubuntu WSL.
if [[ "$(uname -s)" != "Linux" ]]; then
  printf '%s\n' 'Use Ubuntu WSL/Linux, not Windows Python or Node.' >&2
  return 1 2>/dev/null || exit 1
fi
PHYSICALAI_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHYSICALAI_CACHE_KEY="$(printf '%s' "$PHYSICALAI_ROOT" | sha256sum | cut -c1-16)"
export PHYSICALAI_WSL_CACHE="${PHYSICALAI_WSL_CACHE:-$HOME/.cache/physicalai-workshop/$PHYSICALAI_CACHE_KEY}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PHYSICALAI_WSL_CACHE/venv}"
export PHYSICALAI_ROOT
