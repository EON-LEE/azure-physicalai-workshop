#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/dev-env.sh"
cd "$PHYSICALAI_ROOT"
uv sync --locked --group dev
mkdir -p "$PHYSICALAI_WSL_CACHE/web"
cp apps/web/package.json apps/web/package-lock.json "$PHYSICALAI_WSL_CACHE/web/"
npm ci --prefix "$PHYSICALAI_WSL_CACHE/web" --no-audit --no-fund
target="$PHYSICALAI_WSL_CACHE/web/node_modules"
link="$PHYSICALAI_ROOT/apps/web/node_modules"
if [[ -L "$link" ]]; then
  if [[ "$(readlink -f "$link")" != "$(readlink -f "$target")" ]]; then
    printf '%s\n' 'Existing node_modules points elsewhere; inspect it before changing it.' >&2
    exit 1
  fi
elif [[ -e "$link" ]]; then
  printf '%s\n' 'Existing node_modules is not a managed link; no directory was deleted.' >&2
  exit 1
else
  ln -s "$target" "$link"
fi
printf 'Development dependencies ready in native Linux storage: %s\n' "$PHYSICALAI_WSL_CACHE"
