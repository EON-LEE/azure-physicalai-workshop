#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/dev-env.sh"
cd "$PHYSICALAI_ROOT"
uv run --locked ruff check agents apps/api simulation scripts tests contracts
uv run --locked ruff format --check agents apps/api simulation scripts tests contracts
uv run --locked pytest -q --junitxml=test-results/runtime.xml
uv run --locked python tests/export_api_contract.py
node --experimental-strip-types scripts/check-api-contract.mjs
(
  cd apps/web
  npm run typecheck
  npm test -- --reporter=dot --reporter=junit --outputFile=test-results/vitest.xml
  npm run build
)
