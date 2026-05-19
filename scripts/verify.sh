#!/usr/bin/env bash
set -euo pipefail

cd api
uv run ruff check app tests
uv run pytest
cd ..

bash ./scripts/evidence-bundle.test.sh
bash ./scripts/client-handoff-gate.test.sh
bash ./scripts/research-architecture-check.sh
./node_modules/.bin/tsc --noEmit
bun test
bash ./scripts/build.sh
