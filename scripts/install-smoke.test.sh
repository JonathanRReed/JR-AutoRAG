#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

JR_INSTALL_SMOKE_SKIP_SYNC=1 \
JR_INSTALL_SMOKE_API_PORT="${JR_INSTALL_SMOKE_API_PORT:-8124}" \
JR_INSTALL_SMOKE_WEB_PORT="${JR_INSTALL_SMOKE_WEB_PORT:-3100}" \
  bash "${ROOT_DIR}/scripts/install-smoke.sh"
