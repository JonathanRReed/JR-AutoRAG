#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_DIR="${ROOT_DIR}/api"
VENV_PATH="${VENV_PATH:-${ROOT_DIR}/.venv}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"

if [[ -x "${VENV_PATH}/bin/python" ]]; then
  API_PYTHON="${VENV_PATH}/bin/python"
else
  API_PYTHON="$(command -v python3)"
fi

cleanup() {
  if [[ -n "${API_PID:-}" ]]; then
    kill "${API_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT

echo "Starting API at http://${API_HOST}:${API_PORT}..."
(
  cd "${API_DIR}"
  PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::UserWarning:resource_tracker}" \
    PYTHONPATH=. "${API_PYTHON}" -m uvicorn app.main:app --reload --host "${API_HOST}" --port "${API_PORT}"
) &
API_PID=$!

echo "Starting web UI on http://localhost:3000..."
cd "${ROOT_DIR}"
bun dev
