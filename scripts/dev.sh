#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_DIR="${ROOT_DIR}/api"
VENV_PATH="${VENV_PATH:-${ROOT_DIR}/.venv}"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
WEB_PORT="${PORT:-3000}"

API_RUNNER=()
if [[ -x "${API_DIR}/.venv/bin/python" ]]; then
  API_RUNNER=("${API_DIR}/.venv/bin/python" -m)
elif command -v uv >/dev/null 2>&1; then
  API_RUNNER=(uv run --project "${API_DIR}" python -m)
elif [[ -x "${VENV_PATH}/bin/python" ]]; then
  API_RUNNER=("${VENV_PATH}/bin/python" -m)
elif command -v python3 >/dev/null 2>&1; then
  API_RUNNER=("$(command -v python3)" -m)
else
  printf 'Python 3 or uv is required to start the API.\n' >&2
  exit 1
fi

cleanup() {
  if [[ -n "${API_PID:-}" ]]; then
    kill "${API_PID}" 2>/dev/null || true
  fi
}

trap cleanup EXIT

echo "Starting API at http://${API_HOST}:${API_PORT}..."
(
  cd "${ROOT_DIR}"
  PYTHONWARNINGS="${PYTHONWARNINGS:-ignore::UserWarning:resource_tracker}" \
    PYTHONPATH="${API_DIR}" "${API_RUNNER[@]}" uvicorn app.main:app --reload --host "${API_HOST}" --port "${API_PORT}"
) &
API_PID=$!

echo "Starting web UI on http://localhost:${WEB_PORT}..."
cd "${ROOT_DIR}"
BUN_PUBLIC_API_BASE_URL="${BUN_PUBLIC_API_BASE_URL:-http://${API_HOST}:${API_PORT}}" \
  VITE_API_BASE_URL="${VITE_API_BASE_URL:-http://${API_HOST}:${API_PORT}}" \
  PORT="${WEB_PORT}" \
  env \
    -u npm_command \
    -u npm_config_user_agent \
    -u npm_execpath \
    -u npm_node_execpath \
    bun dev
