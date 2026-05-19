#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_PORT="${JR_INTERVIEW_API_PORT:-8130}"
WEB_PORT="${JR_INTERVIEW_WEB_PORT:-3220}"
API_BASE_URL="${JR_INTERVIEW_API_BASE_URL:-http://127.0.0.1:${API_PORT}}"
WEB_BASE_URL="${JR_INTERVIEW_WEB_BASE_URL:-http://127.0.0.1:${WEB_PORT}}"
API_KEY="${JR_INTERVIEW_API_KEY:-$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')}"
DATA_DIR="${JR_INTERVIEW_DATA_DIR:-${ROOT_DIR}/.tmp/interview-demo-data}"
TMP_DIR="$(mktemp -d)"
SMOKE_ONLY=0

usage() {
  cat <<'EOF'
Usage: bash scripts/interview-demo.sh [--smoke]

Starts an auth-enabled local JR AutoRAG interview demo, seeds the demo corpus,
and prints the exact URL and API key. Use --smoke to verify startup and exit.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      SMOKE_ONLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cleanup() {
  kill "${API_PID:-}" "${WEB_PID:-}" >/dev/null 2>&1 || true
  wait "${API_PID:-}" "${WEB_PID:-}" >/dev/null 2>&1 || true
  if [[ "${JR_INTERVIEW_KEEP_LOGS:-0}" == "1" ]]; then
    printf 'interview_demo_logs=%s\n' "${TMP_DIR}"
  else
    rm -rf "${TMP_DIR}"
  fi
}
trap cleanup EXIT INT TERM

need_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$1" >&2
    exit 127
  fi
}

resolve_bun() {
  local path_entry
  IFS=":" read -r -a path_entries <<< "${PATH}"
  for path_entry in "${path_entries[@]}"; do
    case "${path_entry}" in
      */node_modules/.bin) continue ;;
    esac
    if [[ -x "${path_entry}/bun" ]]; then
      printf '%s\n' "${path_entry}/bun"
      return 0
    fi
  done
  command -v bun
}

check_port_available() {
  local port="$1"
  local label="$2"
  if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1; then
    printf '%s port %s is already in use. Set JR_INTERVIEW_%s_PORT to an available port.\n' \
      "${label}" "${port}" "$(printf '%s' "${label}" | tr '[:lower:]' '[:upper:]')" >&2
    exit 1
  fi
}

wait_for_url() {
  local label="$1"
  local url="$2"
  local output="$3"
  local pid="$4"
  local code
  for _ in {1..120}; do
    code="$(curl -sS -o /dev/null -w '%{http_code}' "${url}" 2>/dev/null || true)"
    if [[ "${code}" == "200" ]]; then
      return 0
    fi
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      printf '%s server exited before becoming ready.\n' "${label}" >&2
      tail -n 120 "${output}" >&2 || true
      exit 1
    fi
    sleep 0.25
  done
  printf '%s server did not become ready at %s.\n' "${label}" "${url}" >&2
  tail -n 120 "${output}" >&2 || true
  exit 1
}

need_command bun
need_command curl
need_command python3
need_command uv
BUN_BIN="$(resolve_bun)"

check_port_available "${API_PORT}" "api"
check_port_available "${WEB_PORT}" "web"

rm -rf "${DATA_DIR}"
mkdir -p "${DATA_DIR}"

(
  cd "${ROOT_DIR}"
  env \
    AUTORAG_AUTH_ENABLED=true \
    AUTORAG_API_KEYS="${API_KEY}" \
    AUTORAG_EXPOSE=false \
    AUTORAG_ALLOWED_ORIGINS="${WEB_BASE_URL}" \
    AUTORAG_RATE_LIMIT_ENABLED=true \
    AUTORAG_RAGFUZZ_ENABLED=false \
    AUTORAG_PII_REDACT=true \
    JR_DEMO_MODE=1 \
    JR_DATA_DIR="${DATA_DIR}" \
    PYTHONPATH="${ROOT_DIR}/api" \
    uv run --project api uvicorn app.main:app --host 127.0.0.1 --port "${API_PORT}"
) > "${TMP_DIR}/api.log" 2>&1 &
API_PID="$!"

wait_for_url "api" "${API_BASE_URL}/healthz" "${TMP_DIR}/api.log" "${API_PID}"

curl -fsS -X DELETE -H "X-API-Key: ${API_KEY}" "${API_BASE_URL}/onboarding/demo" >/dev/null || true
curl -fsS -X POST -H "X-API-Key: ${API_KEY}" "${API_BASE_URL}/onboarding/demo/seed" >/dev/null

(
  cd "${ROOT_DIR}"
  env -i \
    PATH="${PATH}" \
    HOME="${HOME}" \
    BUN_PUBLIC_API_BASE_URL="${API_BASE_URL}" \
    PORT="${WEB_PORT}" \
    "${BUN_BIN}" src/index.ts
) > "${TMP_DIR}/web.log" 2>&1 &
WEB_PID="$!"

wait_for_url "web" "${WEB_BASE_URL}/" "${TMP_DIR}/web.log" "${WEB_PID}"

printf 'interview_demo=ready\n'
printf 'ui_url=%s\n' "${WEB_BASE_URL}"
printf 'api_url=%s\n' "${API_BASE_URL}"
printf 'api_key=%s\n' "${API_KEY}"
printf 'data_dir=%s\n' "${DATA_DIR}"
printf 'notes=Use the API key field in the UI if prompted. Press Ctrl-C here to stop and clean up.\n'

if [[ "${SMOKE_ONLY}" == "1" ]]; then
  exit 0
fi

while kill -0 "${API_PID}" >/dev/null 2>&1 && kill -0 "${WEB_PID}" >/dev/null 2>&1; do
  sleep 1
done

printf 'interview_demo=stopped_unexpectedly\n' >&2
tail -n 80 "${TMP_DIR}/api.log" >&2 || true
tail -n 80 "${TMP_DIR}/web.log" >&2 || true
exit 1
