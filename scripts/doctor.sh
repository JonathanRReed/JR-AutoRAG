#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_DIR="${ROOT_DIR}/api"
JSON_MODE=0
CHECKS_FILE="$(mktemp)"

cleanup() {
  rm -f "${CHECKS_FILE}"
}
trap cleanup EXIT

for arg in "$@"; do
  case "${arg}" in
    --json) JSON_MODE=1 ;;
    -h|--help)
      printf 'Usage: bash scripts/doctor.sh [--json]\n'
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "${arg}" >&2
      exit 2
      ;;
  esac
done

python_bin() {
  if [[ -x "${API_DIR}/.venv/bin/python" ]]; then
    printf '%s\n' "${API_DIR}/.venv/bin/python"
  elif [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    printf '%s\n' "${ROOT_DIR}/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    command -v python3
  else
    return 1
  fi
}

add_check() {
  local id="$1"
  local status="$2"
  local message="$3"
  local detail="${4:-}"
  local py
  py="$(python_bin 2>/dev/null || true)"
  if [[ -z "${py}" ]]; then
    printf '{"id":"%s","status":"%s","message":"%s","detail":"%s"}\n' \
      "${id}" "${status}" "${message}" "${detail}" >> "${CHECKS_FILE}"
    return
  fi
  "${py}" - "${CHECKS_FILE}" "${id}" "${status}" "${message}" "${detail}" <<'PY'
import json
import sys

path, check_id, status, message, detail = sys.argv[1:6]
with open(path, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "id": check_id,
        "status": status,
        "message": message,
        "detail": detail,
    }, sort_keys=True) + "\n")
PY
}

version_at_least() {
  local actual="$1"
  local minimum="$2"
  local py
  py="$(python_bin 2>/dev/null || true)"
  [[ -n "${py}" ]] || return 1
  "${py}" - "${actual}" "${minimum}" <<'PY'
import re
import sys

def parts(value: str) -> tuple[int, ...]:
    found = re.findall(r"\d+", value)
    return tuple(int(item) for item in found[:3])

actual = parts(sys.argv[1])
minimum = parts(sys.argv[2])
raise SystemExit(0 if actual >= minimum else 1)
PY
}

bun_version() {
  if [[ "${npm_config_user_agent:-}" =~ bun/([^[:space:]]+) ]]; then
    printf '%s\n' "${BASH_REMATCH[1]}"
    return
  fi
  local bun_bin
  bun_bin="$(command -v bun 2>/dev/null || true)"
  [[ -n "${bun_bin}" ]] || return 1
  env \
    -u npm_command \
    -u npm_config_user_agent \
    -u npm_execpath \
    -u npm_node_execpath \
    "${bun_bin}" --version 2>/dev/null
}

check_bun() {
  if ! command -v bun >/dev/null 2>&1; then
    add_check "bun" "fail" "Bun is not installed" "Install Bun 1.4 or newer, then run bun install."
    return
  fi
  local version
  version="$(bun_version || true)"
  version="${version:-unknown}"
  if version_at_least "${version}" "1.4.0"; then
    add_check "bun" "pass" "Bun ${version} is available" "$(command -v bun)"
  else
    add_check "bun" "warn" "Bun ${version} is older than the recommended 1.4.0" "Upgrade Bun before production installs."
  fi
}

check_python() {
  local py
  py="$(python_bin 2>/dev/null || true)"
  if [[ -z "${py}" ]]; then
    add_check "python" "fail" "Python 3 is not available" "Install Python 3.11 or newer."
    return
  fi
  local version
  version="$("${py}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
  if version_at_least "${version}" "3.11.0"; then
    add_check "python" "pass" "Python ${version} is available" "${py}"
  else
    add_check "python" "fail" "Python ${version} is older than 3.11" "${py}"
  fi
}

check_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    add_check "uv" "warn" "uv is not installed" "Install uv, then run uv sync --project api --all-groups."
    return
  fi
  local version
  version="$(uv --version 2>/dev/null | awk '{print $2}')"
  version="${version:-unknown}"
  add_check "uv" "pass" "uv ${version} is available" "$(command -v uv)"
}

check_api_dependencies() {
  local py
  py="$(python_bin 2>/dev/null || true)"
  if [[ -z "${py}" ]]; then
    add_check "api_dependencies" "fail" "Cannot check API dependencies without Python" "Install Python dependencies with uv sync --project api --all-groups."
    return
  fi
  if (cd "${API_DIR}" && "${py}" - <<'PY' >/dev/null 2>&1
import importlib

for module in ("fastapi", "pydantic", "httpx", "numpy", "sklearn"):
    importlib.import_module(module)
PY
  ); then
    add_check "api_dependencies" "pass" "Core API dependencies import successfully" "Checked FastAPI, Pydantic, httpx, NumPy, and scikit-learn."
  else
    add_check "api_dependencies" "fail" "Core API dependencies are missing" "Run uv sync --project api --all-groups."
  fi
}

check_command() {
  local id="$1"
  local label="$2"
  local command_name="$3"
  local install_hint="$4"
  if command -v "${command_name}" >/dev/null 2>&1; then
    add_check "${id}" "pass" "${label} is available" "$(command -v "${command_name}")"
  else
    add_check "${id}" "warn" "${label} is not installed" "${install_hint}"
  fi
}

check_port() {
  local id="$1"
  local label="$2"
  local port="$3"
  if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1; then
    add_check "${id}" "warn" "${label} port ${port} is already in use" "Set API_PORT or PORT to an available port before starting."
  else
    add_check "${id}" "pass" "${label} port ${port} is available" ""
  fi
}

check_data_dir() {
  local data_dir="${JR_DATA_DIR:-${ROOT_DIR}/data}"
  local parent
  parent="$(dirname "${data_dir}")"
  if [[ -d "${data_dir}" && -w "${data_dir}" ]]; then
    add_check "data_dir" "pass" "Data directory is writable" "${data_dir}"
  elif [[ ! -e "${data_dir}" && -w "${parent}" ]]; then
    add_check "data_dir" "pass" "Data directory can be created" "${data_dir}"
  else
    add_check "data_dir" "fail" "Data directory is not writable" "${data_dir}"
  fi
}

check_auth_env() {
  if [[ "${AUTORAG_AUTH_ENABLED:-false}" =~ ^(1|true|yes)$ ]] && [[ -z "${AUTORAG_API_KEYS:-}" ]]; then
    add_check "auth_keys" "fail" "Auth is enabled but AUTORAG_API_KEYS is empty" "Set AUTORAG_API_KEYS before exposing the API."
  else
    add_check "auth_keys" "pass" "Auth environment is internally consistent" "AUTORAG_AUTH_ENABLED=${AUTORAG_AUTH_ENABLED:-false}"
  fi
}

check_security_posture() {
  local exposed=0
  local auth_enabled=0
  local keys_configured=0
  local rate_enabled=1
  local wildcard_cors=0
  local default_rate="true"

  [[ "${AUTORAG_EXPOSE:-false}" =~ ^(1|true|yes)$ ]] && exposed=1
  [[ "${AUTORAG_AUTH_ENABLED:-false}" =~ ^(1|true|yes)$ ]] && auth_enabled=1
  [[ -n "${AUTORAG_API_KEYS:-}" ]] && keys_configured=1
  [[ "${JR_DEMO_MODE:-false}" =~ ^(1|true|yes)$ ]] && default_rate="false"
  if [[ "${AUTORAG_RATE_LIMIT_ENABLED:-${default_rate}}" =~ ^(0|false|no)$ ]]; then
    rate_enabled=0
  fi
  case ",${AUTORAG_ALLOWED_ORIGINS:-}," in
    *,\*,*) wildcard_cors=1 ;;
  esac

  if [[ "${exposed}" -eq 1 && ( "${auth_enabled}" -eq 0 || "${keys_configured}" -eq 0 ) ]]; then
    add_check "security_posture" "fail" "Exposed mode is not safe without API-key auth" "Set AUTORAG_AUTH_ENABLED=true and AUTORAG_API_KEYS before AUTORAG_EXPOSE=true."
  elif [[ "${exposed}" -eq 1 && "${rate_enabled}" -eq 0 ]]; then
    add_check "security_posture" "fail" "Exposed mode is not safe without rate limiting" "Set AUTORAG_RATE_LIMIT_ENABLED=true before client-network installs."
  elif [[ "${exposed}" -eq 1 && "${wildcard_cors}" -eq 1 ]]; then
    add_check "security_posture" "fail" "Exposed mode must not use wildcard CORS" "Set AUTORAG_ALLOWED_ORIGINS to exact trusted origins."
  elif [[ "${auth_enabled}" -eq 0 ]]; then
    add_check "security_posture" "warn" "Security posture is local-demo only" "Enable API-key auth before a client install."
  else
    add_check "security_posture" "pass" "Security posture is ready for guarded local install" "Auth=${auth_enabled}, exposed=${exposed}, rate_limit=${rate_enabled}"
  fi
}

check_provider() {
  if [[ "${JR_DOCTOR_SKIP_PROVIDERS:-0}" =~ ^(1|true|yes)$ ]]; then
    add_check "local_providers" "warn" "Local provider check skipped" "Unset JR_DOCTOR_SKIP_PROVIDERS to check Ollama and LM Studio."
    return
  fi
  if ! command -v curl >/dev/null 2>&1; then
    add_check "local_providers" "warn" "curl is unavailable, provider discovery was skipped" "Install curl or use the in-app provider scan."
    return
  fi
  local ollama_ok=0
  local lmstudio_ok=0
  curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && ollama_ok=1 || true
  curl -fsS --max-time 2 http://127.0.0.1:1234/v1/models >/dev/null 2>&1 && lmstudio_ok=1 || true
  if [[ "${ollama_ok}" -eq 1 || "${lmstudio_ok}" -eq 1 ]]; then
    add_check "local_providers" "pass" "At least one local model provider is reachable" "Ollama=${ollama_ok}, LM Studio=${lmstudio_ok}"
  else
    add_check "local_providers" "warn" "No local model provider is reachable" "Start Ollama on 11434 or LM Studio on 1234 for generated answers."
  fi
}

emit_report() {
  local py
  py="$(python_bin 2>/dev/null || true)"
  if [[ -z "${py}" ]]; then
    printf 'Python is required to render the doctor report.\n' >&2
    exit 1
  fi
  "${py}" - "${CHECKS_FILE}" "${JSON_MODE}" <<'PY'
from __future__ import annotations

import datetime as dt
import json
import sys

checks_path = sys.argv[1]
json_mode = sys.argv[2] == "1"

checks = []
with open(checks_path, encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if line:
            checks.append(json.loads(line))

status = "pass"
if any(item["status"] == "fail" for item in checks):
    status = "fail"
elif any(item["status"] == "warn" for item in checks):
    status = "warn"

payload = {
    "product": "JR AutoRAG",
    "generated_at": dt.datetime.now(dt.UTC).isoformat(),
    "summary": {
        "status": status,
        "passed": sum(1 for item in checks if item["status"] == "pass"),
        "warnings": sum(1 for item in checks if item["status"] == "warn"),
        "failed": sum(1 for item in checks if item["status"] == "fail"),
    },
    "checks": checks,
}

if json_mode:
    print(json.dumps(payload, indent=2, sort_keys=True))
else:
    print("JR AutoRAG Doctor")
    print(f"Summary: {payload['summary']['status'].upper()} "
          f"({payload['summary']['passed']} pass, "
          f"{payload['summary']['warnings']} warn, "
          f"{payload['summary']['failed']} fail)")
    for item in checks:
        detail = f" [{item['detail']}]" if item.get("detail") else ""
        print(f"- {item['status'].upper():4} {item['id']}: {item['message']}{detail}")
PY
}

check_bun
check_python
check_uv
check_api_dependencies
check_command "ocr_tesseract" "Tesseract OCR" "tesseract" "Install tesseract for scanned PDF OCR."
check_command "ocr_poppler" "Poppler PDF tools" "pdftotext" "Install poppler for PDF text extraction and preview."
check_port "port_api" "API" "${API_PORT:-8000}"
check_port "port_web" "Web" "${PORT:-3000}"
check_data_dir
check_auth_env
check_security_posture
check_provider
emit_report
