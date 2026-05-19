#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

json_file="${TMP_DIR}/doctor.json"
JR_DOCTOR_SKIP_PROVIDERS=1 bash "${ROOT_DIR}/scripts/doctor.sh" --json > "${json_file}"
if [[ ! -s "${json_file}" ]]; then
  printf 'doctor JSON output was empty\n' >&2
  JR_DOCTOR_SKIP_PROVIDERS=1 bash "${ROOT_DIR}/scripts/doctor.sh" --json >&2 || true
  exit 1
fi

python3 - "${json_file}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
assert payload["product"] == "JR AutoRAG"
assert payload["summary"]["status"] in {"pass", "warn", "fail"}
checks = {item["id"]: item for item in payload["checks"]}
required = {
    "bun",
    "python",
    "uv",
    "api_dependencies",
    "ocr_tesseract",
    "ocr_poppler",
    "port_api",
    "port_web",
    "data_dir",
    "security_posture",
}
missing = required - checks.keys()
assert not missing, f"missing checks: {sorted(missing)}"
for item in checks.values():
    assert item["status"] in {"pass", "warn", "fail"}
    assert item["message"]
PY

human_output="$(JR_DOCTOR_SKIP_PROVIDERS=1 bash "${ROOT_DIR}/scripts/doctor.sh")"
[[ "${human_output}" == *"JR AutoRAG Doctor"* ]]
[[ "${human_output}" == *"Summary:"* ]]

lifecyle_json_file="${TMP_DIR}/doctor-lifecycle.json"
JR_DOCTOR_SKIP_PROVIDERS=1 \
  npm_config_user_agent="bun/1.3.13 npm/? node/v24.3.0 darwin arm64" \
  npm_command="run-script" \
  bash "${ROOT_DIR}/scripts/doctor.sh" --json > "${lifecyle_json_file}"
python3 - "${lifecyle_json_file}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
bun_check = next(item for item in payload["checks"] if item["id"] == "bun")
assert "Bun  " not in bun_check["message"]
assert "unknown" not in bun_check["message"].lower()
PY

exposed_json_file="${TMP_DIR}/doctor-exposed.json"
JR_DOCTOR_SKIP_PROVIDERS=1 \
  AUTORAG_EXPOSE=true \
  AUTORAG_AUTH_ENABLED=false \
  bash "${ROOT_DIR}/scripts/doctor.sh" --json > "${exposed_json_file}"
python3 - "${exposed_json_file}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
checks = {item["id"]: item for item in payload["checks"]}
security = checks["security_posture"]
assert security["status"] == "fail"
assert "exposed" in security["message"].lower()
PY
