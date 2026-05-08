#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_BASE_URL="${JR_EVIDENCE_API_BASE_URL:-http://127.0.0.1:8000}"
OUTPUT_ROOT="${JR_EVIDENCE_OUTPUT_DIR:-${ROOT_DIR}/evidence/install}"
DOCTOR_CMD="${JR_EVIDENCE_DOCTOR_CMD:-bash ./scripts/doctor.sh --json}"
RESEARCH_DOC="${JR_EVIDENCE_RESEARCH_DOC:-${ROOT_DIR}/docs/architecture/research-backed-rag-architecture.md}"

usage() {
  cat <<'EOF'
Usage: bash scripts/evidence-bundle.sh [--api-base-url URL] [--output-dir DIR]

Collects a redacted JR AutoRAG install evidence bundle from a running API:
- doctor.json from bun run doctor -- --json
- readyz.json from GET /readyz
- config-policy.json from GET /config/policy
- security-posture.json from GET /security/posture
- install-report.json from GET /install/report
- manifest.json and SHA256SUMS

Environment overrides:
  JR_EVIDENCE_API_BASE_URL
  JR_EVIDENCE_OUTPUT_DIR
  JR_EVIDENCE_DOCTOR_CMD
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --api-base-url)
      API_BASE_URL="${2:?missing URL}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_ROOT="${2:?missing directory}"
      shift 2
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

need_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$1" >&2
    exit 127
  fi
}

need_command curl
need_command python3

API_BASE_URL="${API_BASE_URL%/}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
bundle_dir="${OUTPUT_ROOT}/${timestamp}-install-evidence"
if [[ -e "${bundle_dir}" ]]; then
  bundle_dir="${OUTPUT_ROOT}/${timestamp}-install-evidence-$$"
fi
mkdir -p "${bundle_dir}"

doctor_file="${bundle_dir}/doctor.json"
metadata_file="${bundle_dir}/http-metadata.jsonl"
research_file="${bundle_dir}/research-architecture.md"

if [[ ! -s "${RESEARCH_DOC}" ]]; then
  printf 'Missing research architecture document: %s\n' "${RESEARCH_DOC}" >&2
  exit 1
fi
cp "${RESEARCH_DOC}" "${research_file}"

(
  cd "${ROOT_DIR}"
  bash -c "${DOCTOR_CMD}"
) > "${doctor_file}"

fetch_json() {
  local endpoint="$1"
  local output="$2"
  local status
  local url="${API_BASE_URL}${endpoint}"
  status="$(curl -sS -o "${output}" -w '%{http_code}' "${url}")"
  python3 - "${metadata_file}" "${endpoint}" "${status}" "${output}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

metadata_path = Path(sys.argv[1])
endpoint = sys.argv[2]
status = int(sys.argv[3])
output = Path(sys.argv[4])
entry = {
    "endpoint": endpoint,
    "status": status,
    "file": output.name,
}
with metadata_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(entry, sort_keys=True) + "\n")
PY
  if [[ "${status}" -lt 200 || "${status}" -ge 300 ]]; then
    printf 'Endpoint %s returned HTTP %s\n' "${endpoint}" "${status}" >&2
    exit 1
  fi
}

fetch_json "/readyz" "${bundle_dir}/readyz.json"
fetch_json "/config/policy" "${bundle_dir}/config-policy.json"
fetch_json "/security/posture" "${bundle_dir}/security-posture.json"
fetch_json "/install/report" "${bundle_dir}/install-report.json"

python3 - "${bundle_dir}" "${API_BASE_URL}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

bundle = Path(sys.argv[1])
api_base_url = sys.argv[2]
parsed = urlparse(api_base_url)
api_port = str(parsed.port or (443 if parsed.scheme == "https" else 80))

core_files = [
    "doctor.json",
    "readyz.json",
    "config-policy.json",
    "security-posture.json",
    "install-report.json",
    "research-architecture.md",
]


def load_json(name: str) -> object:
    path = bundle / name
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{name} is not valid JSON: {exc}") from exc


def file_meta(name: str) -> dict[str, object]:
    path = bundle / name
    data = path.read_bytes()
    return {
        "path": name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


doctor = load_json("doctor.json")
readyz = load_json("readyz.json")
policy = load_json("config-policy.json")
security = load_json("security-posture.json")
install_report = load_json("install-report.json")
http_metadata = [
    json.loads(line)
    for line in (bundle / "http-metadata.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]

doctor_summary = doctor.get("summary", {}) if isinstance(doctor, dict) else {}
install_summary = install_report if isinstance(install_report, dict) else {}
if doctor_summary.get("status") == "fail" or int(doctor_summary.get("failed") or 0) > 0:
    raise SystemExit("doctor reported failed checks")
if install_summary.get("status") == "blocked":
    raise SystemExit("install report status is blocked")

manifest = {
    "schema_version": "jr_autorag_evidence_bundle_v1",
    "generated_at": datetime.now(UTC).isoformat(),
    "api_base_url": api_base_url,
    "api_port": api_port,
    "doctor": {
        "summary": doctor_summary,
    },
    "readiness": {
        "level": readyz.get("level") if isinstance(readyz, dict) else None,
        "ready": readyz.get("ready") if isinstance(readyz, dict) else None,
    },
    "security": {
        "level": security.get("level") if isinstance(security, dict) else None,
        "summary": security.get("summary") if isinstance(security, dict) else None,
    },
    "install_report": {
        "status": install_summary.get("status"),
        "summary": install_summary.get("summary"),
        "schema_version": install_summary.get("schema_version"),
    },
    "policy": {
        "deployment_profile": policy.get("deployment_profile") if isinstance(policy, dict) else None,
    },
    "http": http_metadata,
    "artifacts": {name: file_meta(name) for name in core_files},
}

(bundle / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

sum_files = core_files + ["manifest.json"]
lines = []
for name in sum_files:
    data = (bundle / name).read_bytes()
    lines.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
(bundle / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

printf 'Evidence bundle written:\n'
printf '%s\n' "${bundle_dir}"
