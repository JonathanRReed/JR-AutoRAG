#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

cat > "${TMP_DIR}/fake_api.py" <<'PY'
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

port_file = Path(__import__("sys").argv[1])

payloads = {
    "/readyz": {"ready": True, "level": "degraded", "checks": {}},
    "/config/policy": {
        "deployment_profile": "local_only",
        "data_policy": {"classification": "client_confidential"},
    },
    "/security/posture": {
        "level": "needs_attention",
        "summary": "Local posture needs client hardening.",
        "settings": {"auth_enabled": False, "api_keys_configured": False},
        "checks": [],
        "recommendations": [],
    },
    "/install/report": {
        "schema_version": "install_report_v1",
        "status": "warn",
        "summary": "Client handoff still needs evidence work.",
        "redaction": {"secrets": "redacted"},
    },
}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path not in payloads:
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(payloads[self.path]).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_: object) -> None:
        return


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
port_file.write_text(str(server.server_port), encoding="utf-8")
server.serve_forever()
PY

python3 "${TMP_DIR}/fake_api.py" "${TMP_DIR}/port" &
SERVER_PID="$!"
for _ in {1..50}; do
  [[ -s "${TMP_DIR}/port" ]] && break
  sleep 0.1
done
[[ -s "${TMP_DIR}/port" ]]

port="$(cat "${TMP_DIR}/port")"
output_dir="${TMP_DIR}/evidence"
doctor_json='{"product":"JR AutoRAG","summary":{"status":"warn","failed":0},"checks":[]}'

JR_EVIDENCE_API_BASE_URL="http://127.0.0.1:${port}" \
JR_EVIDENCE_OUTPUT_DIR="${output_dir}" \
JR_EVIDENCE_DOCTOR_CMD="printf '%s\n' '${doctor_json}'" \
  bash "${ROOT_DIR}/scripts/evidence-bundle.sh" > "${TMP_DIR}/stdout"

bundle_path="$(tail -n 1 "${TMP_DIR}/stdout")"
[[ -d "${bundle_path}" ]]
[[ "${bundle_path}" == "${output_dir}"/* ]]

for file in doctor.json readyz.json config-policy.json security-posture.json install-report.json research-architecture.md manifest.json SHA256SUMS; do
  [[ -s "${bundle_path}/${file}" ]]
done

python3 - "${bundle_path}" <<'PY'
import json
import sys
from pathlib import Path

bundle = Path(sys.argv[1])
manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))

assert manifest["schema_version"] == "jr_autorag_evidence_bundle_v1"
assert manifest["api_base_url"] == "http://127.0.0.1:" + manifest["api_port"]
assert manifest["doctor"]["summary"]["status"] == "warn"
assert manifest["install_report"]["status"] == "warn"
assert set(manifest["artifacts"]) >= {
    "doctor.json",
    "readyz.json",
    "config-policy.json",
    "security-posture.json",
    "install-report.json",
    "research-architecture.md",
}
for artifact in manifest["artifacts"].values():
    assert artifact["sha256"]
    assert artifact["bytes"] > 0
assert "install-report.json" in (bundle / "SHA256SUMS").read_text(encoding="utf-8")
PY
