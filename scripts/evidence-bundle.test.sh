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
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

port_file = Path(__import__("sys").argv[1])
expected_api_key = os.environ.get("TEST_EVIDENCE_API_KEY", "")
protected_paths = {
    "/config/policy",
    "/security/posture",
    "/install/report",
    "/evaluation/runs?limit=50",
    "/evaluation/runs/client-ready-test/report",
}

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
    "/evaluation/runs?limit=50": [
        {
            "run_id": "client-ready-test",
            "golden_set_name": "client_readiness",
            "timestamp": "2026-05-12T00:00:00+00:00",
            "retrieval_metrics": {"recall_at_k": 1.0, "mrr": 1.0, "ndcg": 1.0, "citation_coverage": 1.0},
            "answer_metrics": {"faithfulness": 1.0, "completeness": 1.0, "refusal_accuracy": 1.0, "coherence": 1.0},
            "duration_ms": 12.0,
            "report_path": "data/eval_reports/client-ready-test.json",
            "report_sha256": "abc123",
            "audit": {
                "golden_set": {
                    "tag_counts": {
                        "client-readiness": 9,
                        "mixed-format": 1,
                        "prompt-injection": 1,
                        "abstention": 1,
                        "binary-retrieval": 1,
                        "agentic-retrieval": 1,
                        "poisoned-document": 1,
                        "knowledge-extraction": 1,
                        "graph-retrieval": 1
                    }
                }
            },
        }
    ],
    "/evaluation/runs/client-ready-test/report": {
        "run_id": "client-ready-test",
        "golden_set_name": "client_readiness",
        "report_sha256": "abc123",
        "audit": {
            "golden_set": {
                "tag_counts": {
                    "client-readiness": 9,
                    "mixed-format": 1,
                    "prompt-injection": 1,
                    "abstention": 1,
                    "binary-retrieval": 1,
                    "agentic-retrieval": 1,
                    "poisoned-document": 1,
                    "knowledge-extraction": 1,
                    "graph-retrieval": 1
                }
            }
        },
    },
}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in protected_paths and expected_api_key and self.headers.get("X-API-Key") != expected_api_key:
            self.send_response(401)
            self.end_headers()
            return
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

TEST_EVIDENCE_API_KEY="test-evidence-key" python3 "${TMP_DIR}/fake_api.py" "${TMP_DIR}/port" &
SERVER_PID="$!"
for _ in {1..50}; do
  [[ -s "${TMP_DIR}/port" ]] && break
  sleep 0.1
done
[[ -s "${TMP_DIR}/port" ]]

port="$(cat "${TMP_DIR}/port")"
output_dir="${TMP_DIR}/evidence"
doctor_json='{"product":"JR AutoRAG","summary":{"status":"warn","failed":0},"checks":[]}'
install_smoke_output='install_smoke=pass api=200 web=200 proxy={"ok": true} same_origin={"profile": "Default"}'
container_manifest_output='container_manifest=pass'
research_check_output='research_architecture=pass checked_paths=32'
secret_scan_output='secret_scan=pass'
supply_chain_output='supply_chain=pass output=test audit_level=high'

JR_EVIDENCE_API_BASE_URL="http://127.0.0.1:${port}" \
JR_EVIDENCE_API_KEY="test-evidence-key" \
JR_EVIDENCE_OUTPUT_DIR="${output_dir}" \
JR_EVIDENCE_DOCTOR_CMD="printf '%s\n' '${doctor_json}'" \
JR_EVIDENCE_INSTALL_SMOKE_CMD="printf '%s\n' '${install_smoke_output}'" \
JR_EVIDENCE_CONTAINER_MANIFEST_CMD="printf '%s\n' '${container_manifest_output}'" \
JR_EVIDENCE_RESEARCH_CHECK_CMD="printf '%s\n' '${research_check_output}'" \
JR_EVIDENCE_SECRET_SCAN_CMD="printf '%s\n' '${secret_scan_output}'" \
JR_EVIDENCE_SUPPLY_CHAIN_CMD="printf '%s\n' '${supply_chain_output}' && printf '%s\n' '{\"schema_version\":\"jr_autorag_supply_chain_v1\",\"artifacts\":{}}' > \"\${JR_SUPPLY_CHAIN_OUTPUT_DIR}/supply-chain-manifest.json\" && printf '%s\n' '{\"bomFormat\":\"CycloneDX\",\"specVersion\":\"1.5\",\"components\":[]}' > \"\${JR_SUPPLY_CHAIN_OUTPUT_DIR}/python-sbom.cdx.json\" && printf '%s\n' 'fake uv export log' > \"\${JR_SUPPLY_CHAIN_OUTPUT_DIR}/python-sbom-export.log\" && printf '%s\n' '{}' > \"\${JR_SUPPLY_CHAIN_OUTPUT_DIR}/web-audit.json\" && printf '%s\n' 'fake bun audit log' > \"\${JR_SUPPLY_CHAIN_OUTPUT_DIR}/web-audit.log\" && printf '%s\n' 'fake dependency tree' > \"\${JR_SUPPLY_CHAIN_OUTPUT_DIR}/web-dependencies.txt\"" \
  bash "${ROOT_DIR}/scripts/evidence-bundle.sh" > "${TMP_DIR}/stdout"

bundle_path="$(tail -n 1 "${TMP_DIR}/stdout")"
[[ -d "${bundle_path}" ]]
[[ "${bundle_path}" == "${output_dir}"/* ]]

for file in doctor.json install-smoke.txt container-manifest.txt research-architecture-check.txt secret-scan.txt supply-chain.txt supply-chain-manifest.json python-sbom.cdx.json python-sbom-export.log web-audit.json web-audit.log web-dependencies.txt readyz.json config-policy.json security-posture.json install-report.json evaluation-runs.json client-readiness-report.json research-architecture.md manifest.json SHA256SUMS; do
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
    "install-smoke.txt",
    "container-manifest.txt",
    "research-architecture-check.txt",
    "secret-scan.txt",
    "supply-chain.txt",
    "supply-chain-manifest.json",
    "python-sbom.cdx.json",
    "python-sbom-export.log",
    "web-audit.json",
    "web-audit.log",
    "web-dependencies.txt",
    "readyz.json",
    "config-policy.json",
    "security-posture.json",
    "install-report.json",
    "evaluation-runs.json",
    "client-readiness-report.json",
    "research-architecture.md",
}
assert manifest["client_readiness"]["status"] == "present"
assert manifest["client_readiness"]["run_id"] == "client-ready-test"
assert manifest["client_readiness"]["report_sha256"] == "abc123"
assert manifest["verification"]["install_smoke"]["path"] == "install-smoke.txt"
assert manifest["verification"]["container_manifest"]["path"] == "container-manifest.txt"
assert manifest["verification"]["research_architecture"]["path"] == "research-architecture-check.txt"
assert manifest["verification"]["secret_scan"]["path"] == "secret-scan.txt"
assert manifest["verification"]["supply_chain"]["path"] == "supply-chain.txt"
for artifact in manifest["artifacts"].values():
    assert artifact["sha256"]
    assert artifact["bytes"] > 0
assert "install_smoke=pass" in (bundle / "install-smoke.txt").read_text(encoding="utf-8")
assert "container_manifest=pass" in (bundle / "container-manifest.txt").read_text(encoding="utf-8")
assert "research_architecture=pass" in (bundle / "research-architecture-check.txt").read_text(encoding="utf-8")
assert "secret_scan=pass" in (bundle / "secret-scan.txt").read_text(encoding="utf-8")
assert "supply_chain=pass" in (bundle / "supply-chain.txt").read_text(encoding="utf-8")
assert "install-report.json" in (bundle / "SHA256SUMS").read_text(encoding="utf-8")
assert "evaluation-runs.json" in (bundle / "SHA256SUMS").read_text(encoding="utf-8")
assert "client-readiness-report.json" in (bundle / "SHA256SUMS").read_text(encoding="utf-8")
assert "python-sbom.cdx.json" in (bundle / "SHA256SUMS").read_text(encoding="utf-8")
PY
