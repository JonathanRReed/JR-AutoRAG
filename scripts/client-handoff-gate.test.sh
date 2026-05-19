#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

make_bundle() {
  local dir="$1"
  local security_level="$2"
  local install_status="$3"
  local completeness="$4"
  mkdir -p "${dir}"

  cat > "${dir}/doctor.json" <<'JSON'
{"summary":{"status":"pass","failed":0},"checks":[]}
JSON
  printf '%s\n' 'install_smoke=pass api=200 web=200' > "${dir}/install-smoke.txt"
  printf '%s\n' 'container_manifest=pass' > "${dir}/container-manifest.txt"
  printf '%s\n' 'research_architecture=pass checked_paths=57' > "${dir}/research-architecture-check.txt"
  printf '%s\n' 'secret_scan=pass' > "${dir}/secret-scan.txt"
  printf '%s\n' 'supply_chain=pass output=test audit_level=high' > "${dir}/supply-chain.txt"
  printf '%s\n' '{"schema_version":"jr_autorag_supply_chain_v1","artifacts":{}}' > "${dir}/supply-chain-manifest.json"
  printf '%s\n' '{"bomFormat":"CycloneDX","specVersion":"1.5","components":[]}' > "${dir}/python-sbom.cdx.json"
  printf '%s\n' '{}' > "${dir}/web-audit.json"
  printf '%s\n' 'web dependencies' > "${dir}/web-dependencies.txt"
  printf '%s\n' '{"ready":true,"level":"ready","checks":{}}' > "${dir}/readyz.json"
  printf '%s\n' '{"deployment_profile":"local_only"}' > "${dir}/config-policy.json"
  printf '{"level":"%s","summary":"ready","settings":{"auth_enabled":true,"api_keys_configured":true,"rate_limit_enabled":true,"wildcard_cors":false},"checks":[],"recommendations":[]}\n' "${security_level}" > "${dir}/security-posture.json"
  cat > "${dir}/evaluation-runs.json" <<'JSON'
[{"run_id":"client-ready-test","golden_set_name":"client_readiness","report_sha256":"abc123"}]
JSON
  cat > "${dir}/client-readiness-report.json" <<JSON
{
  "run_id": "client-ready-test",
  "golden_set_name": "client_readiness",
  "report_sha256": "abc123",
  "retrieval_metrics": {
    "recall_at_k": 1.0,
    "mrr": 1.0,
    "ndcg": 1.0,
    "citation_coverage": 1.0
  },
  "answer_metrics": {
    "faithfulness": 1.0,
    "completeness": ${completeness},
    "refusal_accuracy": 1.0,
    "coherence": 1.0
  },
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
  }
}
JSON
  cat > "${dir}/install-report.json" <<JSON
{
  "schema_version": "install_report_v1",
  "status": "${install_status}",
  "summary": "ready",
  "evidence": [
    {"id":"security_posture","status":"present"},
    {"id":"readiness","status":"present"},
    {"id":"client_readiness_benchmark","status":"present"}
  ],
  "actions": []
}
JSON
  printf '%s\n' '# research architecture' > "${dir}/research-architecture.md"
  python3 - "${dir}" "${security_level}" "${install_status}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

bundle = Path(sys.argv[1])
security_level = sys.argv[2]
install_status = sys.argv[3]
core_files = [
    "doctor.json",
    "install-smoke.txt",
    "container-manifest.txt",
    "research-architecture-check.txt",
    "secret-scan.txt",
    "supply-chain.txt",
    "supply-chain-manifest.json",
    "python-sbom.cdx.json",
    "web-audit.json",
    "web-dependencies.txt",
    "readyz.json",
    "config-policy.json",
    "security-posture.json",
    "install-report.json",
    "evaluation-runs.json",
    "client-readiness-report.json",
    "research-architecture.md",
]


def file_meta(name: str) -> dict[str, object]:
    data = (bundle / name).read_bytes()
    return {"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


manifest = {
    "schema_version": "jr_autorag_evidence_bundle_v1",
    "generated_at": datetime.now(UTC).isoformat(),
    "doctor": {"summary": {"status": "pass", "failed": 0}},
    "install_report": {"status": install_status},
    "security": {"level": security_level},
    "client_readiness": {
        "status": "present",
        "run_id": "client-ready-test",
        "report_sha256": "abc123",
        "golden_set_name": "client_readiness",
    },
    "artifacts": {name: file_meta(name) for name in core_files},
}
(bundle / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
lines = []
for name in core_files + ["manifest.json"]:
    data = (bundle / name).read_bytes()
    lines.append(f"{hashlib.sha256(data).hexdigest()}  {name}")
(bundle / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

passing_bundle="${TMP_DIR}/passing"
make_bundle "${passing_bundle}" "client_ready" "ready" "0.95"
bash "${ROOT_DIR}/scripts/client-handoff-gate.sh" "${passing_bundle}" > "${TMP_DIR}/pass.out"
grep -q 'client_handoff_gate=pass' "${TMP_DIR}/pass.out"

weak_bundle="${TMP_DIR}/weak"
make_bundle "${weak_bundle}" "client_ready" "ready" "0.40"
if bash "${ROOT_DIR}/scripts/client-handoff-gate.sh" "${weak_bundle}" > "${TMP_DIR}/weak.out" 2>"${TMP_DIR}/weak.err"; then
  printf 'weak bundle unexpectedly passed\n' >&2
  exit 1
fi
grep -q 'completeness' "${TMP_DIR}/weak.err"

local_bundle="${TMP_DIR}/local"
make_bundle "${local_bundle}" "local_only" "ready" "0.95"
bash "${ROOT_DIR}/scripts/client-handoff-gate.sh" "${local_bundle}" > "${TMP_DIR}/local.out"
grep -q 'client_handoff_gate=pass' "${TMP_DIR}/local.out"

unauth_local_bundle="${TMP_DIR}/unauth-local"
make_bundle "${unauth_local_bundle}" "local_only" "ready" "0.95"
python3 - "${unauth_local_bundle}/security-posture.json" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["settings"]["auth_enabled"] = False
payload["settings"]["api_keys_configured"] = False
path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
PY
if bash "${ROOT_DIR}/scripts/client-handoff-gate.sh" "${unauth_local_bundle}" > "${TMP_DIR}/unauth-local.out" 2>"${TMP_DIR}/unauth-local.err"; then
  printf 'unauthenticated local-only bundle unexpectedly passed strict gate\n' >&2
  exit 1
fi
grep -q 'security posture must be client_ready or authenticated local_only' "${TMP_DIR}/unauth-local.err"

printf 'client_handoff_gate_test=pass\n'
