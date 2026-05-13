#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/client-handoff-gate.sh BUNDLE_DIR

Validates a generated JR AutoRAG evidence bundle for strict client handoff.
This is intentionally stricter than local evaluation:
- bundle hashes must match SHA256SUMS
- doctor must have zero failed checks
- install report must be ready
- security posture must be client_ready
- client_readiness report must be present and meet quality gates
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

bundle_dir="${1:-}"
if [[ -z "${bundle_dir}" ]]; then
  usage >&2
  exit 2
fi

python3 - "${bundle_dir}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

bundle = Path(sys.argv[1])
issues: list[str] = []

REQUIRED_ARTIFACTS = {
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
    "manifest.json",
    "SHA256SUMS",
}
REQUIRED_CLIENT_TAGS = {
    "client-readiness",
    "mixed-format",
    "prompt-injection",
    "abstention",
    "binary-retrieval",
    "agentic-retrieval",
    "poisoned-document",
    "knowledge-extraction",
    "graph-retrieval",
}
CLIENT_METRIC_THRESHOLDS = {
    ("retrieval_metrics", "recall_at_k"): 0.70,
    ("retrieval_metrics", "citation_coverage"): 0.85,
    ("answer_metrics", "faithfulness"): 0.90,
    ("answer_metrics", "completeness"): 0.70,
    ("answer_metrics", "refusal_accuracy"): 0.95,
}


def load_json(name: str) -> dict | list:
    path = bundle / name
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        issues.append(f"missing {name}")
    except json.JSONDecodeError as exc:
        issues.append(f"{name} is not valid JSON: {exc}")
    return {}


def as_dict(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def metric(report: dict, section: str, name: str) -> float:
    value = as_dict(report.get(section)).get(name)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if not bundle.is_dir():
    raise SystemExit(f"bundle directory not found: {bundle}")

for name in sorted(REQUIRED_ARTIFACTS):
    path = bundle / name
    if not path.is_file() or path.stat().st_size == 0:
        issues.append(f"missing or empty artifact: {name}")

manifest = as_dict(load_json("manifest.json"))
doctor = as_dict(load_json("doctor.json"))
security = as_dict(load_json("security-posture.json"))
install_report = as_dict(load_json("install-report.json"))
client_report = as_dict(load_json("client-readiness-report.json"))

if manifest.get("schema_version") != "jr_autorag_evidence_bundle_v1":
    issues.append("manifest schema_version must be jr_autorag_evidence_bundle_v1")

sha_file = bundle / "SHA256SUMS"
if sha_file.exists():
    for line in sha_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, _, name = line.partition("  ")
        path = bundle / name
        if not path.exists():
            issues.append(f"hash references missing file: {name}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            issues.append(f"hash mismatch: {name}")

doctor_summary = as_dict(doctor.get("summary"))
if doctor_summary.get("status") == "fail" or int(doctor_summary.get("failed") or 0) > 0:
    issues.append("doctor has failed checks")

if as_dict(manifest.get("install_report")).get("status") != "ready" or install_report.get("status") != "ready":
    issues.append("install report must be ready")

if as_dict(manifest.get("security")).get("level") != "client_ready" or security.get("level") != "client_ready":
    issues.append("security posture must be client_ready")

if as_dict(manifest.get("client_readiness")).get("status") != "present":
    issues.append("manifest client_readiness status must be present")

report_sha = client_report.get("report_sha256")
manifest_sha = as_dict(manifest.get("client_readiness")).get("report_sha256")
if not report_sha or report_sha != manifest_sha:
    issues.append("client_readiness report sha must match manifest")

for section, name in CLIENT_METRIC_THRESHOLDS:
    actual = metric(client_report, section, name)
    threshold = CLIENT_METRIC_THRESHOLDS[(section, name)]
    if actual < threshold:
        issues.append(f"client_readiness {section}.{name} {actual:.3f} below {threshold:.3f}")

tag_counts = as_dict(as_dict(as_dict(client_report.get("audit")).get("golden_set")).get("tag_counts"))
missing_tags = sorted(tag for tag in REQUIRED_CLIENT_TAGS if int(tag_counts.get(tag) or 0) <= 0)
if missing_tags:
    issues.append(f"client_readiness missing tags: {', '.join(missing_tags)}")

evidence = {
    item.get("id"): item
    for item in install_report.get("evidence", [])
    if isinstance(item, dict)
}
if as_dict(evidence.get("client_readiness_benchmark")).get("status") != "present":
    issues.append("install report client_readiness_benchmark evidence must be present")
if as_dict(evidence.get("security_posture")).get("status") != "present":
    issues.append("install report security_posture evidence must be present")
if as_dict(evidence.get("readiness")).get("status") != "present":
    issues.append("install report readiness evidence must be present")
if any(as_dict(item).get("status") == "missing" for item in evidence.values()):
    issues.append("install report still has missing evidence")

for name, marker in {
    "install-smoke.txt": "install_smoke=pass",
    "container-manifest.txt": "container_manifest=pass",
    "research-architecture-check.txt": "research_architecture=pass",
    "secret-scan.txt": "secret_scan=pass",
    "supply-chain.txt": "supply_chain=pass",
}.items():
    path = bundle / name
    if path.exists() and marker not in path.read_text(encoding="utf-8"):
        issues.append(f"{name} does not contain {marker}")

if issues:
    for issue in issues:
        print(f"client_handoff_gate=fail reason={issue}", file=sys.stderr)
    raise SystemExit(1)

print(f"client_handoff_gate=pass bundle={bundle}")
PY
