#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${JR_SUPPLY_CHAIN_OUTPUT_DIR:-}"
AUDIT_LEVEL="${JR_SUPPLY_CHAIN_AUDIT_LEVEL:-high}"

usage() {
  cat <<'EOF'
Usage: bash scripts/supply-chain-evidence.sh [--output-dir DIR] [--audit-level LEVEL]

Generates supply-chain receipts for client handoff:
- python-sbom.cdx.json from uv.lock in CycloneDX 1.5 format
- web-audit.json from bun audit
- web-dependencies.txt from bun list --all
- supply-chain-manifest.json with hashes and summary counts

Environment overrides:
  JR_SUPPLY_CHAIN_OUTPUT_DIR
  JR_SUPPLY_CHAIN_AUDIT_LEVEL
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-dir)
      OUTPUT_DIR="${2:?missing directory}"
      shift 2
      ;;
    --audit-level)
      AUDIT_LEVEL="${2:?missing audit level}"
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

case "${AUDIT_LEVEL}" in
  low|moderate|high|critical) ;;
  *)
    printf 'Invalid audit level: %s\n' "${AUDIT_LEVEL}" >&2
    exit 2
    ;;
esac

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

need_command bun
need_command python3
need_command uv
BUN_BIN="$(resolve_bun)"

if [[ -z "${OUTPUT_DIR}" ]]; then
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  OUTPUT_DIR="${ROOT_DIR}/evidence/supply-chain/${timestamp}"
fi
mkdir -p "${OUTPUT_DIR}"

python_sbom="${OUTPUT_DIR}/python-sbom.cdx.json"
python_export_log="${OUTPUT_DIR}/python-sbom-export.log"
web_audit="${OUTPUT_DIR}/web-audit.json"
web_audit_log="${OUTPUT_DIR}/web-audit.log"
web_deps="${OUTPUT_DIR}/web-dependencies.txt"
manifest="${OUTPUT_DIR}/supply-chain-manifest.json"

if ! uv export \
  --project "${ROOT_DIR}/api" \
  --format cyclonedx1.5 \
  --locked \
  --all-groups \
  --output-file "${python_sbom}" > "${python_export_log}" 2>&1; then
  printf 'Python SBOM export failed. See %s\n' "${python_export_log}" >&2
  tail -n 80 "${python_export_log}" >&2 || true
  exit 1
fi

if ! audit_output="$(
  cd "${ROOT_DIR}"
  NO_COLOR=1 "${BUN_BIN}" audit --audit-level "${AUDIT_LEVEL}" --json 2> "${web_audit_log}"
)"; then
  printf 'Bun audit failed. See %s and %s\n' "${web_audit}" "${web_audit_log}" >&2
  tail -n 80 "${web_audit_log}" >&2 || true
  printf '%s\n' "${audit_output:-}" >&2
  exit 1
fi
if [[ -z "${audit_output}" ]]; then
  audit_output="{}"
fi
printf '%s\n' "${audit_output}" > "${web_audit}"

(
  cd "${ROOT_DIR}"
  NO_COLOR=1 "${BUN_BIN}" list --all
) > "${web_deps}"

python3 - "${OUTPUT_DIR}" "${AUDIT_LEVEL}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

output = Path(sys.argv[1])
audit_level = sys.argv[2]

python_sbom = json.loads((output / "python-sbom.cdx.json").read_text(encoding="utf-8"))
if python_sbom.get("bomFormat") != "CycloneDX":
    raise SystemExit("python-sbom.cdx.json is not a CycloneDX document")
if python_sbom.get("specVersion") != "1.5":
    raise SystemExit("python-sbom.cdx.json is not CycloneDX 1.5")

web_audit = json.loads((output / "web-audit.json").read_text(encoding="utf-8"))
if not isinstance(web_audit, dict):
    raise SystemExit("web-audit.json must be a JSON object")


def file_meta(name: str) -> dict[str, object]:
    data = (output / name).read_bytes()
    return {
        "path": name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def advisory_count(payload: dict[str, object]) -> int:
    if isinstance(payload.get("advisories"), dict):
        return len(payload["advisories"])
    if isinstance(payload.get("vulnerabilities"), dict):
        return len(payload["vulnerabilities"])
    if isinstance(payload.get("vulnerabilities"), list):
        return len(payload["vulnerabilities"])
    return len(payload)


manifest = {
    "schema_version": "jr_autorag_supply_chain_v1",
    "generated_at": datetime.now(UTC).isoformat(),
    "audit_level": audit_level,
    "python": {
        "sbom_format": python_sbom.get("bomFormat"),
        "sbom_spec_version": python_sbom.get("specVersion"),
        "component_count": len(python_sbom.get("components") or []),
    },
    "web": {
        "audit_advisory_count": advisory_count(web_audit),
    },
    "artifacts": {
        name: file_meta(name)
        for name in [
            "python-sbom.cdx.json",
            "python-sbom-export.log",
            "web-audit.json",
            "web-audit.log",
            "web-dependencies.txt",
        ]
    },
}

(output / "supply-chain-manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY

printf 'supply_chain=pass output=%s audit_level=%s\n' "${OUTPUT_DIR}" "${AUDIT_LEVEL}"
