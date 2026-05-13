#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKIP_CONTAINER_SMOKE=0

usage() {
  cat <<'EOF'
Usage: bash scripts/release-gate.sh [--skip-container-smoke]

Runs the full local release gate for a client-installable JR AutoRAG build:
- local doctor regression
- install smoke regression
- container manifest invariants
- secret scan
- supply-chain evidence generation
- aggregate code, test, research, handoff, typecheck, and build verification
- container build and runtime smoke unless explicitly skipped
- git diff whitespace check

Use --skip-container-smoke only when Docker is unavailable on the local machine.
CI and release candidates should run without this flag.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-container-smoke)
      SKIP_CONTAINER_SMOKE=1
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

cd "${ROOT_DIR}"

pick_free_ports() {
  python3 - <<'PY'
from __future__ import annotations

import socket

sockets = []
try:
    for _ in range(2):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        sockets.append(sock)
    print(" ".join(str(sock.getsockname()[1]) for sock in sockets))
finally:
    for sock in sockets:
        sock.close()
PY
}

env -i PATH="${PATH}" HOME="${HOME}" bash ./scripts/doctor.test.sh
read -r install_api_port install_web_port < <(pick_free_ports)
JR_INSTALL_SMOKE_SKIP_SYNC=1 \
JR_INSTALL_SMOKE_API_PORT="${install_api_port}" \
JR_INSTALL_SMOKE_WEB_PORT="${install_web_port}" \
  bash ./scripts/install-smoke.sh
bash ./scripts/container-manifest-check.sh
bash ./scripts/secret-scan.sh

tmp_supply_chain="$(mktemp -d)"
cleanup() {
  rm -rf "${tmp_supply_chain}"
}
trap cleanup EXIT
JR_SUPPLY_CHAIN_OUTPUT_DIR="${tmp_supply_chain}" bash ./scripts/supply-chain-evidence.sh

bash ./scripts/verify.sh

if [[ "${SKIP_CONTAINER_SMOKE}" == "1" ]]; then
  echo "release_gate_container_smoke=skipped reason=operator_requested"
else
  bash ./scripts/container-build-smoke.sh
fi

git diff --check
echo "release_gate=pass"
