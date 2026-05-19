#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${JR_RELEASE_OUTPUT_DIR:-${ROOT_DIR}/dist/release}"
STAGE_ROOT="${ROOT_DIR}/.tmp/release-stage"
VERSION="$(
  cd "${ROOT_DIR}"
  python3 - <<'PY'
from __future__ import annotations

import tomllib

with open("api/pyproject.toml", "rb") as fh:
    print(tomllib.load(fh)["project"]["version"])
PY
)"
NAME="jr-autorag-local-enterprise-${VERSION}"
ARTIFACT="${OUTPUT_DIR}/${NAME}.tar.gz"
CHECKSUM="${ARTIFACT}.sha256"

need_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$1" >&2
    exit 127
  fi
}

need_command bun
need_command python3
need_command rsync
need_command shasum
need_command tar

cd "${ROOT_DIR}"
bun run build

rm -rf "${STAGE_ROOT}"
mkdir -p "${STAGE_ROOT}/${NAME}" "${OUTPUT_DIR}"

rsync -a \
  --exclude ".git/" \
  --exclude ".github/" \
  --exclude ".DS_Store" \
  --exclude ".tmp/" \
  --exclude "node_modules/" \
  --exclude "api/.venv/" \
  --exclude ".venv/" \
  --exclude "venv/" \
  --exclude "data/" \
  --exclude "api/data/" \
  --exclude "evidence/" \
  --exclude "dist/release/" \
  --exclude "output/" \
  --exclude "test-results/" \
  --exclude "playwright-report/" \
  --exclude "*.log" \
  "${ROOT_DIR}/" "${STAGE_ROOT}/${NAME}/"

tar -czf "${ARTIFACT}" -C "${STAGE_ROOT}" "${NAME}"
shasum -a 256 "${ARTIFACT}" > "${CHECKSUM}"
cp "${CHECKSUM}" "${OUTPUT_DIR}/SHA256SUMS"

printf 'release_artifact=%s\n' "${ARTIFACT}"
printf 'release_checksum=%s\n' "${CHECKSUM}"
cat "${CHECKSUM}"
