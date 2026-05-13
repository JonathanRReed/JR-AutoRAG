#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 - "${ROOT_DIR}" <<'PY'
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])


def read(path: str) -> str:
    full = root / path
    if not full.exists():
        raise SystemExit(f"missing required file: {path}")
    return full.read_text(encoding="utf-8")


api_dockerfile = read("api/Dockerfile")
web_dockerfile = read("Dockerfile.web")
compose = read("docker-compose.yml")
dockerignore = read(".dockerignore")
api_dockerignore = read("api/.dockerignore")
json.loads(read("package.json"))

required_api_patterns = {
    "uv binary image": r"COPY\s+--from=ghcr\.io/astral-sh/uv:[^\s]+\s+/uv\s+/uvx\s+/bin/",
    "locked project manifest copy": r"COPY\s+pyproject\.toml\s+uv\.lock\s+\./",
    "locked production sync": r"RUN\s+uv\s+sync\s+--locked\s+--no-dev\s+--no-install-project",
    "venv runtime path": r'ENV\s+PATH="/app/\.venv/bin:\$\{PATH\}"',
    "cpu torch backend": r"UV_TORCH_BACKEND=cpu",
    "api app copy": r"COPY\s+app\s+\./app",
}
for label, pattern in required_api_patterns.items():
    if not re.search(pattern, api_dockerfile):
        raise SystemExit(f"api/Dockerfile missing {label}")

for forbidden in ("pip install", "requirements.txt", "uv pip install --system"):
    if forbidden in api_dockerfile:
        raise SystemExit(f"api/Dockerfile must not use {forbidden}")

if "FROM oven/bun:" not in web_dockerfile:
    raise SystemExit("Dockerfile.web must use the official Bun image")
if "bun install --frozen-lockfile" not in web_dockerfile:
    raise SystemExit("Dockerfile.web must use frozen Bun installs")
if "--production=false" in web_dockerfile:
    raise SystemExit("Dockerfile.web must not pass --production=false to bun install")
if 'CMD ["bun", "start"]' not in web_dockerfile:
    raise SystemExit("Dockerfile.web must run the production Bun start command")

container_smoke = read("scripts/container-build-smoke.sh")
for required in (
    "docker build -t \"${API_IMAGE}\"",
    "docker run --rm \"${API_IMAGE}\" python -c",
    "docker network create",
    "/healthz",
    "/readyz",
    "/__api/healthz",
    "docker build -t \"${WEB_IMAGE}\"",
    "web-assets.txt",
):
    if required not in container_smoke:
        raise SystemExit(f"container build smoke missing {required}")

required_compose = [
    "build: ./api",
    "dockerfile: Dockerfile.web",
    "BUN_PUBLIC_API_BASE_URL: http://api:8000",
    "JR_DATA_DIR: /data",
    "api-data:/data",
]
for value in required_compose:
    if value not in compose:
        raise SystemExit(f"docker-compose.yml missing {value}")

required_ignores = {
    "node_modules",
    "api/.venv",
    ".venv",
    ".env",
    ".env.*",
    "data",
    "api/data",
    "evidence",
}
ignore_lines = {line.strip() for line in dockerignore.splitlines() if line.strip() and not line.startswith("#")}
missing_ignores = sorted(required_ignores - ignore_lines)
if missing_ignores:
    raise SystemExit(f".dockerignore missing entries: {missing_ignores}")

required_api_ignores = {
    ".venv",
    "venv",
    "data/",
    "tests/",
    ".env",
    ".env.*",
    "*.log",
}
api_ignore_lines = {line.strip() for line in api_dockerignore.splitlines() if line.strip() and not line.startswith("#")}
missing_api_ignores = sorted(required_api_ignores - api_ignore_lines)
if missing_api_ignores:
    raise SystemExit(f"api/.dockerignore missing entries: {missing_api_ignores}")

print("container_manifest=pass")
PY
