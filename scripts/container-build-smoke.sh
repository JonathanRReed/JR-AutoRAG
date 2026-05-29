#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_IMAGE="${API_IMAGE:-jr-autorag-api:local}"
WEB_IMAGE="${WEB_IMAGE:-jr-autorag-web:local}"
API_PORT="${API_PORT:-8017}"
WEB_PORT="${WEB_PORT:-3017}"
API_CONTAINER="jr-autorag-api-smoke-$$"
WEB_CONTAINER="jr-autorag-web-smoke-$$"
NETWORK_NAME="jr-autorag-smoke-$$"
TMP_DIR="$(mktemp -d)"

cleanup() {
  docker rm -f "${API_CONTAINER}" "${WEB_CONTAINER}" >/dev/null 2>&1 || true
  docker network rm "${NETWORK_NAME}" >/dev/null 2>&1 || true
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

need() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing required command: $1" >&2
    exit 1
  fi
}

wait_for_url() {
  local url="$1"
  local output="$2"
  local label="$3"

  for _ in {1..30}; do
    if curl -fsS "${url}" -o "${output}"; then
      return 0
    fi
    sleep 1
  done

  echo "${label} did not become ready at ${url}" >&2
  docker logs "${API_CONTAINER}" "${WEB_CONTAINER}" 2>/dev/null || true
  exit 1
}

need docker
need curl
need python3

docker build -t "${API_IMAGE}" "${ROOT_DIR}/api"
docker run --rm "${API_IMAGE}" python -c "from app.main import app; print('api_container_import=pass')"
docker network create "${NETWORK_NAME}" >/dev/null
docker run -d --rm --network "${NETWORK_NAME}" --name "${API_CONTAINER}" -p "127.0.0.1:${API_PORT}:8000" "${API_IMAGE}" >/dev/null
wait_for_url "http://127.0.0.1:${API_PORT}/healthz" "${TMP_DIR}/healthz.json" "API"
wait_for_url "http://127.0.0.1:${API_PORT}/readyz" "${TMP_DIR}/readyz.json" "API readiness"

python3 - "${TMP_DIR}/healthz.json" "${TMP_DIR}/readyz.json" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

health = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ready = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

if health.get("status") != "ok":
    raise SystemExit(f"unexpected healthz response: {health}")
if ready.get("ready") is not True:
    raise SystemExit(f"unexpected readyz response: {ready}")
PY

docker build -t "${WEB_IMAGE}" -f "${ROOT_DIR}/Dockerfile.web" "${ROOT_DIR}"
docker run -d --rm \
  --network "${NETWORK_NAME}" \
  --name "${WEB_CONTAINER}" \
  -e "BUN_PUBLIC_API_BASE_URL=http://${API_CONTAINER}:8000" \
  -e "VITE_API_BASE_URL=http://${API_CONTAINER}:8000" \
  -e "BUN_PUBLIC_BROWSER_API_BASE_URL=http://127.0.0.1:${API_PORT}" \
  -e "VITE_BROWSER_API_BASE_URL=http://127.0.0.1:${API_PORT}" \
  -p "127.0.0.1:${WEB_PORT}:3000" \
  "${WEB_IMAGE}" >/dev/null
wait_for_url "http://127.0.0.1:${WEB_PORT}/" "${TMP_DIR}/web.html" "Web"

if ! grep -q '<div id="root"></div>' "${TMP_DIR}/web.html"; then
  echo "web root container was not served" >&2
  exit 1
fi

python3 - "${TMP_DIR}/web.html" "http://127.0.0.1:${WEB_PORT}/" >"${TMP_DIR}/web-assets.txt" <<'PY'
from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse


class AssetParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.assets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "script" and values.get("src"):
            self.assets.append(values["src"] or "")
        if tag == "link" and values.get("rel") == "stylesheet" and values.get("href"):
            self.assets.append(values["href"] or "")


parser = AssetParser()
parser.feed(Path(sys.argv[1]).read_text(encoding="utf-8"))
base = sys.argv[2]
for asset in parser.assets:
    parsed = urlparse(urljoin(base, asset))
    print(parsed.geturl())
PY

while IFS= read -r asset_url; do
  [ -n "${asset_url}" ] || continue
  curl -fsSI "${asset_url}" >/dev/null
done <"${TMP_DIR}/web-assets.txt"

echo "container_build_smoke=pass"
