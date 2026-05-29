#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_DIR="${ROOT_DIR}/api"
TMP_DIR="$(mktemp -d)"
API_PORT="${JR_INSTALL_SMOKE_API_PORT:-8123}"
WEB_PORT="${JR_INSTALL_SMOKE_WEB_PORT:-3099}"
SKIP_SYNC="${JR_INSTALL_SMOKE_SKIP_SYNC:-0}"

cleanup() {
  kill "${API_PID:-}" "${WEB_PID:-}" >/dev/null 2>&1 || true
  wait "${API_PID:-}" "${WEB_PID:-}" >/dev/null 2>&1 || true
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

need_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Missing required command: %s\n' "$1" >&2
    exit 127
  fi
}

check_port_available() {
  local port="$1"
  local label="$2"
  if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1; then
    printf '%s port %s is already in use. Set JR_INSTALL_SMOKE_%s_PORT to an available port.\n' \
      "${label}" "${port}" "$(printf '%s' "${label}" | tr '[:lower:]' '[:upper:]')" >&2
    exit 1
  fi
}

need_command bun
need_command curl
need_command python3
need_command uv

check_port_available "${API_PORT}" "api"
check_port_available "${WEB_PORT}" "web"

if [[ "${SKIP_SYNC}" != "1" ]]; then
  uv sync --project "${API_DIR}" --all-groups --locked
fi

(
  cd "${API_DIR}"
  uv run python - <<'PY'
from app.main import app

routes = {getattr(route, "path", "") for route in app.routes}
required = {"/healthz", "/readyz", "/install/report", "/security/posture"}
missing = sorted(required - routes)
if missing:
    raise SystemExit(f"missing required API routes: {missing}")
print("api_import=pass")
PY
)

cat > "${TMP_DIR}/fake_api.py" <<'PY'
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

port = int(sys.argv[1])
ready_file = Path(sys.argv[2])


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.startswith("/api/install-smoke"):
            payload = {"ok": True, "path": self.path}
        elif self.path == "/healthz":
            payload = {"status": "ok"}
        elif self.path == "/config":
            payload = {"profile": "Default", "provider_profiles": [], "provider": {"name": "mock", "base_url": "mock"}}
        else:
            self.send_response(404)
            self.end_headers()
            return

        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_: object) -> None:
        return


server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
ready_file.write_text("ready", encoding="utf-8")
server.serve_forever()
PY

python3 "${TMP_DIR}/fake_api.py" "${API_PORT}" "${TMP_DIR}/api-ready" &
API_PID="$!"

(
  cd "${ROOT_DIR}"
  env -i \
    PATH="${PATH}" \
    HOME="${HOME}" \
    BUN_PUBLIC_API_BASE_URL="http://127.0.0.1:${API_PORT}" \
    VITE_API_BASE_URL="http://127.0.0.1:${API_PORT}" \
    BUN_PUBLIC_BROWSER_API_BASE_URL="http://127.0.0.1:${API_PORT}" \
    VITE_BROWSER_API_BASE_URL="http://127.0.0.1:${API_PORT}" \
    PORT="${WEB_PORT}" \
    bun --hot src/index.ts
) > "${TMP_DIR}/web.log" 2>&1 &
WEB_PID="$!"

for _ in {1..80}; do
  if ! kill -0 "${WEB_PID}" >/dev/null 2>&1; then
    printf 'Install smoke failed. web server exited before becoming ready.\n' >&2
    printf 'Web server log:\n' >&2
    tail -n 80 "${TMP_DIR}/web.log" >&2 || true
    wait "${WEB_PID}" || true
    exit 1
  fi
  api_code="$(curl -sS -o "${TMP_DIR}/api-health.json" -w '%{http_code}' "http://127.0.0.1:${API_PORT}/healthz" 2>/dev/null || true)"
  web_code="$(curl -sS -o "${TMP_DIR}/web.html" -w '%{http_code}' "http://127.0.0.1:${WEB_PORT}/" 2>/dev/null || true)"
  proxy_body="$(curl -sS "http://127.0.0.1:${WEB_PORT}/api/install-smoke?check=proxy" 2>/dev/null || true)"
  if [[ "${api_code}" == "200" && "${web_code}" == "200" && "${proxy_body}" == *'"ok": true'* && "${proxy_body}" == *'/api/install-smoke?check=proxy'* ]]; then
    printf 'install_smoke=pass api=%s web=%s proxy=%s\n' "${api_code}" "${web_code}" "${proxy_body}"
    exit 0
  fi
  sleep 0.25
done

printf 'Install smoke failed. api=%s web=%s proxy=%s same_origin=%s\n' "${api_code:-none}" "${web_code:-none}" "${proxy_body:-none}" "${same_origin_body:-none}" >&2
printf 'Web server log:\n' >&2
tail -n 80 "${TMP_DIR}/web.log" >&2 || true
exit 1
