#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python3 - "${ROOT_DIR}" <<'PY'
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])

result = subprocess.run(
    ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
    cwd=root,
    check=True,
    capture_output=True,
    text=True,
)

skip_exact = {
    "api/uv.lock",
    "bun.lock",
    "package-lock.json",
}
skip_prefixes = (
    "node_modules/",
    "dist/",
    "api/.venv/",
    ".venv/",
    "audit/evidence/raw/",
)
forbidden_paths = (
    "api/data/",
    "data/",
    "evidence/",
)

secret_patterns: list[tuple[str, re.Pattern[str]]] = [
    ("private key block", re.compile(r"-----BEGIN (?:RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY-----")),
    ("OpenAI API key", re.compile(r"\bsk-[A-Za-z0-9_-]{32,}\b")),
    ("Anthropic API key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{32,}\b")),
    ("OpenRouter API key", re.compile(r"\bsk-or-v1-[A-Za-z0-9_-]{32,}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{30,}\b")),
    ("AWS access key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b")),
]

placeholder_terms = (
    "...",
    "example",
    "placeholder",
    "your-",
    "<",
    "${",
    "redacted",
)

findings: list[str] = []

for rel in result.stdout.splitlines():
    if not rel or rel in skip_exact or rel.startswith(skip_prefixes):
        continue
    if rel.startswith(forbidden_paths):
        findings.append(f"{rel}: tracked runtime/client data path")
        continue

    path = root / rel
    if not path.is_file():
        continue
    try:
        data = path.read_bytes()
    except OSError as exc:
        findings.append(f"{rel}: could not read file: {exc}")
        continue
    if b"\0" in data:
        continue
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        continue

    for line_no, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()
        for label, pattern in secret_patterns:
            match = pattern.search(line)
            if not match:
                continue
            value = match.group(0)
            if any(term in lowered or term in value.lower() for term in placeholder_terms):
                continue
            findings.append(f"{rel}:{line_no}: possible {label}")

if findings:
    print("secret_scan=fail")
    for finding in findings:
        print(finding)
    raise SystemExit(1)

print("secret_scan=pass")
PY
