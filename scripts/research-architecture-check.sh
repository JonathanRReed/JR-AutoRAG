#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC="${ROOT_DIR}/docs/architecture/research-backed-rag-architecture.md"

python3 - "${ROOT_DIR}" "${DOC}" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
doc = Path(sys.argv[2])
text = doc.read_text(encoding="utf-8")

required_markers = [
    "Last verified: 2026-05-12.",
    "https://arxiv.org/abs/2506.00054",
    "https://arxiv.org/abs/2501.09136",
    "https://arxiv.org/abs/2602.03442",
    "https://arxiv.org/abs/2510.13910",
    "https://arxiv.org/abs/2510.10114",
    "https://arxiv.org/abs/2507.09477",
    "https://arxiv.org/abs/2501.14342",
    "https://arxiv.org/abs/2603.21654",
    "https://arxiv.org/abs/2602.09319",
    "https://arxiv.org/abs/2601.09985",
    "https://arxiv.org/abs/2505.00105",
    "https://arxiv.org/abs/2511.13057",
    "https://arxiv.org/abs/2408.08067",
    "Agentic hierarchical retrieval interfaces",
    "Agentic capability benchmarks",
    "Handoff-gated robustness benchmark",
    "RAG threat and leakage controls",
    "Memory-efficient retrieval modes",
    "Fine-grained evaluation receipts",
]

missing = [marker for marker in required_markers if marker not in text]
if missing:
    raise SystemExit(f"research architecture missing required markers: {missing}")

checked_paths: list[str] = []
for match in re.finditer(r"`([^`]+)`", text):
    value = match.group(1)
    if not value.startswith(("api/", "src/", "scripts/", "Public/", "README.md")):
        continue
    checked_paths.append(value)
    if not (root / value).exists():
        raise SystemExit(f"research architecture references missing path: {value}")

if len(checked_paths) < 25:
    raise SystemExit(f"research architecture path coverage is too thin: {len(checked_paths)}")

print(f"research_architecture=pass checked_paths={len(checked_paths)}")
PY
