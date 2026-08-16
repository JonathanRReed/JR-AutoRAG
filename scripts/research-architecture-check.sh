#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOC="${ROOT_DIR}/UPGRADE-2026.md"

python3 - "${ROOT_DIR}" "${DOC}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

root = Path(sys.argv[1])
doc = Path(sys.argv[2])
if not doc.is_file():
    raise SystemExit(f"missing current architecture record: {doc}")

text = doc.read_text(encoding="utf-8")
required_markers = (
    "# JR-AutoRAG Upgrade Plan",
    "Supersedes the original",
    "### Phase 2: SOTA Retrieval Upgrades",
    "Late Chunking",
    "Per-Query Hybrid Weights",
    "Contextual Enrichment as Default",
    "### Phase 3: Eval Gates CI Integration",
    "### Phase 4: Security Hardening",
    "Canary Token Manager",
    "Poisoned Chunk Scanner",
    "### Phase 5: UI/UX P0 Fixes",
    "## 1. Where the project actually is",
    "## 2. Remaining work",
)
missing_markers = [marker for marker in required_markers if marker not in text]
if missing_markers:
    raise SystemExit(
        f"current architecture record is missing required markers: {missing_markers}"
    )

required_files: dict[str, tuple[str, ...]] = {
    "README.md": (),
    "SECURITY.md": (),
    "UPGRADE-2026.md": (),
    "api/app/core/chunking.py": ("class LateChunker",),
    "api/app/core/hybrid_retrieval.py": ("class AutoHybridWeights",),
    "api/app/core/contextual_enrichment.py": ("class ContextualEnricher",),
    "api/app/core/prompt_guard.py": (
        "class CanaryTokenManager",
        "class PoisonedChunkScanner",
    ),
    "api/app/core/eval_gates.py": ("class GatedEvaluator",),
    "api/app/core/ingest.py": (),
    "api/tests/core/test_sota_retrieval_upgrades.py": (),
    "api/tests/core/test_security_hardening.py": (),
    "api/tests/core/test_contextual_enrichment.py": (),
    "src/frontend.tsx": (),
    "scripts/evidence-bundle.sh": (),
    "scripts/research-architecture-check.sh": (),
}

for relative_path, required_symbols in required_files.items():
    file_path = root / relative_path
    if not file_path.is_file():
        raise SystemExit(f"current architecture references missing path: {relative_path}")
    if not required_symbols:
        continue
    source = file_path.read_text(encoding="utf-8")
    for symbol in required_symbols:
        if symbol not in source:
            raise SystemExit(
                f"current architecture path {relative_path} is missing symbol: {symbol}"
            )

print(f"research_architecture=pass checked_paths={len(required_files)}")
PY
