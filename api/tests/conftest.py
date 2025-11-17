"""Test configuration shared across suites."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the FastAPI "app" package (api/app) is importable when pytest runs
PROJECT_API_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_API_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_API_ROOT))
