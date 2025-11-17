"""Configuration persistence utilities."""

from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Optional
import json

from ..schemas.config import AppConfig


class ConfigStore:
    """JSON-backed AppConfig persistence.

    Persists a single configuration profile to disk so the admin console can
    read/write settings without restarting the FastAPI process.
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path or Path.cwd() / "data" / "config.json")
        self._lock = RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self.write(AppConfig())

    def read(self) -> AppConfig:
        with self._lock:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return AppConfig.model_validate(data)

    def write(self, cfg: AppConfig) -> AppConfig:
        payload = cfg.model_dump(mode="json")
        with self._lock:
            self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return cfg
