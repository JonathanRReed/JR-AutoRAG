"""Tests for persisted configuration handling."""

from __future__ import annotations

import json

from app.core.config_store import ConfigStore
from app.schemas.config import AppConfig


def test_config_store_repairs_persisted_zero_dense_k(tmp_path) -> None:
    path = tmp_path / "config.json"
    payload = AppConfig().model_dump(mode="json")
    payload["retrieval"]["dense_k"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")
    store = ConfigStore(path)

    cfg = store.read()

    assert cfg.retrieval.dense_k == 1
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["retrieval"]["dense_k"] == 1
