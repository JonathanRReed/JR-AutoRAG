from __future__ import annotations

from typing import Any

import pytest

from app.schemas.config import AppConfig
from app.core.trace_replay import TraceReplayer


class FakeOrchestrator:
    def __init__(self) -> None:
        self._config: AppConfig | None = AppConfig(profile="Live")
        self._provider = None
        self.rebuild_profiles: list[str] = []
        self.answer_profiles: list[str | None] = []

    def rebuild(self, config: AppConfig) -> None:
        self._config = config
        self.rebuild_profiles.append(config.profile)

    async def answer(self, query: str, trace_id: str) -> dict[str, Any]:
        self.answer_profiles.append(self._config.profile if self._config else None)
        return {
            "answer": f"answer for {query}",
            "trace_bundle": {
                "trace_id": trace_id,
                "query": query,
                "answer": f"answer for {query}",
                "retrieval": [],
                "reranker": [],
                "prompts": {},
                "latency": {},
                "citations": [],
                "confidence": {},
                "config_snapshot": self._config.model_dump(mode="json")
                if self._config
                else {},
            },
        }


@pytest.mark.asyncio
async def test_trace_replay_applies_snapshot_and_restores_live_config() -> None:
    orchestrator = FakeOrchestrator()
    replay_config = AppConfig(profile="Replay")
    result = await TraceReplayer(orchestrator).replay(
        {
            "trace_id": "original",
            "query": "What changed?",
            "answer": "old",
            "retrieval": [],
            "reranker": [],
            "prompts": {},
            "latency": {},
            "citations": [],
            "confidence": {},
            "config_snapshot": replay_config.model_dump(mode="json"),
        },
        compare=False,
    )

    assert result.success is True
    assert result.config_snapshot_applied is True
    assert result.config_snapshot_error is None
    assert orchestrator.answer_profiles == ["Replay"]
    assert orchestrator._config is not None
    assert orchestrator._config.profile == "Live"
    assert orchestrator.rebuild_profiles == ["Replay", "Live"]


@pytest.mark.asyncio
async def test_trace_replay_skips_snapshot_with_redacted_secret() -> None:
    orchestrator = FakeOrchestrator()
    snapshot = AppConfig(profile="Replay").model_dump(mode="json")
    snapshot["provider"] = {
        "name": "OpenAI-compatible",
        "base_url": "http://localhost:11434",
        "api_key": "[redacted]",
    }

    result = await TraceReplayer(orchestrator).replay(
        {
            "trace_id": "original",
            "query": "What changed?",
            "config_snapshot": snapshot,
        },
        compare=False,
    )

    assert result.success is True
    assert result.config_snapshot_applied is False
    assert result.config_snapshot_error is not None
    assert "redacted secret" in result.config_snapshot_error
    assert orchestrator.answer_profiles == ["Live"]
    assert orchestrator.rebuild_profiles == []
