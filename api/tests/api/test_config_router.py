from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.core.providers import ProviderError
from app.routers import config as config_router
from app.routers.config import list_models, update_config
from app.schemas.config import AppConfig, DeploymentProfile, ProviderConfig


class RecordingConfigStore:
    def __init__(self) -> None:
        self.writes: list[AppConfig] = []

    def read(self) -> AppConfig:
        return AppConfig()

    def write(self, cfg: AppConfig) -> AppConfig:
        self.writes.append(cfg)
        return cfg


class RecordingContainer:
    def __init__(self, *, fail_apply: bool = False) -> None:
        self.config_store = RecordingConfigStore()
        self.fail_apply = fail_apply
        self.events: list[str] = []

    def prepare_config_for_storage(self, cfg: AppConfig) -> AppConfig:
        self.events.append("prepare")
        return cfg

    def apply_config(self, _cfg: AppConfig) -> None:
        self.events.append("apply")
        if self.fail_apply:
            raise ProviderError("Ollama Cloud requires an API key")


def test_update_config_does_not_persist_when_apply_fails() -> None:
    container = RecordingContainer(fail_apply=True)
    cfg = AppConfig(
        deployment_profile=DeploymentProfile.CLOUD_ACCELERATED,
        provider=ProviderConfig(
            name="Ollama Cloud",
            base_url="https://ollama.com",
            planner_model="llama3",
            generator_model="llama3",
            gatherer_model="llama3",
            api_key="",
        )
    )

    with pytest.raises(HTTPException) as exc_info:
        update_config(cfg, container=container)

    assert exc_info.value.status_code == 400
    assert "Ollama Cloud requires an API key" in str(exc_info.value.detail)
    assert container.config_store.writes == []
    assert container.events == ["prepare", "apply"]


def test_update_config_persists_after_successful_apply() -> None:
    container = RecordingContainer()
    cfg = AppConfig()

    stored = update_config(cfg, container=container)

    assert stored == cfg
    assert container.config_store.writes == [cfg]
    assert container.events == ["prepare", "apply"]


@pytest.mark.asyncio
async def test_model_discovery_uses_active_deployment_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    container = RecordingContainer()
    discover = AsyncMock(return_value=[])
    monkeypatch.setattr(config_router, "discover_models", discover)
    provider = ProviderConfig(name="OpenAI", base_url="https://api.openai.com/v1")

    result = await list_models(provider, container=container)

    assert result == []
    discover.assert_awaited_once_with(
        provider,
        deployment_profile=DeploymentProfile.LOCAL_ONLY,
    )
