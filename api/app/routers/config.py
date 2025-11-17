from fastapi import APIRouter, Depends, HTTPException, Query

from ..schemas.config import AppConfig, ProviderConfig, ProviderProfile
from ..services import get_container, ServiceContainer
from ..core.providers import discover_models, ProviderError

router = APIRouter()


@router.get("", response_model=AppConfig)
def read_config(container: ServiceContainer = Depends(get_container)):
    return container.config_store.read()


@router.put("", response_model=AppConfig)
def update_config(
    cfg: AppConfig,
    container: ServiceContainer = Depends(get_container),
    active_profile: str | None = Query(default=None, description="Optional provider profile to activate"),
):
    try:
        if active_profile and cfg.provider_profiles:
            profile = next((p for p in cfg.provider_profiles if p.name == active_profile), None)
            if not profile:
                raise HTTPException(status_code=404, detail=f"Profile '{active_profile}' not found")
            cfg.provider = profile.provider
        stored = container.config_store.write(cfg)
        container.orchestrator.rebuild(stored)
        return stored
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/models", response_model=list[str])
async def list_models(payload: ProviderConfig):
    try:
        models = await discover_models(payload)
        return models
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
