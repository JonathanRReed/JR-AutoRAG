
from fastapi import APIRouter, Depends, HTTPException, Query

from ..core.providers import ProviderError, discover_models
from ..schemas.config import RETRIEVAL_PRESETS, AppConfig, ProviderConfig, RetrievalDefaults
from ..services import ServiceContainer, get_container

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
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/models", response_model=list[str])
async def list_models(payload: ProviderConfig):
    try:
        models = await discover_models(payload)
        return models
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.get("/presets", response_model=dict[str, RetrievalDefaults])
def list_presets():
    """List available retrieval presets (fast, balanced, thorough)."""
    return RETRIEVAL_PRESETS


@router.post("/presets/{preset_name}", response_model=AppConfig)
def apply_preset(
    preset_name: str,
    container: ServiceContainer = Depends(get_container),
):
    """Apply a retrieval preset to the current configuration."""
    preset_name_lower = preset_name.lower()
    if preset_name_lower not in RETRIEVAL_PRESETS:
        raise HTTPException(
            status_code=404,
            detail=f"Preset '{preset_name}' not found. Available: {list(RETRIEVAL_PRESETS.keys())}"
        )

    cfg = container.config_store.read()
    cfg.retrieval = RETRIEVAL_PRESETS[preset_name_lower].model_copy()
    stored = container.config_store.write(cfg)
    container.orchestrator.rebuild(stored)
    return stored
