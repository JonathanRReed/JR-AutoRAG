"""Provider helper endpoints for discovering local runtimes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..core.providers import discover_local_providers
from ..schemas.config import LocalProviderInfo

router = APIRouter(prefix="/providers", tags=["providers"])


@router.get("/local", response_model=list[LocalProviderInfo])
async def list_local_providers() -> list[LocalProviderInfo]:
    try:
        providers = await discover_local_providers()
    except Exception as exc:  # pragma: no cover - unexpected runtime failures
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if not providers:
        raise HTTPException(status_code=404, detail="No provider probes configured")
    if any(provider.status == "error" for provider in providers) and not all(
        provider.status == "error" for provider in providers
    ):
        # Mixed success; surface warning but still return 200 payload
        # Clients can inspect provider.status to show warnings.
        return providers
    return providers
