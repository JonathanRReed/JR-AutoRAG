"""Provider helper endpoints for discovering local and cloud runtimes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

from ..core.providers import (
    discover_local_providers,
    OpenRouterProvider,
    OllamaCloudProvider,
    ProviderError,
)
from ..schemas.config import LocalProviderInfo, ProviderKind

router = APIRouter(prefix="/providers", tags=["providers"])


class OpenRouterStatus(BaseModel):
    """OpenRouter provider status."""
    available: bool
    api_key_configured: bool
    default_model: str
    error_message: str | None = None


class OpenRouterModel(BaseModel):
    """OpenRouter model info."""
    id: str
    name: str
    context_length: int | None = None
    pricing: dict[str, Any] | None = None


class OpenRouterTestRequest(BaseModel):
    """Request to test OpenRouter connection."""
    model: str | None = None
    prompt: str = "Say 'hello' in one word."


class OpenRouterTestResponse(BaseModel):
    """Response from OpenRouter test."""
    success: bool
    model: str
    response: str | None = None
    error: str | None = None
    latency_ms: float | None = None


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
        return providers
    return providers


def _extract_openrouter_key(
    x_openrouter_key: str | None,
    authorization: str | None,
) -> str | None:
    """Prefer explicit header, fallback to Bearer auth."""
    if x_openrouter_key:
        return x_openrouter_key
    if authorization and authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    return None


@router.get("/openrouter/status", response_model=OpenRouterStatus)
async def openrouter_status(
    x_openrouter_key: str | None = Header(default=None, convert_underscores=False),
    authorization: str | None = Header(default=None),
) -> OpenRouterStatus:
    """Check OpenRouter provider status and API key configuration."""
    provided_key = _extract_openrouter_key(x_openrouter_key, authorization)
    provider = OpenRouterProvider(api_key=provided_key)
    api_key = provider.api_key
    
    if not api_key:
        return OpenRouterStatus(
            available=False,
            api_key_configured=False,
            default_model=provider.default_model or "openai/gpt-4o-mini",
            error_message="OpenRouter API key not configured",
        )
    
    try:
        models = await provider.list_models()
        return OpenRouterStatus(
            available=True,
            api_key_configured=True,
            default_model=provider.default_model or "openai/gpt-4o-mini",
        )
    except ProviderError as exc:
        return OpenRouterStatus(
            available=False,
            api_key_configured=True,
            default_model=provider.default_model or "openai/gpt-4o-mini",
            error_message=str(exc),
        )


@router.get("/openrouter/models", response_model=list[OpenRouterModel])
async def list_openrouter_models(
    x_openrouter_key: str | None = Header(default=None, convert_underscores=False),
    authorization: str | None = Header(default=None),
) -> list[OpenRouterModel]:
    """List available models from OpenRouter."""
    provided_key = _extract_openrouter_key(x_openrouter_key, authorization)
    provider = OpenRouterProvider(api_key=provided_key)
    api_key = provider.api_key

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="OpenRouter API key not configured",
        )
    
    try:
        models = await provider.list_models()
        return [
            OpenRouterModel(
                id=m.get("id", ""),
                name=m.get("name", m.get("id", "")),
                context_length=m.get("context_length"),
                pricing=m.get("pricing"),
            )
            for m in models
        ]
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _extract_ollama_key(
    x_ollama_key: str | None,
    authorization: str | None,
) -> str | None:
    """Prefer explicit header, fallback to Bearer auth."""
    if x_ollama_key:
        return x_ollama_key
    if authorization and authorization.lower().startswith("bearer "):
        return authorization.split(" ", 1)[1].strip()
    return None


@router.get("/ollama-cloud/models", response_model=list[str])
async def list_ollama_cloud_models(
    x_ollama_key: str | None = Header(default=None, convert_underscores=False),
    authorization: str | None = Header(default=None),
) -> list[str]:
    """List available models from Ollama Cloud."""
    provided_key = _extract_ollama_key(x_ollama_key, authorization)

    try:
        provider = OllamaCloudProvider(api_key=provided_key)
    except ProviderError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    try:
        models = await provider.list_models()
        return models
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/openrouter/test", response_model=OpenRouterTestResponse)
async def test_openrouter(
    request: OpenRouterTestRequest,
    x_openrouter_key: str | None = Header(default=None, convert_underscores=False),
    authorization: str | None = Header(default=None),
) -> OpenRouterTestResponse:
    """Test OpenRouter connection with a simple prompt."""
    import time
    
    provided_key = _extract_openrouter_key(x_openrouter_key, authorization)
    provider = OpenRouterProvider(api_key=provided_key)
    api_key = provider.api_key
    model = request.model or provider.default_model or "openai/gpt-4o-mini"
    
    if not api_key:
        return OpenRouterTestResponse(
            success=False,
            model=model,
            error="OpenRouter API key not configured",
        )
    
    start = time.perf_counter()
    try:
        response = await provider.chat(
            messages=[{"role": "user", "content": request.prompt}],
            model=model,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return OpenRouterTestResponse(
            success=True,
            model=model,
            response=response,
            latency_ms=latency_ms,
        )
    except ProviderError as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        return OpenRouterTestResponse(
            success=False,
            model=model,
            error=str(exc),
            latency_ms=latency_ms,
        )
