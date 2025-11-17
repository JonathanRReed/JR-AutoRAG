"""Provider abstractions for Ollama, LM Studio, and cloud endpoints."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import httpx

from ..schemas.config import LocalProviderInfo, ProviderConfig, ProviderKind


class ProviderError(RuntimeError):
    """Raised when a provider request fails."""


class LLMProvider:
    """Minimal interface for chat/complete operations."""

    def __init__(self, base_url: str, default_model: Optional[str] = None) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.default_model = default_model

    async def chat(self, messages: Iterable[Dict[str, Any]], **kwargs: Any) -> str:
        raise NotImplementedError

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError


class _HTTPProvider(LLMProvider):
    """Shared utilities for HTTP based providers."""

    endpoint: str = ""

    async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(f"Provider error {exc.response.status_code}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Provider request failed: {exc}") from exc


class OllamaProvider(_HTTPProvider):
    async def chat(self, messages: Iterable[Dict[str, Any]], **kwargs: Any) -> str:
        model = kwargs.get("model") or self.default_model or "llama3"
        payload = {
            "model": model,
            "messages": list(messages),
            "stream": False,
        }
        data = await self._post("/api/chat", payload)
        return data.get("message", {}).get("content", "")

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        model = kwargs.get("model") or self.default_model or "llama3"
        payload = {"model": model, "prompt": prompt, "stream": False}
        data = await self._post("/api/generate", payload)
        return data.get("response", "")


class LMStudioProvider(_HTTPProvider):
    async def chat(self, messages: Iterable[Dict[str, Any]], **kwargs: Any) -> str:
        model = kwargs.get("model") or self.default_model or "gpt-3.5-turbo"
        payload = {"model": model, "messages": list(messages)}
        data = await self._post("/v1/chat/completions", payload)
        choices = data.get("choices") or []
        return choices[0]["message"]["content"] if choices else ""

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        model = kwargs.get("model") or self.default_model or "gpt-3.5-turbo-instruct"
        payload = {"model": model, "prompt": prompt}
        data = await self._post("/v1/completions", payload)
        choices = data.get("choices") or []
        return choices[0].get("text", "") if choices else ""


class CloudProvider(_HTTPProvider):
    def __init__(self, base_url: str, default_model: Optional[str] = None, api_key: Optional[str] = None) -> None:
        super().__init__(base_url, default_model)
        self.api_key = api_key

    async def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        try:
            async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
                response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(f"Cloud provider error {exc.response.status_code}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Cloud provider request failed: {exc}") from exc

    async def chat(self, messages: Iterable[Dict[str, Any]], **kwargs: Any) -> str:
        model = kwargs.get("model") or self.default_model or "gpt-4o-mini"
        payload = {"model": model, "messages": list(messages)}
        data = await self._post("/v1/chat/completions", payload)
        choices = data.get("choices") or []
        return choices[0]["message"]["content"] if choices else ""

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        model = kwargs.get("model") or self.default_model or "gpt-4o-mini"
        payload = {"model": model, "prompt": prompt}
        data = await self._post("/v1/completions", payload)
        choices = data.get("choices") or []
        return choices[0].get("text", "") if choices else ""


@dataclass(slots=True)
class ProviderFactory:
    """Build provider clients based on `ProviderConfig`."""

    api_key: Optional[str] = None

    def build(self, cfg: ProviderConfig) -> LLMProvider:
        name = (cfg.name or "").lower()
        if "ollama" in name:
            return OllamaProvider(str(cfg.base_url), cfg.planner_model or cfg.generator_model)
        if "lm" in name or "studio" in name:
            return LMStudioProvider(str(cfg.base_url), cfg.generator_model)
        return CloudProvider(str(cfg.base_url), cfg.generator_model, api_key=cfg.api_key or self.api_key)


async def discover_models(cfg: ProviderConfig) -> List[str]:
    """Fetch available model names for a provider."""

    base = str(cfg.base_url).rstrip("/")
    headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else None
    async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
        kind = (cfg.name or "").lower()
        try:
            if "ollama" in kind:
                resp = await client.get(f"{base}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                return [model.get("name", "") for model in data.get("models", []) if model.get("name")]
            if "lm" in kind or "studio" in kind:
                resp = await client.get(f"{base}/v1/models")
                resp.raise_for_status()
                data = resp.json()
                return [item.get("id", "") for item in data.get("data", []) if item.get("id")]
            # Fallback for OpenAI-compatible clouds
            resp = await client.get(f"{base}/v1/models")
            resp.raise_for_status()
            data = resp.json()
            return [item.get("id", "") for item in data.get("data", []) if item.get("id")]
        except httpx.HTTPError as exc:
            raise ProviderError(f"Failed to discover models: {exc}") from exc


_DEFAULT_OLLAMA_URL = os.environ.get("JR_OLLAMA_URL", "http://localhost:11434")
_DEFAULT_LMSTUDIO_URL = os.environ.get("JR_LMSTUDIO_URL", "http://localhost:1234")


async def _probe_ollama(base_url: str) -> LocalProviderInfo:
    base = base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=3.0) as client:
        try:
            tags = await client.get(f"{base}/api/tags")
            tags.raise_for_status()
            data = tags.json()
        except (httpx.HTTPError, httpx.RequestError) as exc:
            raise ProviderError(f"Ollama tags request failed: {exc}") from exc

        models = [model.get("name", "") for model in data.get("models", []) if model.get("name")]

        running: List[str] = []
        try:
            ps_resp = await client.get(f"{base}/api/ps")
            ps_resp.raise_for_status()
            running = [model.get("model", "") for model in ps_resp.json().get("models", []) if model.get("model")]
        except (httpx.HTTPError, httpx.RequestError):
            running = []

        version: Optional[str] = None
        try:
            version_resp = await client.get(f"{base}/api/version")
            version_resp.raise_for_status()
            version = version_resp.json().get("version")
        except (httpx.HTTPError, httpx.RequestError):
            version = None

    return LocalProviderInfo(
        kind=ProviderKind.OLLAMA,
        name="Ollama",
        base_url=base_url,
        models=models,
        running=[m for m in running if m],
        version=version,
    )


async def _probe_lmstudio(base_url: str) -> LocalProviderInfo:
    base = base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=3.0) as client:
        try:
            models_resp = await client.get(f"{base}/api/v0/models")
            models_resp.raise_for_status()
        except (httpx.HTTPError, httpx.RequestError) as exc:
            raise ProviderError(f"LM Studio models request failed: {exc}") from exc

        payload = models_resp.json()
        entries = payload.get("data", [])
        models = [entry.get("id", "") for entry in entries if entry.get("id")]
        running = [entry.get("id", "") for entry in entries if entry.get("state") == "loaded" and entry.get("id")]

        version: Optional[str] = None
        try:
            version_resp = await client.get(f"{base}/api/v0/version")
            version_resp.raise_for_status()
            version = version_resp.json().get("version")
        except (httpx.HTTPError, httpx.RequestError):
            version = None

    return LocalProviderInfo(
        kind=ProviderKind.LM_STUDIO,
        name="LM Studio",
        base_url=base_url,
        models=models,
        running=running,
        version=version,
    )


async def discover_local_providers() -> List[LocalProviderInfo]:
    probes = [
        (ProviderKind.OLLAMA, "Ollama", _DEFAULT_OLLAMA_URL, _probe_ollama),
        (ProviderKind.LM_STUDIO, "LM Studio", _DEFAULT_LMSTUDIO_URL, _probe_lmstudio),
    ]

    providers: List[LocalProviderInfo] = []
    for kind, name, base, func in probes:
        if not base:
            continue
        try:
            info = await func(base)
            info.status = "ok"
            providers.append(info)
        except ProviderError as exc:
            providers.append(
                LocalProviderInfo(
                    kind=kind,
                    name=name,
                    base_url=base,
                    models=[],
                    running=[],
                    status="error",
                    error_message=str(exc),
                )
            )
    return providers
