"""Provider abstractions for Ollama, LM Studio, and cloud endpoints."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx

from ..schemas.config import LocalProviderInfo, ProviderConfig, ProviderKind
from .secrets_vault import get_secrets_vault

DEFAULT_PROVIDER_TIMEOUT = 300.0
DEFAULT_STREAM_TIMEOUT = 300.0
DEFAULT_CLOUD_TIMEOUT = 120.0

_shared_async_client: httpx.AsyncClient | None = None


def get_shared_client() -> httpx.AsyncClient:
    """Get or create a shared httpx AsyncClient with connection pooling."""
    global _shared_async_client
    if _shared_async_client is None or _shared_async_client.is_closed:
        _shared_async_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=30.0,
                read=DEFAULT_PROVIDER_TIMEOUT,
                write=60.0,
                pool=5.0,
            ),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30.0,
            ),
        )
    return _shared_async_client


async def close_shared_client() -> None:
    """Close the shared httpx client (call on app shutdown)."""
    global _shared_async_client
    if _shared_async_client is not None:
        await _shared_async_client.aclose()
        _shared_async_client = None


@asynccontextmanager
async def get_client(timeout: float | httpx.Timeout | None = None, headers: dict[str, str] | None = None):
    """Get a client for making requests.

    Uses the shared client with connection pooling for better performance.
    For custom timeout/headers, creates a temporary client.
    """
    if timeout is None and headers is None:
        yield get_shared_client()
    else:
        effective_timeout = timeout if isinstance(timeout, httpx.Timeout) else httpx.Timeout(
            connect=30.0,
            read=timeout or DEFAULT_PROVIDER_TIMEOUT,
            write=60.0,
            pool=5.0,
        )
        async with httpx.AsyncClient(timeout=effective_timeout, headers=headers) as client:
            yield client


def _get_timeout(env_key: str, default: float) -> float:
    raw = os.environ.get(env_key)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _infer_secret_key_name(name: str, base_url: str) -> str:
    lowered = (name or "").lower()
    base = (base_url or "").lower()
    if "openrouter" in lowered or "openrouter" in base:
        return "OPENROUTER_API_KEY"
    if "ollama.com" in base or "ollama cloud" in lowered or "ollama_cloud" in lowered:
        return "OLLAMA_API_KEY"
    if "lm" in lowered or "studio" in lowered:
        return "LM_STUDIO_API_KEY"
    if "openai" in lowered or "api.openai.com" in base:
        return "OPENAI_API_KEY"
    if "anthropic" in lowered or "claude" in lowered:
        return "ANTHROPIC_API_KEY"
    if "google" in lowered or "gemini" in lowered:
        return "GOOGLE_API_KEY"
    sanitized = "".join(ch if ch.isalnum() else "_" for ch in (name or "PROVIDER").upper())
    return f"{sanitized}_API_KEY"


def resolve_provider_api_key(
    name: str,
    base_url: str,
    api_key: str | None = None,
    fallback: str | None = None,
) -> str | None:
    if api_key:
        trimmed = api_key.strip()
        if trimmed:
            return trimmed
    if fallback:
        trimmed = fallback.strip()
        if trimmed:
            return trimmed
    vault = get_secrets_vault()
    return vault.get(_infer_secret_key_name(name, base_url))


class ProviderError(RuntimeError):
    """Raised when a provider request fails."""


class LLMProvider:
    """Minimal interface for chat/complete operations."""

    def __init__(self, base_url: str, default_model: str | None = None) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.default_model = default_model

    async def chat(self, messages: Iterable[dict[str, Any]], **kwargs: Any) -> str:
        raise NotImplementedError

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        raise NotImplementedError

    async def chat_stream(self, messages: Iterable[dict[str, Any]], **kwargs: Any) -> AsyncIterator[str]:
        raise NotImplementedError

    # Optional: surface token-level stats for uncertainty monitoring.
    # Subclasses can override; default returns empty stats.
    def get_token_stats(self, text: str) -> dict[str, float | None]:  # pragma: no cover - optional hook
        return {"avg_logprob": None, "entropy": None, "logit_margin": None}


class _HTTPProvider(LLMProvider):
    """Shared utilities for HTTP based providers."""

    endpoint: str = ""

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            timeout = _get_timeout("JR_PROVIDER_TIMEOUT", DEFAULT_PROVIDER_TIMEOUT)
            async with get_client(timeout) as client:
                response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(f"Provider error {exc.response.status_code}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Provider request failed ({type(exc).__name__}): {exc}") from exc


class OllamaProvider(_HTTPProvider):
    """Provider for local Ollama instances."""

    def __init__(self, base_url: str, default_model: str | None = None, api_key: str | None = None) -> None:
        super().__init__(base_url, default_model)
        self.api_key = api_key

    def _get_headers(self) -> dict[str, str] | None:
        if self.api_key:
            return {
                "Authorization": f"Bearer {self.api_key}",
                "X-API-Key": self.api_key,
                "X-Ollama-Key": self.api_key,
            }
        return None

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = self._get_headers()
        try:
            timeout = _get_timeout("JR_PROVIDER_TIMEOUT", DEFAULT_PROVIDER_TIMEOUT)
            async with get_client(timeout, headers) as client:
                response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(f"Ollama error {exc.response.status_code}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Ollama request failed ({type(exc).__name__}): {exc}") from exc

    async def chat(self, messages: Iterable[dict[str, Any]], **kwargs: Any) -> str:
        model = kwargs.get("model") or self.default_model or "llama3"
        payload = {
            "model": model,
            "messages": list(messages),
            "stream": False,
        }
        data = await self._post("/api/chat", payload)
        return data.get("message", {}).get("content", "")

    async def chat_stream(self, messages: Iterable[dict[str, Any]], **kwargs: Any):
        model = kwargs.get("model") or self.default_model or "llama3"
        payload = {
            "model": model,
            "messages": list(messages),
            "stream": True,
        }
        url = f"{self.base_url}/api/chat"
        headers = self._get_headers()
        try:
            timeout = _get_timeout("JR_PROVIDER_STREAM_TIMEOUT", DEFAULT_STREAM_TIMEOUT)
            async with httpx.AsyncClient(timeout=timeout, headers=headers) as client, client.stream(
                "POST", url, json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    message = data.get("message", {})
                    content = message.get("content", "")
                    if content:
                        yield content
                    if data.get("done") is True:
                        break
        except httpx.HTTPStatusError as exc:
            raise ProviderError(f"Provider error {exc.response.status_code}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Provider request failed ({type(exc).__name__}): {exc}") from exc

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        model = kwargs.get("model") or self.default_model or "llama3"
        payload = {"model": model, "prompt": prompt, "stream": False}
        data = await self._post("/api/generate", payload)
        return data.get("response", "")


class OllamaCloudProvider(OllamaProvider):
    """Provider for Ollama Cloud service (https://ollama.com).

    Ollama Cloud provides access to models without requiring local GPU.
    Uses the same API as local Ollama but requires authentication.

    Environment variables:
    - OLLAMA_API_KEY: API key from https://ollama.com/settings/keys

    Features:
    - Free tier available with hourly/weekly limits
    - No data retention
    - Access to large models that may not fit on local GPUs
    """

    OLLAMA_CLOUD_URL = "https://ollama.com"

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str | None = None,
    ) -> None:
        resolved_key = resolve_provider_api_key("ollama cloud", self.OLLAMA_CLOUD_URL, api_key)
        super().__init__(self.OLLAMA_CLOUD_URL, default_model or "llama3", resolved_key)
        if not self.api_key:
            raise ProviderError("Ollama Cloud requires an API key. Set OLLAMA_API_KEY or provide api_key parameter.")

    async def list_models(self) -> list[str]:
        """Fetch available cloud models from Ollama."""
        headers = self._get_headers()
        try:
            async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
                response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model.get("name", "") for model in data.get("models", []) if model.get("name")]
        except httpx.HTTPError as exc:
            raise ProviderError(f"Failed to list Ollama Cloud models: {exc}") from exc


class LMStudioProvider(_HTTPProvider):
    async def chat(self, messages: Iterable[dict[str, Any]], **kwargs: Any) -> str:
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
    def __init__(self, base_url: str, default_model: str | None = None, api_key: str | None = None) -> None:
        super().__init__(base_url, default_model)
        self.api_key = api_key

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        try:
            timeout = _get_timeout("JR_PROVIDER_TIMEOUT", DEFAULT_CLOUD_TIMEOUT)
            async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
                response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(f"Cloud provider error {exc.response.status_code}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Cloud provider request failed ({type(exc).__name__}): {exc}") from exc

    async def chat(self, messages: Iterable[dict[str, Any]], **kwargs: Any) -> str:
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


class OpenRouterProvider(_HTTPProvider):
    """Provider for OpenRouter - unified API for 300+ cloud models.

    OpenRouter provides access to models from OpenAI, Anthropic, Google, Meta,
    Mistral, and many others through a single API endpoint.

    Environment variables:
    - OPENROUTER_API_KEY: API key for authentication
    """

    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: str | None = None,
        default_model: str | None = None,
    ) -> None:
        super().__init__(self.OPENROUTER_BASE_URL, default_model or "openai/gpt-4o-mini")
        self.api_key = resolve_provider_api_key("openrouter", self.OPENROUTER_BASE_URL, api_key)

    def _get_headers(self) -> dict[str, str]:
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        headers = self._get_headers()
        try:
            timeout = _get_timeout("JR_PROVIDER_TIMEOUT", DEFAULT_CLOUD_TIMEOUT)
            async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
                response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            raise ProviderError(f"OpenRouter error {exc.response.status_code}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenRouter request failed ({type(exc).__name__}): {exc}") from exc

    async def chat(self, messages: Iterable[dict[str, Any]], **kwargs: Any) -> str:
        model = kwargs.get("model") or self.default_model
        payload = {
            "model": model,
            "messages": list(messages),
        }
        if kwargs.get("temperature") is not None:
            payload["temperature"] = kwargs["temperature"]
        if kwargs.get("max_tokens") is not None:
            payload["max_tokens"] = kwargs["max_tokens"]

        data = await self._post("/chat/completions", payload)
        choices = data.get("choices") or []
        return choices[0]["message"]["content"] if choices else ""

    async def chat_stream(self, messages: Iterable[dict[str, Any]], **kwargs: Any) -> AsyncIterator[str]:
        model = kwargs.get("model") or self.default_model
        payload = {
            "model": model,
            "messages": list(messages),
            "stream": True,
        }
        url = f"{self.base_url}/chat/completions"
        headers = self._get_headers()
        try:
            timeout = _get_timeout("JR_PROVIDER_STREAM_TIMEOUT", DEFAULT_STREAM_TIMEOUT)
            async with httpx.AsyncClient(timeout=timeout, headers=headers) as client, client.stream(
                "POST", url, json=payload
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]  # Remove "data: " prefix
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
        except httpx.HTTPStatusError as exc:
            raise ProviderError(f"OpenRouter error {exc.response.status_code}: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenRouter request failed ({type(exc).__name__}): {exc}") from exc

    async def complete(self, prompt: str, **kwargs: Any) -> str:
        messages = [{"role": "user", "content": prompt}]
        return await self.chat(messages, **kwargs)

    async def list_models(self) -> list[dict[str, Any]]:
        """Fetch available models from OpenRouter."""
        headers = self._get_headers()
        try:
            async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
                response = await client.get(f"{self.base_url}/models")
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        except httpx.HTTPError as exc:
            raise ProviderError(f"Failed to list OpenRouter models: {exc}") from exc


@dataclass
class ProviderFactory:
    """Build provider clients based on `ProviderConfig`."""

    api_key: str | None = None

    def build(self, cfg: ProviderConfig) -> LLMProvider:
        name = (cfg.name or "").lower()
        base_url = str(cfg.base_url).lower()
        resolved_key = resolve_provider_api_key(cfg.name, str(cfg.base_url), cfg.api_key, self.api_key)

        # Check for Ollama Cloud (ollama.com URL or explicit "ollama cloud" name)
        if "ollama.com" in base_url or "ollama_cloud" in name or "ollama cloud" in name:
            return OllamaCloudProvider(
                api_key=resolved_key,
                default_model=cfg.planner_model or cfg.generator_model,
            )
        # Local Ollama
        if "ollama" in name:
            return OllamaProvider(str(cfg.base_url), cfg.planner_model or cfg.generator_model, resolved_key)
        if "lm" in name or "studio" in name:
            return LMStudioProvider(str(cfg.base_url), cfg.generator_model)
        if "openrouter" in name:
            return OpenRouterProvider(
                api_key=resolved_key,
                default_model=cfg.generator_model,
            )
        return CloudProvider(str(cfg.base_url), cfg.generator_model, api_key=resolved_key)

    def get_default_provider(self) -> LLMProvider | None:
        """Get a default local provider for background tasks like GraphRAG.

        Probes Ollama and LM Studio endpoints synchronously.
        Returns None if no local provider is available.
        """
        import httpx

        # Try Ollama first
        ollama_url = os.environ.get("JR_OLLAMA_URL", "http://localhost:11434")
        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(f"{ollama_url}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                if models:
                    return OllamaProvider(ollama_url, models[0])
        except (httpx.HTTPError, httpx.RequestError):
            pass

        # Try LM Studio
        lmstudio_url = os.environ.get("JR_LMSTUDIO_URL", "http://localhost:1234")
        try:
            with httpx.Client(timeout=2.0) as client:
                resp = client.get(f"{lmstudio_url}/v1/models")
                resp.raise_for_status()
                data = resp.json()
                models = [m.get("id") for m in data.get("data", []) if m.get("id")]
                if models:
                    return LMStudioProvider(lmstudio_url, models[0])
        except (httpx.HTTPError, httpx.RequestError):
            pass

        return None


async def discover_models(cfg: ProviderConfig) -> list[str]:
    """Fetch available model names for a provider."""

    base = str(cfg.base_url).rstrip("/")
    base_lower = base.lower()
    api_key = resolve_provider_api_key(cfg.name, base, cfg.api_key)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None

    def _models_url(base_url: str) -> str:
        if base_url.endswith("/v1"):
            return f"{base_url}/models"
        return f"{base_url}/v1/models"

    async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
        kind = (cfg.name or "").lower()
        try:
            if "ollama" in kind:
                resp = await client.get(f"{base}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                return [model.get("name", "") for model in data.get("models", []) if model.get("name")]
            if "openrouter" in kind or "openrouter.ai" in base_lower:
                if not api_key:
                    raise ProviderError("OpenRouter requires an API key.")
                resp = await client.get(_models_url(base))
                resp.raise_for_status()
                data = resp.json()
                return [item.get("id", "") for item in data.get("data", []) if item.get("id")]
            if "lm" in kind or "studio" in kind:
                resp = await client.get(_models_url(base))
                resp.raise_for_status()
                data = resp.json()
                return [item.get("id", "") for item in data.get("data", []) if item.get("id")]
            # Fallback for OpenAI-compatible clouds
            resp = await client.get(_models_url(base))
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

        running: list[str] = []
        try:
            ps_resp = await client.get(f"{base}/api/ps")
            ps_resp.raise_for_status()
            running = [model.get("model", "") for model in ps_resp.json().get("models", []) if model.get("model")]
        except (httpx.HTTPError, httpx.RequestError):
            running = []

        version: str | None = None
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

        version: str | None = None
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


async def discover_local_providers() -> list[LocalProviderInfo]:
    probes = [
        (ProviderKind.OLLAMA, "Ollama", _DEFAULT_OLLAMA_URL, _probe_ollama),
        (ProviderKind.LM_STUDIO, "LM Studio", _DEFAULT_LMSTUDIO_URL, _probe_lmstudio),
    ]

    providers: list[LocalProviderInfo] = []
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
