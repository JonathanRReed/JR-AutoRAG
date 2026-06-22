import pytest
import respx
import httpx

from app.core.providers import ProviderFactory, OllamaProvider, LMStudioProvider


@pytest.fixture
def factory():
    return ProviderFactory()


@respx.mock
def test_get_default_provider_ollama_available(factory, monkeypatch):
    """Test when Ollama is available, it should return OllamaProvider."""
    # Ensure default environment variables are used
    monkeypatch.delenv("JR_OLLAMA_URL", raising=False)
    monkeypatch.delenv("JR_LMSTUDIO_URL", raising=False)

    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": [{"name": "llama3:latest"}]})
    )

    provider = factory.get_default_provider()

    assert isinstance(provider, OllamaProvider)
    assert provider.base_url == "http://localhost:11434"
    assert provider.default_model == "llama3:latest"


@respx.mock
def test_get_default_provider_lmstudio_available(factory, monkeypatch):
    """Test when Ollama is unavailable but LM Studio is available, it returns LMStudioProvider."""
    monkeypatch.delenv("JR_OLLAMA_URL", raising=False)
    monkeypatch.delenv("JR_LMSTUDIO_URL", raising=False)

    # Mock Ollama to fail
    respx.get("http://localhost:11434/api/tags").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    # Mock LM Studio to succeed
    respx.get("http://localhost:1234/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "lmstudio-model-v1"}]})
    )

    provider = factory.get_default_provider()

    assert isinstance(provider, LMStudioProvider)
    assert provider.base_url == "http://localhost:1234"
    assert provider.default_model == "lmstudio-model-v1"


@respx.mock
def test_get_default_provider_both_unavailable(factory, monkeypatch):
    """Test when both providers are unavailable, it returns None."""
    monkeypatch.delenv("JR_OLLAMA_URL", raising=False)
    monkeypatch.delenv("JR_LMSTUDIO_URL", raising=False)

    respx.get("http://localhost:11434/api/tags").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )
    respx.get("http://localhost:1234/v1/models").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )

    provider = factory.get_default_provider()

    assert provider is None


@respx.mock
def test_get_default_provider_no_models(factory, monkeypatch):
    """Test when endpoints return valid JSON but no models are available, it skips to next or returns None."""
    monkeypatch.delenv("JR_OLLAMA_URL", raising=False)
    monkeypatch.delenv("JR_LMSTUDIO_URL", raising=False)

    # Ollama has no models
    respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(200, json={"models": []})
    )
    # LM Studio has no models
    respx.get("http://localhost:1234/v1/models").mock(
        return_value=httpx.Response(200, json={"data": []})
    )

    provider = factory.get_default_provider()

    assert provider is None


@respx.mock
def test_get_default_provider_custom_urls(factory, monkeypatch):
    """Test custom environment variables are respected for URLs."""
    custom_ollama = "http://custom-ollama:11434"
    custom_lmstudio = "http://custom-lmstudio:1234"

    monkeypatch.setenv("JR_OLLAMA_URL", custom_ollama)
    monkeypatch.setenv("JR_LMSTUDIO_URL", custom_lmstudio)

    # Mock Ollama to fail
    respx.get(f"{custom_ollama}/api/tags").mock(
        side_effect=httpx.ConnectError("Connection refused")
    )
    # Mock LM Studio to succeed
    respx.get(f"{custom_lmstudio}/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "custom-model"}]})
    )

    provider = factory.get_default_provider()

    assert isinstance(provider, LMStudioProvider)
    assert provider.base_url == custom_lmstudio
    assert provider.default_model == "custom-model"
