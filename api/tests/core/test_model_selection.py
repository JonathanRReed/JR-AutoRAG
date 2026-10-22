import pytest
import respx
from app.core.providers import OllamaProvider, ProviderFactory


@respx.mock
def test_background_chat_never_chooses_embedding_model(monkeypatch):
    monkeypatch.delenv("JR_OLLAMA_URL", raising=False)
    respx.get("http://localhost:11434/api/tags").respond(200, json={"models": [
        {"name": "bge-m3:latest"}, {"name": "all-minilm:latest"}, {"name": "qwen3-embedding:4b"}, {"name": "qwen3.5:4b"}, {"name": "llama3:latest"}]})
    assert ProviderFactory().get_default_provider().default_model == "qwen3.5:4b"


@pytest.mark.asyncio
async def test_ollama_preserves_generation_controls():
    provider = OllamaProvider("http://localhost:11434", "qwen3.5:4b")
    payloads = []
    async def post(path, payload):
        payloads.append(payload)
        return {"message": {"content": "ok"}}
    provider._post = post
    assert await provider.chat([{"role":"user", "content":"test"}], temperature=0.0,
                               num_predict=128, think=False, format="json") == "ok"
    assert payloads[0]["options"] == {"temperature": 0.0, "num_predict": 128}
    assert payloads[0]["think"] is False
    assert payloads[0]["format"] == "json"


@pytest.mark.asyncio
async def test_ollama_completion_preserves_generation_controls():
    provider = OllamaProvider("http://localhost:11434", "qwen3.5:4b")
    payloads = []

    async def post(path, payload):
        assert path == "/api/generate"
        payloads.append(payload)
        return {"response": "ok"}

    provider._post = post
    assert await provider.complete("test", temperature=0.0, num_predict=128,
                                   think=False, format="json", unknown="ignored") == "ok"
    assert payloads[0]["options"] == {"temperature": 0.0, "num_predict": 128}
    assert payloads[0]["think"] is False
    assert payloads[0]["format"] == "json"
    assert "unknown" not in payloads[0]
