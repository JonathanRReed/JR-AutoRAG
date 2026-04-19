"""Integration tests for JR AutoRAG FastAPI app."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import ServiceContainer, get_container


@pytest.fixture()
def client():
    with tempfile.TemporaryDirectory() as tmpdir:
        container = ServiceContainer(base_path=Path(tmpdir))

        def override_container() -> ServiceContainer:
            return container

        app.dependency_overrides[get_container] = override_container
        yield TestClient(app)
        app.dependency_overrides.clear()


def test_config_roundtrip(client: TestClient) -> None:
    resp = client.get("/config")
    assert resp.status_code == 200
    config = resp.json()
    config["profile"] = "Smoke"
    update = client.put("/config", json=config)
    assert update.status_code == 200
    assert update.json()["profile"] == "Smoke"


def test_config_rejects_cloud_provider_in_local_only_mode(client: TestClient) -> None:
    resp = client.get("/config")
    assert resp.status_code == 200
    config = resp.json()
    config["deployment_profile"] = "local_only"
    config["provider"] = {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "generator_model": "openai/gpt-4o-mini",
    }

    update = client.put("/config", json=config)
    assert update.status_code in {400, 422}
    detail = update.json()["detail"]
    if isinstance(detail, list):
        detail = " ".join(str(item) for item in detail)
    assert "local-only" in str(detail).lower()


def test_policy_endpoint_exposes_client_data_policy(client: TestClient) -> None:
    resp = client.get("/config/policy")
    assert resp.status_code == 200
    body = resp.json()
    assert body["deployment_profile"] == "local_only"
    assert body["data_policy"]["classification"] == "client_confidential"
    assert body["data_policy"]["managed_cloud_hosting_allowed"] is False
    assert body["data_policy"]["external_model_calls_allowed"] is False
    assert body["guardrails"]["pii_redaction_required"] is True


def test_client_safe_profile_accepts_private_provider(client: TestClient) -> None:
    resp = client.get("/config")
    assert resp.status_code == 200
    config = resp.json()
    config["deployment_profile"] = "client_safe"
    config["provider"] = {
        "name": "Client Ollama",
        "base_url": "http://10.0.0.5:11434",
        "generator_model": "llama3.1",
    }

    update = client.put("/config", json=config)
    assert update.status_code == 200
    body = update.json()
    assert body["deployment_profile"] == "client_safe"
    assert body["data_policy"]["storage_boundary"] == "client_owned"


def test_hybrid_profile_allows_longer_retention(client: TestClient) -> None:
    resp = client.get("/config")
    assert resp.status_code == 200
    config = resp.json()
    config["deployment_profile"] = "hybrid"
    config["data_policy"]["document_retention_days"] = 365
    config["data_policy"]["trace_retention_days"] = 180

    update = client.put("/config", json=config)
    assert update.status_code == 200
    body = update.json()
    assert body["deployment_profile"] == "hybrid"
    assert body["data_policy"]["document_retention_days"] == 365


def test_client_safe_profile_rejects_long_retention(client: TestClient) -> None:
    resp = client.get("/config")
    assert resp.status_code == 200
    config = resp.json()
    config["deployment_profile"] = "client_safe"
    config["data_policy"]["document_retention_days"] = 365

    update = client.put("/config", json=config)
    assert update.status_code in {400, 422}
    detail = update.json()["detail"]
    if isinstance(detail, list):
        detail = " ".join(str(item) for item in detail)
    assert "client-safe retention" in str(detail).lower()


def test_client_safe_profile_rejects_public_cloud_provider(client: TestClient) -> None:
    resp = client.get("/config")
    assert resp.status_code == 200
    config = resp.json()
    config["deployment_profile"] = "client_safe"
    config["provider"] = {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "generator_model": "openai/gpt-4o-mini",
    }

    update = client.put("/config", json=config)
    assert update.status_code in {400, 422}
    detail = update.json()["detail"]
    if isinstance(detail, list):
        detail = " ".join(str(item) for item in detail)
    assert "client-safe" in str(detail).lower()


def test_client_safe_profile_rejects_cloud_backend(client: TestClient) -> None:
    resp = client.get("/config")
    assert resp.status_code == 200
    config = resp.json()
    config["deployment_profile"] = "client_safe"
    config["backends"]["llm"] = {
        "subsystem": "llm",
        "backend_id": "llm.cloud.openrouter",
        "label": "OpenRouter",
        "capabilities": {
            "mode": "cloud",
            "requires_network": True,
        },
    }

    update = client.put("/config", json=config)
    assert update.status_code in {400, 422}
    detail = update.json()["detail"]
    if isinstance(detail, list):
        detail = " ".join(str(item) for item in detail)
    assert "client-safe" in str(detail).lower()


def test_document_ingest_query_and_evaluation(client: TestClient) -> None:
    payload = {
        "title": "Intro",
        "text": "JR AutoRAG lets admins build RAG pipelines.",
        "sync": True,
        "langextract_profile_override": "generic_entities_v1",
        "langextract_prompt_override": "Extract factual entities only.",
    }
    ingest = client.post("/documents/text", json=payload)
    assert ingest.status_code == 200
    data = ingest.json()
    assert data["chunk_count"] >= 1

    docs = client.get("/documents")
    assert docs.status_code == 200
    assert len(docs.json()) == 1
    assert docs.json()[0]["metadata"]["langextract_status"] == "disabled"

    question = {"question": "What is JR AutoRAG?", "conversation_id": "smoke-session"}
    query_resp = client.post("/query", json=question)
    assert query_resp.status_code == 200
    query_data = query_resp.json()
    assert "answer" in query_data
    assert query_data["chunks"], "expected evidence chunks in response"
    assert query_data["metrics"]["conversation_id"] == "smoke-session"

    eval_payload = {"name": "SmokeTest", "questions": ["What is JR AutoRAG?"]}
    eval_resp = client.post("/evaluation", json=eval_payload)
    assert eval_resp.status_code == 200
    eval_data = eval_resp.json()
    assert eval_data["responses"], "evaluation should include responses"
    assert eval_data["average_coverage"] >= 0


def test_upload_accepts_langextract_override_fields(client: TestClient) -> None:
    upload = client.post(
        "/documents/upload",
        data={
            "title": "Upload Intro",
            "sync": "true",
            "ocr_policy": "dedicated_ocr",
            "langextract_profile_override": "contract_terms_v1",
            "langextract_prompt_override": "Extract obligations and deadlines.",
        },
        files={"file": ("intro.txt", b"Upload text for ingestion.", "text/plain")},
    )
    assert upload.status_code == 200
    body = upload.json()
    assert body["chunk_count"] >= 1

    docs = client.get("/documents")
    assert docs.status_code == 200
    assert docs.json()[0]["metadata"]["ocr_policy"] == "dedicated_ocr"
