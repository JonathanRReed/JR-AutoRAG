"""Integration tests for JR AutoRAG FastAPI app."""

from __future__ import annotations

from datetime import datetime, timezone
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.core.security_middleware import _resolve_route_timeout
from app.schemas.query import QueryResponse
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


def test_onboarding_demo_seed_query_and_clear(client: TestClient) -> None:
    initial = client.get("/onboarding")
    assert initial.status_code == 200
    initial_body = initial.json()
    assert initial_body["demo_seeded"] is False
    assert initial_body["sample_documents"], "expected demo sample documents"
    assert initial_body["example_queries"], "expected demo example queries"

    seed = client.post("/onboarding/demo/seed")
    assert seed.status_code == 200
    seed_body = seed.json()
    assert seed_body["seeded"], "expected at least one demo document to be seeded"
    assert seed_body["document_count"] >= len(seed_body["seeded"])

    docs = client.get("/documents")
    assert docs.status_code == 200
    demo_docs = [doc for doc in docs.json() if doc["metadata"].get("demo_corpus") == "true"]
    assert demo_docs
    assert {doc["metadata"].get("retention") for doc in demo_docs} == {"disposable"}

    query_resp = client.post(
        "/query",
        json={
            "question": "What should an evaluator notice first about JR AutoRAG?",
            "query_mode": "grounded",
            "conversation_id": "demo-test",
        },
    )
    assert query_resp.status_code == 200
    query_body = query_resp.json()
    assert query_body["chunks"], "demo query should retrieve evidence"
    assert query_body["metrics"]["conversation_id"] == "demo-test"

    clear = client.delete("/onboarding/demo")
    assert clear.status_code == 200
    assert clear.json()["deleted"] == len(demo_docs)

    after = client.get("/documents")
    assert after.status_code == 200
    assert not [doc for doc in after.json() if doc["metadata"].get("demo_corpus") == "true"]


def test_streaming_query_accepts_grounded_mode(client: TestClient) -> None:
    seed = client.post("/onboarding/demo/seed")
    assert seed.status_code == 200

    with client.stream(
        "POST",
        "/query/stream",
        json={
            "question": "Compare hybrid search and RAPTOR",
            "query_mode": "grounded",
            "conversation_id": "stream-test",
        },
    ) as response:
        assert response.status_code == 200
        body = "\n".join(response.iter_lines())

    assert "data:" in body
    assert '"type": "result"' in body
    assert "stream-test" in body


def test_query_mode_rejects_unknown_value(client: TestClient) -> None:
    resp = client.post("/query", json={"question": "What is JR AutoRAG?", "query_mode": "offline_web"})
    assert resp.status_code == 422


def test_query_response_accepts_datetime_step_fields() -> None:
    now = datetime.now(timezone.utc)
    response = QueryResponse(
        answer="ok",
        chunks=[],
        sources=[],
        trace_id="trace-1",
        metrics={},
        steps=[
            {
                "name": "generate",
                "duration_ms": 1.0,
                "started_at": now,
                "completed_at": now,
            }
        ],
    )

    dumped = response.model_dump(mode="json")
    assert isinstance(dumped["steps"][0]["started_at"], str)


def test_query_stream_uses_specific_timeout() -> None:
    assert _resolve_route_timeout("/query/stream") == 300
    assert _resolve_route_timeout("/query") == 300


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
