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

    question = {"question": "What is JR AutoRAG?"}
    query_resp = client.post("/query", json=question)
    assert query_resp.status_code == 200
    query_data = query_resp.json()
    assert "answer" in query_data
    assert query_data["chunks"], "expected evidence chunks in response"

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
            "langextract_profile_override": "contract_terms_v1",
            "langextract_prompt_override": "Extract obligations and deadlines.",
        },
        files={"file": ("intro.txt", b"Upload text for ingestion.", "text/plain")},
    )
    assert upload.status_code == 200
    body = upload.json()
    assert body["chunk_count"] >= 1
