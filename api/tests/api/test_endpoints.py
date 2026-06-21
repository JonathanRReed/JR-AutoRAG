"""Integration tests for JR AutoRAG FastAPI app."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core import auth as auth_module
from app.core.auth import APIKeyAuth
from app.core.document_acl import DocumentACL, get_acl_store
from app.core.documents import Document
from app.core.golden_eval import AnswerMetrics, EvalRunResult, EvalRunStore, GoldenSetStore, RetrievalMetrics
from app.main import app
from app.routers import config as config_router
from app.routers import evaluation
from app.core.security_middleware import _resolve_required_scope, _resolve_route_timeout
from app.schemas.query import QueryResponse
from app.services import ServiceContainer, get_container
from app.state import get_orchestrator, set_orchestrator


@pytest.fixture()
def client():
    with tempfile.TemporaryDirectory() as tmpdir:
        container = ServiceContainer(base_path=Path(tmpdir))
        previous_golden_store = evaluation._golden_store
        previous_eval_run_store = evaluation._eval_run_store
        previous_evaluator = evaluation._evaluator
        evaluation._golden_store = GoldenSetStore(Path(tmpdir) / "golden_sets.json")
        evaluation._eval_run_store = EvalRunStore(Path(tmpdir) / "eval_runs.json")
        evaluation._evaluator = None

        def override_container() -> ServiceContainer:
            return container

        app.dependency_overrides[get_container] = override_container
        yield TestClient(app)
        app.dependency_overrides.clear()
        evaluation._golden_store = previous_golden_store
        evaluation._eval_run_store = previous_eval_run_store
        evaluation._evaluator = previous_evaluator


def test_sensitive_report_routes_require_admin_scope() -> None:
    assert _resolve_required_scope("/install/report", "GET") == "admin"
    assert _resolve_required_scope("/evaluation/runs/run-1/report", "GET") == "admin"
    assert _resolve_required_scope("/evaluation/runs", "GET") == "read"


def test_config_roundtrip(client: TestClient) -> None:
    resp = client.get("/config")
    assert resp.status_code == 200
    config = resp.json()
    config["profile"] = "Smoke"
    update = client.put("/config", json=config)
    assert update.status_code == 200
    assert update.json()["profile"] == "Smoke"


def test_config_rejects_zero_dense_k(client: TestClient) -> None:
    resp = client.get("/config")
    assert resp.status_code == 200
    config = resp.json()
    config["retrieval"]["dense_k"] = 0

    update = client.put("/config", json=config)

    assert update.status_code == 422
    detail = str(update.json()["detail"]).lower()
    assert "dense_k" in detail


def test_model_download_rejects_unconfigured_model(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloaded: list[str] = []
    monkeypatch.setattr(config_router, "_download_model", downloaded.append)

    resp = client.post(
        "/config/models/download",
        json={"kind": "embedding", "model": "bigscience/bloom"},
    )

    assert resp.status_code == 403
    assert downloaded == []
    assert "configured model" in resp.json()["detail"]


def test_model_download_allows_configured_model(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloaded: list[str] = []
    monkeypatch.setattr(config_router, "_download_model", downloaded.append)

    resp = client.post(
        "/config/models/download",
        json={"kind": "embedding", "model": "BAAI/bge-base-en-v1.5"},
    )

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model": "BAAI/bge-base-en-v1.5"}
    assert downloaded == ["BAAI/bge-base-en-v1.5"]


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


def test_config_rejects_remote_active_profile_in_local_only_mode(client: TestClient) -> None:
    resp = client.get("/config")
    assert resp.status_code == 200
    config = resp.json()
    config["deployment_profile"] = "local_only"
    config["provider"] = {
        "name": "Ollama",
        "base_url": "http://localhost:11434",
        "generator_model": "llama3",
    }
    config["provider_profiles"] = [
        {
            "name": "remote",
            "provider": {
                "name": "Attacker",
                "base_url": "https://attacker.example/v1",
                "generator_model": "remote-model",
            },
        }
    ]

    update = client.put("/config?active_profile=remote", json=config)
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


def test_readyz_uses_runtime_status_contract(client: TestClient) -> None:
    resp = client.get("/readyz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ready"] is True
    assert body["level"] in {"ready", "degraded"}
    assert body["checks"]["orchestrator"]["status"] == "ok"
    assert body["checks"]["document_store"]["details"] == {}
    assert body["checks"]["document_store"]["message"] is None
    assert body["checks"]["retrieval_index"]["status"] == "ok"
    assert body["checks"]["retrieval_index"]["details"] == {}


def test_readyz_returns_503_without_orchestrator(client: TestClient) -> None:
    previous = get_orchestrator()
    set_orchestrator(None)
    try:
        resp = client.get("/readyz")
    finally:
        if previous is not None:
            set_orchestrator(previous)

    assert resp.status_code == 503
    body = resp.json()
    assert body["ready"] is False
    assert body["level"] == "not_ready"
    assert body["checks"]["orchestrator"]["status"] == "fail"
    assert body["checks"]["orchestrator"]["message"] is None
    assert body["checks"]["orchestrator"]["details"] == {}


def test_security_posture_reports_local_install_defaults(client: TestClient) -> None:
    resp = client.get("/security/posture")
    assert resp.status_code == 200
    body = resp.json()
    assert body["level"] in {"local_only", "needs_attention", "client_ready"}
    checks = {item["id"]: item for item in body["checks"]}
    assert checks["auth"]["status"] in {"pass", "warn"}
    assert checks["exposure"]["status"] == "pass"
    assert checks["cors"]["status"] == "pass"
    assert checks["prompt_injection"]["status"] == "pass"
    assert body["settings"]["exposed_mode"] is False
    assert body["settings"]["auth_enabled"] is False
    assert "recommendations" in body


def test_install_report_collects_redacted_handoff_evidence(client: TestClient) -> None:
    resp = client.get("/install/report")
    assert resp.status_code == 200
    body = resp.json()
    assert body["schema_version"] == "install_report_v1"
    assert body["status"] in {"ready", "warn", "blocked"}
    assert body["readiness"]["level"] in {"ready", "degraded", "not_ready"}
    assert body["security"]["settings"]["api_keys_configured"] is False
    assert body["corpus"]["document_count"] == 0
    assert body["redaction"]["secrets"]
    assert "backends" not in body["policy"]
    assert "fallbacks" not in body["policy"]
    assert "backend_count" in body["policy"]
    evidence = {item["id"]: item for item in body["evidence"]}
    assert evidence["security_posture"]["endpoint"] == "/security/posture"
    assert "prompt-injection" in evidence["security_posture"]["detail"]
    assert evidence["readiness"]["endpoint"] == "/readyz"
    assert evidence["quality_receipt"]["status"] == "missing"
    assert evidence["client_readiness_benchmark"]["status"] == "missing"
    actions = {item["id"] for item in body["actions"]}
    assert "ingest-corpus" in actions
    assert "run-golden-eval" in actions
    assert "run-client-readiness-benchmark" in actions


def test_install_report_accepts_client_readiness_receipt(
    tmp_path: Path,
    client: TestClient,
) -> None:
    run_store = EvalRunStore(tmp_path / "eval_runs.json")
    result = EvalRunResult(
        run_id="client-ready-run",
        golden_set_name="client_readiness",
        timestamp=datetime.now(timezone.utc),
        retrieval_metrics=RetrievalMetrics(recall_at_k=1.0, mrr=1.0, ndcg=1.0, citation_coverage=1.0),
        answer_metrics=AnswerMetrics(faithfulness=1.0, completeness=1.0, refusal_accuracy=1.0, coherence=1.0),
        audit={
            "schema_version": "eval_run_audit_v1",
            "golden_set": {
                "name": "client_readiness",
                "case_count": 9,
                "tag_counts": {
                    "client-readiness": 9,
                    "mixed-format": 1,
                    "prompt-injection": 1,
                    "abstention": 1,
                    "binary-retrieval": 1,
                    "agentic-retrieval": 1,
                    "poisoned-document": 1,
                    "knowledge-extraction": 1,
                    "graph-retrieval": 1,
                },
            },
        },
    )
    run_store.save_run(result)

    previous_store = evaluation._eval_run_store
    evaluation._eval_run_store = run_store
    try:
        resp = client.get("/install/report")
    finally:
        evaluation._eval_run_store = previous_store

    assert resp.status_code == 200
    body = resp.json()
    evidence = {item["id"]: item for item in body["evidence"]}
    assert evidence["client_readiness_benchmark"]["status"] == "present"
    assert evidence["client_readiness_benchmark"]["sha256"]
    actions = {item["id"] for item in body["actions"]}
    assert "run-client-readiness-benchmark" not in actions


def test_install_report_warns_on_weak_client_readiness_metrics(
    tmp_path: Path,
    client: TestClient,
) -> None:
    run_store = EvalRunStore(tmp_path / "eval_runs.json")
    result = EvalRunResult(
        run_id="client-weak-run",
        golden_set_name="client_readiness",
        timestamp=datetime.now(timezone.utc),
        retrieval_metrics=RetrievalMetrics(recall_at_k=0.2, mrr=1.0, ndcg=1.0, citation_coverage=0.3),
        answer_metrics=AnswerMetrics(faithfulness=0.4, completeness=0.5, refusal_accuracy=1.0, coherence=1.0),
        audit={
            "schema_version": "eval_run_audit_v1",
            "golden_set": {
                "name": "client_readiness",
                "case_count": 9,
                "tag_counts": {
                    "client-readiness": 9,
                    "mixed-format": 1,
                    "prompt-injection": 1,
                    "abstention": 1,
                    "binary-retrieval": 1,
                    "agentic-retrieval": 1,
                    "poisoned-document": 1,
                    "knowledge-extraction": 1,
                    "graph-retrieval": 1,
                },
            },
        },
    )
    run_store.save_run(result)

    previous_store = evaluation._eval_run_store
    evaluation._eval_run_store = run_store
    try:
        resp = client.get("/install/report")
    finally:
        evaluation._eval_run_store = previous_store

    assert resp.status_code == 200
    body = resp.json()
    evidence = {item["id"]: item for item in body["evidence"]}
    assert evidence["client_readiness_benchmark"]["status"] == "warn"
    assert "failed metrics" in evidence["client_readiness_benchmark"]["detail"]
    actions = {item["id"] for item in body["actions"]}
    assert "run-client-readiness-benchmark" in actions


def test_install_report_fails_closed_when_exposed_without_auth(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    monkeypatch.setenv("AUTORAG_EXPOSE", "true")
    monkeypatch.setenv("AUTORAG_AUTH_ENABLED", "false")

    resp = client.get("/install/report")
    assert resp.status_code == 503
    assert "Refusing unauthenticated access" in resp.json()["detail"]


def test_exposed_mode_blocks_public_docs(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    monkeypatch.setenv("AUTORAG_EXPOSE", "true")
    resp = client.get("/docs")
    assert resp.status_code == 404
    assert "disabled" in resp.json()["detail"].lower()


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


def test_onboarding_scope_resolution_is_method_specific() -> None:
    assert _resolve_required_scope("/onboarding", "GET") == "read"
    assert _resolve_required_scope("/onboarding/demo/seed", "POST") == "write"
    assert _resolve_required_scope("/onboarding/demo", "DELETE") == "write"


def test_onboarding_demo_mutations_require_write_scope(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    auth = APIKeyAuth(enabled=True)
    read_key, _ = auth.generate_key("read-only", scopes=["read"])
    write_key, _ = auth.generate_key("writer", scopes=["write"])
    monkeypatch.setattr(auth_module, "_auth_instance", auth)

    read_headers = {"X-API-Key": read_key}
    write_headers = {"X-API-Key": write_key}

    onboarding = client.get("/onboarding", headers=read_headers)
    assert onboarding.status_code == 200

    seed_denied = client.post("/onboarding/demo/seed", headers=read_headers)
    assert seed_denied.status_code == 403

    seed_allowed = client.post("/onboarding/demo/seed", headers=write_headers)
    assert seed_allowed.status_code == 200
    assert seed_allowed.json()["seeded"]

    clear_denied = client.delete("/onboarding/demo", headers=read_headers)
    assert clear_denied.status_code == 403

    clear_allowed = client.delete("/onboarding/demo", headers=write_headers)
    assert clear_allowed.status_code == 200
    assert clear_allowed.json()["deleted"] > 0


def test_onboarding_demo_respects_document_write_acl(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    auth = APIKeyAuth(enabled=True)
    write_key, _ = auth.generate_key("writer", scopes=["write"])
    monkeypatch.setattr(auth_module, "_auth_instance", auth)
    get_acl_store().clear()

    container = app.dependency_overrides[get_container]()
    protected_doc = Document(
        id="protected-demo-title",
        title="JR AutoRAG Evaluation Brief",
        text="Private admin-owned content",
        metadata={"demo_corpus": "false"},
    )
    container.document_store.upsert(protected_doc)
    get_acl_store().set(DocumentACL.create_private(protected_doc.id, owner="admin-user"))

    response = client.post("/onboarding/demo/seed", headers={"X-API-Key": write_key})

    assert response.status_code == 403
    assert container.document_store.get(protected_doc.id).metadata["demo_corpus"] == "false"

    writer_id = hashlib.sha256(write_key.encode()).hexdigest()[:16]
    get_acl_store().set(DocumentACL.create_private(protected_doc.id, owner=writer_id))

    allowed = client.post("/onboarding/demo/seed", headers={"X-API-Key": write_key})

    assert allowed.status_code == 200
    assert container.document_store.get(protected_doc.id).metadata["demo_corpus"] == "true"


def test_onboarding_demo_clear_respects_document_write_acl(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    auth = APIKeyAuth(enabled=True)
    write_key, _ = auth.generate_key("writer", scopes=["write"])
    monkeypatch.setattr(auth_module, "_auth_instance", auth)
    get_acl_store().clear()

    container = app.dependency_overrides[get_container]()
    protected_doc = Document(
        id="protected-demo-doc",
        title="Protected Demo",
        text="Private demo-tagged content",
        metadata={"demo_corpus": "true"},
    )
    container.document_store.upsert(protected_doc)
    get_acl_store().set(DocumentACL.create_private(protected_doc.id, owner="admin-user"))

    response = client.delete("/onboarding/demo", headers={"X-API-Key": write_key})

    assert response.status_code == 403
    assert container.document_store.get(protected_doc.id) is not None

    writer_id = hashlib.sha256(write_key.encode()).hexdigest()[:16]
    get_acl_store().set(DocumentACL.create_private(protected_doc.id, owner=writer_id))

    allowed = client.delete("/onboarding/demo", headers={"X-API-Key": write_key})

    assert allowed.status_code == 200
    assert allowed.json()["deleted"] == 1
    assert container.document_store.get(protected_doc.id) is None


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


def test_demo_seed_supports_client_readiness_benchmark(client: TestClient) -> None:
    seed = client.post("/onboarding/demo/seed")
    assert seed.status_code == 200

    builtins = client.post("/evaluation/golden-sets/builtins")
    assert builtins.status_code == 200

    run = client.post("/evaluation/batch/client_readiness")
    assert run.status_code == 200
    run_body = run.json()
    assert run_body["retrieval_metrics"]["recall_at_k"] >= 0.70
    assert run_body["retrieval_metrics"]["citation_coverage"] >= 0.85
    assert run_body["answer_metrics"]["faithfulness"] >= 0.90
    assert run_body["answer_metrics"]["completeness"] >= 0.70

    report = client.get("/install/report")
    assert report.status_code == 200
    evidence = {item["id"]: item for item in report.json()["evidence"]}
    assert evidence["client_readiness_benchmark"]["status"] == "present"


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


def test_scoped_conversation_id_is_bound_to_principal() -> None:
    from app.routers.query import _scoped_conversation_id

    alice_key = _scoped_conversation_id("alice", "shared-session")
    bob_key = _scoped_conversation_id("bob", "shared-session")

    assert alice_key is not None
    assert bob_key is not None
    assert alice_key != bob_key
    assert "shared-session" not in alice_key
    assert _scoped_conversation_id("alice", None) is None
