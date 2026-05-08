from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.golden_eval import (
    EvalRunStore,
    GoldenSetEvaluator,
    GoldenSetStore,
    GoldenTestCase,
)
from app.main import app
from app.routers import evaluation
from app.core.orchestrator import Orchestrator


class FakeAuditOrchestrator:
    async def answer(self, question: str) -> dict:
        return {
            "answer": f"JR AutoRAG answers {question} with local evidence [1].",
            "sources": [{"id": "doc-a"}],
            "trace_id": "trace-a",
            "metrics": {"faithfulness": 0.92, "coherence": 0.87},
        }

    def get_eval_audit_context(self) -> dict:
        return {
            "corpus": {
                "fingerprint": "corpus-test-fingerprint",
                "document_count": 1,
                "chunk_count": 3,
            },
            "runtime_profile": {
                "deployment_profile": "local_only",
                "embedding_model": "BAAI/bge-base-en-v1.5",
                "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            },
            "config_snapshot": {
                "retrieval": {
                    "hybrid": True,
                    "use_reranking": True,
                }
            },
        }


@pytest.mark.asyncio
async def test_golden_eval_run_writes_auditable_report(tmp_path: Path) -> None:
    golden_store = GoldenSetStore(tmp_path / "golden_sets.json")
    run_store = EvalRunStore(tmp_path / "eval_runs.json")
    golden_store.create_set(
        "enterprise_smoke",
        [
            GoldenTestCase(
                id="case-a",
                question="What is JR AutoRAG?",
                expected_source_ids=["doc-a"],
                expected_answer_points=["local evidence"],
                tags=["smoke", "b2b"],
            )
        ],
    )

    evaluator = GoldenSetEvaluator(golden_store=golden_store, run_store=run_store)
    result = await evaluator.run_batch(FakeAuditOrchestrator(), "enterprise_smoke")

    assert result.audit["schema_version"] == "eval_run_audit_v1"
    assert result.audit["corpus"]["fingerprint"] == "corpus-test-fingerprint"
    assert result.audit["runtime_profile"]["deployment_profile"] == "local_only"
    assert result.audit["golden_set"]["case_count"] == 1
    assert result.audit["golden_set"]["fingerprint"]
    assert result.report_path
    assert result.report_sha256

    report_path = Path(result.report_path)
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["run_id"] == result.run_id
    assert report["audit"]["corpus"]["fingerprint"] == "corpus-test-fingerprint"
    assert report["report_sha256"] == result.report_sha256

    persisted = run_store.get_run(result.run_id)
    assert persisted is not None
    assert persisted.audit["corpus"]["document_count"] == 1
    assert persisted.report_sha256 == result.report_sha256


@pytest.mark.asyncio
async def test_eval_run_store_lists_report_artifact_metadata(tmp_path: Path) -> None:
    golden_store = GoldenSetStore(tmp_path / "golden_sets.json")
    run_store = EvalRunStore(tmp_path / "eval_runs.json")
    golden_store.create_set("enterprise_smoke", [GoldenTestCase(id="case-a", question="Q")])

    evaluator = GoldenSetEvaluator(golden_store=golden_store, run_store=run_store)
    result = await evaluator.run_batch(FakeAuditOrchestrator(), "enterprise_smoke")
    run_id = result.run_id
    [summary] = run_store.list_runs()
    assert summary["run_id"] == run_id
    assert summary["report_path"]
    assert summary["report_sha256"]


def test_eval_config_redaction_catches_secret_shaped_keys() -> None:
    payload = {
        "provider": {
            "API_KEY": "sk-test",
            "SecretToken": "secret-token",
            "base_url": "http://127.0.0.1:11434",
        },
        "profiles": [{"password": "pw", "name": "local"}],
    }

    redacted = Orchestrator._redacted_config_snapshot(payload)

    assert redacted["provider"]["API_KEY"] == "[redacted]"
    assert redacted["provider"]["SecretToken"] == "[redacted]"
    assert redacted["provider"]["base_url"] == "http://127.0.0.1:11434"
    assert redacted["profiles"][0]["password"] == "[redacted]"


@pytest.mark.asyncio
async def test_eval_report_endpoint_returns_saved_artifact(tmp_path: Path) -> None:
    golden_store = GoldenSetStore(tmp_path / "golden_sets.json")
    run_store = EvalRunStore(tmp_path / "eval_runs.json")
    golden_store.create_set("enterprise_smoke", [GoldenTestCase(id="case-a", question="Q")])
    result = await GoldenSetEvaluator(
        golden_store=golden_store,
        run_store=run_store,
    ).run_batch(FakeAuditOrchestrator(), "enterprise_smoke")

    previous_store = evaluation._eval_run_store
    evaluation._eval_run_store = run_store
    try:
        client = TestClient(app)
        response = client.get(f"/evaluation/runs/{result.run_id}/report")
    finally:
        evaluation._eval_run_store = previous_store

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == result.run_id
    assert payload["report_sha256"] == result.report_sha256
    assert payload["audit"]["schema_version"] == "eval_run_audit_v1"
