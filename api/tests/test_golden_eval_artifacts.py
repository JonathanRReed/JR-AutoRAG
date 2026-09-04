from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.core import golden_eval as golden_eval_module
from app.core.golden_eval import (
    EvalRunStore,
    GoldenSetEvaluator,
    GoldenSetStore,
    GoldenTestCase,
    compute_refusal_accuracy,
)
from app.main import app
from app.routers import evaluation
from app.core.orchestrator import Orchestrator
from app.core.eval_gates import BUILTIN_DATASETS, install_builtin_datasets


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


class FakeNestedMetricsOrchestrator(FakeAuditOrchestrator):
    async def answer(self, question: str) -> dict:
        payload = await super().answer(question)
        payload["metrics"] = {
            "ragas": {
                "faithfulness": 0.91,
                "overall_score": 0.84,
            }
        }
        return payload


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
    assert result.audit["golden_set"]["tag_counts"] == {"b2b": 1, "smoke": 1}
    assert result.audit["golden_set"]["case_tags"]["case-a"] == ["smoke", "b2b"]
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
async def test_golden_eval_reads_nested_ragas_metrics(tmp_path: Path) -> None:
    golden_store = GoldenSetStore(tmp_path / "golden_sets.json")
    run_store = EvalRunStore(tmp_path / "eval_runs.json")
    golden_store.create_set(
        "nested_metrics",
        [
            GoldenTestCase(
                id="case-a",
                question="What is JR AutoRAG?",
                expected_source_ids=["doc-a"],
                expected_answer_points=["local evidence"],
            )
        ],
    )

    result = await GoldenSetEvaluator(
        golden_store=golden_store,
        run_store=run_store,
    ).run_batch(FakeNestedMetricsOrchestrator(), "nested_metrics")

    assert result.answer_metrics.faithfulness == pytest.approx(0.91)
    assert result.answer_metrics.coherence == pytest.approx(0.84)


@pytest.mark.asyncio
async def test_golden_eval_batches_case_tasks(tmp_path: Path) -> None:
    golden_store = GoldenSetStore(tmp_path / "golden_sets.json")
    run_store = EvalRunStore(tmp_path / "eval_runs.json")
    golden_store.create_set(
        "batching",
        [GoldenTestCase(id=f"case-{i}", question=f"Q{i}") for i in range(7)],
    )
    gather_batch_sizes: list[int] = []

    original_gather = golden_eval_module.asyncio.gather

    async def recording_gather(*aws, **kwargs):
        gather_batch_sizes.append(len(aws))
        return await original_gather(*aws, **kwargs)

    with patch.object(golden_eval_module.asyncio, "gather", recording_gather):
        result = await GoldenSetEvaluator(
            golden_store=golden_store,
            run_store=run_store,
        ).run_batch(FakeAuditOrchestrator(), "batching", max_concurrent=3)

    assert len(result.individual_results) == 7
    assert gather_batch_sizes == [3, 3, 1]


@pytest.mark.asyncio
async def test_eval_run_store_lists_report_artifact_metadata(tmp_path: Path) -> None:
    golden_store = GoldenSetStore(tmp_path / "golden_sets.json")
    run_store = EvalRunStore(tmp_path / "eval_runs.json")
    golden_store.create_set(
        "enterprise_smoke", [GoldenTestCase(id="case-a", question="Q")]
    )

    evaluator = GoldenSetEvaluator(golden_store=golden_store, run_store=run_store)
    result = await evaluator.run_batch(FakeAuditOrchestrator(), "enterprise_smoke")
    run_id = result.run_id
    [summary] = run_store.list_runs()
    assert summary["run_id"] == run_id
    assert "report_path" not in summary
    assert summary["report_sha256"]
    assert "corpus" not in summary["audit"]
    assert "config_snapshot" not in summary["audit"]

    [sensitive_summary] = run_store.list_runs(include_sensitive=True)
    assert sensitive_summary["report_path"]
    assert (
        sensitive_summary["audit"]["corpus"]["fingerprint"] == "corpus-test-fingerprint"
    )


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


def test_refusal_accuracy_scores_refusal_sensitive_cases() -> None:
    assert (
        compute_refusal_accuracy(
            "That secret is not available in the current corpus.", ["abstention"]
        )
        == 1.0
    )
    assert (
        compute_refusal_accuracy("The answer is 42.", ["knowledge-extraction"]) == 0.0
    )
    assert (
        compute_refusal_accuracy("Normal grounded answer.", ["client-readiness"]) == 1.0
    )


@pytest.mark.asyncio
async def test_eval_report_endpoint_returns_saved_artifact(tmp_path: Path) -> None:
    golden_store = GoldenSetStore(tmp_path / "golden_sets.json")
    run_store = EvalRunStore(tmp_path / "eval_runs.json")
    golden_store.create_set(
        "enterprise_smoke", [GoldenTestCase(id="case-a", question="Q")]
    )
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


def test_builtin_client_readiness_golden_set_can_be_installed(tmp_path: Path) -> None:
    previous_store = evaluation._golden_store
    evaluation._golden_store = GoldenSetStore(tmp_path / "golden_sets.json")
    try:
        client = TestClient(app)
        response = client.post("/evaluation/golden-sets/builtins")
    finally:
        evaluation._golden_store = previous_store

    assert response.status_code == 200
    payload = response.json()
    sets = {item["name"]: item["count"] for item in payload["sets"]}
    assert sets["client_readiness"] == 9
    assert sets["adversarial"] == 3


def test_builtin_client_readiness_golden_set_refreshes_stale_copy(
    tmp_path: Path,
) -> None:
    store = GoldenSetStore(tmp_path / "golden_sets.json")
    stale_cases = BUILTIN_DATASETS["client_readiness"][:6]
    store.create_set("client_readiness", stale_cases)

    installed = install_builtin_datasets(store)
    refreshed = store.get_set("client_readiness")

    assert installed >= 1
    assert len(refreshed) == 9
    tag_counts = {tag for case in refreshed for tag in case.tags}
    assert "poisoned-document" in tag_counts
    assert "knowledge-extraction" in tag_counts
    assert "graph-retrieval" in tag_counts


def test_default_eval_stores_use_jr_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("JR_DATA_DIR", str(tmp_path))

    golden_store = GoldenSetStore()
    run_store = EvalRunStore()

    assert golden_store._path == tmp_path / "golden_sets.json"
    assert run_store._path == tmp_path / "eval_runs.json"
