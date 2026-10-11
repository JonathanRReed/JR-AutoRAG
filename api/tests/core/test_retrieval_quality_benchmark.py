"""Keep the regression benchmark's metrics honest and reproducible."""

import importlib.util
import math
from pathlib import Path

import pytest


def benchmark_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "retrieval-quality-benchmark.py"
    assert path.is_file(), "The reproducible quality benchmark is missing"
    spec = importlib.util.spec_from_file_location("retrieval_quality_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_duplicate_evidence_does_not_inflate_recall_or_ndcg():
    metrics = benchmark_module().evidence_metrics(["a", "copy", "b"], [["a", "copy"], ["b"]], 3)
    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["ndcg_at_k"] == pytest.approx((1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3)))


def test_ndcg_ideal_is_cut_off_at_requested_k():
    metrics = benchmark_module().evidence_metrics(["a"], [["a"], ["b"]], 1)
    assert metrics["recall_at_k"] == 0.5
    assert metrics["ndcg_at_k"] == 1.0


def test_irrelevant_first_result_reduces_mrr():
    metrics = benchmark_module().evidence_metrics(["noise", "a"], [["a"]], 2)
    assert metrics["mrr"] == 0.5
    assert metrics["ndcg_at_k"] == pytest.approx(1 / math.log2(3))


@pytest.mark.parametrize("retrieved, expected", [([], 1.0), (["noise"], 0.0)])
def test_unanswerable_queries_have_a_separate_abstention_metric(retrieved, expected):
    metrics = benchmark_module().evidence_metrics(retrieved, [], 2)
    assert metrics["no_evidence_accuracy"] == expected
    assert metrics["recall_at_k"] is None


@pytest.mark.asyncio
async def test_benchmark_keeps_selected_source_and_valid_spans(tmp_path):
    module = benchmark_module()
    case = {
        "id": "scoped", "query": "Orion launch", "top_k": 1,
        "documents": [{"id": "selected", "title": "Selected", "text": "Orion launch date is October 11."}],
        "distractors": {"count": 25, "text": "Orion launch launch launch date is October 4."},
        "scope": ["selected"], "expected_evidence_groups": [["selected"]],
        "answer_points": ["October 11"], "config": {"rerank_top_k": 5},
    }
    result = await module.evaluate_case(case, tmp_path)
    assert result["retrieved_ids"] == ["selected"]
    assert result["recall_at_k"] == 1.0
    assert result["evidence_answer_point_coverage"] == 1.0
    assert result["source_span_fidelity"] == 1.0


@pytest.mark.parametrize("unsafe", [
    {"embedding_model": "remote-model"}, {"use_reranking": True},
    {"use_colbert": True}, {"graph": True},
])
@pytest.mark.asyncio
async def test_benchmark_rejects_model_enabled_fixture_controls(tmp_path, unsafe):
    module = benchmark_module()
    case = {"id": "unsafe", "query": "orion", "top_k": 1, "documents": [],
            "expected_evidence_groups": [], "answer_points": [], "config": unsafe}
    with pytest.raises(ValueError, match="offline"):
        await module.evaluate_case(case, tmp_path)
