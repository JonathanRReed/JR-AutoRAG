"""Synthetic score contracts; no model downloads, user documents, or API calls."""

from dataclasses import replace

import numpy as np
import pytest

from app.core.cache import CacheManager
from app.core.chunking import ChunkingStrategy
from app.core.compression import ContextCompressor
from app.core.documents import DocumentStore
from app.core.evidence_contract import EvidenceContractResult
from app.core.gatherer import EvidenceChunk, Gatherer, aggregate_evidence
from app.core.hybrid_retrieval import HybridConfig, HybridRetrievalEngine
from app.core.orchestrator import Orchestrator
from app.core.persistence import DiskQueryCache, QueryCacheConfig
from app.core.planner import Planner
from app.core.providers import ProviderFactory
from app.core.retrieval_evaluator import EvaluationResult, RetrievalEvaluator, RetrievalVerdict
from app.core.smart_planner import PlanStep, RetrievalPlan, compute_marginal_gain
from app.core.telemetry import TelemetryStore
from app.schemas.config import AppConfig, RetrievalDefaults


@pytest.mark.parametrize("score", [0.02, 1.0, 10.0, -100.0, 1000.0])
def test_marginal_gain_is_novelty_fraction_independent_of_raw_score(score):
    old = [{"text": "Repeated evidence"}]
    new = [{"text": "Repeated evidence", "score": score}] * 9
    new.append({"text": "Unique evidence", "score": score})
    assert compute_marginal_gain(old, new) == pytest.approx(0.1)


@pytest.mark.parametrize("old,new,expected", [
    ([], [], 0.0),
    ([], [{"text": " "}, {}], 0.0),
    ([], [{"text": "alpha"}] * 3, 1 / 3),
    ([{"text": "ALPHA"}], [{"chunk_text": " alpha "}], 0.0),
    ([{"id": "a", "text": "original"}], [{"id": "a", "text": "trimmed"}], 0.0),
    ([{"text": "a" * 100 + "x"}], [{"text": "a" * 100 + "y"}], 1.0),
    ([{"text": "old"}], [{"text": "new"}] * 3 + [{"text": ""}], 1 / 3),
])
def test_marginal_gain_handles_empty_duplicate_and_full_content(old, new, expected):
    assert compute_marginal_gain(old, new) == pytest.approx(expected)


def test_heuristic_verdict_does_not_use_raw_retrieval_scores():
    evaluator = RetrievalEvaluator()
    results = [evaluator.evaluate_heuristic("Orion launch date", [
        EvidenceChunk("a", "A", "Orion launch.", score),
    ]) for score in [0.02, 1.0, 10.0, -100.0]]
    assert len({result.verdict for result in results}) == 1
    assert len({result.confidence for result in results}) == 1
    assert results[0].verdict == RetrievalVerdict.AMBIGUOUS
    assert results[0].confidence == pytest.approx(0.4 * (2 / 3) + 0.3)


def test_compression_uses_text_relevance_and_stable_evidence_order():
    compressor = ContextCompressor()
    orders = []
    for score in [0.02, 10.0, -100.0]:
        context = compressor.compress_extractive([
            EvidenceChunk("a", "A", "Orion launch is approved.", 0.01),
            EvidenceChunk("b", "B", "Orion launch is scheduled.", score),
            EvidenceChunk("c", "C", "Pottery conventions vary.", 10_000.0),
        ], "Orion launch")
        orders.append([citation["id"] for citation in context.citations])
    assert orders == [["a", "b", "c"]] * 3


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    cache = CacheManager()
    disk = DiskQueryCache(QueryCacheConfig(db_path=tmp_path / "answers.db"))
    monkeypatch.setattr("app.core.orchestrator.get_cache_manager", lambda: cache)
    monkeypatch.setattr("app.core.orchestrator.get_disk_query_cache", lambda: disk)
    store = DocumentStore(tmp_path / "documents.db")
    for name in ["alpha", "beta", "charlie"]:
        store.add(title=name, text=f"Orion launch {name} is approved.")
    engine = HybridRetrievalEngine(store, HybridConfig(
        embedding_model="", use_reranking=False, recency_weight=0.0,
        chunking_strategy=ChunkingStrategy.FIXED,
    ), persist_path=None)
    engine._cache_manager = CacheManager()
    engine.build()
    config = AppConfig(retrieval=RetrievalDefaults(
        embedding_model="", use_reranking=False, dense_k=2, sparse_k=2,
        compression=False, multi_resolution=False, enforce_evidence_contract=False,
        flare_generation=False,
    ))
    orchestrator = Orchestrator(Planner(config), engine, Gatherer(engine),
                              ProviderFactory(), TelemetryStore(tmp_path / "traces.json"))
    orchestrator.rebuild(config)
    plan = RetrievalPlan([PlanStep("Orion launch", 2, 2, 20, False)],
                         1600, 0.75, iterative=True, max_iterations=3, stop_threshold=0.1)
    monkeypatch.setattr(orchestrator._planner, "plan", lambda query: plan)
    original_budget = orchestrator._budget_planner.plan
    monkeypatch.setattr(orchestrator._budget_planner, "plan", lambda **kwargs: replace(
        original_budget(**kwargs), max_iterations=3, use_colbert=False, use_rerank=False,
        estimated_latency_ms=30_000,
    ))
    return orchestrator, engine


class RefiningEvaluator(RetrievalEvaluator):
    async def evaluate(self, query, chunks, provider=None):
        return EvaluationResult(RetrievalVerdict.LOW_COVERAGE, 0.6, "Synthetic refinement",
                                suggested_query="launch Orion")


@pytest.mark.parametrize("reranked", [False, True])
@pytest.mark.parametrize("scale,shift", [(1.0, 0.0), (100.0, 0.0), (1.0, -100.0)])
@pytest.mark.asyncio
async def test_actual_mixed_channel_iterations_keep_best_rank_and_raw_diagnostics(
    pipeline, monkeypatch, reranked, scale, shift,
):
    orchestrator, engine = pipeline
    monkeypatch.setattr(engine, "_dense_search", lambda *args: [])
    def sparse(query, *args):
        pairs = [(0, 3.0), (1, 2.0)] if query == "Orion launch" else [(2, 300.0), (0, 200.0)]
        return [(idx, score * scale + shift) for idx, score in pairs]
    monkeypatch.setattr(engine, "_sparse_search", sparse)
    monkeypatch.setattr(engine, "_literal_search", lambda query, *args: (
        ([(0, 9.0), (1, 8.0)], 2) if query == "Orion launch" else ([], 0)
    ))
    if reranked:
        class NumericalReranker:
            def predict(self, pairs):
                scores = {"alpha": 30.0, "beta": 20.0} if pairs[0][0] == "Orion launch" else {"charlie": 300.0, "alpha": 200.0}
                return np.array([scores[text.split()[2]] * scale + shift for _, text in pairs])
        engine._reranker = NumericalReranker()
        engine._config.use_reranking = True
        original_budget = orchestrator._budget_planner.plan
        monkeypatch.setattr(orchestrator._budget_planner, "plan", lambda **kwargs: replace(
            original_budget(**kwargs), use_rerank=True,
        ))
    first = await engine.query("Orion launch", top_k=2)
    second = await engine.query("launch Orion", top_k=2)
    assert first[0].retrieval_method == "hybrid"
    assert second[0].retrieval_method == "sparse"
    assert second[0].score == pytest.approx(300.0 * scale + shift)
    first_score = first[0].score
    orchestrator._retrieval_evaluator = RefiningEvaluator()
    result = await orchestrator.answer("Orion launch")
    assert [chunk["title"] for chunk in result["chunks"]][:3] == ["alpha", "charlie", "beta"]
    alpha = next(chunk for chunk in result["chunks"] if chunk["title"] == "alpha")
    assert alpha["score"] == pytest.approx(first_score)
    gatherer = next(step["details"] for step in result["steps"] if step["name"] == "gatherer")
    assert len(gatherer["iterations"]) == 3
    assert gatherer["iterations"][-1]["marginal_gain"] == 0.0
    assert gatherer["total_iterations"] == len(gatherer["iterations"])


def test_aggregation_preserves_independent_list_ranks_and_supplement_fallback():
    primary = [EvidenceChunk("a", "A", "alpha", 0.02, retrieval_rank=1),
               EvidenceChunk("b", "B", "beta", 0.01, retrieval_rank=2)]
    slot_fill = [EvidenceChunk("c", "C", "charlie", 500.0, retrieval_rank=1),
                 EvidenceChunk("a", "A", "alpha", 400.0, retrieval_rank=2)]
    clarification = [EvidenceChunk("b", "B", "beta clarified", -20.0, retrieval_rank=1)]
    supplements = [EvidenceChunk("graph", "Graph", "graph context", 1000.0),
                   EvidenceChunk("raptor", "RAPTOR", "summary context", 0.8),
                   EvidenceChunk("empty", "Empty", " ", 2000.0)]
    chunks = aggregate_evidence(primary + supplements + slot_fill + clarification * 3)
    assert [chunk.id for chunk in chunks] == ["a", "b", "c", "graph", "raptor"]
    assert chunks[0] is primary[0]
    assert chunks[1] is clarification[0]
    assert aggregate_evidence([]) == []


@pytest.mark.parametrize("strategy", ["simple", "extractive"])
def test_unranked_corrective_context_is_available_without_reserved_token_budget(strategy):
    ranked = EvidenceChunk("a", "A", "Orion launch is approved.", 0.02, retrieval_rank=1)
    graph = EvidenceChunk("graph", "Graph", "Orion launch is scheduled.", 1000.0)
    chunks = aggregate_evidence([graph, ranked])
    compressor = ContextCompressor()
    enough = compressor.compress(chunks, "Orion launch", max_tokens=20, strategy=strategy)
    limited = compressor.compress(chunks, "Orion launch", max_tokens=7, strategy=strategy)
    assert [citation["id"] for citation in enough.citations] == ["a", "graph"]
    # Documented limit: supplements have no reserved budget and can be omitted.
    assert [citation["id"] for citation in limited.citations] == ["a"]


@pytest.mark.asyncio
async def test_multi_resolution_context_cannot_reorder_ranked_evidence(pipeline, monkeypatch):
    orchestrator, engine = pipeline
    orchestrator._config.retrieval.multi_resolution = True
    monkeypatch.setattr(orchestrator, "_multi_resolution_expand", lambda *args: [
        EvidenceChunk("context", "Context", "Orion launch context.", 1000.0),
    ])
    result = await orchestrator.answer("Orion launch")
    assert result["chunks"][-1]["id"] == "context"
    assert result["chunks"][0]["id"] != "context"


@pytest.mark.asyncio
async def test_contract_retry_uses_same_aggregation_policy(pipeline, monkeypatch):
    orchestrator, engine = pipeline
    orchestrator._config.retrieval.enforce_evidence_contract = True
    monkeypatch.setattr(engine, "_dense_search", lambda *args: [])
    monkeypatch.setattr(engine, "_literal_search", lambda *args: ([], 0))
    monkeypatch.setattr(engine, "_sparse_search", lambda query, *args: (
        [(0, 0.02), (1, 0.01)] if query == "Orion launch" else [(2, 500.0), (0, 400.0)]
    ))
    checks = 0
    def verify(answer, chunks):
        nonlocal checks
        checks += 1
        return EvidenceContractResult([], [], 1.0, checks > 1, ["launch Orion"])
    monkeypatch.setattr(orchestrator._evidence_contract, "verify_answer", verify)
    result = await orchestrator.answer("Orion launch")
    assert checks == 2
    assert [chunk["title"] for chunk in result["chunks"]] == ["alpha", "charlie", "beta"]
    assert result["chunks"][0]["score"] == 0.02


@pytest.mark.asyncio
async def test_gatherer_and_aggregation_preserve_mmr_order(pipeline, monkeypatch):
    _, engine = pipeline
    engine._embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    monkeypatch.setattr(engine, "_dense_search", lambda *args: [(0, 0.9), (1, 0.8), (2, 0.7)])
    monkeypatch.setattr(engine, "_sparse_search", lambda *args: [])
    monkeypatch.setattr(engine, "_literal_search", lambda *args: ([], 0))
    evidence = await Gatherer(engine).gather("Orion launch", top_k=3, routing_params={"diversity": 0.8})
    assert [chunk.score for chunk in evidence.chunks] == [0.9, 0.7, 0.8]
    assert [chunk.retrieval_rank for chunk in evidence.chunks] == [1, 2, 3]
    assert aggregate_evidence(evidence.chunks) == evidence.chunks


@pytest.mark.parametrize("coverage_gap", [None, "query", "plan"])
@pytest.mark.parametrize("empty_second", [False, True])
@pytest.mark.asyncio
async def test_novelty_stop_requires_query_and_plan_coverage(pipeline, monkeypatch, coverage_gap, empty_second):
    orchestrator, engine = pipeline
    class CoverageEvaluator(RefiningEvaluator):
        def evaluate_coverage(self, query, chunks, **kwargs):
            return RetrievalVerdict.CORRECT, (0.0 if coverage_gap == "query" else 1.0), []
        def evaluate_plan_coverage(self, queries, chunks, **kwargs):
            return (0.0 if coverage_gap == "plan" else 1.0), []
    orchestrator._retrieval_evaluator = CoverageEvaluator()
    monkeypatch.setattr(engine, "_dense_search", lambda *args: [])
    monkeypatch.setattr(engine, "_literal_search", lambda *args: ([], 0))
    monkeypatch.setattr(engine, "_sparse_search", lambda query, *args: (
        [] if empty_second and query != "Orion launch" else [(0, 10.0)]
    ))
    result = await orchestrator.answer("Orion launch")
    details = next(step["details"] for step in result["steps"] if step["name"] == "gatherer")
    assert len(details["iterations"]) == (2 if coverage_gap is None else 3)
    assert details["iterations"][1]["marginal_gain"] == 0.0
    assert ("stopped_reason" in details["iterations"][1]) == (coverage_gap is None)


@pytest.mark.parametrize("threshold,iterations", [(0.1, 3), (0.11, 2)])
@pytest.mark.asyncio
async def test_novelty_threshold_remains_strict_in_orchestration(pipeline, monkeypatch, threshold, iterations):
    orchestrator, engine = pipeline
    plan = orchestrator._planner.plan("Orion launch")
    plan.stop_threshold = threshold
    plan.steps[0].dense_k = 10
    orchestrator._retrieval_evaluator = RefiningEvaluator()
    monkeypatch.setattr(engine, "_dense_search", lambda *args: [])
    monkeypatch.setattr(engine, "_literal_search", lambda *args: ([], 0))
    monkeypatch.setattr(engine, "_sparse_search", lambda query, *args: (
        [(0, 10.0)] if query == "Orion launch" else [(0, 10.0)] * 9 + [(2, 5.0)]
    ))
    result = await orchestrator.answer("Orion launch")
    details = next(step["details"] for step in result["steps"] if step["name"] == "gatherer")
    assert len(details["iterations"]) == iterations
    assert details["iterations"][1]["marginal_gain"] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_sparse_fallback_scores_remain_diagnostic_and_gatherer_records_order(tmp_path):
    store = DocumentStore(tmp_path / "fallback.db")
    store.add(title="Movement", text="go steadily up.")
    engine = HybridRetrievalEngine(store, HybridConfig(
        embedding_model="", use_reranking=False, recency_weight=0.0,
    ), persist_path=None)
    engine.build()
    engine._bm25 = None
    results = await engine.query("go up")
    assert [result.retrieval_method for result in results] == ["sparse"]
    assert [result.score for result in results] == [2.0]
    evidence = await Gatherer(engine).gather("go up", top_k=2)
    assert [getattr(chunk, "retrieval_rank", None) for chunk in evidence.chunks] == [1]
    assert evidence.chunks[0].score == 2.0
