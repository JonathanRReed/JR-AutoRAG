"""Real no-provider answer path: evidence, source identity, and abstention.

These checks exercise the deterministic context-summary fallback. They do not
measure generated-answer faithfulness or substitute for local-model evaluation.
"""

import pytest
import numpy as np

from app.core.bq_hybrid_retrieval import BQHybridRetrievalEngine
from app.core.bq_retrieval import BQRetrievalConfig, RetrievalModeV2
from app.core.cache import CacheManager
from app.core.chunking import ChunkingStrategy
from app.core.documents import DocumentStore
from app.core.gatherer import Gatherer
from app.core.hybrid_retrieval import HybridConfig, HybridRetrievalEngine
from app.core.orchestrator import Orchestrator
from app.core.persistence import DiskQueryCache, QueryCacheConfig
from app.core.planner import Planner
from app.core.providers import ProviderFactory
from app.core.telemetry import TelemetryStore
from app.schemas.config import AppConfig, RetrievalDefaults


def answer_pipeline(tmp_path, monkeypatch, documents):
    cache = CacheManager()
    disk_cache = DiskQueryCache(QueryCacheConfig(db_path=tmp_path / "query-cache.db"))
    monkeypatch.setattr("app.core.orchestrator.get_cache_manager", lambda: cache)
    monkeypatch.setattr("app.core.orchestrator.get_disk_query_cache", lambda: disk_cache)
    store = DocumentStore(tmp_path / "documents.db")
    docs = [store.add(title=title, text=text) for title, text in documents]
    engine = HybridRetrievalEngine(store, HybridConfig(
        embedding_model="", use_reranking=False, recency_weight=0.0,
        chunking_strategy=ChunkingStrategy.FIXED, rerank_top_k=5,
    ), persist_path=None)
    engine._cache_manager = CacheManager()
    engine.build()
    config = AppConfig(retrieval=RetrievalDefaults(
        embedding_model="", use_reranking=False, dense_k=1, sparse_k=1,
        rerank_pool=5, top_n=1, planner_mode="simple", multi_resolution=False,
        enforce_evidence_contract=False, flare_generation=False,
    ))
    orchestrator = Orchestrator(Planner(config), engine, Gatherer(engine),
                                ProviderFactory(), TelemetryStore(tmp_path / "traces.json"))
    orchestrator.rebuild(config)
    return orchestrator, docs


@pytest.mark.asyncio
async def test_selected_source_fact_and_citation_survive_stronger_distractors(tmp_path, monkeypatch):
    pipeline, docs = answer_pipeline(tmp_path, monkeypatch, [
        (f"Distractor {i}", "Orion launch date launch date launch date is October 4.")
        for i in range(25)
    ] + [("Approved launch", "Orion launch date is October 11.")])

    result = await pipeline.answer("What is the Orion launch date?", document_ids=[docs[-1].id])

    assert "October 11" in result["answer"]
    assert "October 4" not in result["answer"]
    assert "[1]" in result["answer"]
    assert result["sources"]
    assert all(source["id"].startswith(docs[-1].id + "-") for source in result["sources"])


@pytest.mark.asyncio
async def test_substring_only_answer_abstains_without_false_citation(tmp_path, monkeypatch):
    pipeline, _ = answer_pipeline(tmp_path, monkeypatch, [("Maps", "Cartography describes maps.")])

    result = await pipeline.answer("Where is art described?")

    assert result["sources"] == []
    assert result["grounding"]["no_evidence_response"]["found_evidence"] is False
    assert "Cartography" not in result["answer"]


@pytest.mark.asyncio
async def test_answer_built_before_update_is_not_cached_under_new_corpus_version(tmp_path, monkeypatch):
    pipeline, docs = answer_pipeline(tmp_path, monkeypatch, [
        ("Approved launch", "Orion launch date is October 4."),
    ])
    updated = False

    def update_during_generation(stage):
        nonlocal updated
        if stage == "generation" and not updated:
            updated = True
            docs[0].text = "Orion launch date is October 11."
            pipeline._retrieval._docs.upsert(docs[0])
            pipeline._retrieval.build()

    query = "What is the Orion launch date?"
    first = await pipeline.answer(query, on_stage=update_during_generation)
    second = await pipeline.answer(query)

    assert updated and "October 4" in first["answer"]
    assert "October 11" in second["answer"]
    assert "October 4" not in second["answer"]


@pytest.mark.asyncio
async def test_disk_answer_cache_includes_retrieval_configuration(tmp_path, monkeypatch):
    pipeline, _ = answer_pipeline(tmp_path, monkeypatch, [
        ("Approved launch", "Orion launch date is October 11."),
    ])
    query = "What is the Orion launch date?"
    await pipeline.answer(query)
    config = pipeline._config.model_copy(deep=True)
    config.retrieval.top_n = 2
    pipeline.rebuild(config)

    result = await pipeline.answer(query)

    assert not result.get("from_cache", False)
    repeated = await pipeline.answer(query)
    assert repeated.get("from_cache") is True


@pytest.mark.asyncio
async def test_source_update_during_answer_cache_lookup_drops_stale_hit(tmp_path, monkeypatch):
    import app.core.orchestrator as orchestrator_module

    pipeline, docs = answer_pipeline(tmp_path, monkeypatch, [
        ("Approved launch", "Orion launch date is October 4."),
    ])
    query = "What is the Orion launch date?"
    await pipeline.answer(query)
    orchestrator_module.get_cache_manager().queries.invalidate_all()
    disk = orchestrator_module.get_disk_query_cache()
    original = disk.get

    def update_after_lookup(*args, **kwargs):
        cached = original(*args, **kwargs)
        docs[0].text = "Orion launch date is October 11."
        pipeline._retrieval._docs.upsert(docs[0])
        pipeline._retrieval.build()
        return cached

    with monkeypatch.context() as scoped:
        scoped.setattr(disk, "get", update_after_lookup)
        result = await pipeline.answer(query)

    assert "October 11" in result["answer"]
    assert "October 4" not in result["answer"]
    assert not result.get("from_cache", False)


@pytest.mark.parametrize("embedding_dim", [None, 7])
@pytest.mark.asyncio
async def test_permanently_unavailable_binary_mode_keeps_answer_cache(tmp_path, monkeypatch, embedding_dim):
    pipeline, docs = answer_pipeline(tmp_path, monkeypatch, [
        ("Approved launch", "Orion launch date is October 11."),
    ])
    engine = BQHybridRetrievalEngine(pipeline._retrieval._docs, pipeline._retrieval._config,
        BQRetrievalConfig(default_mode=RetrievalModeV2.BINARY), persist_path=None)
    engine._cache_manager = CacheManager()
    if embedding_dim is not None:
        class UnalignedEmbedder:
            def encode(self, texts, **kwargs):
                return np.ones((len(texts), embedding_dim), dtype=np.float32)
        engine._embedder = UnalignedEmbedder()
    engine.build()
    assert not engine._bq_ready
    pipeline._retrieval = engine
    pipeline._gatherer = Gatherer(engine)
    pipeline.rebuild(pipeline._config)
    version = engine.get_corpus_version()
    query = "What is the Orion launch date?"

    first = await pipeline.answer(query, document_ids=[docs[0].id])
    repeated = await pipeline.answer(query, document_ids=[docs[0].id])

    assert "October 11" in first["answer"] and first["sources"]
    assert repeated.get("from_cache") is True
    assert "October 11" in repeated["answer"]
    assert engine.get_corpus_version() == version
    assert "cache_write_skipped" not in first["metrics"]
