"""Offline in-process binary-store regressions for shared evidence invariants."""

import asyncio
import threading

import numpy as np
import pytest

from app.core.bq_hybrid_retrieval import BQHybridRetrievalEngine
from app.core.bq_retrieval import BQRetrievalConfig, RetrievalModeV2
from app.core.cache import CacheManager
from app.core.chunking import ChunkingStrategy
from app.core.documents import Document, DocumentStore
from app.core.hybrid_retrieval import HybridConfig


class NumericalEmbedder:
    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 8), dtype=np.float32)


def binary_engine(tmp_path):
    store = DocumentStore(tmp_path / "documents.db")
    docs = [store.add(title="Approved launch", text="Orion launch date is October 11.")]
    engine = BQHybridRetrievalEngine(store, HybridConfig(
        embedding_model="", use_reranking=False, chunking_strategy=ChunkingStrategy.FIXED,
    ), BQRetrievalConfig(default_mode=RetrievalModeV2.BINARY, fallback_enabled=False), persist_path=None)
    engine._cache_manager = CacheManager()
    engine._embedder = NumericalEmbedder()
    engine.build()
    return engine, docs


@pytest.mark.asyncio
async def test_binary_pending_source_update_cannot_return_stale_evidence(tmp_path):
    engine, docs = binary_engine(tmp_path)
    assert await engine.query("orion launch")
    engine._docs.upsert(Document(docs[0].id, docs[0].title, "New source.", dict(docs[0].metadata)))
    assert await engine.query("orion launch") == []


@pytest.mark.asyncio
async def test_binary_source_edit_after_snapshot_is_detected(tmp_path, monkeypatch):
    engine, docs = binary_engine(tmp_path)
    original = engine._docs.list
    changed = False

    def edit_after_snapshot():
        nonlocal changed
        snapshot = original()
        if not changed:
            changed = True
            engine._docs.upsert(Document(docs[0].id, docs[0].title, "New source.", dict(docs[0].metadata)))
        return snapshot

    monkeypatch.setattr(engine._docs, "list", edit_after_snapshot)
    assert await engine.query("orion launch") == []


@pytest.mark.asyncio
async def test_binary_background_update_is_fail_closed_until_binary_store_is_ready(tmp_path, monkeypatch):
    engine, _ = binary_engine(tmp_path)
    started, release = threading.Event(), threading.Event()
    original = engine._build_bq_index

    def blocked_binary_update():
        started.set()
        assert release.wait(5)
        original()

    monkeypatch.setattr(engine, "_build_bq_index", blocked_binary_update)
    errors = []

    def rebuild():
        try:
            engine.build()
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=rebuild)
    worker.start()
    assert started.wait(3)
    try:
        assert await asyncio.wait_for(engine.query("orion launch"), 1) == []
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive() and not errors
    assert await engine.query("orion launch")


@pytest.mark.parametrize("top_k", [0, -1])
@pytest.mark.asyncio
async def test_binary_nonpositive_budget_returns_no_evidence(tmp_path, top_k):
    engine, _ = binary_engine(tmp_path)
    assert await engine.query("orion launch", top_k=top_k) == []


def test_binary_query_vector_cache_is_model_scoped(tmp_path):
    engine, _ = binary_engine(tmp_path)
    engine._config.embedding_model = "synthetic-a"
    assert len(engine._embed_query_cached("orion")) == 8

    class LargerEmbedder:
        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 16), dtype=np.float32)

    engine._config.embedding_model = "synthetic-b"
    engine._embedder = LargerEmbedder()
    assert len(engine._embed_query_cached("orion")) == 16


def test_binary_query_cache_pairs_key_with_captured_embedder(tmp_path, monkeypatch):
    engine, _ = binary_engine(tmp_path)
    original = engine._cache_manager.embeddings.get
    engine._config.embedding_model = "synthetic-a"

    class LargerEmbedder:
        def encode(self, texts, **kwargs):
            return np.ones((len(texts), 16), dtype=np.float32)

    def switch_model_during_cache_lookup(key):
        engine._config.embedding_model = "synthetic-b"
        engine._embedder = LargerEmbedder()
        return original(key)

    monkeypatch.setattr(engine._cache_manager.embeddings, "get", switch_model_during_cache_lookup)
    assert len(engine._embed_query_cached("orion")) == 8
    monkeypatch.setattr(engine._cache_manager.embeddings, "get", original)
    engine._config.embedding_model = "synthetic-a"
    engine._embedder = NumericalEmbedder()
    assert len(engine._embed_query_cached("orion")) == 8
