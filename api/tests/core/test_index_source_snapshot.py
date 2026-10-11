"""A build must not publish chunks from a source snapshot that changed."""

import numpy as np
import pytest

from app.core.bq_hybrid_retrieval import BQHybridRetrievalEngine
from app.core.cache import CacheManager
from app.core.documents import DocumentStore
from app.core.hybrid_retrieval import HybridConfig, HybridRetrievalEngine


class MutatingEmbedder:
    def __init__(self):
        self.on_encode = None

    def encode(self, texts, **kwargs):
        callback, self.on_encode = self.on_encode, None
        if callback:
            callback()
        return np.ones((len(texts), 8), dtype=np.float32)


@pytest.mark.parametrize("engine_class", [HybridRetrievalEngine, BQHybridRetrievalEngine])
@pytest.mark.parametrize("persist", [False, True])
@pytest.mark.parametrize("incremental", [False, True])
@pytest.mark.parametrize("mutation", ["edit", "add", "delete"])
@pytest.mark.asyncio
async def test_changed_build_snapshot_is_discarded_and_next_query_recovers(
    tmp_path, engine_class, persist, incremental, mutation,
):
    store = DocumentStore(tmp_path / "documents.db")
    original = store.add(title="Orion", text="Orion launches Monday.")
    store.add(title="Stable", text="Stable deployment is approved.")
    engine = engine_class(store, HybridConfig(embedding_model="", use_reranking=False),
                          persist_path=str(tmp_path / "indexes") if persist else None)
    engine._cache_manager = CacheManager()
    embedder = MutatingEmbedder()
    engine._embedder = embedder
    engine.build()

    def change_source():
        if mutation == "edit":
            original.text = "Orion launches Friday."
            store.upsert(original)
        elif mutation == "add":
            store.add(title="New", text="New orbital evidence arrived.")
        else:
            store.delete(original.id)

    embedder.on_encode = change_source
    if incremental:
        original.text = "Orion launches Tuesday."
        store.upsert(original)
        engine.index_documents([original])
    else:
        engine.build()

    assert embedder.on_encode is None, "The source mutation must run during embedding"
    assert not engine.get_chunk_records(), "Stale in-memory chunks remained installed"
    assert engine._embeddings is None and engine._bm25 is None
    assert not engine._indexed_source_hashes
    if isinstance(engine, BQHybridRetrievalEngine):
        assert not engine._bq_ready

    results = await engine.query("deployment launches evidence", top_k=10)
    assert results
    assert engine._indexed_source_hashes == engine._source_hashes(store.list())
    assert {doc_id for doc_id, _ in engine.get_chunk_records()} == {doc.id for doc in store.list()}
    for result in results:
        current = store.get_by_title(result.document.title)
        assert current is not None
        assert result.chunk_text in current.text


@pytest.mark.parametrize("engine_class", [HybridRetrievalEngine, BQHybridRetrievalEngine])
@pytest.mark.parametrize("persist", [False, True])
@pytest.mark.asyncio
async def test_sparse_only_build_also_discards_changed_source(tmp_path, engine_class, persist):
    store = DocumentStore(tmp_path / "documents.db")
    doc = store.add(title="Orion", text="Orion launches Monday.")
    engine = engine_class(store, HybridConfig(embedding_model="", use_reranking=False),
                          persist_path=str(tmp_path / "indexes") if persist else None)
    engine._cache_manager = CacheManager()

    def edit_source(stage, current, total, message=None):
        if stage == "loading_documents":
            doc.text = "Orion launches Friday."
            store.upsert(doc)

    engine.build(on_progress=edit_source)
    assert engine._embedder is None
    assert not engine.get_chunk_records()
    results = await engine.query("orion launch")
    assert results and results[0].chunk_text == doc.text
