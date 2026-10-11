"""Optional binary-store failures must preserve coherent float evidence."""

import numpy as np
import pytest

from app.core.binary_vector_store import MilvusVectorStore
from app.core.bq_hybrid_retrieval import BQHybridRetrievalEngine
from app.core.bq_retrieval import BQRetrievalConfig, RetrievalModeV2
from app.core.cache import CacheManager
from app.core.chunking import ChunkingStrategy
from app.core.documents import DocumentStore
from app.core.hybrid_retrieval import HybridConfig


class NumericalEmbedder:
    def encode(self, texts, **kwargs):
        return np.ones((len(texts), 8), dtype=np.float32)


def make_binary_engine(tmp_path, text="Orion launch date is October 11.", strategy=ChunkingStrategy.FIXED):
    store = DocumentStore(tmp_path / "documents.db")
    doc = store.add(title="Approved launch", text=text)
    engine = BQHybridRetrievalEngine(store, HybridConfig(
        embedding_model="", use_reranking=False, chunking_strategy=strategy,
    ), BQRetrievalConfig(default_mode=RetrievalModeV2.BINARY, fallback_enabled=False),
        persist_path=str(tmp_path / "indexes"))
    engine._cache_manager = CacheManager()
    engine._embedder = NumericalEmbedder()
    engine.build()
    return engine, doc


STORE_FAILURES = [
    (operation, method)
    for operation in ("build", "load", "configuration", "query")
    for method in ("connect", "clear", "bulk_insert", "build_index")
] + [("incremental", method)
     for method in ("connect", "delete_by_doc_id", "bulk_insert", "build_index")]


@pytest.mark.parametrize("operation, method", STORE_FAILURES)
@pytest.mark.asyncio
async def test_optional_binary_failure_keeps_float_index_and_fallback(tmp_path, monkeypatch, operation, method):
    engine, doc = make_binary_engine(tmp_path)
    engine._bq_store = None
    engine._bq_initialized = False
    engine._bq_ready = False
    failures = []

    def fail_store_operation(*args, **kwargs):
        failures.append(method)
        raise RuntimeError(f"synthetic binary {method} failure")

    monkeypatch.setattr(MilvusVectorStore, method, fail_store_operation)
    raised = None
    try:
        if operation == "build":
            engine.build()
        elif operation == "load":
            assert engine.load_index()
        elif operation == "incremental":
            doc.text = "Orion launch date is October 12."
            engine._docs.upsert(doc)
            engine.index_documents([doc])
        elif operation == "configuration":
            engine.set_bq_config(engine._bq_config)
        else:
            await engine.query("orion launch")
    except RuntimeError as error:
        raised = error

    assert failures, "The injected optional-store failure was not exercised"
    assert engine.get_chunk_records(), "Healthy base chunks were discarded by optional binary failure"
    assert engine._bm25 is not None and engine._embeddings is not None
    assert engine._tokenized_corpus == [engine._tokenize(chunk.text) for _, chunk in engine.get_chunk_records()]
    assert raised is None
    assert not engine._bq_ready and engine._last_bq_error
    results = await engine.query("orion launch", routing_params={"retrieval_mode": "float32"})
    assert results and results[0].chunk_text == doc.text
    if method != "delete_by_doc_id":
        fallback = await engine.query("orion launch")
        assert fallback and fallback[0].retrieval_method != "binary"
        assert fallback[0].chunk_text == doc.text


@pytest.mark.parametrize("strategy", [ChunkingStrategy.FIXED, ChunkingStrategy.SEMANTIC])
@pytest.mark.asyncio
async def test_long_binary_chunk_preserves_exact_source_span(tmp_path, strategy):
    text = "Orion launch date is October 11. " + "Longterm " * 8000
    engine, doc = make_binary_engine(tmp_path, text=text, strategy=strategy)
    assert len(engine.get_chunk_records()[0][1].text) > 65535

    results = await engine.query("orion launch", top_k=1)

    assert results, "A valid oversized stored chunk must not be mistaken for a source edit"
    assert results[0].retrieval_method == "binary"
    assert doc.text[results[0].start_char:results[0].end_char] == results[0].chunk_text
    assert len(results[0].chunk_text) > 65535
    doc.text = doc.text[:-20] + "Changed source tail."
    engine._docs.upsert(doc)
    assert await engine.query("orion launch") == []


@pytest.mark.asyncio
async def test_failed_base_build_still_clears_partial_index(tmp_path, monkeypatch):
    from app.core.chunking import FixedChunker

    engine, _ = make_binary_engine(tmp_path)
    engine._docs.add(title="Other", text="Unrelated pottery notes.")

    class FailingChunker:
        calls = 0

        def chunk(self, text):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("synthetic base indexing failure")
            return FixedChunker().chunk(text)

    monkeypatch.setattr("app.core.hybrid_retrieval.get_chunker", lambda **kwargs: FailingChunker())
    with pytest.raises(RuntimeError, match="base indexing failure"):
        engine.build()
    assert engine.get_chunk_records() == []
    assert engine._bm25 is engine._embeddings is None
    assert not engine._bq_ready
