import numpy as np
import pytest
from app.core.cache import CacheManager
from app.core.documents import DocumentStore
from app.core.hybrid_retrieval import HybridConfig, HybridRetrievalEngine


def test_document_norms_are_reused_until_index_mutates(tmp_path, monkeypatch):
    engine = HybridRetrievalEngine(DocumentStore(tmp_path / "docs.db"),
        HybridConfig(embedding_model="", use_reranking=False), persist_path=None)
    engine._cache_manager = CacheManager()
    class Embedder:
        def encode(self, texts, **kwargs):
            return np.array([[1., 0.]])
    engine._embedder = Embedder()
    engine._embeddings = np.array([[1., 0.], [0., 1.], [1., 1.]])
    calls = []
    original = np.linalg.norm
    def norm(array, *args, **kwargs):
        if np.asarray(array).ndim == 2:
            calls.append(True)
        return original(array, *args, **kwargs)
    monkeypatch.setattr(np.linalg, "norm", norm)
    first = engine._dense_search("one", 2)
    assert engine._dense_search("two", 2) == first
    assert [row[0] for row in first] == [0, 2]
    assert len(calls) == 1
    with engine._mutating_index():
        engine._embeddings = np.array([[0., 1.], [1., 0.], [1., 1.]])
    assert engine._dense_search("three", 2)[0][0] == 1
    assert len(calls) == 2


@pytest.mark.parametrize("broken_rows", [None, np.empty((0, 2))])
def test_incremental_recovery_does_not_misalign_dense_rows(tmp_path, monkeypatch, broken_rows):
    from app.core.documents import Document
    engine = HybridRetrievalEngine(DocumentStore(tmp_path / "docs.db"),
        HybridConfig(embedding_model="", use_reranking=False), persist_path=None)
    monkeypatch.setattr(engine, "_init_models", lambda: None)
    class Embedder:
        def encode(self, texts, **kwargs):
            return np.array([[1., 0.] if "Existing" in text else [0., 1.] for text in texts])
    engine._embedder = Embedder()
    old = Document("old", "Old", "Existing document content.", {})
    engine._docs.upsert(old)
    engine.index_documents([old])
    assert len(engine._chunks) == 1
    engine._embeddings = broken_rows  # Previous failure left an incomplete dense index.
    new = Document("new", "New", "New document content.", {})
    engine._docs.upsert(new)
    engine.index_documents([new])
    assert len(engine._chunks) == 2
    assert engine._embeddings is not None
    assert len(engine._embeddings) == len(engine._chunks)

    assert np.array_equal(engine._embeddings, [[1., 0.], [0., 1.]])
