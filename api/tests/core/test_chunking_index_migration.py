"""Persisted pre-source-span chunks must be rebuilt on upgrade."""

import hashlib
import json
import time

import numpy as np

from app.core.chunking import Chunk, ChunkingStrategy
from app.core.documents import DocumentStore
from app.core.hybrid_retrieval import HybridConfig, HybridRetrievalEngine
from app.core.persistence import IndexMetadata, IndexPersistence


def test_old_chunking_index_is_rejected_without_loading_stale_spans(tmp_path):
    store = DocumentStore(tmp_path / "documents.db")
    doc = store.add(title="Orion", text="Orion launches October 11.", metadata={})
    config = HybridConfig(embedding_model="", use_reranking=False, chunking_strategy=ChunkingStrategy.FIXED)
    legacy_config = {
        "embedding_model": config.embedding_model,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "chunking_strategy": str(config.chunking_strategy),
    }
    legacy_config_hash = hashlib.sha256(json.dumps(legacy_config, sort_keys=True).encode()).hexdigest()[:16]
    corpus_version = hashlib.sha256(doc.id.encode()).hexdigest()[:16]
    path = tmp_path / "indexes"
    persistence = IndexPersistence(path)
    persistence.save_dense_index(
        "default", np.array([[1.0, 0.0]], dtype=np.float32),
        [(doc.id, Chunk(text=doc.text, index=0, start_char=2, end_char=len(doc.text) + 2))],
        IndexMetadata(corpus_version=corpus_version, config_hash=legacy_config_hash,
                      chunk_count=1, created_at=time.time(), model_name=""),
    )

    engine = HybridRetrievalEngine(store, config, persist_path=str(path))

    assert engine.load_index() is False
    assert engine.get_chunk_records() == []
    engine.build()
    assert all(doc.text[chunk.start_char:chunk.end_char] == chunk.text for _, chunk in engine.get_chunk_records())


def test_editing_source_without_changing_id_invalidates_persisted_index(tmp_path):
    store = DocumentStore(tmp_path / "documents.db")
    doc = store.add(title="Orion", text="Orion launches October 4.", metadata={})
    config = HybridConfig(embedding_model="", use_reranking=False, chunking_strategy=ChunkingStrategy.FIXED)
    path = tmp_path / "indexes"
    engine = HybridRetrievalEngine(store, config, persist_path=str(path))
    engine.build()
    engine._embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    assert engine.save_index()
    doc.text = "Orion launches October 11."
    store.upsert(doc)

    reopened = HybridRetrievalEngine(store, config, persist_path=str(path))
    assert reopened.load_index() is False


def test_persistence_fingerprint_describes_indexed_snapshot_not_later_store_edit(tmp_path):
    from app.core.documents import Document

    store = DocumentStore(tmp_path / "documents.db")
    doc = store.add(title="Schedule", text="Orion launch date is October 4.", metadata={})
    config = HybridConfig(embedding_model="", use_reranking=False, chunking_strategy=ChunkingStrategy.FIXED)
    path = tmp_path / "indexes"
    engine = HybridRetrievalEngine(store, config, persist_path=str(path))

    class NumericalEmbedder:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)

    engine._embedder = NumericalEmbedder()

    def edit_at_chunk_completion(stage, current, total):
        if stage == "chunking_documents" and current == total:
            store.upsert(Document(doc.id, doc.title, "New source.", dict(doc.metadata)))

    engine.build(on_progress=edit_at_chunk_completion)

    reopened = HybridRetrievalEngine(store, config, persist_path=str(path))
    assert reopened.load_index() is False
