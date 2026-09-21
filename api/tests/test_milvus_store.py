"""Tests for Milvus Vector Store module.

Tests cover:
- MilvusConfig dataclass
- MilvusChunk and MilvusSearchResult dataclasses
- IndexStats
- Store initialization and validation
- Mock-based search tests (without actual Milvus connection)
"""

import pytest

from app.core.binary_quantization import BQConfig
from app.core.binary_vector_store import (
    IndexStats,
    MilvusChunk,
    MilvusConfig,
    MilvusSearchResult,
    MilvusVectorStore,
    get_milvus_store,
    is_milvus_available,
)


class TestMilvusConfig:
    def test_default_config(self):
        config = MilvusConfig()
        assert config.host == "localhost"
        assert config.port == 19530
        assert config.collection_name == "jr_autorag_chunks_bq"
        assert config.index_type == "BIN_FLAT"
        assert config.metric_type == "HAMMING"

    def test_custom_config(self):
        config = MilvusConfig(
            host="milvus.example.com",
            port=19531,
            collection_name="custom_collection",
            index_type="BIN_IVF_FLAT",
            nlist=256,
        )
        assert config.host == "milvus.example.com"
        assert config.port == 19531
        assert config.index_type == "BIN_IVF_FLAT"
        assert config.nlist == 256

    def test_to_dict(self):
        config = MilvusConfig()
        d = config.to_dict()
        assert d["host"] == "localhost"
        assert d["port"] == 19530
        assert d["collection_name"] == "jr_autorag_chunks_bq"


class TestMilvusChunk:
    def test_basic_chunk(self):
        chunk = MilvusChunk(
            doc_id="doc1",
            chunk_id="doc1-0",
            source="/path/to/doc.pdf",
            text="This is the chunk text.",
            metadata={"page": 1},
        )
        assert chunk.doc_id == "doc1"
        assert chunk.chunk_id == "doc1-0"
        assert chunk.text == "This is the chunk text."
        assert chunk.metadata["page"] == 1

    def test_chunk_with_embedding(self):
        chunk = MilvusChunk(
            doc_id="doc1",
            chunk_id="doc1-0",
            source="test.txt",
            text="Test",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        )
        assert chunk.embedding is not None
        assert len(chunk.embedding) == 8


class TestMilvusSearchResult:
    def test_basic_result(self):
        result = MilvusSearchResult(
            id=1,
            doc_id="doc1",
            chunk_id="doc1-0",
            source="test.txt",
            text="Result text",
            metadata={},
            distance=100.0,
        )
        assert result.id == 1
        assert result.distance == 100.0

    def test_score_conversion(self):
        # Lower distance = higher score
        result1 = MilvusSearchResult(
            id=1,
            doc_id="d1",
            chunk_id="c1",
            source="",
            text="",
            metadata={},
            distance=0.0,
        )
        result2 = MilvusSearchResult(
            id=2,
            doc_id="d2",
            chunk_id="c2",
            source="",
            text="",
            metadata={},
            distance=100.0,
        )

        # Distance 0 -> score 1.0
        assert result1.score == 1.0

        # Distance 100 -> score ~0.0099
        assert result2.score < result1.score
        assert result2.score == pytest.approx(1.0 / 101.0)


class TestIndexStats:
    def test_stats_creation(self):
        stats = IndexStats(
            count=1000,
            index_type="BIN_FLAT",
            metric="HAMMING",
            dim_bits=768,
            storage_estimate_bytes=96000,
            collection_name="test_collection",
        )
        assert stats.count == 1000
        assert stats.dim_bits == 768


class TestMilvusVectorStore:
    def test_init_valid_dimension(self):
        store = MilvusVectorStore(embedding_dim=768)
        assert store._embedding_dim == 768
        assert store._binary_dim == 96

    def test_init_invalid_dimension_raises(self):
        with pytest.raises(ValueError, match="divisible by 8"):
            MilvusVectorStore(embedding_dim=769)

    def test_init_with_config(self):
        config = MilvusConfig(collection_name="custom")
        bq_config = BQConfig(normalize=True)
        store = MilvusVectorStore(
            config=config,
            embedding_dim=384,
            bq_config=bq_config,
        )
        assert store._config.collection_name == "custom"
        assert store._bq_config.normalize is True
        assert store._binary_dim == 48

    def test_validate_query_compatibility_valid(self):
        store = MilvusVectorStore(embedding_dim=768)
        store._embedding_version = "v1"

        assert store.validate_query_compatibility(768, "v1") is True

    def test_validate_query_compatibility_dim_mismatch(self):
        store = MilvusVectorStore(embedding_dim=768)

        with pytest.raises(ValueError, match="dimension"):
            store.validate_query_compatibility(384)

    def test_validate_query_compatibility_version_mismatch(self):
        store = MilvusVectorStore(embedding_dim=768)
        store._embedding_version = "v1"

        with pytest.raises(ValueError, match="version"):
            store.validate_query_compatibility(768, "v2")

    def test_get_stats_no_collection(self):
        store = MilvusVectorStore(embedding_dim=768)
        stats = store.get_stats()

        assert stats.count == 0
        assert stats.index_type == "BIN_FLAT"
        assert stats.metric == "HAMMING"


class TestGetMilvusStore:
    def test_factory_function(self):
        store = get_milvus_store(embedding_dim=768)
        assert isinstance(store, MilvusVectorStore)
        assert store._embedding_dim == 768


class TestIsMilvusAvailable:
    def test_availability_check(self):
        # Pure Python implementation - always available
        result = is_milvus_available()
        assert result is True


class TestBinaryVectorStoreSearch:
    """Integration tests for the pure Python binary vector store."""

    def test_insert_and_search(self):
        store = MilvusVectorStore(embedding_dim=8)
        store.connect()
        store.create_collection()

        # Insert chunks with embeddings
        chunks = [
            MilvusChunk(
                doc_id="doc1",
                chunk_id="doc1-0",
                source="test.txt",
                text="First chunk about cats",
                embedding=[1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
            ),
            MilvusChunk(
                doc_id="doc1",
                chunk_id="doc1-1",
                source="test.txt",
                text="Second chunk about dogs",
                embedding=[-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
            ),
            MilvusChunk(
                doc_id="doc2",
                chunk_id="doc2-0",
                source="test2.txt",
                text="Third chunk about cats and dogs",
                embedding=[1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
            ),
        ]

        ids = store.insert(chunks)
        assert len(ids) == 3
        store.build_index()

        # Search with query similar to first chunk
        query = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]
        results = store.search(query, top_k=2)

        assert len(results) == 2
        # First result should be exact match (distance 0)
        assert results[0].chunk_id == "doc1-0"
        assert results[0].distance == 0.0
        assert results[0].score == 1.0

    def test_search_with_filter(self):
        store = MilvusVectorStore(embedding_dim=8)
        store.connect()
        store.create_collection()

        chunks = [
            MilvusChunk(
                doc_id="doc1",
                chunk_id="doc1-0",
                source="test.txt",
                text="Doc1 chunk",
                embedding=[1.0] * 8,
            ),
            MilvusChunk(
                doc_id="doc2",
                chunk_id="doc2-0",
                source="test2.txt",
                text="Doc2 chunk",
                embedding=[1.0] * 8,
            ),
        ]

        store.insert(chunks)
        store.build_index()

        # Search with filter
        results = store.search([1.0] * 8, top_k=5, filter_expr='doc_id == "doc2"')

        assert len(results) == 1
        assert results[0].doc_id == "doc2"

    def test_search_binary_filters_before_distance_for_missing_doc(self, monkeypatch):
        store = MilvusVectorStore(embedding_dim=8)
        store.connect()
        store.create_collection()
        store.insert(
            [
                MilvusChunk(
                    doc_id="doc1",
                    chunk_id="c1",
                    source="",
                    text="",
                    embedding=[1.0] * 8,
                ),
            ]
        )
        store.build_index()

        def fail_if_called(*args, **kwargs):
            raise AssertionError(
                "distance calculation should be skipped when no documents match"
            )

        monkeypatch.setattr(store, "_hamming_distance_batch", fail_if_called)

        results = store.search_binary(b"\xff", top_k=5, document_ids=["missing"])

        assert results == []

    def test_search_binary_scans_matching_documents_once(self, monkeypatch):
        store = MilvusVectorStore(embedding_dim=8)
        store.connect()
        store.create_collection()
        store.insert(
            [
                MilvusChunk(
                    doc_id="doc1",
                    chunk_id="c1",
                    source="",
                    text="",
                    embedding=[1.0] * 8,
                ),
                MilvusChunk(
                    doc_id="doc2",
                    chunk_id="c2",
                    source="",
                    text="",
                    embedding=[-1.0] * 8,
                ),
                MilvusChunk(
                    doc_id="doc3",
                    chunk_id="c3",
                    source="",
                    text="",
                    embedding=[1.0] * 8,
                ),
            ]
        )
        store.build_index()

        calls = []
        original = store._hamming_distance_batch

        def count_call(query, vectors):
            calls.append(vectors.shape[0])
            return original(query, vectors)

        monkeypatch.setattr(store, "_hamming_distance_batch", count_call)

        results = store.search_binary(
            b"\xff", top_k=5, document_ids=["doc1", "doc2", "missing"]
        )

        assert len(results) == 2
        assert {result.doc_id for result in results} == {"doc1", "doc2"}
        assert calls == [2]

    def test_delete_by_doc_id(self):
        store = MilvusVectorStore(embedding_dim=8)
        store.connect()
        store.create_collection()

        chunks = [
            MilvusChunk(
                doc_id="doc1", chunk_id="c1", source="", text="", embedding=[1.0] * 8
            ),
            MilvusChunk(
                doc_id="doc1", chunk_id="c2", source="", text="", embedding=[1.0] * 8
            ),
            MilvusChunk(
                doc_id="doc2", chunk_id="c3", source="", text="", embedding=[1.0] * 8
            ),
        ]

        store.insert(chunks)
        assert store.count() == 3

        deleted = store.delete_by_doc_id("doc1")
        assert deleted == 2
        assert store.count() == 1

    def test_clear(self):
        store = MilvusVectorStore(embedding_dim=8)
        store.connect()
        store.create_collection()

        store.insert(
            [
                MilvusChunk(
                    doc_id="d1", chunk_id="c1", source="", text="", embedding=[1.0] * 8
                )
            ]
        )
        assert store.count() == 1

        store.clear()
        assert store.count() == 0


class TestMilvusStorePersistence:
    def test_save_and_load_persistence(self, tmp_path):
        persist_file = tmp_path / "store_index.json"
        config = MilvusConfig(persist_path=str(persist_file))

        store1 = MilvusVectorStore(config=config, embedding_dim=8)
        store1.connect()
        store1.create_collection()

        chunks = [
            MilvusChunk(
                doc_id="doc1",
                chunk_id="c1",
                source="src1",
                text="text 1",
                metadata={"key": "val1"},
                embedding=[1.0] * 8,
            ),
            MilvusChunk(
                doc_id="doc2",
                chunk_id="c2",
                source="src2",
                text="text 2",
                metadata={"key": "val2"},
                embedding=[-1.0] * 8,
            ),
        ]
        store1.insert(chunks)
        assert store1.count() == 2
        store1.disconnect()
        assert persist_file.exists()

        # Load store from persisted JSON file
        store2 = MilvusVectorStore(config=config, embedding_dim=8)
        store2.connect()
        assert store2.count() == 2

        results = store2.search(query_embedding=[1.0] * 8, top_k=2)
        assert len(results) == 2
        assert results[0].doc_id == "doc1"
        assert results[0].metadata == {"key": "val1"}

        # Confirm the file is valid JSON and not pickle
        import json

        with open(persist_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "chunks" in data
        assert len(data["chunks"]) == 2
        assert "bq_vector" in data["chunks"][0]
