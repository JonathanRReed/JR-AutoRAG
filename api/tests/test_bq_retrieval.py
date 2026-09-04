"""Tests for BQ Retrieval Service module.

Tests cover:
- RetrievalModeV2 enum
- RetrievalTimings and RetrievalDebug dataclasses
- RetrievedChunk dataclass
- BQRetrievalConfig
- BQRetrievalService initialization and configuration
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.core.bq_retrieval import (
    BQRetrievalConfig,
    BQRetrievalService,
    RetrievalDebug,
    RetrievalModeV2,
    RetrievalTimings,
    RetrievedChunk,
    get_bq_retrieval_service,
)


class TestRetrievalModeV2:
    def test_enum_values(self):
        assert RetrievalModeV2.FLOAT32.value == "float32"
        assert RetrievalModeV2.BINARY.value == "binary"

    def test_from_string_binary(self):
        assert RetrievalModeV2.from_string("binary") == RetrievalModeV2.BINARY
        assert RetrievalModeV2.from_string("bq") == RetrievalModeV2.BINARY
        assert RetrievalModeV2.from_string("hamming") == RetrievalModeV2.BINARY
        assert RetrievalModeV2.from_string("BINARY") == RetrievalModeV2.BINARY

    def test_from_string_float32(self):
        assert RetrievalModeV2.from_string("float32") == RetrievalModeV2.FLOAT32
        assert RetrievalModeV2.from_string("dense") == RetrievalModeV2.FLOAT32
        assert RetrievalModeV2.from_string("unknown") == RetrievalModeV2.FLOAT32


class TestRetrievalTimings:
    def test_default_timings(self):
        timings = RetrievalTimings()
        assert timings.t_embed_query_ms == 0.0
        assert timings.t_total_ms == 0.0

    def test_to_dict(self):
        timings = RetrievalTimings(
            t_embed_query_ms=10.5,
            t_milvus_search_ms=5.2,
            t_total_ms=20.0,
        )
        d = timings.to_dict()
        assert d["t_embed_query_ms"] == 10.5
        assert d["t_milvus_search_ms"] == 5.2
        assert d["t_total_ms"] == 20.0


class TestRetrievalDebug:
    def test_basic_debug(self):
        timings = RetrievalTimings(t_total_ms=15.0)
        debug = RetrievalDebug(
            mode="binary",
            top_k=5,
            candidates_searched=50,
            results_returned=5,
            timings=timings,
            distances=[10.0, 20.0, 30.0, 40.0, 50.0],
            chunk_ids=["c1", "c2", "c3", "c4", "c5"],
        )
        assert debug.mode == "binary"
        assert debug.top_k == 5
        assert len(debug.distances) == 5

    def test_to_dict(self):
        timings = RetrievalTimings()
        debug = RetrievalDebug(
            mode="float32",
            top_k=5,
            candidates_searched=5,
            results_returned=3,
            timings=timings,
        )
        d = debug.to_dict()
        assert d["mode"] == "float32"
        assert d["top_k"] == 5
        assert d["results_returned"] == 3
        assert "timings" in d

    def test_fallback_info(self):
        timings = RetrievalTimings()
        debug = RetrievalDebug(
            mode="binary->float32",
            top_k=5,
            candidates_searched=50,
            results_returned=5,
            timings=timings,
            fallback_triggered=True,
            fallback_reason="High distance: 600.0",
        )
        assert debug.fallback_triggered is True
        assert "High distance" in debug.fallback_reason


class TestRetrievedChunk:
    def test_basic_chunk(self):
        chunk = RetrievedChunk(
            chunk_id="doc1-0",
            doc_id="doc1",
            text="This is the retrieved text.",
            score=0.95,
            source="test.pdf",
            metadata={"page": 1},
        )
        assert chunk.chunk_id == "doc1-0"
        assert chunk.score == 0.95
        assert chunk.metadata["page"] == 1

    def test_to_dict(self):
        chunk = RetrievedChunk(
            chunk_id="c1",
            doc_id="d1",
            text="Text",
            score=0.8,
        )
        d = chunk.to_dict()
        assert d["chunk_id"] == "c1"
        assert d["doc_id"] == "d1"
        assert d["score"] == 0.8


class TestBQRetrievalConfig:
    def test_default_config(self):
        config = BQRetrievalConfig()
        assert config.default_mode == RetrievalModeV2.BINARY
        assert config.top_k == 5
        assert config.two_stage_enabled is False
        assert config.fallback_enabled is True

    def test_custom_config(self):
        config = BQRetrievalConfig(
            default_mode=RetrievalModeV2.FLOAT32,
            top_k=10,
            two_stage_enabled=True,
            stage1_candidates=100,
        )
        assert config.default_mode == RetrievalModeV2.FLOAT32
        assert config.top_k == 10
        assert config.two_stage_enabled is True
        assert config.stage1_candidates == 100

    def test_to_dict(self):
        config = BQRetrievalConfig()
        d = config.to_dict()
        assert d["default_mode"] == "binary"
        assert d["top_k"] == 5
        assert "milvus_config" in d
        assert "bq_config" in d


class TestBQRetrievalService:
    def test_init_basic(self):
        service = BQRetrievalService()
        assert service._config is not None
        assert service._milvus_store is None
        assert service._milvus_initialized is False

    def test_init_with_config(self):
        config = BQRetrievalConfig(top_k=10)
        service = BQRetrievalService(config=config)
        assert service._config.top_k == 10

    def test_init_with_float32_engine(self):
        mock_engine = MagicMock()
        mock_engine._reranker = MagicMock()

        service = BQRetrievalService(float32_engine=mock_engine)
        assert service._float32_engine is mock_engine
        assert service._reranker is mock_engine._reranker

    def test_get_index_stats_empty(self):
        service = BQRetrievalService()
        stats = service.get_index_stats()

        # No stores initialized, should return empty dict
        assert isinstance(stats, dict)

    def test_index_documents_from_docs_path(self, tmp_path, monkeypatch):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "handbook.md").write_text(
            "# Handbook\n\nInstall locally. Keep client data in the client-owned data volume.",
            encoding="utf-8",
        )
        (docs_dir / "notes.txt").write_text(
            "Use API-key auth before client exposure.", encoding="utf-8"
        )
        (docs_dir / "ignored.pdf").write_text("unsupported", encoding="utf-8")

        class FakeStore:
            def __init__(self):
                self.inserted = []
                self.index_built = False

            def bulk_insert(self, chunks):
                self.inserted = chunks
                return list(range(1, len(chunks) + 1))

            def build_index(self):
                self.index_built = True

        store = FakeStore()
        service = BQRetrievalService(embed_fn=lambda _text: [0.1] * 768)
        monkeypatch.setattr(service, "_ensure_milvus", lambda: store)

        result = service.index_documents(docs_path=str(docs_dir))

        assert result["mode"] == "binary"
        assert result["chunks_indexed"] == 2
        assert result["documents_scanned"] == 2
        assert result["collection"] == service._config.milvus_config.collection_name
        assert store.index_built is True
        assert {chunk.source for chunk in store.inserted} == {
            "handbook.md",
            "notes.txt",
        }
        assert all(chunk.embedding == [0.1] * 768 for chunk in store.inserted)
        assert all(
            chunk.metadata["index_source"] == "docs_path" for chunk in store.inserted
        )

    def test_index_documents_reports_invalid_docs_path_without_milvus(
        self, monkeypatch
    ):
        service = BQRetrievalService(embed_fn=lambda _text: [0.1] * 768)
        ensure_milvus = MagicMock(
            side_effect=AssertionError("should not initialize Milvus")
        )
        monkeypatch.setattr(service, "_ensure_milvus", ensure_milvus)

        result = service.index_documents(docs_path="/does/not/exist")

        assert result["mode"] == "binary"
        assert result["chunks_indexed"] == 0
        assert "does not exist" in result["error"]
        ensure_milvus.assert_not_called()


class TestBQRetrievalServiceWithMockEngine:
    @pytest.fixture
    def mock_engine(self):
        engine = MagicMock()
        engine._embedder = MagicMock()
        engine._embedder.encode.return_value = np.random.randn(768)
        engine._reranker = None
        engine._chunks = []

        # Mock retrieve method
        mock_result = MagicMock()
        mock_result.chunk_id = "c1"
        mock_result.chunk_text = "Test chunk"
        mock_result.score = 0.9
        mock_result.document = MagicMock()
        mock_result.document.id = "d1"
        mock_result.document.title = "Test Doc"
        mock_result.document.metadata = {}
        mock_result.start_char = 0
        mock_result.end_char = 100
        engine.retrieve.return_value = [mock_result]

        return engine

    def test_retrieve_float32_mode(self, mock_engine):
        config = BQRetrievalConfig(default_mode=RetrievalModeV2.FLOAT32)
        service = BQRetrievalService(
            config=config,
            float32_engine=mock_engine,
        )

        chunks, debug = service.retrieve("test query", mode="float32")

        assert debug.mode == "float32"
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "c1"
        mock_engine.retrieve.assert_called_once()

    def test_retrieve_with_custom_k(self, mock_engine):
        config = BQRetrievalConfig(default_mode=RetrievalModeV2.FLOAT32)
        service = BQRetrievalService(
            config=config,
            float32_engine=mock_engine,
        )

        chunks, debug = service.retrieve("test query", k=10, mode="float32")

        assert debug.top_k == 10


class TestGetBQRetrievalService:
    def test_factory_function(self):
        service = get_bq_retrieval_service()
        assert isinstance(service, BQRetrievalService)

    def test_factory_with_config(self):
        config = BQRetrievalConfig(top_k=15)
        service = get_bq_retrieval_service(config=config)
        assert service._config.top_k == 15


class TestQueryRequestDocumentIds:
    def test_document_ids_have_max_length(self):
        from pydantic import ValidationError

        from app.schemas.query import MAX_DOCUMENT_IDS, QueryRequest

        with pytest.raises(ValidationError):
            QueryRequest(
                question="What changed?",
                document_ids=[str(i) for i in range(MAX_DOCUMENT_IDS + 1)],
            )
