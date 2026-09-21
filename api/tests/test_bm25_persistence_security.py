import json
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from app.core.chunking import Chunk
from app.core.persistence import IndexMetadata, IndexPersistence


def test_bm25_save_and_load_json(tmp_path):
    persistence = IndexPersistence(base_path=tmp_path)
    index_name = "test_index"

    corpus = [["hello", "world"], ["foo", "bar", "hello"]]
    bm25 = BM25Okapi(corpus, k1=1.2, b=0.8, epsilon=0.2)
    metadata = IndexMetadata(
        corpus_version="v1",
        config_hash="hash123",
        chunk_count=2,
        created_at=time.time(),
        model_name="bm25",
    )

    # Save sparse index
    saved_path = persistence.save_sparse_index(
        index_name=index_name,
        bm25=bm25,
        tokenized_corpus=corpus,
        metadata=metadata,
    )

    assert saved_path.exists()
    assert saved_path.suffix == ".json"

    # Verify files on disk are JSON (not pickle)
    bm25_json_path = tmp_path / f"{index_name}_bm25.json"
    tokenized_json_path = tmp_path / f"{index_name}_tokenized.json"

    with open(bm25_json_path, "r") as f:
        bm25_data = json.load(f)
        assert bm25_data["k1"] == 1.2
        assert bm25_data["b"] == 0.8
        assert bm25_data["epsilon"] == 0.2

    with open(tokenized_json_path, "r") as f:
        tokenized_data = json.load(f)
        assert tokenized_data == corpus

    # Verify loading BM25 index does NOT call pickle.load
    with patch("pickle.load") as mock_pickle_load:
        loaded_bm25, loaded_tokenized, loaded_meta = persistence.load_sparse_index(
            index_name
        )
        assert mock_pickle_load.call_count == 0

    assert loaded_bm25 is not None
    assert loaded_tokenized == corpus
    assert loaded_meta.corpus_version == "v1"

    # Check that reconstructed BM25 produces identical scores
    scores_orig = bm25.get_scores(["hello"])
    scores_loaded = loaded_bm25.get_scores(["hello"])
    assert list(scores_orig) == list(scores_loaded)


def test_delete_index_cleans_json_and_pkl_files(tmp_path):
    persistence = IndexPersistence(base_path=tmp_path)
    index_name = "test_delete"

    # Create dummy json and pkl files
    json_bm25 = tmp_path / f"{index_name}_bm25.json"
    json_tokenized = tmp_path / f"{index_name}_tokenized.json"
    legacy_pkl_bm25 = tmp_path / f"{index_name}_bm25.pkl"

    json_bm25.write_text("{}")
    json_tokenized.write_text("[]")
    legacy_pkl_bm25.write_text("dummy")

    persistence.delete_index(index_name)

    assert not json_bm25.exists()
    assert not json_tokenized.exists()
    assert not legacy_pkl_bm25.exists()


def test_legacy_pickle_fallback_rebuilds_without_unpickling(tmp_path):
    """Test that when only legacy pickle files exist, load_sparse_index returns None (no unpickling)

    and HybridRetrievalEngine safely rebuilds BM25 index from chunks.
    """
    from app.core.hybrid_retrieval import HybridConfig, HybridRetrievalEngine

    persistence = IndexPersistence(base_path=tmp_path)
    index_name = "test_legacy"

    # Save legacy pickle file
    legacy_pkl_bm25 = tmp_path / f"{index_name}_bm25.pkl"
    legacy_pkl_bm25.write_text("malicious pickle data")

    # load_sparse_index should return None because .json files do not exist
    with patch("pickle.load") as mock_pickle_load:
        bm25, tokenized, meta = persistence.load_sparse_index(index_name)
        assert bm25 is None
        assert tokenized is None
        assert mock_pickle_load.call_count == 0

    # Test HybridRetrievalEngine rebuild behavior when loading index
    mock_doc_store = MagicMock()
    engine = HybridRetrievalEngine(
        documents=mock_doc_store, config=HybridConfig(), persist_path=str(tmp_path)
    )
    c1 = Chunk(text="Python search index", index=0, start_char=0, end_char=19)
    c2 = Chunk(text="BM25 sparse retrieval", index=1, start_char=0, end_char=21)
    engine._chunks = [("doc1", c1), ("doc2", c2)]

    # Mock index validity and dense index loading so load_index proceeds to sparse index section
    dummy_embeddings = np.zeros((2, 384))
    with (
        patch("app.core.persistence.IndexPersistence.is_valid", return_value=True),
        patch(
            "app.core.persistence.IndexPersistence.load_dense_index"
        ) as mock_load_dense,
    ):
        mock_load_dense.return_value = (
            dummy_embeddings,
            [("doc1", c1), ("doc2", c2)],
            IndexMetadata(
                corpus_version="v1", config_hash="h", chunk_count=2, created_at=1.0
            ),
        )
        with patch("pickle.load") as mock_pickle_load:
            engine.load_index(index_name)
            assert mock_pickle_load.call_count == 0

    # Ensure BM25 retriever was rebuilt and works
    assert engine._bm25 is not None
    scores = engine._bm25.get_scores(["python"])
    assert len(scores) == 2


def test_invalid_json_schema_raises_error(tmp_path):
    persistence = IndexPersistence(base_path=tmp_path)
    index_name = "test_invalid_schema"

    bm25_json = tmp_path / f"{index_name}_bm25.json"
    tokenized_json = tmp_path / f"{index_name}_tokenized.json"
    metadata_json = tmp_path / f"{index_name}_sparse_metadata.json"

    bm25_json.write_text('{"k1": "invalid_type"}')
    tokenized_json.write_text('{"invalid": "schema_not_list"}')
    metadata_json.write_text(
        '{"corpus_version": "v1", "config_hash": "h", "chunk_count": 1, "created_at": 1.0, "model_name": ""}'
    )

    with pytest.raises(ValueError, match="Invalid tokenized_corpus schema"):
        persistence.load_sparse_index(index_name)
