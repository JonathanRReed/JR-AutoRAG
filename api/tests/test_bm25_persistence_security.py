import json
import time

from rank_bm25 import BM25Okapi
from unittest.mock import patch

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
        loaded_bm25, loaded_tokenized, loaded_meta = persistence.load_sparse_index(index_name)
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
