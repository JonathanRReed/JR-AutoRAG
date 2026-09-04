import json
import pickle
import time
from rank_bm25 import BM25Okapi
from app.core.persistence import IndexMetadata, IndexPersistence


def test_sparse_index_saves_and_loads_json(tmp_path):
    persistence = IndexPersistence(base_path=str(tmp_path))
    index_name = "test_idx"

    tokenized_corpus = [["hello", "world"], ["foo", "bar"]]
    bm25 = BM25Okapi(tokenized_corpus)
    metadata = IndexMetadata(
        corpus_version="v1",
        config_hash="cfg1",
        chunk_count=2,
        created_at=time.time(),
        model_name="bm25",
    )

    # Save sparse index
    persistence.save_sparse_index(index_name, bm25, tokenized_corpus, metadata)

    # Verify tokenized corpus is stored as JSON file, not pickle
    json_file = tmp_path / f"{index_name}_tokenized.json"
    pkl_file = tmp_path / f"{index_name}_tokenized.pkl"

    assert json_file.exists()
    assert not pkl_file.exists()

    with open(json_file, "r", encoding="utf-8") as f:
        loaded_json_data = json.load(f)
    assert loaded_json_data == tokenized_corpus

    # Load sparse index
    loaded_bm25, loaded_tokenized, loaded_meta = persistence.load_sparse_index(
        index_name
    )
    assert loaded_tokenized == tokenized_corpus
    assert loaded_meta.corpus_version == "v1"


def test_sparse_index_legacy_pickle_fallback(tmp_path):
    persistence = IndexPersistence(base_path=str(tmp_path))
    index_name = "legacy_idx"

    tokenized_corpus = [["legacy", "tokens"]]
    bm25 = BM25Okapi(tokenized_corpus)
    metadata = IndexMetadata(
        corpus_version="v0",
        config_hash="cfg0",
        chunk_count=1,
        created_at=time.time(),
        model_name="bm25",
    )

    # Manually create legacy pickle tokenized file and other files
    bm25_path = tmp_path / f"{index_name}_bm25.pkl"
    legacy_tokenized_path = tmp_path / f"{index_name}_tokenized.pkl"
    metadata_path = tmp_path / f"{index_name}_sparse_metadata.json"

    with open(bm25_path, "wb") as f:
        pickle.dump(bm25, f)
    with open(legacy_tokenized_path, "wb") as f:
        pickle.dump(tokenized_corpus, f)
    with open(metadata_path, "w") as f:
        json.dump(metadata.to_dict(), f)

    # Load sparse index from legacy pickle file
    loaded_bm25, loaded_tokenized, loaded_meta = persistence.load_sparse_index(
        index_name
    )
    assert loaded_tokenized == tokenized_corpus
    assert loaded_meta.corpus_version == "v0"

    # Test delete_index cleans up both json and pkl tokenized paths if they exist
    persistence.delete_index(index_name)
    assert not legacy_tokenized_path.exists()
