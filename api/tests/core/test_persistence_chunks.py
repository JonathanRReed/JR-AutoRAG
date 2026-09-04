import json
import pickle
import numpy as np
import pytest
from pathlib import Path

from app.core.chunking import Chunk
from app.core.persistence import IndexPersistence, IndexMetadata


@pytest.fixture
def temp_persistence(tmp_path: Path) -> IndexPersistence:
    return IndexPersistence(base_path=tmp_path)


def test_save_and_load_dense_index_json(
    temp_persistence: IndexPersistence, tmp_path: Path
):
    index_name = "test_index"
    embeddings = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    chunk1 = Chunk(
        text="Hello world",
        index=0,
        start_char=0,
        end_char=11,
        metadata={"source": "test"},
    )
    chunk2 = Chunk(text="Foo bar", index=1, start_char=12, end_char=19, metadata=None)
    chunks = [("doc1", chunk1), ("doc2", chunk2)]
    metadata = IndexMetadata(
        created_at=123456.0,
        corpus_version="v1",
        config_hash="hash1",
        chunk_count=2,
        model_name="test-model",
    )

    temp_persistence.save_dense_index(index_name, embeddings, chunks, metadata)

    # Check that chunks file is JSON
    chunks_json_path = tmp_path / f"{index_name}_chunks.json"
    assert chunks_json_path.exists()

    with open(chunks_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 2
    assert data[0][0] == "doc1"
    assert data[0][1]["text"] == "Hello world"

    # Test load
    loaded_embeddings, loaded_chunks, loaded_metadata = (
        temp_persistence.load_dense_index(index_name)
    )
    assert loaded_embeddings is not None
    assert np.array_equal(loaded_embeddings, embeddings)
    assert loaded_chunks is not None
    assert len(loaded_chunks) == 2
    assert loaded_chunks[0][0] == "doc1"
    assert isinstance(loaded_chunks[0][1], Chunk)
    assert loaded_chunks[0][1].text == "Hello world"
    assert loaded_chunks[0][1].metadata == {"source": "test"}
    assert loaded_metadata.corpus_version == "v1"


def test_load_dense_index_legacy_pickle_fallback(
    temp_persistence: IndexPersistence, tmp_path: Path
):
    index_name = "legacy_index"
    embeddings = np.array([[0.5, 0.6]], dtype=np.float32)
    chunk1 = Chunk(text="Legacy chunk", index=0, start_char=0, end_char=12)
    chunks = [("doc_legacy", chunk1)]
    metadata = IndexMetadata(
        created_at=123456.0,
        corpus_version="v1",
        config_hash="hash1",
        chunk_count=1,
        model_name="test-model",
    )

    # Save embeddings & metadata manually
    np.save(str(tmp_path / f"{index_name}_embeddings.npy"), embeddings)
    with open(tmp_path / f"{index_name}_metadata.json", "w") as f:
        json.dump(metadata.to_dict(), f)

    # Save legacy pickle file
    legacy_pkl_path = tmp_path / f"{index_name}_chunks.pkl"
    with open(legacy_pkl_path, "wb") as f:
        pickle.dump(chunks, f)

    # Load should fallback to pickle file
    loaded_embeddings, loaded_chunks, loaded_metadata = (
        temp_persistence.load_dense_index(index_name)
    )
    assert loaded_embeddings is not None
    assert loaded_chunks is not None
    assert len(loaded_chunks) == 1
    assert loaded_chunks[0][0] == "doc_legacy"
    assert loaded_chunks[0][1].text == "Legacy chunk"
