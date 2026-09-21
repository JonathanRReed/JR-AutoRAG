"""Regression coverage for validating persisted binary indexes before use."""

import copy
import json

import pytest

from app.core.binary_vector_store import MilvusChunk, MilvusConfig, MilvusVectorStore


@pytest.mark.parametrize(
    "mutation",
    [
        "dimension",
        "quantization",
        "short_vector",
        "non_string_vector",
        "duplicate_id",
        "next_id_collision",
        "invalid_metadata",
        "invalid_second_chunk",
    ],
)
def test_invalid_index_preserves_loaded_state(tmp_path, mutation):
    path = tmp_path / "index.json"
    store = MilvusVectorStore(
        config=MilvusConfig(persist_path=str(path)), embedding_dim=8
    )
    store.connect()
    store.insert(
        [
            MilvusChunk(
                doc_id="doc1", chunk_id="c1", source="source", text="first",
                bq_vector=b"\xff",
            ),
            MilvusChunk(
                doc_id="doc2", chunk_id="c2", source="source", text="second",
                bq_vector=b"\x00",
            ),
        ]
    )
    store._save_to_disk()
    before = copy.deepcopy(store._chunks)
    next_id = store._next_id
    data = json.loads(path.read_text(encoding="utf-8"))

    if mutation == "dimension":
        data["embedding_dim"] = 16
    elif mutation == "quantization":
        data["bq_config"] = {}
    elif mutation == "short_vector":
        data["chunks"][0]["bq_vector"] = ""
    elif mutation == "non_string_vector":
        data["chunks"][0]["bq_vector"] = [255]
    elif mutation == "duplicate_id":
        data["chunks"][1]["id"] = data["chunks"][0]["id"]
    elif mutation == "next_id_collision":
        data["next_id"] = 1
    elif mutation == "invalid_metadata":
        data["chunks"][0]["metadata"] = []
    else:
        data["chunks"][1]["bq_vector"] = "not-hex"

    path.write_text(json.dumps(data), encoding="utf-8")
    assert store._load_from_disk() is False
    assert store._chunks == before
    assert store._next_id == next_id
    assert path.read_text(encoding="utf-8") == json.dumps(data)


def test_valid_empty_index_loads(tmp_path):
    path = tmp_path / "empty.json"
    config = MilvusConfig(persist_path=str(path))
    first = MilvusVectorStore(config=config, embedding_dim=8)
    first._save_to_disk()
    second = MilvusVectorStore(config=config, embedding_dim=8)
    assert second._load_from_disk() is True
    assert second.count() == 0
    assert second._next_id == 1
