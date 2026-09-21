"""Unreadable legacy or corrupt indexes must never be overwritten on shutdown."""

import json
from unittest.mock import patch

import pytest

from app.core.binary_vector_store import MilvusChunk, MilvusConfig, MilvusVectorStore


@pytest.mark.parametrize("payload", [b"\x80\x04legacy-pickle", b"{broken", b"[]"])
def test_failed_connect_preserves_existing_index_on_disconnect(tmp_path, payload):
    path = tmp_path / "index.data"
    path.write_bytes(payload)
    store = MilvusVectorStore(MilvusConfig(persist_path=str(path)), embedding_dim=8)
    with patch("pickle.load", side_effect=AssertionError("Must not unpickle")):
        try:
            with pytest.raises(ValueError, match="rebuild"):
                store.connect()
            assert not store._connected
            with pytest.raises(ValueError, match="overwrite"):
                store._save_to_disk()
        finally:
            store.disconnect()
    assert path.read_bytes() == payload


def test_new_index_round_trip_is_unchanged(tmp_path):
    path = tmp_path / "index.json"
    config = MilvusConfig(persist_path=str(path))
    store = MilvusVectorStore(config, embedding_dim=8)
    assert store.connect()
    store.insert([MilvusChunk("doc", "chunk", "source", "text", bq_vector=b"\xff")])
    store.disconnect()
    loaded = MilvusVectorStore(config, embedding_dim=8)
    assert loaded.connect()
    try:
        assert loaded.count() == 1
        assert loaded.search_binary(b"\xff")[0].doc_id == "doc"
    finally:
        loaded.disconnect()


def test_failed_reload_cannot_overwrite_existing_file(tmp_path):
    path = tmp_path / "index.json"
    store = MilvusVectorStore(MilvusConfig(persist_path=str(path)), embedding_dim=8)
    store.connect()
    store.insert([MilvusChunk("doc", "chunk", "source", "text", bq_vector=b"\xff")])
    payload = b"{corrupt-after-connect"
    path.write_bytes(payload)
    assert not store._load_from_disk()
    assert store.count() == 1
    with pytest.raises(ValueError, match="overwrite"):
        store.disconnect()
    assert not store._connected
    assert path.read_bytes() == payload


def test_archiving_bad_index_allows_explicit_rebuild(tmp_path):
    path = tmp_path / "index.json"
    path.write_bytes(b"legacy-pickle")
    store = MilvusVectorStore(MilvusConfig(persist_path=str(path)), embedding_dim=8)
    with pytest.raises(ValueError, match="rebuild"):
        store.connect()
    archived = path.with_suffix(".legacy")
    path.rename(archived)
    assert store.connect()
    store.insert([MilvusChunk("doc", "chunk", "source", "text", bq_vector=b"\xff")])
    store.disconnect()
    assert archived.read_bytes() == b"legacy-pickle"
    assert json.loads(path.read_text())["chunks"][0]["doc_id"] == "doc"
