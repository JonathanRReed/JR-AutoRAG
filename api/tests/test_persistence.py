"""Embedding cache format and malformed-entry regression tests."""

import json
import pickle
import time
from unittest.mock import patch

import pytest

from app.core.persistence import CacheConfig, DiskEmbeddingCache


def exploding_func():
    raise AssertionError("Unpickling executed unsafe code")


class ExplodingPayload:
    def __reduce__(self):
        return exploding_func, ()


@pytest.fixture
def cache(tmp_path):
    instance = DiskEmbeddingCache(CacheConfig(db_path=tmp_path / "embedding_cache.db"))
    try:
        yield instance
    finally:
        instance.close()


def insert_payload(cache, text, payload):
    key, text_hash = cache._make_key(text)
    conn = cache._get_conn()
    conn.execute(
        """
        INSERT OR REPLACE INTO embeddings
        (key, model, text_hash, embedding, created_at, hit_count)
        VALUES (?, ?, ?, ?, ?, 0)
        """,
        (key, "default", text_hash, payload, time.time()),
    )
    conn.commit()


def test_disk_embedding_cache_json_serialization(cache):
    embedding = [0.1, 0.25, 0.5, 0.75, -1.0]
    with (
        patch("pickle.dumps", side_effect=AssertionError("pickle.dumps called")),
        patch("pickle.loads", side_effect=AssertionError("pickle.loads called")),
    ):
        cache.set("test query text", embedding)
        assert cache.get("test query text") == embedding

    row = cache._get_conn().execute("SELECT embedding FROM embeddings").fetchone()
    assert row is not None
    assert isinstance(row[0], (bytes, str))
    assert json.loads(row[0]) == embedding


def test_disk_embedding_cache_legacy_pickle_returns_none_without_unpickling(cache):
    # __reduce__ would actually invoke the target if unpickled, unlike a tuple
    # containing a function reference. Never execute this payload in the test.
    payload = pickle.dumps(ExplodingPayload())
    insert_payload(cache, "legacy text", payload)
    with patch("pickle.loads", side_effect=AssertionError("pickle.loads called")) as loads:
        assert cache.get("legacy text") is None
        loads.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        b'{"',
        b'{"a": 1}',
        b"[]",
        b'[1.0, "2.0", 3.0]',
        b"[1.0, true, 3.0]",
        b"[1.0, NaN, 3.0]",
        b"[1.0, Infinity, 3.0]",
        b"\xff\xfe",
        b"null",
        b"[1" + b"0" * 400 + b"]",
    ],
)
def test_disk_embedding_cache_malformed_json_and_invalid_vectors(cache, payload):
    insert_payload(cache, "invalid", payload)
    with patch("pickle.loads", side_effect=AssertionError("pickle.loads called")):
        assert cache.get("invalid") is None
