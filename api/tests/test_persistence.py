import json
import pickle
import tempfile
import time
from pathlib import Path

from app.core.persistence import CacheConfig, DiskEmbeddingCache


class ExplodingTarget:
    def __call__(self):
        raise RuntimeError("Unpickling executed unsafe code!")


def exploding_func():
    raise RuntimeError("Unpickling executed unsafe code!")


def test_disk_embedding_cache_json_serialization():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "embedding_cache.db"
        config = CacheConfig(db_path=db_path)
        cache = DiskEmbeddingCache(config)

        sample_embedding = [0.1, 0.25, 0.5, 0.75, -1.0]
        cache.set("test query text", sample_embedding)

        # Retrieve and verify equality
        retrieved = cache.get("test query text")
        assert retrieved == sample_embedding

        # Verify underlying stored data is valid JSON
        conn = cache._get_conn()
        cursor = conn.execute("SELECT embedding FROM embeddings")
        row = cursor.fetchone()
        assert row is not None
        raw_bytes = row[0]
        assert isinstance(raw_bytes, (bytes, str))

        cache.close()


def test_disk_embedding_cache_legacy_pickle_returns_none_without_unpickling():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "embedding_cache.db"
        config = CacheConfig(db_path=db_path)
        cache = DiskEmbeddingCache(config)

        # Insert legacy pickled bytes directly into SQLite DB
        text = "legacy text"
        key, text_hash = cache._make_key(text)
        # Pickle payload that calls exploding_func upon unpickling
        pickled_bytes = pickle.dumps((exploding_func, ()))

        conn = cache._get_conn()
        conn.execute(
            """
            INSERT INTO embeddings (key, model, text_hash, embedding, created_at, hit_count)
            VALUES (?, ?, ?, ?, ?, 0)
            """,
            (key, config.model_name, text_hash, pickled_bytes, time.time()),
        )
        conn.commit()

        # Should treat legacy pickled bytes as cache miss (returns None) without triggering function call
        result = cache.get(text)
        assert result is None

        cache.close()


def test_disk_embedding_cache_malformed_json_and_invalid_vectors():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "embedding_cache.db"
        config = CacheConfig(db_path=db_path)
        cache = DiskEmbeddingCache(config)

        conn = cache._get_conn()

        def insert_payload(text, raw_bytes):
            key, text_hash = cache._make_key(text)
            conn.execute(
                """
                INSERT OR REPLACE INTO embeddings (key, model, text_hash, embedding, created_at, hit_count)
                VALUES (?, ?, ?, ?, ?, 0)
                """,
                (key, config.model_name, text_hash, raw_bytes, time.time()),
            )
            conn.commit()

        # Malformed JSON
        insert_payload("malformed", b'{"')
        assert cache.get("malformed") is None

        # Non-list JSON object
        insert_payload("dict_json", b'{"a": 1}')
        assert cache.get("dict_json") is None

        # Empty list
        insert_payload("empty_list", b"[]")
        assert cache.get("empty_list") is None

        # Non-numeric elements (string in vector)
        insert_payload("string_element", b'[1.0, "2.0", 3.0]')
        assert cache.get("string_element") is None

        # Boolean elements
        insert_payload("bool_element", b"[1.0, true, 3.0]")
        assert cache.get("bool_element") is None

        # NaN / Infinity floats
        insert_payload("nan_element", json.dumps([1.0, float("nan"), 3.0]).encode("utf-8"))
        assert cache.get("nan_element") is None

        insert_payload("inf_element", json.dumps([1.0, float("inf"), 3.0]).encode("utf-8"))
        assert cache.get("inf_element") is None

        cache.close()
