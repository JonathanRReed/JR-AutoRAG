"""Security unit tests for DiskEmbeddingCache deserialization."""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

from app.core.persistence import CacheConfig, DiskEmbeddingCache


def test_disk_embedding_cache_json_roundtrip(tmp_path: Path) -> None:
    """Test that embedding vectors are serialized and deserialized via JSON."""
    db_path = tmp_path / "embedding_cache.db"
    cache = DiskEmbeddingCache(CacheConfig(db_path=db_path))

    test_text = "secure query embedding"
    test_embedding = [0.1, 0.25, -0.5, 0.99]

    cache.set(test_text, test_embedding)

    # Verify raw database content is valid UTF-8 JSON
    key, _ = cache._make_key(test_text)
    conn = cache._get_conn()
    cursor = conn.execute("SELECT embedding FROM embeddings WHERE key = ?", (key,))
    raw_bytes = cursor.fetchone()[0]

    decoded_json = json.loads(raw_bytes.decode("utf-8"))
    assert decoded_json == test_embedding

    # Verify get returns correct embedding
    retrieved = cache.get(test_text)
    assert retrieved == test_embedding

    cache.close()


def test_disk_embedding_cache_legacy_pickle_fallback(tmp_path: Path) -> None:
    """Test that legacy pickle-encoded cache entries still deserialize safely via fallback."""
    db_path = tmp_path / "embedding_cache.db"
    cache = DiskEmbeddingCache(CacheConfig(db_path=db_path))

    test_text = "legacy pickle query"
    legacy_embedding = [0.5, -0.25, 0.123]
    key, text_hash = cache._make_key(test_text)

    # Insert pickle-serialized byte payload directly into database
    pickle_bytes = pickle.dumps(legacy_embedding)
    conn = cache._get_conn()
    conn.execute(
        """
        INSERT INTO embeddings (key, model, text_hash, embedding, created_at, hit_count)
        VALUES (?, ?, ?, ?, ?, 0)
    """,
        (key, cache._config.model_name, text_hash, pickle_bytes, time.time()),
    )
    conn.commit()

    # Get should deserialize legacy pickle data without failing
    retrieved = cache.get(test_text)
    assert retrieved == legacy_embedding

    cache.close()


def test_disk_embedding_cache_malformed_payload_graceful_handling(tmp_path: Path) -> None:
    """Test that corrupt or unparseable byte payloads return None rather than crashing."""
    db_path = tmp_path / "embedding_cache.db"
    cache = DiskEmbeddingCache(CacheConfig(db_path=db_path))

    test_text = "corrupt payload query"
    key, text_hash = cache._make_key(test_text)

    corrupt_bytes = b"not a json or pickle payload \x80\xff\xfe"
    conn = cache._get_conn()
    conn.execute(
        """
        INSERT INTO embeddings (key, model, text_hash, embedding, created_at, hit_count)
        VALUES (?, ?, ?, ?, ?, 0)
    """,
        (key, cache._config.model_name, text_hash, corrupt_bytes, time.time()),
    )
    conn.commit()

    retrieved = cache.get(test_text)
    assert retrieved is None

    cache.close()
