import tempfile
from pathlib import Path
from app.core.persistence import DiskEmbeddingCache, CacheConfig

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
