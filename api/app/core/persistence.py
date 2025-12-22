"""Index persistence for FAISS/BM25 indexes and disk-backed caching.

This module provides:
- DiskEmbeddingCache: SQLite-backed embedding cache with model+text hash keys
- IndexPersistence: Save/load FAISS and BM25 indexes to disk
- Cache invalidation with corpus version + config hash
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rank_bm25 import BM25Okapi


# ============================================================================
# Disk-Backed Embedding Cache
# ============================================================================

@dataclass
class CacheConfig:
    """Configuration for disk cache."""
    db_path: Path
    model_name: str = "default"
    max_entries: int = 100000
    ttl_days: int = 30
    
    @property
    def corpus_version_key(self) -> str:
        """Key for corpus version tracking."""
        return f"__corpus_version__{self.model_name}"


class DiskEmbeddingCache:
    """SQLite-backed embedding cache with model+text hash keys.
    
    Persists embeddings across restarts with automatic expiration.
    Keys are (model_name, text_hash) to prevent model mismatch issues.
    """
    
    def __init__(self, config: CacheConfig | None = None) -> None:
        if config is None:
            config = CacheConfig(db_path=Path("data/embedding_cache.db"))
        
        self._config = config
        self._db_path = config.db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                key TEXT PRIMARY KEY,
                model TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                embedding BLOB NOT NULL,
                created_at REAL NOT NULL,
                hit_count INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_model_hash 
            ON embeddings(model, text_hash)
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        conn.commit()
    
    def _get_conn(self) -> sqlite3.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        return self._conn
    
    def _make_key(self, text: str, model: str | None = None) -> tuple[str, str]:
        """Create cache key from text and model."""
        model = model or self._config.model_name
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:32]
        key = f"{model}:{text_hash}"
        return key, text_hash
    
    def get(self, text: str, model: str | None = None) -> list[float] | None:
        """Get cached embedding for text."""
        key, _ = self._make_key(text, model)
        conn = self._get_conn()
        
        cursor = conn.execute(
            "SELECT embedding, created_at FROM embeddings WHERE key = ?",
            (key,)
        )
        row = cursor.fetchone()
        
        if row is None:
            return None
        
        # Check TTL
        created_at = row[1]
        if time.time() - created_at > self._config.ttl_days * 86400:
            conn.execute("DELETE FROM embeddings WHERE key = ?", (key,))
            conn.commit()
            return None
        
        # Update hit count
        conn.execute(
            "UPDATE embeddings SET hit_count = hit_count + 1 WHERE key = ?",
            (key,)
        )
        
        # Deserialize embedding
        embedding_bytes = row[0]
        return pickle.loads(embedding_bytes)
    
    def set(
        self,
        text: str,
        embedding: list[float],
        model: str | None = None,
    ) -> None:
        """Cache embedding for text."""
        model = model or self._config.model_name
        key, text_hash = self._make_key(text, model)
        conn = self._get_conn()
        
        # Enforce max entries
        cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
        count = cursor.fetchone()[0]
        
        if count >= self._config.max_entries:
            # Remove oldest 10%
            to_remove = int(self._config.max_entries * 0.1)
            conn.execute("""
                DELETE FROM embeddings WHERE key IN (
                    SELECT key FROM embeddings 
                    ORDER BY created_at ASC 
                    LIMIT ?
                )
            """, (to_remove,))
        
        # Serialize and store
        embedding_bytes = pickle.dumps(embedding)
        conn.execute("""
            INSERT OR REPLACE INTO embeddings 
            (key, model, text_hash, embedding, created_at, hit_count)
            VALUES (?, ?, ?, ?, ?, 0)
        """, (key, model, text_hash, embedding_bytes, time.time()))
        conn.commit()
    
    def get_many(
        self,
        texts: list[str],
        model: str | None = None,
    ) -> list[list[float] | None]:
        """Get cached embeddings for multiple texts."""
        return [self.get(text, model) for text in texts]
    
    def set_many(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        model: str | None = None,
    ) -> None:
        """Cache multiple embeddings."""
        for text, embedding in zip(texts, embeddings):
            self.set(text, embedding, model)
    
    def invalidate_by_model(self, model: str) -> int:
        """Invalidate all entries for a model."""
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM embeddings WHERE model = ?",
            (model,)
        )
        conn.commit()
        return cursor.rowcount
    
    def set_corpus_version(self, version: str) -> None:
        """Set corpus version for invalidation tracking."""
        conn = self._get_conn()
        conn.execute("""
            INSERT OR REPLACE INTO metadata (key, value, updated_at)
            VALUES (?, ?, ?)
        """, (self._config.corpus_version_key, version, time.time()))
        conn.commit()
    
    def get_corpus_version(self) -> str | None:
        """Get current corpus version."""
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (self._config.corpus_version_key,)
        )
        row = cursor.fetchone()
        return row[0] if row else None
    
    def clear(self) -> None:
        """Clear all cached embeddings."""
        conn = self._get_conn()
        conn.execute("DELETE FROM embeddings")
        conn.commit()
    
    def cleanup_expired(self) -> int:
        """Remove expired entries."""
        cutoff = time.time() - self._config.ttl_days * 86400
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM embeddings WHERE created_at < ?",
            (cutoff,)
        )
        conn.commit()
        return cursor.rowcount
    
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        conn = self._get_conn()
        cursor = conn.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(hit_count) as total_hits,
                COUNT(DISTINCT model) as models
            FROM embeddings
        """)
        row = cursor.fetchone()
        
        return {
            "total_entries": row[0],
            "total_hits": row[1] or 0,
            "models": row[2],
            "db_path": str(self._db_path),
            "max_entries": self._config.max_entries,
        }
    
    def close(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


# ============================================================================
# Index Persistence
# ============================================================================

@dataclass
class IndexMetadata:
    """Metadata for persisted indexes."""
    corpus_version: str
    config_hash: str
    chunk_count: int
    created_at: float
    model_name: str = ""
    
    def to_dict(self) -> dict:
        return {
            "corpus_version": self.corpus_version,
            "config_hash": self.config_hash,
            "chunk_count": self.chunk_count,
            "created_at": self.created_at,
            "model_name": self.model_name,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "IndexMetadata":
        return cls(**data)


class IndexPersistence:
    """Persist and load FAISS/BM25 indexes to disk."""
    
    def __init__(self, base_path: Path | str = "data/indexes") -> None:
        self._base_path = Path(base_path)
        self._base_path.mkdir(parents=True, exist_ok=True)
    
    def _metadata_path(self, index_name: str) -> Path:
        return self._base_path / f"{index_name}_metadata.json"
    
    def _embeddings_path(self, index_name: str) -> Path:
        return self._base_path / f"{index_name}_embeddings.npy"
    
    def _chunks_path(self, index_name: str) -> Path:
        return self._base_path / f"{index_name}_chunks.pkl"
    
    def _bm25_path(self, index_name: str) -> Path:
        return self._base_path / f"{index_name}_bm25.pkl"
    
    def _tokenized_path(self, index_name: str) -> Path:
        return self._base_path / f"{index_name}_tokenized.pkl"
    
    def compute_config_hash(self, config: dict) -> str:
        """Compute hash of config for invalidation."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def compute_corpus_version(self, doc_ids: list[str]) -> str:
        """Compute version hash from document IDs."""
        sorted_ids = sorted(doc_ids)
        ids_str = ",".join(sorted_ids)
        return hashlib.sha256(ids_str.encode()).hexdigest()[:16]
    
    def save_dense_index(
        self,
        index_name: str,
        embeddings: np.ndarray,
        chunks: list[tuple[str, Any]],  # (doc_id, chunk)
        metadata: IndexMetadata,
    ) -> Path:
        """Save dense embeddings and chunks to disk."""
        # Save embeddings as numpy array
        embeddings_path = self._embeddings_path(index_name)
        np.save(str(embeddings_path), embeddings)
        
        # Save chunks
        chunks_path = self._chunks_path(index_name)
        with open(chunks_path, "wb") as f:
            pickle.dump(chunks, f)
        
        # Save metadata
        metadata_path = self._metadata_path(index_name)
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        return embeddings_path
    
    def load_dense_index(
        self,
        index_name: str,
    ) -> tuple[np.ndarray | None, list | None, IndexMetadata | None]:
        """Load dense embeddings and chunks from disk."""
        embeddings_path = self._embeddings_path(index_name)
        chunks_path = self._chunks_path(index_name)
        metadata_path = self._metadata_path(index_name)
        
        if not all(p.exists() for p in [embeddings_path, chunks_path, metadata_path]):
            return None, None, None
        
        # Load embeddings
        embeddings = np.load(str(embeddings_path))
        
        # Load chunks
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        
        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = IndexMetadata.from_dict(json.load(f))
        
        return embeddings, chunks, metadata
    
    def save_sparse_index(
        self,
        index_name: str,
        bm25: "BM25Okapi",
        tokenized_corpus: list[list[str]],
        metadata: IndexMetadata,
    ) -> Path:
        """Save BM25 index to disk."""
        # Save BM25 object
        bm25_path = self._bm25_path(index_name)
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25, f)
        
        # Save tokenized corpus
        tokenized_path = self._tokenized_path(index_name)
        with open(tokenized_path, "wb") as f:
            pickle.dump(tokenized_corpus, f)
        
        # Save metadata
        metadata_path = self._metadata_path(f"{index_name}_sparse")
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        return bm25_path
    
    def load_sparse_index(
        self,
        index_name: str,
    ) -> tuple[Any | None, list | None, IndexMetadata | None]:
        """Load BM25 index from disk."""
        bm25_path = self._bm25_path(index_name)
        tokenized_path = self._tokenized_path(index_name)
        metadata_path = self._metadata_path(f"{index_name}_sparse")
        
        if not all(p.exists() for p in [bm25_path, tokenized_path, metadata_path]):
            return None, None, None
        
        # Load BM25
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
        
        # Load tokenized corpus
        with open(tokenized_path, "rb") as f:
            tokenized_corpus = pickle.load(f)
        
        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = IndexMetadata.from_dict(json.load(f))
        
        return bm25, tokenized_corpus, metadata
    
    def is_valid(
        self,
        index_name: str,
        expected_corpus_version: str,
        expected_config_hash: str,
    ) -> bool:
        """Check if saved index is valid for current corpus/config."""
        metadata_path = self._metadata_path(index_name)
        if not metadata_path.exists():
            return False
        
        try:
            with open(metadata_path, "r") as f:
                metadata = IndexMetadata.from_dict(json.load(f))
            
            return (
                metadata.corpus_version == expected_corpus_version
                and metadata.config_hash == expected_config_hash
            )
        except Exception:
            return False
    
    def delete_index(self, index_name: str) -> None:
        """Delete all files for an index."""
        paths = [
            self._embeddings_path(index_name),
            self._chunks_path(index_name),
            self._bm25_path(index_name),
            self._tokenized_path(index_name),
            self._metadata_path(index_name),
            self._metadata_path(f"{index_name}_sparse"),
        ]
        for path in paths:
            if path.exists():
                path.unlink()
    
    def save_graph(self, index_name: str, graph_data: dict[str, Any], metadata: IndexMetadata) -> Path:
        """Save GraphRAG data to disk."""
        path = self._base_path / f"{index_name}_graph.pkl"
        with open(path, "wb") as f:
            pickle.dump(graph_data, f)
        
        # Save metadata for graph
        metadata_path = self._metadata_path(f"{index_name}_graph")
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
        
        return path

    def load_graph(self, index_name: str) -> tuple[dict[str, Any] | None, IndexMetadata | None]:
        """Load GraphRAG data from disk."""
        path = self._base_path / f"{index_name}_graph.pkl"
        metadata_path = self._metadata_path(f"{index_name}_graph")
        
        if not path.exists() or not metadata_path.exists():
            return None, None
            
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        with open(metadata_path, "r") as f:
            metadata = IndexMetadata.from_dict(json.load(f))
            
        return data, metadata

    def save_trees(self, index_name: str, trees: dict[str, Any], metadata: IndexMetadata) -> Path:
        """Save RAPTOR hierarchical trees to disk."""
        path = self._base_path / f"{index_name}_trees.pkl"
        with open(path, "wb") as f:
            pickle.dump(trees, f)
            
        # Save metadata for trees
        metadata_path = self._metadata_path(f"{index_name}_trees")
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
            
        return path

    def load_trees(self, index_name: str) -> tuple[dict[str, Any] | None, IndexMetadata | None]:
        """Load RAPTOR hierarchical trees from disk."""
        path = self._base_path / f"{index_name}_trees.pkl"
        metadata_path = self._metadata_path(f"{index_name}_trees")
        
        if not path.exists() or not metadata_path.exists():
            return None, None
            
        with open(path, "rb") as f:
            trees = pickle.load(f)
            
        with open(metadata_path, "r") as f:
            metadata = IndexMetadata.from_dict(json.load(f))
            
        return trees, metadata

    def list_indexes(self) -> list[str]:
        """List all saved indexes."""
        indexes = set()
        for path in self._base_path.glob("*_metadata.json"):
            name = path.stem.replace("_metadata", "").replace("_sparse", "").replace("_graph", "").replace("_trees", "")
            indexes.add(name)
        return sorted(indexes)


__all__ = [
    "CacheConfig",
    "DiskEmbeddingCache",
    "IndexMetadata",
    "IndexPersistence",
]
