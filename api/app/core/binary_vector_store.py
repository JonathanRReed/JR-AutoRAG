"""Pure Python Binary Vector Store for JR AutoRAG v2.

This module provides:
- In-memory binary vector store with HAMMING distance search
- No external dependencies (no Docker, no Milvus)
- Optional disk persistence
- Bulk insert and search operations

Runs entirely in-process with numpy for fast Hamming distance computation.
"""

from __future__ import annotations

import json
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

from .binary_quantization import (
    BQConfig,
    float32_to_binary,
    batch_float32_to_binary,
    get_binary_dimension,
    validate_dimension,
)

if TYPE_CHECKING:
    pass


def is_milvus_available() -> bool:
    """Always returns True - we use in-memory store, no external deps needed."""
    return True


@dataclass
class MilvusConfig:
    """Configuration for binary vector store (kept for API compatibility)."""
    
    # These are kept for API compatibility but not used for external connections
    host: str = "localhost"
    port: int = 19530
    uri: str = ""
    
    # Collection settings
    collection_name: str = "jr_autorag_chunks_bq"
    
    # Index settings
    index_type: str = "BIN_FLAT"  # Only BIN_FLAT supported in pure Python
    metric_type: str = "HAMMING"
    nlist: int = 128  # Not used in pure Python implementation
    
    # Search settings
    nprobe: int = 16  # Not used in pure Python implementation
    
    # Persistence settings (new)
    persist_path: str = ""  # Path to save/load index, empty = in-memory only
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            "uri": self.uri,
            "collection_name": self.collection_name,
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "nlist": self.nlist,
            "nprobe": self.nprobe,
            "persist_path": self.persist_path,
        }


@dataclass
class MilvusChunk:
    """A chunk to be stored in the binary vector store."""
    doc_id: str
    chunk_id: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None  # Float32 embedding (will be quantized)
    bq_vector: bytes | None = None  # Pre-quantized binary vector


@dataclass
class MilvusSearchResult:
    """Result from binary vector search."""
    id: int
    doc_id: str
    chunk_id: str
    source: str
    text: str
    metadata: dict[str, Any]
    distance: float  # Hamming distance (lower is better)
    
    @property
    def score(self) -> float:
        """Convert Hamming distance to similarity score (higher is better)."""
        return 1.0 / (1.0 + self.distance)


@dataclass 
class IndexStats:
    """Statistics about the binary vector index."""
    count: int
    index_type: str
    metric: str
    dim_bits: int
    storage_estimate_bytes: int
    collection_name: str
    embedding_version: str = ""
    quantization_version: str = ""


@dataclass
class _StoredChunk:
    """Internal representation of a stored chunk."""
    id: int
    doc_id: str
    chunk_id: str
    source: str
    text: str
    metadata: dict[str, Any]
    bq_vector: bytes


class MilvusVectorStore:
    """Pure Python in-memory binary vector store.
    
    Implements HAMMING distance search without any external dependencies.
    Compatible with the JR AutoRAG v2 design.
    """
    
    def __init__(
        self,
        config: MilvusConfig | None = None,
        embedding_dim: int = 768,
        bq_config: BQConfig | None = None,
    ) -> None:
        """Initialize binary vector store.
        
        Args:
            config: Store configuration
            embedding_dim: Dimension of float32 embeddings (must be divisible by 8)
            bq_config: Binary quantization configuration
        """
        self._config = config or MilvusConfig()
        self._embedding_dim = embedding_dim
        self._bq_config = bq_config or BQConfig()
        
        if not validate_dimension(embedding_dim):
            raise ValueError(
                f"Embedding dimension {embedding_dim} must be divisible by 8 "
                "for binary quantization."
            )
        
        self._binary_dim = get_binary_dimension(embedding_dim)
        
        # In-memory storage
        self._chunks: list[_StoredChunk] = []
        self._next_id: int = 1
        self._connected = False
        
        # Binary vectors as numpy array for fast search
        self._vectors: np.ndarray | None = None
        self._vectors_dirty = True
        
        # Versioning for drift control
        self._embedding_version = ""
        self._quantization_version = self._bq_config.version
    
    def connect(self) -> bool:
        """Connect to store (no-op for in-memory, loads from disk if persist_path set)."""
        self._connected = True
        
        if self._config.persist_path:
            self._load_from_disk()
        
        return True
    
    def disconnect(self) -> None:
        """Disconnect from store (saves to disk if persist_path set)."""
        if self._config.persist_path:
            self._save_to_disk()
        self._connected = False
    
    def _ensure_connected(self) -> None:
        """Ensure store is connected."""
        if not self._connected:
            self.connect()
    
    def create_collection(self, drop_existing: bool = False) -> bool:
        """Create/reset the collection.
        
        Args:
            drop_existing: If True, clear existing data
            
        Returns:
            True always (in-memory store is always ready)
        """
        self._ensure_connected()
        
        if drop_existing:
            self._chunks = []
            self._next_id = 1
            self._vectors = None
            self._vectors_dirty = True
            print(f"Cleared collection: {self._config.collection_name}")
        
        print(f"Collection ready: {self._config.collection_name}")
        return True
    
    def build_index(self) -> bool:
        """Build the binary vector index (precompute numpy array)."""
        self._rebuild_vectors()
        print(f"Built BIN_FLAT index with HAMMING metric ({len(self._chunks)} vectors)")
        return True
    
    def _rebuild_vectors(self) -> None:
        """Rebuild the numpy array of binary vectors for fast search."""
        if not self._chunks:
            self._vectors = None
            self._vectors_dirty = False
            return
        
        # Convert bytes to numpy array of uint8
        vectors = []
        for chunk in self._chunks:
            vec = np.frombuffer(chunk.bq_vector, dtype=np.uint8)
            vectors.append(vec)
        
        self._vectors = np.array(vectors, dtype=np.uint8)
        self._vectors_dirty = False
    
    def insert(self, chunks: list[MilvusChunk]) -> list[int]:
        """Insert chunks into the store.
        
        Args:
            chunks: List of chunks to insert
            
        Returns:
            List of inserted IDs
        """
        self._ensure_connected()
        
        if not chunks:
            return []
        
        ids = []
        for chunk in chunks:
            # Get or compute binary vector
            if chunk.bq_vector is not None:
                bq_vector = chunk.bq_vector
            elif chunk.embedding is not None:
                bq_vector = float32_to_binary(chunk.embedding, self._bq_config)
            else:
                raise ValueError(f"Chunk {chunk.chunk_id} has no embedding or bq_vector")
            
            stored = _StoredChunk(
                id=self._next_id,
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                text=chunk.text[:65535],
                metadata=chunk.metadata,
                bq_vector=bq_vector,
            )
            
            self._chunks.append(stored)
            ids.append(self._next_id)
            self._next_id += 1
        
        self._vectors_dirty = True
        return ids
    
    def bulk_insert(
        self,
        chunks: list[MilvusChunk],
        batch_size: int = 1000,
    ) -> list[int]:
        """Bulk insert chunks.
        
        Args:
            chunks: List of chunks to insert
            batch_size: Batch size (for progress reporting only)
            
        Returns:
            List of all inserted IDs
        """
        all_ids = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            ids = self.insert(batch)
            all_ids.extend(ids)
            print(f"Inserted batch {i // batch_size + 1}: {len(ids)} chunks")
        
        # Rebuild index after bulk insert
        self._rebuild_vectors()
        
        return all_ids
    
    def _hamming_distance_batch(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """Compute Hamming distance between query and all vectors.
        
        Args:
            query: Query vector as uint8 array
            vectors: Matrix of vectors as uint8 array (n_vectors x n_bytes)
            
        Returns:
            Array of Hamming distances
        """
        # XOR query with all vectors
        xor = np.bitwise_xor(vectors, query)
        
        # Count set bits using lookup table
        # This is faster than np.unpackbits for large arrays
        lookup = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
        bit_counts = lookup[xor]
        
        # Sum across bytes to get total Hamming distance
        return bit_counts.sum(axis=1)
    
    def search(
        self,
        query_embedding: list[float] | np.ndarray,
        top_k: int = 5,
        filter_expr: str | None = None,
    ) -> list[MilvusSearchResult]:
        """Search for similar chunks using binary vector search.
        
        Args:
            query_embedding: Float32 query embedding (will be quantized)
            top_k: Number of results to return
            filter_expr: Optional filter (doc_id == "xxx" supported)
            
        Returns:
            List of search results sorted by Hamming distance (ascending)
        """
        self._ensure_connected()
        
        if not self._chunks:
            return []
        
        # Rebuild vectors if dirty
        if self._vectors_dirty or self._vectors is None:
            self._rebuild_vectors()
        
        if self._vectors is None:
            return []
        
        # Quantize query embedding
        query_bq = float32_to_binary(query_embedding, self._bq_config)
        query_arr = np.frombuffer(query_bq, dtype=np.uint8)
        
        # Compute Hamming distances
        distances = self._hamming_distance_batch(query_arr, self._vectors)
        
        # Apply filter if specified
        valid_indices = np.arange(len(self._chunks))
        if filter_expr:
            # Simple filter parsing: doc_id == "xxx"
            if 'doc_id ==' in filter_expr:
                doc_id = filter_expr.split('==')[1].strip().strip('"\'')
                valid_indices = np.array([
                    i for i, c in enumerate(self._chunks) if c.doc_id == doc_id
                ])
                if len(valid_indices) == 0:
                    return []
                distances = distances[valid_indices]
        
        # Get top-k indices
        if len(distances) <= top_k:
            sorted_indices = np.argsort(distances)
        else:
            # Use argpartition for efficiency when k << n
            partition_indices = np.argpartition(distances, top_k)[:top_k]
            sorted_indices = partition_indices[np.argsort(distances[partition_indices])]
        
        # Build results
        results = []
        for idx in sorted_indices[:top_k]:
            chunk_idx = valid_indices[idx] if filter_expr else idx
            chunk = self._chunks[chunk_idx]
            results.append(MilvusSearchResult(
                id=chunk.id,
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                text=chunk.text,
                metadata=chunk.metadata,
                distance=float(distances[idx]),
            ))
        
        return results
    
    def search_binary(
        self,
        query_bq: bytes,
        top_k: int = 5,
        filter_expr: str | None = None,
    ) -> list[MilvusSearchResult]:
        """Search using pre-quantized binary query vector."""
        self._ensure_connected()
        
        if not self._chunks:
            return []
        
        if self._vectors_dirty or self._vectors is None:
            self._rebuild_vectors()
        
        if self._vectors is None:
            return []
        
        query_arr = np.frombuffer(query_bq, dtype=np.uint8)
        distances = self._hamming_distance_batch(query_arr, self._vectors)
        
        valid_indices = np.arange(len(self._chunks))
        if filter_expr and 'doc_id ==' in filter_expr:
            doc_id = filter_expr.split('==')[1].strip().strip('"\'')
            valid_indices = np.array([
                i for i, c in enumerate(self._chunks) if c.doc_id == doc_id
            ])
            if len(valid_indices) == 0:
                return []
            distances = distances[valid_indices]
        
        if len(distances) <= top_k:
            sorted_indices = np.argsort(distances)
        else:
            partition_indices = np.argpartition(distances, top_k)[:top_k]
            sorted_indices = partition_indices[np.argsort(distances[partition_indices])]
        
        results = []
        for idx in sorted_indices[:top_k]:
            chunk_idx = valid_indices[idx] if filter_expr else idx
            chunk = self._chunks[chunk_idx]
            results.append(MilvusSearchResult(
                id=chunk.id,
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                text=chunk.text,
                metadata=chunk.metadata,
                distance=float(distances[idx]),
            ))
        
        return results
    
    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all chunks for a document."""
        self._ensure_connected()
        
        original_count = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.doc_id != doc_id]
        deleted = original_count - len(self._chunks)
        
        if deleted > 0:
            self._vectors_dirty = True
        
        return deleted
    
    def count(self) -> int:
        """Get total number of chunks."""
        return len(self._chunks)
    
    def get_stats(self) -> IndexStats:
        """Get index statistics."""
        count = self.count()
        storage_bytes = count * self._binary_dim
        
        return IndexStats(
            count=count,
            index_type="BIN_FLAT",
            metric="HAMMING",
            dim_bits=self._embedding_dim,
            storage_estimate_bytes=storage_bytes,
            collection_name=self._config.collection_name,
            embedding_version=self._embedding_version,
            quantization_version=self._quantization_version,
        )
    
    def clear(self) -> None:
        """Clear all data."""
        self._chunks = []
        self._next_id = 1
        self._vectors = None
        self._vectors_dirty = True
    
    def set_embedding_version(self, version: str) -> None:
        """Set embedding model version for drift control."""
        self._embedding_version = version
    
    def validate_query_compatibility(
        self,
        query_dim: int,
        embedding_version: str = "",
    ) -> bool:
        """Validate query compatibility with indexed data."""
        if query_dim != self._embedding_dim:
            raise ValueError(
                f"Query dimension {query_dim} does not match index dimension {self._embedding_dim}"
            )
        
        if embedding_version and self._embedding_version:
            if embedding_version != self._embedding_version:
                raise ValueError(
                    f"Query embedding version '{embedding_version}' does not match "
                    f"index version '{self._embedding_version}'"
                )
        
        return True
    
    def _save_to_disk(self) -> None:
        """Save index to disk."""
        if not self._config.persist_path:
            return
        
        path = Path(self._config.persist_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "chunks": self._chunks,
            "next_id": self._next_id,
            "embedding_dim": self._embedding_dim,
            "embedding_version": self._embedding_version,
            "quantization_version": self._quantization_version,
            "bq_config": self._bq_config.to_dict(),
        }
        
        with open(path, "wb") as f:
            pickle.dump(data, f)
        
        print(f"Saved {len(self._chunks)} chunks to {path}")
    
    def _load_from_disk(self) -> bool:
        """Load index from disk."""
        if not self._config.persist_path:
            return False
        
        path = Path(self._config.persist_path)
        if not path.exists():
            return False
        
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            
            self._chunks = data["chunks"]
            self._next_id = data["next_id"]
            self._embedding_version = data.get("embedding_version", "")
            self._quantization_version = data.get("quantization_version", "")
            self._vectors_dirty = True
            
            print(f"Loaded {len(self._chunks)} chunks from {path}")
            return True
        except Exception as e:
            print(f"Failed to load index from {path}: {e}")
            return False


def get_milvus_store(
    config: MilvusConfig | None = None,
    embedding_dim: int = 768,
    bq_config: BQConfig | None = None,
) -> MilvusVectorStore:
    """Factory function for creating binary vector store."""
    return MilvusVectorStore(
        config=config,
        embedding_dim=embedding_dim,
        bq_config=bq_config,
    )


__all__ = [
    "MilvusConfig",
    "MilvusChunk",
    "MilvusSearchResult",
    "MilvusVectorStore",
    "IndexStats",
    "is_milvus_available",
    "get_milvus_store",
]
