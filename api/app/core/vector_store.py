"""Vector store abstraction and ChromaDB integration.

This module provides:
- Abstract vector store interface
- ChromaDB implementation for persistent storage
- In-memory fallback for development
- Collection management
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING
import uuid

if TYPE_CHECKING:
    pass


@dataclass
class VectorDocument:
    """A document with embedding for vector store."""
    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorSearchResult:
    """Result from vector search."""
    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(ABC):
    """Abstract interface for vector stores."""
    
    @abstractmethod
    def add(self, documents: list[VectorDocument]) -> list[str]:
        """Add documents to the store. Returns IDs."""
        pass
    
    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    def delete(self, ids: list[str]) -> int:
        """Delete documents by ID. Returns count deleted."""
        pass
    
    @abstractmethod
    def get(self, ids: list[str]) -> list[VectorDocument | None]:
        """Get documents by ID."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Get total document count."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all documents."""
        pass


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store for development."""
    
    def __init__(self) -> None:
        self._documents: dict[str, VectorDocument] = {}
    
    def add(self, documents: list[VectorDocument]) -> list[str]:
        ids = []
        for doc in documents:
            doc_id = doc.id or str(uuid.uuid4())
            self._documents[doc_id] = doc
            ids.append(doc_id)
        return ids
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b) or not a:
            return 0.0
        
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot / (norm_a * norm_b)
    
    def search(
        self,
        query_embedding: list[float],
        k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        results = []
        
        for doc_id, doc in self._documents.items():
            # Apply filter
            if filter:
                match = all(
                    doc.metadata.get(key) == value
                    for key, value in filter.items()
                )
                if not match:
                    continue
            
            score = self._cosine_similarity(query_embedding, doc.embedding)
            results.append(VectorSearchResult(
                id=doc_id,
                text=doc.text,
                score=score,
                metadata=doc.metadata,
            ))
        
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:k]
    
    def delete(self, ids: list[str]) -> int:
        count = 0
        for doc_id in ids:
            if doc_id in self._documents:
                del self._documents[doc_id]
                count += 1
        return count
    
    def get(self, ids: list[str]) -> list[VectorDocument | None]:
        return [self._documents.get(doc_id) for doc_id in ids]
    
    def count(self) -> int:
        return len(self._documents)
    
    def clear(self) -> None:
        self._documents.clear()


class ChromaVectorStore(VectorStore):
    """ChromaDB-backed vector store for production."""
    
    def __init__(
        self,
        collection_name: str = "jr_autorag",
        persist_directory: str | None = None,
    ) -> None:
        self._collection_name = collection_name
        self._persist_directory = persist_directory
        self._client = None
        self._collection = None
        self._initialized = False
    
    def _ensure_initialized(self) -> None:
        """Lazy initialization of ChromaDB."""
        if self._initialized:
            return
        
        try:
            import chromadb
            from chromadb.config import Settings
            
            if self._persist_directory:
                self._client = chromadb.Client(Settings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=self._persist_directory,
                    anonymized_telemetry=False,
                ))
            else:
                self._client = chromadb.Client()
            
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._initialized = True
        except ImportError:
            raise RuntimeError(
                "ChromaDB not installed. Install with: pip install chromadb"
            )
    
    def add(self, documents: list[VectorDocument]) -> list[str]:
        self._ensure_initialized()
        
        ids = [doc.id or str(uuid.uuid4()) for doc in documents]
        
        self._collection.add(
            ids=ids,
            documents=[doc.text for doc in documents],
            embeddings=[doc.embedding for doc in documents],
            metadatas=[doc.metadata for doc in documents],
        )
        
        return ids
    
    def search(
        self,
        query_embedding: list[float],
        k: int = 10,
        filter: dict[str, Any] | None = None,
    ) -> list[VectorSearchResult]:
        self._ensure_initialized()
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filter,
        )
        
        search_results = []
        if results and results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                search_results.append(VectorSearchResult(
                    id=doc_id,
                    text=results["documents"][0][i] if results["documents"] else "",
                    score=1 - results["distances"][0][i] if results["distances"] else 0,
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                ))
        
        return search_results
    
    def delete(self, ids: list[str]) -> int:
        self._ensure_initialized()
        
        try:
            self._collection.delete(ids=ids)
            return len(ids)
        except Exception:
            return 0
    
    def get(self, ids: list[str]) -> list[VectorDocument | None]:
        self._ensure_initialized()
        
        results = self._collection.get(ids=ids, include=["documents", "embeddings", "metadatas"])
        
        docs = []
        for i, doc_id in enumerate(results["ids"]):
            docs.append(VectorDocument(
                id=doc_id,
                text=results["documents"][i] if results["documents"] else "",
                embedding=results["embeddings"][i] if results["embeddings"] else [],
                metadata=results["metadatas"][i] if results["metadatas"] else {},
            ))
        
        return docs
    
    def count(self) -> int:
        self._ensure_initialized()
        return self._collection.count()
    
    def clear(self) -> None:
        self._ensure_initialized()
        
        # Delete and recreate collection
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )
    
    def persist(self) -> None:
        """Persist data to disk (if using persistent storage)."""
        if self._client and self._persist_directory:
            self._client.persist()


def get_vector_store(
    backend: str = "memory",
    **kwargs,
) -> VectorStore:
    """Factory for creating vector stores.
    
    Args:
        backend: "memory" or "chromadb"
        **kwargs: Backend-specific options
    
    Returns:
        VectorStore instance
    """
    if backend == "chromadb":
        return ChromaVectorStore(**kwargs)
    else:
        return InMemoryVectorStore()


__all__ = [
    "VectorDocument",
    "VectorSearchResult",
    "VectorStore",
    "InMemoryVectorStore",
    "ChromaVectorStore",
    "get_vector_store",
]
