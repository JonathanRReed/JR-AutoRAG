"""Caching layer for embeddings and query results.

This module provides:
- LRU cache for embeddings
- Query result caching
- TTL-based expiration
- Memory-efficient storage
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar


T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """A cached item with metadata."""
    key: str
    value: T
    created_at: float
    ttl_seconds: float | None
    hit_count: int = 0
    
    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
    
    def touch(self) -> None:
        """Record a cache hit."""
        self.hit_count += 1


class LRUCache(Generic[T]):
    """Least Recently Used cache with optional TTL."""
    
    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: float | None = 3600,  # 1 hour default
    ) -> None:
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._hits = 0
        self._misses = 0
    
    def _make_key(self, key: str) -> str:
        """Create a cache key."""
        return key
    
    def get(self, key: str) -> T | None:
        """Get item from cache."""
        cache_key = self._make_key(key)
        
        if cache_key not in self._cache:
            self._misses += 1
            return None
        
        entry = self._cache[cache_key]
        
        # Check expiration
        if entry.is_expired:
            del self._cache[cache_key]
            self._misses += 1
            return None
        
        # Move to end (most recently used)
        self._cache.move_to_end(cache_key)
        entry.touch()
        self._hits += 1
        
        return entry.value
    
    def set(self, key: str, value: T, ttl: float | None = None) -> None:
        """Set item in cache."""
        cache_key = self._make_key(key)
        
        # Remove if exists
        if cache_key in self._cache:
            del self._cache[cache_key]
        
        # Evict oldest if at capacity
        while len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)
        
        # Add new entry
        self._cache[cache_key] = CacheEntry(
            key=cache_key,
            value=value,
            created_at=time.time(),
            ttl_seconds=ttl if ttl is not None else self._default_ttl,
        )
    
    def delete(self, key: str) -> bool:
        """Delete item from cache."""
        cache_key = self._make_key(key)
        if cache_key in self._cache:
            del self._cache[cache_key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cached items."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        expired = [
            key for key, entry in self._cache.items()
            if entry.is_expired
        ]
        for key in expired:
            del self._cache[key]
        return len(expired)
    
    @property
    def size(self) -> int:
        return len(self._cache)
    
    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    def stats(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }


class EmbeddingCache:
    """Specialized cache for embeddings."""
    
    def __init__(
        self,
        max_size: int = 5000,
        ttl_seconds: float = 86400,  # 24 hours
    ) -> None:
        self._cache: LRUCache[list[float]] = LRUCache(
            max_size=max_size,
            default_ttl=ttl_seconds,
        )
    
    def _hash_text(self, text: str) -> str:
        """Create hash key for text."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def get(self, text: str) -> list[float] | None:
        """Get cached embedding for text."""
        key = self._hash_text(text)
        return self._cache.get(key)
    
    def set(self, text: str, embedding: list[float]) -> None:
        """Cache embedding for text."""
        key = self._hash_text(text)
        self._cache.set(key, embedding)
    
    def get_many(self, texts: list[str]) -> list[list[float] | None]:
        """Get cached embeddings for multiple texts."""
        return [self.get(text) for text in texts]
    
    def set_many(self, texts: list[str], embeddings: list[list[float]]) -> None:
        """Cache multiple embeddings."""
        for text, embedding in zip(texts, embeddings):
            self.set(text, embedding)
    
    def clear(self) -> None:
        self._cache.clear()
    
    def stats(self) -> dict[str, Any]:
        return self._cache.stats()


class QueryCache:
    """Cache for query results."""
    
    def __init__(
        self,
        max_size: int = 500,
        ttl_seconds: float = 1800,  # 30 minutes
    ) -> None:
        self._cache: LRUCache[dict[str, Any]] = LRUCache(
            max_size=max_size,
            default_ttl=ttl_seconds,
        )
    
    def _make_key(self, query: str, config_hash: str = "") -> str:
        """Create cache key from query and config."""
        combined = f"{query}:{config_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]
    
    def get(self, query: str, config_hash: str = "") -> dict[str, Any] | None:
        """Get cached query result."""
        key = self._make_key(query, config_hash)
        return self._cache.get(key)
    
    def set(
        self,
        query: str,
        result: dict[str, Any],
        config_hash: str = "",
    ) -> None:
        """Cache query result."""
        key = self._make_key(query, config_hash)
        self._cache.set(key, result)
    
    def invalidate_all(self) -> None:
        """Invalidate all cached queries."""
        self._cache.clear()

    def clear(self) -> None:
        """Compatibility wrapper for clearing cached queries."""
        self._cache.clear()
    
    def stats(self) -> dict[str, Any]:
        return self._cache.stats()


class CacheManager:
    """Unified cache management."""
    
    def __init__(
        self,
        embedding_cache_size: int = 5000,
        query_cache_size: int = 500,
        embedding_ttl: float = 86400,
        query_ttl: float = 1800,
    ) -> None:
        self.embeddings = EmbeddingCache(
            max_size=embedding_cache_size,
            ttl_seconds=embedding_ttl,
        )
        self.queries = QueryCache(
            max_size=query_cache_size,
            ttl_seconds=query_ttl,
        )
    
    def clear_all(self) -> None:
        """Clear all caches."""
        self.embeddings.clear()
        self.queries.clear()
    
    def cleanup(self) -> dict[str, int]:
        """Cleanup expired entries in all caches."""
        return {
            "embeddings": self.embeddings._cache.cleanup_expired(),
            "queries": self.queries._cache.cleanup_expired(),
        }
    
    def stats(self) -> dict[str, Any]:
        return {
            "embeddings": self.embeddings.stats(),
            "queries": self.queries.stats(),
        }


# Global cache manager instance
_cache_manager: CacheManager | None = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


__all__ = [
    "CacheEntry",
    "LRUCache",
    "EmbeddingCache",
    "QueryCache",
    "CacheManager",
    "get_cache_manager",
]
