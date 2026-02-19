"""Corpus health dashboard API.

This module implements P2.15: Ingestion & Corpus Health Dashboard
- Corpus stats: doc count, chunk count, avg chunk size, embedding status
- Dedupe detection
- Health indicators
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .retrieval import HybridRetrievalEngine

logger = logging.getLogger("autorag.corpus_health")


@dataclass
class CorpusStats:
    """Statistics about the corpus."""

    document_count: int = 0
    chunk_count: int = 0
    avg_chunk_size: int = 0
    total_tokens: int = 0
    embedding_status: str = "unknown"  # ready, building, missing
    index_status: str = "unknown"  # ready, stale, missing

    # Advanced stats
    duplicate_chunks: int = 0
    failed_documents: list[str] = field(default_factory=list)
    last_updated: str | None = None

    def to_dict(self) -> dict:
        return {
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
            "avg_chunk_size": self.avg_chunk_size,
            "total_tokens": self.total_tokens,
            "embedding_status": self.embedding_status,
            "index_status": self.index_status,
            "duplicate_chunks": self.duplicate_chunks,
            "failed_documents": self.failed_documents[:10],  # Limit for API
            "failed_count": len(self.failed_documents),
            "last_updated": self.last_updated,
        }


@dataclass
class HealthCheck:
    """Result of a health check."""

    name: str
    status: str  # healthy, warning, critical
    message: str
    metric: float | int | None = None
    recommendation: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "message": self.message,
            "metric": self.metric,
            "recommendation": self.recommendation,
        }


@dataclass
class CorpusHealthReport:
    """Complete corpus health report."""

    overall_status: str  # healthy, warning, critical
    stats: CorpusStats
    checks: list[HealthCheck]
    recommendations: list[str]

    @property
    def healthy_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "healthy")

    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "warning")

    @property
    def critical_count(self) -> int:
        return sum(1 for c in self.checks if c.status == "critical")

    def to_dict(self) -> dict:
        return {
            "overall_status": self.overall_status,
            "summary": f"{self.healthy_count} healthy, {self.warning_count} warnings, {self.critical_count} critical",
            "stats": self.stats.to_dict(),
            "checks": [c.to_dict() for c in self.checks],
            "recommendations": self.recommendations[:5],
        }


class CorpusHealthChecker:
    """Check corpus health and generate recommendations."""

    def __init__(self, retrieval: HybridRetrievalEngine | None = None) -> None:
        self._retrieval = retrieval

    def get_stats(self) -> CorpusStats:
        """Get current corpus statistics."""
        stats = CorpusStats()

        if not self._retrieval:
            return stats

        # Get document count
        if hasattr(self._retrieval, '_documents'):
            stats.document_count = len(self._retrieval._documents)

        # Get chunk count
        if hasattr(self._retrieval, '_chunks'):
            chunks = self._retrieval._chunks
            stats.chunk_count = len(chunks)

            if chunks:
                sizes = [len(c.text.split()) for c in chunks if hasattr(c, 'text')]
                if sizes:
                    stats.avg_chunk_size = sum(sizes) // len(sizes)
                    stats.total_tokens = int(sum(sizes) * 1.3)  # Rough token estimate

        # Get embedding status
        if hasattr(self._retrieval, 'get_model_status'):
            model_status = self._retrieval.get_model_status()
            stats.embedding_status = model_status.get('embedding_status', 'unknown')

        # Get index status
        if hasattr(self._retrieval, '_faiss') and self._retrieval._faiss is not None:
            stats.index_status = "ready"
        else:
            stats.index_status = "missing"

        return stats

    def run_checks(self) -> list[HealthCheck]:
        """Run all health checks."""
        checks = []
        stats = self.get_stats()

        # Check 1: Document count
        if stats.document_count == 0:
            checks.append(HealthCheck(
                name="Document Count",
                status="critical",
                message="No documents loaded",
                metric=0,
                recommendation="Ingest some documents to get started",
            ))
        elif stats.document_count < 5:
            checks.append(HealthCheck(
                name="Document Count",
                status="warning",
                message=f"Only {stats.document_count} documents - small corpus",
                metric=stats.document_count,
                recommendation="Consider adding more documents for better coverage",
            ))
        else:
            checks.append(HealthCheck(
                name="Document Count",
                status="healthy",
                message=f"{stats.document_count} documents loaded",
                metric=stats.document_count,
            ))

        # Check 2: Chunk size
        if stats.avg_chunk_size > 0:
            if stats.avg_chunk_size < 50:
                checks.append(HealthCheck(
                    name="Chunk Size",
                    status="warning",
                    message=f"Chunks are very small ({stats.avg_chunk_size} words avg)",
                    metric=stats.avg_chunk_size,
                    recommendation="Consider larger chunk sizes for better context",
                ))
            elif stats.avg_chunk_size > 500:
                checks.append(HealthCheck(
                    name="Chunk Size",
                    status="warning",
                    message=f"Chunks are large ({stats.avg_chunk_size} words avg)",
                    metric=stats.avg_chunk_size,
                    recommendation="Consider smaller chunks for more precise retrieval",
                ))
            else:
                checks.append(HealthCheck(
                    name="Chunk Size",
                    status="healthy",
                    message=f"Good chunk size ({stats.avg_chunk_size} words avg)",
                    metric=stats.avg_chunk_size,
                ))

        # Check 3: Index status
        if stats.index_status == "ready":
            checks.append(HealthCheck(
                name="Vector Index",
                status="healthy",
                message="Index is ready",
            ))
        else:
            checks.append(HealthCheck(
                name="Vector Index",
                status="critical",
                message="Index not built",
                recommendation="Ingest documents to build index",
            ))

        # Check 4: Embedding model
        if stats.embedding_status == "ready":
            checks.append(HealthCheck(
                name="Embedding Model",
                status="healthy",
                message="Model loaded and ready",
            ))
        elif stats.embedding_status == "missing":
            checks.append(HealthCheck(
                name="Embedding Model",
                status="critical",
                message="Embedding model not loaded",
                recommendation="Download embedding model in settings",
            ))
        else:
            checks.append(HealthCheck(
                name="Embedding Model",
                status="warning",
                message=f"Status: {stats.embedding_status}",
            ))

        return checks

    def generate_report(self) -> CorpusHealthReport:
        """Generate complete health report."""
        stats = self.get_stats()
        checks = self.run_checks()

        # Determine overall status
        if any(c.status == "critical" for c in checks):
            overall_status = "critical"
        elif any(c.status == "warning" for c in checks):
            overall_status = "warning"
        else:
            overall_status = "healthy"

        # Collect recommendations
        recommendations = [
            c.recommendation for c in checks
            if c.recommendation is not None
        ]

        return CorpusHealthReport(
            overall_status=overall_status,
            stats=stats,
            checks=checks,
            recommendations=recommendations,
        )


# Global instance
_health_checker: CorpusHealthChecker | None = None


def get_corpus_health_checker(
    retrieval: HybridRetrievalEngine | None = None,
) -> CorpusHealthChecker:
    """Get or create global health checker."""
    global _health_checker
    if _health_checker is None or retrieval is not None:
        _health_checker = CorpusHealthChecker(retrieval)
    return _health_checker


__all__ = [
    "CorpusStats",
    "HealthCheck",
    "CorpusHealthReport",
    "CorpusHealthChecker",
    "get_corpus_health_checker",
]
