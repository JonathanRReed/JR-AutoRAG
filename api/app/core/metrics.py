"""RAG metrics collection and analysis.

This module provides comprehensive RAG evaluation metrics:
- Retrieval metrics (precision, recall, MRR, NDCG)
- Generation metrics (answer quality, faithfulness)
- Latency and performance tracking
- Aggregation and visualization helpers
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import statistics


class MetricType(str, Enum):
    """Types of metrics tracked."""
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    LATENCY = "latency"
    QUALITY = "quality"


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval quality."""
    precision_at_k: float  # Relevant / Retrieved
    recall_at_k: float     # Relevant retrieved / Total relevant
    mrr: float             # Mean Reciprocal Rank
    ndcg: float            # Normalized Discounted Cumulative Gain
    hit_rate: float        # Did we find anything relevant?
    coverage: float        # Proportion of query terms covered
    avg_score: float       # Average retrieval score
    
    def to_dict(self) -> dict[str, float]:
        return {
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "hit_rate": self.hit_rate,
            "coverage": self.coverage,
            "avg_score": self.avg_score,
        }


@dataclass
class LatencyMetrics:
    """Metrics for performance."""
    total_ms: float
    planning_ms: float
    retrieval_ms: float
    compression_ms: float
    generation_ms: float
    
    def to_dict(self) -> dict[str, float]:
        return {
            "total_ms": self.total_ms,
            "planning_ms": self.planning_ms,
            "retrieval_ms": self.retrieval_ms,
            "compression_ms": self.compression_ms,
            "generation_ms": self.generation_ms,
        }


@dataclass
class QualityMetrics:
    """Metrics for answer quality."""
    faithfulness: float      # How grounded in context (0-1)
    relevance: float         # How relevant to query (0-1)
    completeness: float      # Did it answer fully (0-1)
    coherence: float         # Is it well-structured (0-1)
    citation_accuracy: float # Are citations correct (0-1)
    
    def to_dict(self) -> dict[str, float]:
        return {
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
            "completeness": self.completeness,
            "coherence": self.coherence,
            "citation_accuracy": self.citation_accuracy,
        }
    
    @property
    def overall(self) -> float:
        """Weighted overall score."""
        weights = {
            "faithfulness": 0.3,
            "relevance": 0.3,
            "completeness": 0.2,
            "coherence": 0.1,
            "citation_accuracy": 0.1,
        }
        return sum(
            getattr(self, k) * v for k, v in weights.items()
        )


@dataclass
class EvaluationResult:
    """Complete evaluation result for a query."""
    query: str
    answer: str
    timestamp: datetime
    retrieval_metrics: RetrievalMetrics
    latency_metrics: LatencyMetrics
    quality_metrics: QualityMetrics | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "timestamp": self.timestamp.isoformat(),
            "retrieval": self.retrieval_metrics.to_dict(),
            "latency": self.latency_metrics.to_dict(),
            "quality": self.quality_metrics.to_dict() if self.quality_metrics else None,
            "metadata": self.metadata,
        }


class MetricsCalculator:
    """Calculates RAG metrics from query results."""
    
    def calculate_retrieval_metrics(
        self,
        retrieved_ids: list[str],
        relevant_ids: set[str],
        scores: list[float],
        k: int = 5,
    ) -> RetrievalMetrics:
        """Calculate retrieval metrics.
        
        Args:
            retrieved_ids: IDs of retrieved chunks
            relevant_ids: IDs of known relevant chunks
            scores: Retrieval scores for each chunk
            k: Cutoff for @k metrics
        """
        if not retrieved_ids:
            return RetrievalMetrics(
                precision_at_k=0, recall_at_k=0, mrr=0, ndcg=0,
                hit_rate=0, coverage=0, avg_score=0
            )
        
        retrieved_k = retrieved_ids[:k]
        
        # Precision@k
        relevant_retrieved = sum(1 for rid in retrieved_k if rid in relevant_ids)
        precision = relevant_retrieved / len(retrieved_k) if retrieved_k else 0
        
        # Recall@k
        recall = relevant_retrieved / len(relevant_ids) if relevant_ids else 0
        
        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        for i, rid in enumerate(retrieved_ids):
            if rid in relevant_ids:
                mrr = 1.0 / (i + 1)
                break
        
        # NDCG (simplified)
        import math
        dcg = 0.0
        for i, rid in enumerate(retrieved_k):
            if rid in relevant_ids:
                dcg += 1.0 / math.log2(i + 2)  # +2 because rank starts at 1
        
        # Ideal DCG (all relevant at top)
        ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant_ids), k)))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0
        
        # Hit rate
        hit_rate = 1.0 if relevant_retrieved > 0 else 0.0
        
        # Coverage (simplified - proportion of chunks with positive score)
        positive_scores = sum(1 for s in scores if s > 0)
        coverage = positive_scores / len(scores) if scores else 0
        
        # Average score
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return RetrievalMetrics(
            precision_at_k=precision,
            recall_at_k=recall,
            mrr=mrr,
            ndcg=ndcg,
            hit_rate=hit_rate,
            coverage=coverage,
            avg_score=avg_score,
        )
    
    def calculate_latency_metrics(
        self,
        steps: list[dict[str, Any]],
    ) -> LatencyMetrics:
        """Calculate latency metrics from pipeline steps."""
        step_times = {s.get("name", ""): s.get("duration_ms", 0) for s in steps}
        
        return LatencyMetrics(
            total_ms=sum(step_times.values()),
            planning_ms=step_times.get("planning", 0),
            retrieval_ms=step_times.get("retrieval", 0),
            compression_ms=step_times.get("compression", 0),
            generation_ms=step_times.get("generation", 0),
        )


@dataclass
class MetricsAggregation:
    """Aggregated metrics over multiple evaluations."""
    count: int
    avg_retrieval: dict[str, float]
    avg_latency: dict[str, float]
    avg_quality: dict[str, float] | None
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float


class MetricsStore:
    """Stores and aggregates evaluation metrics."""
    
    def __init__(self, max_history: int = 1000) -> None:
        self._evaluations: list[EvaluationResult] = []
        self._max_history = max_history
    
    def record(self, evaluation: EvaluationResult) -> None:
        """Record an evaluation result."""
        self._evaluations.append(evaluation)
        
        # Trim if over limit
        if len(self._evaluations) > self._max_history:
            self._evaluations = self._evaluations[-self._max_history:]
    
    def get_recent(self, n: int = 10) -> list[EvaluationResult]:
        """Get recent evaluations."""
        return self._evaluations[-n:]
    
    def aggregate(
        self,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> MetricsAggregation:
        """Aggregate metrics over time period."""
        evaluations = self._evaluations
        
        if since:
            evaluations = [e for e in evaluations if e.timestamp >= since]
        
        if limit:
            evaluations = evaluations[-limit:]
        
        if not evaluations:
            return MetricsAggregation(
                count=0,
                avg_retrieval={},
                avg_latency={},
                avg_quality=None,
                p50_latency_ms=0,
                p95_latency_ms=0,
                p99_latency_ms=0,
            )
        
        # Calculate averages
        retrieval_keys = ["precision_at_k", "recall_at_k", "mrr", "ndcg", "hit_rate"]
        avg_retrieval = {}
        for key in retrieval_keys:
            values = [getattr(e.retrieval_metrics, key) for e in evaluations]
            avg_retrieval[key] = sum(values) / len(values)
        
        latency_keys = ["total_ms", "planning_ms", "retrieval_ms", "generation_ms"]
        avg_latency = {}
        for key in latency_keys:
            values = [getattr(e.latency_metrics, key) for e in evaluations]
            avg_latency[key] = sum(values) / len(values)
        
        # Quality metrics (if available)
        quality_evals = [e for e in evaluations if e.quality_metrics]
        avg_quality = None
        if quality_evals:
            quality_keys = ["faithfulness", "relevance", "completeness", "coherence"]
            avg_quality = {}
            for key in quality_keys:
                values = [getattr(e.quality_metrics, key) for e in quality_evals]
                avg_quality[key] = sum(values) / len(values)
        
        # Latency percentiles
        latencies = sorted(e.latency_metrics.total_ms for e in evaluations)
        n = len(latencies)
        
        def percentile(p: float) -> float:
            idx = int(n * p / 100)
            return latencies[min(idx, n - 1)]
        
        return MetricsAggregation(
            count=len(evaluations),
            avg_retrieval=avg_retrieval,
            avg_latency=avg_latency,
            avg_quality=avg_quality,
            p50_latency_ms=percentile(50),
            p95_latency_ms=percentile(95),
            p99_latency_ms=percentile(99),
        )
    
    def to_dashboard_data(self) -> dict[str, Any]:
        """Format metrics for dashboard display."""
        agg = self.aggregate(limit=100)
        recent = self.get_recent(10)
        
        return {
            "summary": {
                "total_queries": agg.count,
                "avg_precision": agg.avg_retrieval.get("precision_at_k", 0),
                "avg_latency_ms": agg.avg_latency.get("total_ms", 0),
                "p95_latency_ms": agg.p95_latency_ms,
            },
            "retrieval_metrics": agg.avg_retrieval,
            "latency_metrics": agg.avg_latency,
            "quality_metrics": agg.avg_quality,
            "recent_queries": [
                {
                    "query": e.query[:100],
                    "latency_ms": e.latency_metrics.total_ms,
                    "precision": e.retrieval_metrics.precision_at_k,
                }
                for e in recent
            ],
        }


__all__ = [
    "MetricType",
    "RetrievalMetrics",
    "LatencyMetrics",
    "QualityMetrics",
    "EvaluationResult",
    "MetricsCalculator",
    "MetricsAggregation",
    "MetricsStore",
]
