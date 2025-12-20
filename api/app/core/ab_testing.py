"""A/B testing framework for comparing retrieval strategies.

This module enables experimentation with:
- Strategy comparison (hybrid vs dense vs sparse)
- Configuration tuning (chunk sizes, rerank settings)
- Statistical significance testing
"""

from __future__ import annotations

import random
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class ExperimentStatus(str, Enum):
    """Status of an experiment."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentVariant:
    """A variant in an A/B test."""
    id: str
    name: str
    description: str
    config: dict[str, Any]  # Config overrides for this variant
    weight: float = 1.0     # Traffic weight


@dataclass
class ExperimentResult:
    """Result of a single experiment trial."""
    variant_id: str
    query: str
    timestamp: datetime
    latency_ms: float
    precision: float
    recall: float
    quality_score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentStats:
    """Aggregated statistics for a variant."""
    variant_id: str
    sample_count: int
    avg_latency_ms: float
    std_latency_ms: float
    avg_precision: float
    avg_recall: float
    avg_quality: float | None

    def __post_init__(self):
        # Calculate confidence intervals
        pass


@dataclass
class Experiment:
    """An A/B test experiment."""
    id: str
    name: str
    description: str
    variants: list[ExperimentVariant]
    status: ExperimentStatus = ExperimentStatus.DRAFT
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    results: list[ExperimentResult] = field(default_factory=list)

    def get_variant(self, variant_id: str) -> ExperimentVariant | None:
        """Get variant by ID."""
        for v in self.variants:
            if v.id == variant_id:
                return v
        return None


class ABTestingFramework:
    """Framework for running A/B tests on RAG configurations."""

    def __init__(self) -> None:
        self._experiments: dict[str, Experiment] = {}
        self._active_experiment_id: str | None = None

    def create_experiment(
        self,
        name: str,
        description: str,
        variants: list[dict[str, Any]],
    ) -> Experiment:
        """Create a new experiment.

        Args:
            name: Experiment name
            description: What we're testing
            variants: List of variant configs with name, description, config

        Returns:
            Created experiment
        """
        exp_id = str(uuid.uuid4())

        variant_objects = [
            ExperimentVariant(
                id=str(uuid.uuid4()),
                name=v["name"],
                description=v.get("description", ""),
                config=v.get("config", {}),
                weight=v.get("weight", 1.0),
            )
            for v in variants
        ]

        experiment = Experiment(
            id=exp_id,
            name=name,
            description=description,
            variants=variant_objects,
        )

        self._experiments[exp_id] = experiment
        return experiment

    def start_experiment(self, experiment_id: str) -> bool:
        """Start running an experiment."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return False

        exp.status = ExperimentStatus.RUNNING
        exp.started_at = datetime.now(UTC)
        self._active_experiment_id = experiment_id
        return True

    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an experiment."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return False

        exp.status = ExperimentStatus.COMPLETED
        exp.completed_at = datetime.now(UTC)

        if self._active_experiment_id == experiment_id:
            self._active_experiment_id = None

        return True

    def select_variant(self, experiment_id: str | None = None) -> ExperimentVariant | None:
        """Select a variant for the current request.

        Uses weighted random selection based on variant weights.
        """
        exp_id = experiment_id or self._active_experiment_id
        if not exp_id:
            return None

        exp = self._experiments.get(exp_id)
        if not exp or exp.status != ExperimentStatus.RUNNING:
            return None

        # Weighted random selection
        total_weight = sum(v.weight for v in exp.variants)
        r = random.random() * total_weight

        current = 0
        for variant in exp.variants:
            current += variant.weight
            if r <= current:
                return variant

        return exp.variants[-1] if exp.variants else None

    def record_result(
        self,
        experiment_id: str,
        variant_id: str,
        query: str,
        latency_ms: float,
        precision: float,
        recall: float,
        quality_score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record result of a trial."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return

        result = ExperimentResult(
            variant_id=variant_id,
            query=query,
            timestamp=datetime.now(UTC),
            latency_ms=latency_ms,
            precision=precision,
            recall=recall,
            quality_score=quality_score,
            metadata=metadata or {},
        )

        exp.results.append(result)

    def get_variant_stats(self, experiment_id: str) -> list[ExperimentStats]:
        """Get statistics for each variant in an experiment."""
        exp = self._experiments.get(experiment_id)
        if not exp:
            return []

        stats = []
        for variant in exp.variants:
            variant_results = [r for r in exp.results if r.variant_id == variant.id]

            if not variant_results:
                stats.append(ExperimentStats(
                    variant_id=variant.id,
                    sample_count=0,
                    avg_latency_ms=0,
                    std_latency_ms=0,
                    avg_precision=0,
                    avg_recall=0,
                    avg_quality=None,
                ))
                continue

            latencies = [r.latency_ms for r in variant_results]
            precisions = [r.precision for r in variant_results]
            recalls = [r.recall for r in variant_results]
            qualities = [r.quality_score for r in variant_results if r.quality_score is not None]

            stats.append(ExperimentStats(
                variant_id=variant.id,
                sample_count=len(variant_results),
                avg_latency_ms=statistics.mean(latencies),
                std_latency_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
                avg_precision=statistics.mean(precisions),
                avg_recall=statistics.mean(recalls),
                avg_quality=statistics.mean(qualities) if qualities else None,
            ))

        return stats

    def get_winner(self, experiment_id: str, metric: str = "precision") -> ExperimentVariant | None:
        """Determine the winning variant for a metric.

        Args:
            experiment_id: Experiment to analyze
            metric: Metric to compare (precision, recall, latency, quality)

        Returns:
            Best performing variant or None
        """
        stats = self.get_variant_stats(experiment_id)
        if not stats:
            return None

        exp = self._experiments.get(experiment_id)
        if not exp:
            return None

        # Find best by metric
        best_stat = None
        for stat in stats:
            if stat.sample_count == 0:
                continue

            if best_stat is None:
                best_stat = stat
                continue

            if metric == "latency":
                # Lower is better
                if stat.avg_latency_ms < best_stat.avg_latency_ms:
                    best_stat = stat
            elif metric == "precision":
                if stat.avg_precision > best_stat.avg_precision:
                    best_stat = stat
            elif metric == "recall":
                if stat.avg_recall > best_stat.avg_recall:
                    best_stat = stat
            elif metric == "quality" and stat.avg_quality is not None:
                if best_stat.avg_quality is None or stat.avg_quality > best_stat.avg_quality:
                    best_stat = stat
        
        if not best_stat:
            return None
        
        return exp.get_variant(best_stat.variant_id)
    
    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """Get experiment by ID."""
        return self._experiments.get(experiment_id)
    
    def list_experiments(self) -> list[Experiment]:
        """List all experiments."""
        return list(self._experiments.values())


__all__ = [
    "ExperimentStatus",
    "ExperimentVariant",
    "ExperimentResult",
    "ExperimentStats",
    "Experiment",
    "ABTestingFramework",
]
