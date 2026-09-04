"""RAG decision logging for future fine-tuning.

Logs all RAG pipeline decisions for training data collection:
- Gate decisions (retrieve/no-retrieve/clarify)
- Route decisions (single/iterative/graph/raptor)
- Retrieval evaluator verdicts
- Answer quality assessments

This data can be used to fine-tune:
- Gating models (when to retrieve)
- Routing models (how to retrieve)
- Rerankers (with hard negatives)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class GateDecisionLog:
    """Log of a gating decision."""

    query: str
    decision: str  # no_retrieval, single, iterative, clarify
    confidence: float
    reasoning: str
    outcome_quality: float | None = None  # Filled after answer evaluation
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class RouteDecisionLog:
    """Log of a routing decision."""

    query: str
    decision: str  # single, iterative, graph, raptor, hybrid_heavy
    features: dict[str, Any]
    suggested_k: int
    use_rerank: bool
    max_iterations: int
    outcome_quality: float | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class RetrievalVerdictLog:
    """Log of retrieval quality verdict."""

    query: str
    verdict: str  # correct, ambiguous, incorrect, low_coverage
    confidence: float
    chunks_count: int
    iteration: int
    corrective_action: str | None = None
    coverage_ratio: float | None = None
    missing_aspects: list[str] = field(default_factory=list)
    outcome_helped: bool | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class AnswerQualityLog:
    """Log of answer quality assessment."""

    query: str
    answer_length: int
    chunks_used: int
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float
    overall_score: float
    reflection_quality: str
    should_retry: bool
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


@dataclass
class HardNegativeExample:
    """Example for hard-negative training of reranker."""

    query: str
    positive_chunk: str  # Chunk that led to good answer
    positive_score: float
    hard_negative_chunk: str  # Chunk that was retrieved but not helpful
    hard_negative_score: float
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class RAGDecisionLogger:
    """Logger for RAG pipeline decisions for training data collection.

    Collects:
    - Gate decisions for teaching when to retrieve
    - Route decisions for teaching how to retrieve
    - Retrieval verdicts for teaching quality assessment
    - Answer quality for reward signals
    - Hard negatives for reranker training
    """

    def __init__(self, log_dir: str | Path = "data/rag_training_logs"):
        """Initialize logger with output directory.

        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # In-memory buffers
        self._gate_logs: list[GateDecisionLog] = []
        self._route_logs: list[RouteDecisionLog] = []
        self._verdict_logs: list[RetrievalVerdictLog] = []
        self._quality_logs: list[AnswerQualityLog] = []
        self._hard_negatives: list[HardNegativeExample] = []

        # Auto-flush threshold
        self._flush_threshold = 100

    def log_gate_decision(
        self,
        query: str,
        decision: str,
        confidence: float,
        reasoning: str,
    ) -> None:
        """Log a gating decision."""
        log = GateDecisionLog(
            query=query,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
        )
        self._gate_logs.append(log)
        self._maybe_flush()

    def log_route_decision(
        self,
        query: str,
        decision: str,
        features: dict[str, Any],
        suggested_k: int,
        use_rerank: bool,
        max_iterations: int,
    ) -> None:
        """Log a routing decision."""
        log = RouteDecisionLog(
            query=query,
            decision=decision,
            features=features,
            suggested_k=suggested_k,
            use_rerank=use_rerank,
            max_iterations=max_iterations,
        )
        self._route_logs.append(log)
        self._maybe_flush()

    def log_retrieval_verdict(
        self,
        query: str,
        verdict: str,
        confidence: float,
        chunks_count: int,
        iteration: int,
        corrective_action: str | None = None,
        coverage_ratio: float | None = None,
        missing_aspects: list[str] | None = None,
    ) -> None:
        """Log a retrieval quality verdict."""
        log = RetrievalVerdictLog(
            query=query,
            verdict=verdict,
            confidence=confidence,
            chunks_count=chunks_count,
            iteration=iteration,
            corrective_action=corrective_action,
            coverage_ratio=coverage_ratio,
            missing_aspects=missing_aspects or [],
        )
        self._verdict_logs.append(log)
        self._maybe_flush()

    def log_answer_quality(
        self,
        query: str,
        answer_length: int,
        chunks_used: int,
        faithfulness: float,
        answer_relevance: float,
        context_precision: float,
        context_recall: float,
        overall_score: float,
        reflection_quality: str,
        should_retry: bool,
    ) -> None:
        """Log answer quality assessment."""
        log = AnswerQualityLog(
            query=query,
            answer_length=answer_length,
            chunks_used=chunks_used,
            faithfulness=faithfulness,
            answer_relevance=answer_relevance,
            context_precision=context_precision,
            context_recall=context_recall,
            overall_score=overall_score,
            reflection_quality=reflection_quality,
            should_retry=should_retry,
        )
        self._quality_logs.append(log)
        self._maybe_flush()

    def log_hard_negative(
        self,
        query: str,
        positive_chunk: str,
        positive_score: float,
        hard_negative_chunk: str,
        hard_negative_score: float,
    ) -> None:
        """Log a hard negative example for reranker training."""
        example = HardNegativeExample(
            query=query,
            positive_chunk=positive_chunk,
            positive_score=positive_score,
            hard_negative_chunk=hard_negative_chunk,
            hard_negative_score=hard_negative_score,
        )
        self._hard_negatives.append(example)
        self._maybe_flush()

    def update_outcome(
        self,
        query: str,
        outcome_quality: float,
    ) -> None:
        """Update logs with final outcome quality.

        Links quality back to decisions for reward signals.
        """
        # Update gate logs
        for log in self._gate_logs[-10:]:  # Check recent logs
            if log.query == query and log.outcome_quality is None:
                log.outcome_quality = outcome_quality
                break

        # Update route logs
        for log in self._route_logs[-10:]:
            if log.query == query and log.outcome_quality is None:
                log.outcome_quality = outcome_quality
                break

    def _maybe_flush(self) -> None:
        """Flush to disk if threshold reached."""
        total = (
            len(self._gate_logs)
            + len(self._route_logs)
            + len(self._verdict_logs)
            + len(self._quality_logs)
            + len(self._hard_negatives)
        )
        if total >= self._flush_threshold:
            self.flush()

    def flush(self) -> None:
        """Flush all logs to disk."""
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        if self._gate_logs:
            self._write_logs(f"gate_decisions_{timestamp}.jsonl", self._gate_logs)
            self._gate_logs = []

        if self._route_logs:
            self._write_logs(f"route_decisions_{timestamp}.jsonl", self._route_logs)
            self._route_logs = []

        if self._verdict_logs:
            self._write_logs(
                f"retrieval_verdicts_{timestamp}.jsonl", self._verdict_logs
            )
            self._verdict_logs = []

        if self._quality_logs:
            self._write_logs(f"answer_quality_{timestamp}.jsonl", self._quality_logs)
            self._quality_logs = []

        if self._hard_negatives:
            self._write_logs(f"hard_negatives_{timestamp}.jsonl", self._hard_negatives)
            self._hard_negatives = []

    def _write_logs(self, filename: str, logs: list) -> None:
        """Write logs to JSONL file."""
        filepath = self.log_dir / filename
        with open(filepath, "a") as f:
            for log in logs:
                f.write(json.dumps(asdict(log)) + "\n")

    def get_stats(self) -> dict[str, int]:
        """Get logging statistics."""
        return {
            "gate_logs": len(self._gate_logs),
            "route_logs": len(self._route_logs),
            "verdict_logs": len(self._verdict_logs),
            "quality_logs": len(self._quality_logs),
            "hard_negatives": len(self._hard_negatives),
        }


# Global logger instance
_decision_logger: RAGDecisionLogger | None = None


def get_decision_logger(log_dir: str = "data/rag_training_logs") -> RAGDecisionLogger:
    """Get or create the global decision logger."""
    global _decision_logger
    if _decision_logger is None:
        _decision_logger = RAGDecisionLogger(log_dir)
    return _decision_logger


__all__ = [
    "GateDecisionLog",
    "RouteDecisionLog",
    "RetrievalVerdictLog",
    "AnswerQualityLog",
    "HardNegativeExample",
    "RAGDecisionLogger",
    "get_decision_logger",
]
