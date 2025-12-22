"""Evaluation harness for RAG quality metrics.

This module implements P2.12: Evaluation Harness
- Accuracy, citation validity, latency, cost metrics
- Regression tracking (before vs after)
- Integration with evaluator.py
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


logger = logging.getLogger("autorag.eval_harness")


@dataclass
class EvalCase:
    """A single evaluation case."""
    
    id: str
    query: str
    expected_answer: str | None = None
    expected_sources: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of evaluating a single case."""
    
    case_id: str
    query: str
    actual_answer: str
    
    # Quality metrics
    accuracy_score: float = 0.0
    citation_validity: float = 0.0
    answer_relevance: float = 0.0
    
    # Performance
    latency_ms: float = 0.0
    tokens_used: int = 0
    
    # Cost (if applicable)
    estimated_cost_usd: float = 0.0
    
    # Grounding
    grounded: bool = False
    sources_used: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "answer_preview": self.actual_answer[:100] + "..." if len(self.actual_answer) > 100 else self.actual_answer,
            "scores": {
                "accuracy": round(self.accuracy_score, 3),
                "citation_validity": round(self.citation_validity, 3),
                "answer_relevance": round(self.answer_relevance, 3),
            },
            "performance": {
                "latency_ms": round(self.latency_ms),
                "tokens": self.tokens_used,
                "cost_usd": round(self.estimated_cost_usd, 4),
            },
            "grounded": self.grounded,
            "sources_count": len(self.sources_used),
        }


@dataclass
class EvalRun:
    """A complete evaluation run."""
    
    run_id: str
    timestamp: str
    results: list[EvalResult] = field(default_factory=list)
    config_snapshot: dict = field(default_factory=dict)
    
    @property
    def avg_accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.accuracy_score for r in self.results) / len(self.results)
    
    @property
    def avg_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_ms for r in self.results) / len(self.results)
    
    @property
    def total_cost_usd(self) -> float:
        return sum(r.estimated_cost_usd for r in self.results)
    
    @property
    def grounding_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.grounded) / len(self.results)
    
    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "summary": {
                "cases": len(self.results),
                "avg_accuracy": round(self.avg_accuracy, 3),
                "avg_latency_ms": round(self.avg_latency_ms),
                "total_cost_usd": round(self.total_cost_usd, 4),
                "grounding_rate": round(self.grounding_rate, 3),
            },
            "results": [r.to_dict() for r in self.results],
            "config": self.config_snapshot,
        }


@dataclass
class RegressionComparison:
    """Comparison between two eval runs."""
    
    baseline_run_id: str
    current_run_id: str
    
    accuracy_delta: float = 0.0
    latency_delta_ms: float = 0.0
    cost_delta_usd: float = 0.0
    grounding_delta: float = 0.0
    
    regressions: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    
    @property
    def has_regression(self) -> bool:
        return len(self.regressions) > 0
    
    def to_dict(self) -> dict:
        return {
            "baseline": self.baseline_run_id,
            "current": self.current_run_id,
            "deltas": {
                "accuracy": round(self.accuracy_delta, 3),
                "latency_ms": round(self.latency_delta_ms),
                "cost_usd": round(self.cost_delta_usd, 4),
                "grounding_rate": round(self.grounding_delta, 3),
            },
            "has_regression": self.has_regression,
            "regressions": self.regressions,
            "improvements": self.improvements,
        }


class EvalHarness:
    """Run evaluations and track regression."""
    
    def __init__(
        self,
        data_path: Path | None = None,
        query_fn: Callable | None = None,
    ) -> None:
        """Initialize harness.
        
        Args:
            data_path: Path to store eval results
            query_fn: Async function to run queries
        """
        self._data_path = data_path or Path("data/evaluations")
        self._query_fn = query_fn
        self._runs: dict[str, EvalRun] = {}
    
    def load_cases(self, path: Path) -> list[EvalCase]:
        """Load eval cases from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        cases = []
        for item in data:
            cases.append(EvalCase(
                id=item["id"],
                query=item["query"],
                expected_answer=item.get("expected_answer"),
                expected_sources=item.get("expected_sources", []),
                tags=item.get("tags", []),
                metadata=item.get("metadata", {}),
            ))
        return cases
    
    async def run(
        self,
        cases: list[EvalCase],
        config: dict | None = None,
        run_id: str | None = None,
    ) -> EvalRun:
        """Run evaluation on cases.
        
        Args:
            cases: List of eval cases
            config: Optional config snapshot
            run_id: Optional run ID
            
        Returns:
            EvalRun with results
        """
        import uuid
        
        run_id = run_id or str(uuid.uuid4())[:8]
        run = EvalRun(
            run_id=run_id,
            timestamp=datetime.utcnow().isoformat(),
            config_snapshot=config or {},
        )
        
        for case in cases:
            result = await self._evaluate_case(case)
            run.results.append(result)
        
        self._runs[run_id] = run
        self._save_run(run)
        
        return run
    
    async def _evaluate_case(self, case: EvalCase) -> EvalResult:
        """Evaluate a single case."""
        start = time.perf_counter()
        
        # Run query
        answer = ""
        sources = []
        tokens = 0
        grounded = False
        
        if self._query_fn:
            try:
                result = await self._query_fn(case.query)
                answer = result.get("answer", "")
                sources = [c.get("title", "") for c in result.get("chunks", [])]
                tokens = result.get("metrics", {}).get("tokens", 0)
                grounded = result.get("grounding", {}).get("grounded", False)
            except Exception as e:
                logger.error(f"Query failed for case {case.id}: {e}")
                answer = f"[ERROR: {e}]"
        
        latency = (time.perf_counter() - start) * 1000
        
        # Compute scores
        accuracy = self._compute_accuracy(answer, case.expected_answer)
        citation_validity = self._compute_citation_validity(answer, sources)
        relevance = self._compute_relevance(answer, case.query)
        
        return EvalResult(
            case_id=case.id,
            query=case.query,
            actual_answer=answer,
            accuracy_score=accuracy,
            citation_validity=citation_validity,
            answer_relevance=relevance,
            latency_ms=latency,
            tokens_used=tokens,
            grounded=grounded,
            sources_used=sources,
        )
    
    def _compute_accuracy(self, actual: str, expected: str | None) -> float:
        """Compute accuracy score."""
        if not expected:
            return 0.5  # No ground truth
        
        # Simple term overlap
        actual_terms = set(actual.lower().split())
        expected_terms = set(expected.lower().split())
        
        if not expected_terms:
            return 0.5
        
        overlap = len(actual_terms & expected_terms)
        return min(1.0, overlap / len(expected_terms))
    
    def _compute_citation_validity(self, answer: str, sources: list[str]) -> float:
        """Check if citations reference valid sources."""
        import re
        citations = re.findall(r'\[(\d+)\]', answer)
        if not citations:
            return 1.0 if not sources else 0.5
        
        valid = sum(1 for c in citations if int(c) <= len(sources))
        return valid / len(citations) if citations else 1.0
    
    def _compute_relevance(self, answer: str, query: str) -> float:
        """Compute answer relevance to query."""
        query_terms = set(query.lower().split())
        answer_terms = set(answer.lower().split())
        
        if not query_terms:
            return 0.5
        
        overlap = len(query_terms & answer_terms)
        return min(1.0, overlap / len(query_terms))
    
    def compare(
        self,
        baseline_run_id: str,
        current_run_id: str,
    ) -> RegressionComparison:
        """Compare two runs for regression."""
        baseline = self._runs.get(baseline_run_id)
        current = self._runs.get(current_run_id)
        
        if not baseline or not current:
            return RegressionComparison(
                baseline_run_id=baseline_run_id,
                current_run_id=current_run_id,
            )
        
        comparison = RegressionComparison(
            baseline_run_id=baseline_run_id,
            current_run_id=current_run_id,
            accuracy_delta=current.avg_accuracy - baseline.avg_accuracy,
            latency_delta_ms=current.avg_latency_ms - baseline.avg_latency_ms,
            cost_delta_usd=current.total_cost_usd - baseline.total_cost_usd,
            grounding_delta=current.grounding_rate - baseline.grounding_rate,
        )
        
        # Identify regressions (worse) and improvements (better)
        if comparison.accuracy_delta < -0.05:
            comparison.regressions.append(f"Accuracy dropped {-comparison.accuracy_delta:.1%}")
        elif comparison.accuracy_delta > 0.05:
            comparison.improvements.append(f"Accuracy improved {comparison.accuracy_delta:.1%}")
        
        if comparison.latency_delta_ms > 100:
            comparison.regressions.append(f"Latency increased {comparison.latency_delta_ms:.0f}ms")
        elif comparison.latency_delta_ms < -100:
            comparison.improvements.append(f"Latency decreased {-comparison.latency_delta_ms:.0f}ms")
        
        if comparison.grounding_delta < -0.1:
            comparison.regressions.append(f"Grounding rate dropped {-comparison.grounding_delta:.1%}")
        elif comparison.grounding_delta > 0.1:
            comparison.improvements.append(f"Grounding rate improved {comparison.grounding_delta:.1%}")
        
        return comparison
    
    def _save_run(self, run: EvalRun) -> None:
        """Save run to disk."""
        self._data_path.mkdir(parents=True, exist_ok=True)
        path = self._data_path / f"{run.run_id}.json"
        with open(path, "w") as f:
            json.dump(run.to_dict(), f, indent=2)


def get_eval_harness(
    data_path: Path | None = None,
    query_fn: Callable | None = None,
) -> EvalHarness:
    """Create evaluation harness."""
    return EvalHarness(data_path=data_path, query_fn=query_fn)


__all__ = [
    "EvalCase",
    "EvalResult",
    "EvalRun",
    "RegressionComparison",
    "EvalHarness",
    "get_eval_harness",
]
