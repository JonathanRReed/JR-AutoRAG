"""Trace bundle export for offline reproducibility.

Implements Workstream E1: Trace bundles that include config hash, corpus_version,
mode flags, per-step timings, evaluator verdicts, and citation-check results.

These bundles enable:
- Offline reproduction of query decisions
- Debugging and auditing of pipeline behavior
- Comparison of results across configuration changes
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class TraceBundle:
    """Complete trace bundle for reproducibility.

    Contains all information needed to understand and reproduce
    a query's execution through the RAG pipeline.
    """
    # Query and response
    query: str
    answer: str

    # Version and configuration tracking
    corpus_version: str
    config_hash: str
    retrieval_mode_flags: int

    # Pipeline execution details
    steps: list[dict[str, Any]] = field(default_factory=list)

    # Evaluation and verification
    evaluator_verdicts: dict[str, str] = field(default_factory=dict)
    citation_check: dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    total_duration_ms: float = 0.0

    # Metadata
    created_at: str = ""
    bundle_version: str = "1.0"

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Export as formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_compact_json(self) -> str:
        """Export as compact JSON (no indentation)."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, json_str: str) -> TraceBundle:
        """Create TraceBundle from JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TraceBundle:
        """Create TraceBundle from dictionary."""
        return cls(**data)


def create_trace_bundle(
    query: str,
    answer: str,
    steps: list[dict[str, Any]],
    corpus_version: str,
    config_hash: str,
    retrieval_mode: int,
    evaluator_verdicts: dict[str, str] | None = None,
    citation_check: dict[str, Any] | None = None,
    total_duration_ms: float = 0.0,
) -> TraceBundle:
    """Factory function to create a trace bundle from pipeline execution.

    Args:
        query: Original user query
        answer: Generated answer
        steps: List of pipeline step dictionaries with timing info
        corpus_version: Current corpus version for cache invalidation
        config_hash: Hash of retrieval configuration
        retrieval_mode: Bitmask of RetrievalMode flags
        evaluator_verdicts: Dict of evaluator names to verdict strings
        citation_check: Citation verification results dict
        total_duration_ms: Total pipeline execution time

    Returns:
        Populated TraceBundle ready for export
    """
    return TraceBundle(
        query=query,
        answer=answer,
        corpus_version=corpus_version,
        config_hash=config_hash,
        retrieval_mode_flags=retrieval_mode,
        steps=steps,
        evaluator_verdicts=evaluator_verdicts or {},
        citation_check=citation_check or {},
        total_duration_ms=total_duration_ms,
    )


def summarize_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Create a summary of pipeline steps for quick inspection.

    Args:
        steps: List of step dictionaries

    Returns:
        Summary dict with step names, statuses, and timing breakdown
    """
    summary: dict[str, Any] = {
        "step_count": len(steps),
        "steps": [],
        "total_duration_ms": 0.0,
        "slowest_step": None,
    }

    max_duration = 0.0
    for step in steps:
        step_name = step.get("name", "unknown")
        duration = step.get("duration_ms", 0.0)
        status = step.get("status", "unknown")

        summary["steps"].append({
            "name": step_name,
            "status": status,
            "duration_ms": duration,
        })
        summary["total_duration_ms"] += duration

        if duration > max_duration:
            max_duration = duration
            summary["slowest_step"] = step_name

    return summary


__all__ = [
    "TraceBundle",
    "create_trace_bundle",
    "summarize_steps",
]
