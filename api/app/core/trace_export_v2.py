"""Enhanced trace export with config snapshots.

This module implements P2.13: Trace Export Improvements
- Include config snapshot, corpus hash, model IDs, stage timings
- Shareable trace summary (redacted option)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


logger = logging.getLogger("autorag.trace_export")


@dataclass
class TraceExport:
    """Enhanced trace export with full context."""
    
    trace_id: str
    query: str
    answer: str
    timestamp: str
    
    # Stage timings
    stages: list[dict] = field(default_factory=list)
    total_duration_ms: float = 0
    
    # Config snapshot
    config: dict = field(default_factory=dict)
    corpus_hash: str = ""
    
    # Model info
    model_ids: dict = field(default_factory=dict)
    
    # Result metadata
    chunks_used: int = 0
    citations_count: int = 0
    grounded: bool = False
    cache_hit: bool = False
    
    def to_dict(self, redact_content: bool = False) -> dict:
        """Export as dictionary.
        
        Args:
            redact_content: If True, redact query/answer text
        """
        query = "[REDACTED]" if redact_content else self.query
        answer = "[REDACTED]" if redact_content else self.answer
        
        return {
            "meta": {
                "trace_id": self.trace_id,
                "timestamp": self.timestamp,
                "version": "3.0",
            },
            "query": query,
            "answer": answer,
            "execution": {
                "total_duration_ms": round(self.total_duration_ms),
                "stages": self.stages,
            },
            "context": {
                "corpus_hash": self.corpus_hash,
                "model_ids": self.model_ids,
                "config_snapshot": self.config if not redact_content else "[REDACTED]",
            },
            "metrics": {
                "chunks_used": self.chunks_used,
                "citations_count": self.citations_count,
                "grounded": self.grounded,
                "cache_hit": self.cache_hit,
            },
        }
    
    def to_json(self, redact_content: bool = False, indent: int = 2) -> str:
        """Export as JSON string."""
        return json.dumps(self.to_dict(redact_content), indent=indent)
    
    def to_summary(self, redact_content: bool = False) -> str:
        """Generate human-readable summary."""
        lines = [
            f"# Trace Summary: {self.trace_id}",
            f"**Timestamp**: {self.timestamp}",
            f"**Duration**: {self.total_duration_ms:.0f}ms",
            "",
            "## Query",
            self.query if not redact_content else "[REDACTED]",
            "",
            "## Answer",
            (self.answer[:200] + "..." if len(self.answer) > 200 else self.answer) 
                if not redact_content else "[REDACTED]",
            "",
            "## Metrics",
            f"- Chunks used: {self.chunks_used}",
            f"- Citations: {self.citations_count}",
            f"- Grounded: {self.grounded}",
            f"- Cache hit: {self.cache_hit}",
            "",
            "## Stages",
        ]
        
        for stage in self.stages:
            name = stage.get("name", "unknown")
            duration = stage.get("duration_ms", 0)
            status = stage.get("status", "unknown")
            lines.append(f"- {name}: {duration:.0f}ms ({status})")
        
        lines.extend([
            "",
            "## Models",
            f"- Planner: {self.model_ids.get('planner', 'N/A')}",
            f"- Generator: {self.model_ids.get('generator', 'N/A')}",
        ])
        
        return "\n".join(lines)


def compute_corpus_hash(documents: list) -> str:
    """Compute hash of corpus for versioning."""
    if not documents:
        return "empty"
    
    # Hash based on doc IDs and content hashes
    parts = []
    for doc in documents:
        doc_id = getattr(doc, 'id', None) or doc.get('id', str(id(doc)))
        text = getattr(doc, 'text', None) or doc.get('text', '')
        text_hash = hashlib.md5(text[:1000].encode()).hexdigest()[:8]
        parts.append(f"{doc_id}:{text_hash}")
    
    combined = "|".join(sorted(parts))
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


def create_trace_export(
    trace_id: str,
    query: str,
    answer: str,
    pipeline_steps: list,
    config: dict,
    model_ids: dict,
    chunks: list = None,
    documents: list = None,
) -> TraceExport:
    """Create enhanced trace export from query result.
    
    Args:
        trace_id: Unique trace identifier
        query: Original query
        answer: Generated answer
        pipeline_steps: List of PipelineStep dicts
        config: Current configuration
        model_ids: Model IDs used (planner, generator, etc.)
        chunks: Evidence chunks used
        documents: Corpus documents for hashing
        
    Returns:
        TraceExport ready for serialization
    """
    # Extract stage timings
    stages = []
    total_ms = 0
    for step in pipeline_steps:
        if isinstance(step, dict):
            stages.append({
                "name": step.get("name", "unknown"),
                "duration_ms": step.get("duration_ms", 0),
                "status": step.get("status", "unknown"),
            })
            total_ms += step.get("duration_ms", 0)
        else:
            # Handle PipelineStep dataclass
            stages.append({
                "name": getattr(step, "name", "unknown"),
                "duration_ms": getattr(step, "duration_ms", 0),
                "status": getattr(step, "status", "unknown"),
            })
            total_ms += getattr(step, "duration_ms", 0)
    
    # Count citations in answer
    import re
    citations = len(set(re.findall(r'\[(\d+)\]', answer)))
    
    return TraceExport(
        trace_id=trace_id,
        query=query,
        answer=answer,
        timestamp=datetime.utcnow().isoformat(),
        stages=stages,
        total_duration_ms=total_ms,
        config=config,
        corpus_hash=compute_corpus_hash(documents or []),
        model_ids=model_ids,
        chunks_used=len(chunks) if chunks else 0,
        citations_count=citations,
        grounded=citations > 0 and (chunks and len(chunks) > 0),
        cache_hit=False,  # Set by caller
    )


__all__ = [
    "TraceExport",
    "compute_corpus_hash",
    "create_trace_export",
]
