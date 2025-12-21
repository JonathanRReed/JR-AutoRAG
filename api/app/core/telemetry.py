"""Telemetry storage module."""

from __future__ import annotations

import builtins
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any


@dataclass
class PipelineStep:
    """A single step in the RAG pipeline with timing and details."""
    name: str
    started_at: datetime
    completed_at: datetime
    duration_ms: float
    details: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"


@dataclass
class Trace:
    id: str
    started_at: datetime
    completed_at: datetime
    prompt: str
    answer: str
    metrics: dict[str, Any]
    steps: list[PipelineStep] = field(default_factory=list)


class TelemetryStore:
    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path or Path.cwd() / "data" / "traces.json")
        self._lock = RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._traces: list[Trace] = []
        if self._path.exists():
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._traces = [self._decode(item) for item in raw]

    def _decode_step(self, data: dict[str, Any]) -> PipelineStep:
        return PipelineStep(
            name=data["name"],
            started_at=datetime.fromisoformat(data["started_at"]),
            completed_at=datetime.fromisoformat(data["completed_at"]),
            duration_ms=data.get("duration_ms", 0.0),
            details=data.get("details", {}),
            status=data.get("status", "completed"),
        )

    def _decode(self, payload: dict[str, Any]) -> Trace:
        steps = [self._decode_step(s) for s in payload.get("steps", [])]
        return Trace(
            id=payload["id"],
            started_at=datetime.fromisoformat(payload["started_at"]),
            completed_at=datetime.fromisoformat(payload["completed_at"]),
            prompt=payload["prompt"],
            answer=payload["answer"],
            metrics=payload.get("metrics", {}),
            steps=steps,
        )

    def _persist(self) -> None:
        payload = []
        for trace in self._traces:
            data = asdict(trace)
            data["started_at"] = trace.started_at.isoformat()
            data["completed_at"] = trace.completed_at.isoformat()
            # Serialize steps with ISO timestamps
            data["steps"] = []
            for step in trace.steps:
                step_data = asdict(step)
                step_data["started_at"] = step.started_at.isoformat()
                step_data["completed_at"] = step.completed_at.isoformat()
                data["steps"].append(step_data)
            payload.append(data)
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def record(
        self,
        prompt: str,
        answer: str,
        metrics: dict[str, Any] | None = None,
        steps: builtins.list[PipelineStep] | None = None,
        started_at: datetime | None = None,
    ) -> Trace:
        with self._lock:
            now = datetime.utcnow()
            trace = Trace(
                id=str(uuid.uuid4()),
                started_at=started_at or now,
                completed_at=now,
                prompt=prompt,
                answer=answer,
                metrics=metrics or {},
                steps=steps or [],
            )
            self._traces.append(trace)
            self._persist()
            return trace

    def list(self) -> builtins.list[Trace]:
        with self._lock:
            return list(self._traces)

    def export_metrics(self) -> dict:
        """Export aggregated metrics for dashboard."""
        with self._lock:
            if not self._traces:
                return {"total_queries": 0}
            
            latencies = [t.metrics.get("duration_ms", 0) for t in self._traces]
            chunk_counts = [t.metrics.get("context_chunks", 0) for t in self._traces]
            cache_hits = sum(1 for t in self._traces if t.metrics.get("embedding_cache") == "hit")
            
            return {
                "total_queries": len(self._traces),
                "cache_hit_rate": cache_hits / len(self._traces) if self._traces else 0,
                "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
                "p50_latency_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0,
                "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
                "avg_chunks_per_query": sum(chunk_counts) / len(chunk_counts) if chunk_counts else 0,
                "queries_per_hour": self._calculate_queries_per_hour(),
                "quality_distribution": self._calculate_quality_distribution(),
                "rerank_usage_rate": self._calculate_rerank_usage(),
            }

    def _calculate_queries_per_hour(self) -> float:
        """Calculate queries per hour over last 24h."""
        if len(self._traces) < 2:
            return 0.0
        # Sort traces by time just in case
        sorted_traces = sorted(self._traces, key=lambda t: t.started_at)
        first = sorted_traces[0].started_at
        last = sorted_traces[-1].started_at
        
        # Calculate timespan in hours
        hours = (last - first).total_seconds() / 3600
        return len(self._traces) / max(hours, 0.01)

    def _calculate_quality_distribution(self) -> dict:
        """Distribution of answer quality ratings."""
        dist = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
        for t in self._traces:
            quality = t.metrics.get("quality_rating", "unknown")
            if quality in dist:
                dist[quality] += 1
            else:
                dist["unknown"] += 1
        return dist

    def _calculate_rerank_usage(self) -> float:
        """Percentage of queries using reranking or advanced retrieval."""
        advanced_count = 0
        for t in self._traces:
            used_rerank = False
            for step in t.steps:
                if step.name == "retrieval" and step.details.get("reranked", False):
                    used_rerank = True
                    break
            if used_rerank:
                advanced_count += 1
        return advanced_count / len(self._traces) if self._traces else 0.0
