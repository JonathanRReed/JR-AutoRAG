"""Telemetry storage module."""

from __future__ import annotations

import builtins
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
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


PUBLIC_PROVIDER_ERROR_MESSAGE = (
    "Provider request failed; check server logs for details."
)


def sanitize_step_details(
    step_name: str, details: dict[str, Any] | None
) -> dict[str, Any]:
    """Return step details safe to persist or expose through public APIs.

    Generation steps can otherwise carry provider endpoints, deployment/model IDs,
    or raw upstream exception text. Keep operational status fields while replacing
    sensitive connection metadata with coarse, non-identifying values.
    """
    safe_details = dict(details or {})
    if not step_name.startswith("generation"):
        return safe_details

    provider = safe_details.get("provider")
    if provider not in (None, "none", "unknown"):
        safe_details["provider"] = "configured"

    model = safe_details.get("model")
    if model not in (None, "unknown"):
        safe_details["model"] = "configured"

    for key in list(safe_details):
        if "error" in key.lower():
            safe_details[key] = PUBLIC_PROVIDER_ERROR_MESSAGE

    return safe_details


def pipeline_step_to_public_dict(step: PipelineStep) -> dict[str, Any]:
    """Serialize a pipeline step with sanitized details for responses/traces."""
    return {
        "name": step.name,
        "duration_ms": step.duration_ms,
        "details": sanitize_step_details(step.name, step.details),
        "status": step.status,
        "started_at": step.started_at.isoformat(),
        "completed_at": step.completed_at.isoformat(),
    }


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
            details=sanitize_step_details(data["name"], data.get("details", {})),
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
                step_data["details"] = sanitize_step_details(step.name, step.details)
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
            now = datetime.now(UTC)
            safe_steps = [
                PipelineStep(
                    name=step.name,
                    started_at=step.started_at,
                    completed_at=step.completed_at,
                    duration_ms=step.duration_ms,
                    details=sanitize_step_details(step.name, step.details),
                    status=step.status,
                )
                for step in (steps or [])
            ]
            trace = Trace(
                id=str(uuid.uuid4()),
                started_at=started_at or now,
                completed_at=now,
                prompt=prompt,
                answer=answer,
                metrics=metrics or {},
                steps=safe_steps,
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
            embedding_hits = sum(
                int(t.metrics.get("embedding_cache_hits", 0)) for t in self._traces
            )
            embedding_misses = sum(
                int(t.metrics.get("embedding_cache_misses", 0)) for t in self._traces
            )
            embedding_total = embedding_hits + embedding_misses
            cache_hits = sum(
                1 for t in self._traces if t.metrics.get("embedding_cache") == "hit"
            )
            cache_hit_rate = (
                embedding_hits / embedding_total
                if embedding_total > 0
                else (cache_hits / len(self._traces) if self._traces else 0)
            )

            return {
                "total_queries": len(self._traces),
                "cache_hit_rate": cache_hit_rate,
                "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
                "p50_latency_ms": sorted(latencies)[len(latencies) // 2]
                if latencies
                else 0,
                "p95_latency_ms": sorted(latencies)[int(len(latencies) * 0.95)]
                if latencies
                else 0,
                "avg_chunks_per_query": sum(chunk_counts) / len(chunk_counts)
                if chunk_counts
                else 0,
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
        dist = {"high": 0, "medium": 0, "low": 0, "insufficient": 0, "unknown": 0}
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

    def get_stage_latency_breakdown(self) -> dict[str, float]:
        """Get average latency per pipeline stage.

        Returns a dict mapping stage name to average duration in ms.
        Useful for identifying bottlenecks.
        """
        with self._lock:
            stage_totals: dict[str, float] = {}
            stage_counts: dict[str, int] = {}

            for trace in self._traces:
                for step in trace.steps:
                    name = step.name
                    stage_totals[name] = stage_totals.get(name, 0) + step.duration_ms
                    stage_counts[name] = stage_counts.get(name, 0) + 1

            return {
                name: stage_totals[name] / stage_counts[name]
                for name in stage_totals
                if stage_counts[name] > 0
            }

    def get_stage_latency_percentiles(self) -> dict[str, dict[str, float]]:
        """Get latency percentiles per pipeline stage."""

        def percentile(values: list[float], pct: float) -> float:
            if not values:
                return 0.0
            ordered = sorted(values)
            if len(ordered) == 1:
                return ordered[0]
            k = (len(ordered) - 1) * pct
            f = int(k)
            c = min(f + 1, len(ordered) - 1)
            if f == c:
                return ordered[f]
            return ordered[f] + (ordered[c] - ordered[f]) * (k - f)

        with self._lock:
            stage_values: dict[str, list[float]] = {}
            for trace in self._traces:
                for step in trace.steps:
                    stage_values.setdefault(step.name, []).append(step.duration_ms)

            output: dict[str, dict[str, float]] = {}
            for name, values in stage_values.items():
                output[name] = {
                    "p50_ms": round(percentile(values, 0.5), 2),
                    "p95_ms": round(percentile(values, 0.95), 2),
                    "p99_ms": round(percentile(values, 0.99), 2),
                }
            return output

    def get_retrieval_mode_distribution(self) -> dict[str, int]:
        """Distribution of retrieval modes used (standard, RAPTOR, GraphRAG)."""
        with self._lock:
            dist = {"standard": 0, "raptor": 0, "graph": 0, "combined": 0}

            for trace in self._traces:
                mode = trace.metrics.get("retrieval_mode", "standard")
                if mode in dist:
                    dist[mode] += 1
                else:
                    dist["standard"] += 1

            return dist

    def get_flare_trigger_rate(self) -> float:
        """Percentage of queries where FLARE triggered mid-generation retrieval."""
        with self._lock:
            if not self._traces:
                return 0.0

            flare_count = 0
            for trace in self._traces:
                # Check if any FLARE step triggered retrieval
                for step in trace.steps:
                    if (
                        step.name == "flare"
                        and step.details.get("retrievals_triggered", 0) > 0
                    ):
                        flare_count += 1
                        break
                    # Also check metrics
                    if trace.metrics.get("flare_retrievals", 0) > 0:
                        flare_count += 1
                        break

            return flare_count / len(self._traces)

    def get_hallucination_pass_rate(self) -> float:
        """Average hallucination firewall pass rate."""
        with self._lock:
            if not self._traces:
                return 0.0

            pass_rates = []
            for trace in self._traces:
                rate = trace.metrics.get("firewall_pass_rate")
                if rate is not None:
                    pass_rates.append(rate)

            return sum(pass_rates) / len(pass_rates) if pass_rates else 0.0

    def get_full_metrics_export(self) -> dict:
        """Export all metrics for comprehensive dashboard.

        Combines export_metrics with additional stage and feature metrics.
        """
        base = self.export_metrics()
        base.update(
            {
                "stage_latency_breakdown": self.get_stage_latency_breakdown(),
                "stage_latency_percentiles": self.get_stage_latency_percentiles(),
                "retrieval_mode_distribution": self.get_retrieval_mode_distribution(),
                "flare_trigger_rate": self.get_flare_trigger_rate(),
                "hallucination_pass_rate": self.get_hallucination_pass_rate(),
            }
        )
        return base
