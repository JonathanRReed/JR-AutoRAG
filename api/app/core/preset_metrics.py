"""Preset metrics tracking for latency estimation.

This module implements P1.7: Presets Show Changes
- Track last N runs per preset
- Estimate latency and token ranges
- Provide calibration data for UI
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

logger = logging.getLogger("autorag.preset_metrics")


@dataclass
class RunMetrics:
    """Metrics for a single query run."""

    preset: str
    total_duration_ms: float
    tokens_used: int
    chunks_retrieved: int
    cache_hit: bool
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "preset": self.preset,
            "total_duration_ms": self.total_duration_ms,
            "tokens_used": self.tokens_used,
            "chunks_retrieved": self.chunks_retrieved,
            "cache_hit": self.cache_hit,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RunMetrics:
        return cls(
            preset=data["preset"],
            total_duration_ms=data["total_duration_ms"],
            tokens_used=data["tokens_used"],
            chunks_retrieved=data["chunks_retrieved"],
            cache_hit=data.get("cache_hit", False),
            timestamp=data.get("timestamp", time.time()),
        )


@dataclass
class PresetEstimate:
    """Estimated latency and token ranges for a preset."""

    preset: str
    latency_p50_ms: float
    latency_p90_ms: float
    latency_min_ms: float
    latency_max_ms: float
    tokens_avg: int
    tokens_min: int
    tokens_max: int
    sample_count: int

    def to_dict(self) -> dict:
        return {
            "preset": self.preset,
            "latency": {
                "p50_ms": round(self.latency_p50_ms),
                "p90_ms": round(self.latency_p90_ms),
                "min_ms": round(self.latency_min_ms),
                "max_ms": round(self.latency_max_ms),
            },
            "tokens": {
                "avg": self.tokens_avg,
                "min": self.tokens_min,
                "max": self.tokens_max,
            },
            "sample_count": self.sample_count,
        }


class PresetMetricsTracker:
    """Track per-preset metrics for latency estimation.

    Maintains a sliding window of the last N runs per preset,
    and computes latency/token estimates on demand.
    """

    def __init__(
        self,
        max_runs_per_preset: int = 20,
        data_path: Path | None = None,
    ) -> None:
        """Initialize the tracker.

        Args:
            max_runs_per_preset: Max runs to keep per preset
            data_path: Optional path for persisting metrics
        """
        self._max_runs = max_runs_per_preset
        self._data_path = data_path
        self._runs: dict[str, deque[RunMetrics]] = {}
        self._lock = Lock()

        # Load persisted data if available
        if self._data_path and self._data_path.exists():
            self._load()

    def record(self, metrics: RunMetrics) -> None:
        """Record a run's metrics."""
        with self._lock:
            if metrics.preset not in self._runs:
                self._runs[metrics.preset] = deque(maxlen=self._max_runs)

            self._runs[metrics.preset].append(metrics)

            # Persist if path configured
            if self._data_path:
                self._save()

    def record_from_result(
        self,
        preset: str,
        result: dict,
    ) -> None:
        """Record metrics from a query result dict."""
        metrics_data = result.get("metrics", {})

        # Extract relevant metrics
        duration = metrics_data.get("duration_ms") or metrics_data.get("total_duration_ms", 0)
        tokens = metrics_data.get("tokens", 0)
        chunks = len(result.get("chunks", []))
        cache_hit = result.get("from_cache", False) or metrics_data.get("cache_hit", False)

        self.record(RunMetrics(
            preset=preset,
            total_duration_ms=float(duration),
            tokens_used=int(tokens),
            chunks_retrieved=chunks,
            cache_hit=cache_hit,
        ))

    def get_estimate(self, preset: str) -> PresetEstimate | None:
        """Get latency/token estimates for a preset."""
        with self._lock:
            runs = self._runs.get(preset)
            if not runs or len(runs) < 3:
                return None

            # Filter out cache hits for latency estimation
            non_cached = [r for r in runs if not r.cache_hit]
            if len(non_cached) < 2:
                non_cached = list(runs)  # Fall back to all runs

            latencies = sorted(r.total_duration_ms for r in non_cached)
            tokens = [r.tokens_used for r in runs]

            # Calculate percentiles
            n = len(latencies)
            p50_idx = n // 2
            p90_idx = int(n * 0.9)

            return PresetEstimate(
                preset=preset,
                latency_p50_ms=latencies[p50_idx],
                latency_p90_ms=latencies[min(p90_idx, n - 1)],
                latency_min_ms=min(latencies),
                latency_max_ms=max(latencies),
                tokens_avg=sum(tokens) // len(tokens),
                tokens_min=min(tokens),
                tokens_max=max(tokens),
                sample_count=len(runs),
            )

    def get_all_estimates(self) -> dict[str, PresetEstimate]:
        """Get estimates for all tracked presets."""
        results = {}
        with self._lock:
            for preset in self._runs:
                est = self.get_estimate(preset)
                if est:
                    results[preset] = est
        return results

    def _save(self) -> None:
        """Persist metrics to disk."""
        if not self._data_path:
            return

        try:
            data = {
                preset: [m.to_dict() for m in runs]
                for preset, runs in self._runs.items()
            }
            self._data_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._data_path, "w") as f:
                json.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save preset metrics: {e}")

    def _load(self) -> None:
        """Load metrics from disk."""
        if not self._data_path or not self._data_path.exists():
            return

        try:
            with open(self._data_path) as f:
                data = json.load(f)

            for preset, runs_data in data.items():
                self._runs[preset] = deque(
                    (RunMetrics.from_dict(r) for r in runs_data),
                    maxlen=self._max_runs,
                )
        except Exception as e:
            logger.warning(f"Failed to load preset metrics: {e}")

    def clear(self, preset: str | None = None) -> None:
        """Clear metrics for a preset or all presets."""
        with self._lock:
            if preset:
                self._runs.pop(preset, None)
            else:
                self._runs.clear()

            if self._data_path:
                self._save()


# Global instance
_tracker: PresetMetricsTracker | None = None


def get_preset_metrics_tracker(data_path: Path | None = None) -> PresetMetricsTracker:
    """Get or create the global preset metrics tracker."""
    global _tracker
    if _tracker is None:
        default_path = Path("data/preset_metrics.json") if data_path is None else data_path
        _tracker = PresetMetricsTracker(data_path=default_path)
    return _tracker


__all__ = [
    "RunMetrics",
    "PresetEstimate",
    "PresetMetricsTracker",
    "get_preset_metrics_tracker",
]
