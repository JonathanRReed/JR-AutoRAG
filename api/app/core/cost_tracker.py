"""Cost and Latency Tracker: Monitor RAG pipeline costs and performance.

Provides real-time tracking of:
- Token usage and costs per provider
- Latency breakdowns by stage
- Budget enforcement and alerts
- Historical aggregations

Enables cost-aware routing and budget management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any


@dataclass
class TokenUsage:
    """Token usage for a single request."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )


@dataclass
class CostEstimate:
    """Cost estimate for a request."""
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    currency: str = "USD"


@dataclass
class LatencyBreakdown:
    """Latency breakdown by pipeline stage."""
    total_ms: float = 0.0
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    reranking_ms: float = 0.0
    embedding_ms: float = 0.0
    other_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "total": self.total_ms,
            "retrieval": self.retrieval_ms,
            "generation": self.generation_ms,
            "reranking": self.reranking_ms,
            "embedding": self.embedding_ms,
            "other": self.other_ms,
        }


@dataclass
class RequestMetrics:
    """Complete metrics for a single request."""
    request_id: str
    timestamp: datetime
    tokens: TokenUsage
    cost: CostEstimate
    latency: LatencyBreakdown
    provider: str = ""
    model: str = ""
    preset: str = ""
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics over a time period."""
    period_start: datetime
    period_end: datetime
    total_requests: int = 0
    successful_requests: int = 0
    total_tokens: TokenUsage = field(default_factory=TokenUsage)
    total_cost: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    by_provider: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_preset: dict[str, dict[str, Any]] = field(default_factory=dict)


# Cost per 1K tokens by provider/model (simplified pricing)
DEFAULT_PRICING = {
    "openai": {
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    },
    "anthropic": {
        "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    },
    "google": {
        "gemini-pro": {"input": 0.00025, "output": 0.0005},
        "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    },
    "ollama": {
        "default": {"input": 0.0, "output": 0.0},  # Local, no cost
    },
}


class CostLatencyTracker:
    """Track costs and latency for RAG pipeline.

    Features:
    - Per-request token and cost tracking
    - Latency breakdowns by stage
    - Budget alerts and enforcement
    - Aggregated reporting over time windows
    """

    def __init__(
        self,
        pricing: dict[str, dict[str, dict[str, float]]] | None = None,
        budget_limit: float | None = None,
        budget_window_hours: int = 24,
        max_history_size: int = 10000,
    ) -> None:
        """Initialize tracker.

        Args:
            pricing: Custom pricing table
            budget_limit: Optional spending limit for the window
            budget_window_hours: Budget window in hours
            max_history_size: Maximum requests to keep in history
        """
        self.pricing = pricing or DEFAULT_PRICING
        self.budget_limit = budget_limit
        self.budget_window = timedelta(hours=budget_window_hours)
        self.max_history_size = max_history_size

        self._history: list[RequestMetrics] = []
        self._current_window_cost = 0.0
        self._window_start = datetime.now(UTC)

    def estimate_cost(
        self,
        tokens: TokenUsage,
        provider: str,
        model: str,
    ) -> CostEstimate:
        """Estimate cost for token usage."""
        provider_pricing = self.pricing.get(provider.lower(), {})

        # Find model pricing (partial match)
        model_pricing = None
        for model_key, pricing in provider_pricing.items():
            if model_key in model.lower() or model.lower() in model_key:
                model_pricing = pricing
                break

        if not model_pricing:
            model_pricing = provider_pricing.get("default", {"input": 0.0, "output": 0.0})

        input_cost = (tokens.input_tokens / 1000) * model_pricing.get("input", 0.0)
        output_cost = (tokens.output_tokens / 1000) * model_pricing.get("output", 0.0)

        return CostEstimate(
            input_cost=round(input_cost, 6),
            output_cost=round(output_cost, 6),
            total_cost=round(input_cost + output_cost, 6),
        )

    def record_request(
        self,
        request_id: str,
        tokens: TokenUsage,
        latency: LatencyBreakdown,
        provider: str = "",
        model: str = "",
        preset: str = "",
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> RequestMetrics:
        """Record metrics for a completed request.

        Returns:
            RequestMetrics with cost estimate
        """
        cost = self.estimate_cost(tokens, provider, model)

        metrics = RequestMetrics(
            request_id=request_id,
            timestamp=datetime.now(UTC),
            tokens=tokens,
            cost=cost,
            latency=latency,
            provider=provider,
            model=model,
            preset=preset,
            success=success,
            metadata=metadata or {},
        )

        self._history.append(metrics)

        # Trim history if needed
        if len(self._history) > self.max_history_size:
            self._history = self._history[-self.max_history_size:]

        # Update budget tracking
        self._update_budget_window(cost.total_cost)

        return metrics

    def _update_budget_window(self, cost: float) -> None:
        """Update budget window tracking."""
        now = datetime.now(UTC)

        # Check if we need to reset window
        if now - self._window_start > self.budget_window:
            self._window_start = now
            self._current_window_cost = 0.0

        self._current_window_cost += cost

    def is_budget_exceeded(self) -> bool:
        """Check if budget limit is exceeded."""
        if self.budget_limit is None:
            return False
        return self._current_window_cost >= self.budget_limit

    def get_budget_remaining(self) -> float | None:
        """Get remaining budget in current window."""
        if self.budget_limit is None:
            return None
        return max(0.0, self.budget_limit - self._current_window_cost)

    def aggregate(
        self,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> AggregatedMetrics:
        """Aggregate metrics over a time period.

        Args:
            start: Start of period (default: 24 hours ago)
            end: End of period (default: now)

        Returns:
            AggregatedMetrics with summaries
        """
        end = end or datetime.now(UTC)
        start = start or (end - timedelta(hours=24))

        # Filter to time window
        filtered = [
            m for m in self._history
            if start <= m.timestamp <= end
        ]

        if not filtered:
            return AggregatedMetrics(
                period_start=start,
                period_end=end,
            )

        # Calculate aggregates
        total_tokens = TokenUsage()
        total_cost = 0.0
        latencies = []
        by_provider: dict[str, dict[str, Any]] = {}
        by_preset: dict[str, dict[str, Any]] = {}

        for m in filtered:
            total_tokens = total_tokens + m.tokens
            total_cost += m.cost.total_cost
            latencies.append(m.latency.total_ms)

            # By provider
            if m.provider:
                if m.provider not in by_provider:
                    by_provider[m.provider] = {"requests": 0, "cost": 0.0, "tokens": 0}
                by_provider[m.provider]["requests"] += 1
                by_provider[m.provider]["cost"] += m.cost.total_cost
                by_provider[m.provider]["tokens"] += m.tokens.total_tokens

            # By preset
            if m.preset:
                if m.preset not in by_preset:
                    by_preset[m.preset] = {"requests": 0, "avg_latency": 0.0}
                by_preset[m.preset]["requests"] += 1

        # Sort latencies for percentiles
        latencies.sort()
        n = len(latencies)

        return AggregatedMetrics(
            period_start=start,
            period_end=end,
            total_requests=len(filtered),
            successful_requests=sum(1 for m in filtered if m.success),
            total_tokens=total_tokens,
            total_cost=round(total_cost, 4),
            avg_latency_ms=round(sum(latencies) / n, 2) if n else 0.0,
            p50_latency_ms=round(latencies[n // 2], 2) if n else 0.0,
            p95_latency_ms=round(latencies[int(n * 0.95)], 2) if n else 0.0,
            p99_latency_ms=round(latencies[int(n * 0.99)], 2) if n else 0.0,
            by_provider=by_provider,
            by_preset=by_preset,
        )

    def get_recent(self, count: int = 10) -> list[RequestMetrics]:
        """Get the most recent requests."""
        return self._history[-count:]

    def clear_history(self) -> None:
        """Clear all history."""
        self._history.clear()
        self._current_window_cost = 0.0
        self._window_start = datetime.now(UTC)


# Singleton
_tracker: CostLatencyTracker | None = None


def get_cost_tracker(
    budget_limit: float | None = None,
) -> CostLatencyTracker:
    """Get or create the cost/latency tracker."""
    global _tracker
    if _tracker is None:
        _tracker = CostLatencyTracker(budget_limit=budget_limit)
    return _tracker


__all__ = [
    "TokenUsage",
    "CostEstimate",
    "LatencyBreakdown",
    "RequestMetrics",
    "AggregatedMetrics",
    "CostLatencyTracker",
    "get_cost_tracker",
]
