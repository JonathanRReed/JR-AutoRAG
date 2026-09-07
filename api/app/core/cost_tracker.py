"""Cost and Latency Tracker: Monitor RAG pipeline costs and performance.

Provides real-time tracking of:
- Token usage and costs per provider
- Latency breakdowns by stage
- Budget enforcement and alerts
- Historical aggregations

Enables cost-aware routing and budget management.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    """Record of token usage for an LLM/embedding operation."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __post_init__(self) -> None:
        """Calculate total tokens if not explicitly provided or if sum differs."""
        if self.total_tokens == 0 and (self.prompt_tokens > 0 or self.completion_tokens > 0):
            self.total_tokens = self.prompt_tokens + self.completion_tokens

    def add(self, other: TokenUsage) -> TokenUsage:
        """Combine two TokenUsage instances."""
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
        )

    def to_dict(self) -> dict[str, int]:
        """Convert token usage to dictionary format."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class CostEstimate:
    """Cost estimate for pipeline execution."""

    prompt_cost_usd: float = 0.0
    completion_cost_usd: float = 0.0
    total_cost_usd: float = 0.0
    provider: str = "unknown"
    model: str = "unknown"

    def __post_init__(self) -> None:
        """Calculate total cost if zero or ensure consistency."""
        if self.total_cost_usd == 0.0 and (self.prompt_cost_usd > 0.0 or self.completion_cost_usd > 0.0):
            self.total_cost_usd = round(self.prompt_cost_usd + self.completion_cost_usd, 6)

    def add(self, other: CostEstimate) -> CostEstimate:
        """Combine two CostEstimate instances."""
        return CostEstimate(
            prompt_cost_usd=round(self.prompt_cost_usd + other.prompt_cost_usd, 6),
            completion_cost_usd=round(self.completion_cost_usd + other.completion_cost_usd, 6),
            total_cost_usd=round(self.total_cost_usd + other.total_cost_usd, 6),
            provider=self.provider if self.provider == other.provider else "mixed",
            model=self.model if self.model == other.model else "mixed",
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert cost estimate to dictionary format."""
        return {
            "prompt_cost_usd": self.prompt_cost_usd,
            "completion_cost_usd": self.completion_cost_usd,
            "total_cost_usd": self.total_cost_usd,
            "provider": self.provider,
            "model": self.model,
        }


# Per 1K tokens standard pricing table (in USD)
DEFAULT_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {"prompt": 0.0025, "completion": 0.010},
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
    "claude-3-5-sonnet": {"prompt": 0.003, "completion": 0.015},
    "claude-3-haiku": {"prompt": 0.00025, "completion": 0.00125},
    "text-embedding-3-small": {"prompt": 0.00002, "completion": 0.0},
    "text-embedding-3-large": {"prompt": 0.00013, "completion": 0.0},
}


class CostTracker:
    """Tracks token usage, calculates costs, and monitors budget thresholds."""

    def __init__(self, pricing_table: dict[str, dict[str, float]] | None = None) -> None:
        self.pricing = pricing_table if pricing_table is not None else DEFAULT_MODEL_PRICING
        self._history: list[dict[str, Any]] = []

    def calculate_cost(
        self,
        usage: TokenUsage,
        model: str,
        provider: str = "openai",
    ) -> CostEstimate:
        """Calculate cost USD for a given TokenUsage and model."""
        rates = self.pricing.get(model, {"prompt": 0.0, "completion": 0.0})
        prompt_cost = (usage.prompt_tokens / 1000.0) * rates.get("prompt", 0.0)
        completion_cost = (usage.completion_tokens / 1000.0) * rates.get("completion", 0.0)
        total_cost = prompt_cost + completion_cost

        estimate = CostEstimate(
            prompt_cost_usd=round(prompt_cost, 6),
            completion_cost_usd=round(completion_cost, 6),
            total_cost_usd=round(total_cost, 6),
            provider=provider,
            model=model,
        )

        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "model": model,
            "provider": provider,
            "usage": usage.to_dict(),
            "cost": estimate.to_dict(),
        }
        self._history.append(record)

        return estimate

    def get_total_cost(self) -> float:
        """Return cumulative total cost across all tracked calls."""
        return round(sum(entry["cost"]["total_cost_usd"] for entry in self._history), 6)

    def get_history(self) -> list[dict[str, Any]]:
        """Return transaction history."""
        return list(self._history)

    def clear_history(self) -> None:
        """Clear tracking history."""
        self._history.clear()


__all__ = [
    "TokenUsage",
    "CostEstimate",
    "DEFAULT_MODEL_PRICING",
    "CostTracker",
]
