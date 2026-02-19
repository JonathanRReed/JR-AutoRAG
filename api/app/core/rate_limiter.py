"""Token bucket rate limiter for API protection.

Provides simple rate limiting for:
- Per-user request limits
- Global request limits
- Configurable rates and burst capacity

Designed for local/internal hosting with in-memory state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class TokenBucket:
    """Token bucket state for rate limiting."""
    tokens: float
    last_update: float
    capacity: float
    refill_rate: float  # tokens per second

    def update_and_consume(self, tokens_needed: float = 1.0) -> bool:
        """Update bucket and try to consume tokens.

        Returns True if tokens were available and consumed.
        """
        now = time.time()
        elapsed = now - self.last_update

        # Refill tokens based on elapsed time
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_update = now

        # Try to consume
        if self.tokens >= tokens_needed:
            self.tokens -= tokens_needed
            return True
        return False

    @property
    def wait_time(self) -> float:
        """Time until next token is available."""
        if self.tokens >= 1.0:
            return 0.0
        return (1.0 - self.tokens) / self.refill_rate


@dataclass
class RateLimitConfig:
    """Configuration for rate limiter."""
    requests_per_minute: int = 60
    burst_capacity: int = 10
    enabled: bool = True


class RateLimiter:
    """Token bucket rate limiter.

    Features:
    - Per-key rate limiting (user, IP, API key)
    - Configurable rate and burst capacity
    - Clean up of stale buckets
    - Optional global limit

    Usage:
        limiter = RateLimiter(requests_per_minute=60)
        if not limiter.allow("user123"):
            raise RateLimitExceeded()
    """

    # Clean up buckets older than this (seconds)
    BUCKET_TTL = 3600  # 1 hour

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_capacity: int | None = None,
        enabled: bool = True,
    ) -> None:
        """Initialize rate limiter.

        Args:
            requests_per_minute: Sustained rate limit
            burst_capacity: Maximum burst size (defaults to 1/6 of rate)
            enabled: Whether rate limiting is active
        """
        self._requests_per_minute = requests_per_minute
        self._burst_capacity = burst_capacity or max(1, requests_per_minute // 6)
        self._refill_rate = requests_per_minute / 60.0
        self._enabled = enabled

        self._buckets: dict[str, TokenBucket] = {}
        self._last_cleanup = time.time()

    @property
    def enabled(self) -> bool:
        """Check if rate limiting is enabled."""
        return self._enabled

    def _get_bucket(self, key: str) -> TokenBucket:
        """Get or create token bucket for a key."""
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(
                tokens=float(self._burst_capacity),
                last_update=time.time(),
                capacity=float(self._burst_capacity),
                refill_rate=self._refill_rate,
            )
        return self._buckets[key]

    def _cleanup_stale_buckets(self) -> None:
        """Remove buckets that haven't been used recently."""
        now = time.time()
        if now - self._last_cleanup < 300:  # Clean up every 5 minutes
            return

        self._last_cleanup = now
        stale_keys = []

        for key, bucket in self._buckets.items():
            if now - bucket.last_update > self.BUCKET_TTL:
                stale_keys.append(key)

        for key in stale_keys:
            del self._buckets[key]

    def allow(self, key: str = "default", tokens: float = 1.0) -> bool:
        """Check if request is allowed.

        Args:
            key: Identifier for rate limiting (user ID, IP, etc.)
            tokens: Number of tokens to consume (usually 1)

        Returns:
            True if request is allowed
        """
        if not self._enabled:
            return True

        self._cleanup_stale_buckets()
        bucket = self._get_bucket(key)
        return bucket.update_and_consume(tokens)

    def get_wait_time(self, key: str = "default") -> float:
        """Get time until next request is allowed.

        Returns:
            Seconds until a request would be allowed
        """
        if not self._enabled:
            return 0.0

        bucket = self._get_bucket(key)
        # Update without consuming to get current state
        now = time.time()
        elapsed = now - bucket.last_update
        current_tokens = min(bucket.capacity, bucket.tokens + elapsed * bucket.refill_rate)

        if current_tokens >= 1.0:
            return 0.0
        return (1.0 - current_tokens) / bucket.refill_rate

    def get_remaining(self, key: str = "default") -> int:
        """Get remaining requests in current window.

        Returns approximate number of requests available.
        """
        if not self._enabled:
            return 999999

        bucket = self._get_bucket(key)
        now = time.time()
        elapsed = now - bucket.last_update
        current_tokens = min(bucket.capacity, bucket.tokens + elapsed * bucket.refill_rate)

        return int(current_tokens)

    def reset(self, key: str | None = None) -> None:
        """Reset rate limit for a key or all keys."""
        if key is None:
            self._buckets.clear()
        elif key in self._buckets:
            del self._buckets[key]

    def configure(
        self,
        requests_per_minute: int | None = None,
        burst_capacity: int | None = None,
        enabled: bool | None = None,
    ) -> None:
        """Update rate limiter configuration.

        Note: Changes affect new buckets; existing buckets retain old settings.
        """
        if requests_per_minute is not None:
            self._requests_per_minute = requests_per_minute
            self._refill_rate = requests_per_minute / 60.0

        if burst_capacity is not None:
            self._burst_capacity = burst_capacity

        if enabled is not None:
            self._enabled = enabled

    def get_stats(self) -> dict[str, Any]:
        """Get rate limiter statistics."""
        return {
            "enabled": self._enabled,
            "requests_per_minute": self._requests_per_minute,
            "burst_capacity": self._burst_capacity,
            "active_buckets": len(self._buckets),
        }


# Singleton instance
_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance.

    Environment variables:
    - AUTORAG_RATE_LIMIT_ENABLED: Enable/disable rate limiting (default: true)
    - AUTORAG_RATE_LIMIT_RPM: Requests per minute (default: 100)
    - AUTORAG_RATE_LIMIT_BURST: Burst capacity (default: 20)
    """
    global _rate_limiter
    if _rate_limiter is None:
        import os
        enabled = os.environ.get("AUTORAG_RATE_LIMIT_ENABLED", "true").lower() == "true"
        rpm = int(os.environ.get("AUTORAG_RATE_LIMIT_RPM", "100"))
        burst = int(os.environ.get("AUTORAG_RATE_LIMIT_BURST", "20"))
        _rate_limiter = RateLimiter(
            requests_per_minute=rpm,
            burst_capacity=burst,
            enabled=enabled,
        )
    return _rate_limiter


__all__ = [
    "TokenBucket",
    "RateLimitConfig",
    "RateLimiter",
    "get_rate_limiter",
]
