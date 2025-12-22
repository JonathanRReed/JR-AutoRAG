"""Circuit Breakers: Protect against cascading failures.

Implements the circuit breaker pattern for external services:
- Tracks failure rates per service
- Opens circuit when failures exceed threshold
- Half-open state for testing recovery
- Configurable timeouts and retry policies

Prevents overloading failing services and improves resilience.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, TypeVar

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Blocking all requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""
    # Failure thresholds
    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 3  # Successes to close from half-open
    
    # Timing
    open_timeout_seconds: float = 30.0  # Time before entering half-open
    call_timeout_seconds: float = 10.0  # Timeout for each call
    
    # Rate limiting
    max_half_open_calls: int = 3  # Concurrent calls in half-open state


@dataclass
class CircuitBreakerStats:
    """Statistics for a circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0  # Calls rejected due to open circuit
    last_failure_time: float = 0.0
    last_success_time: float = 0.0
    current_state: CircuitState = CircuitState.CLOSED
    state_changes: int = 0


class CircuitBreakerError(Exception):
    """Raised when circuit is open."""
    pass


class CircuitBreaker:
    """Circuit breaker for protecting external service calls.
    
    Usage:
    ```python
    breaker = CircuitBreaker("llm_provider")
    
    async def call_llm():
        async with breaker:
            return await provider.chat(...)
    ```
    
    Or:
    ```python
    result = await breaker.call(provider.chat, messages=[...])
    ```
    """
    
    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
        on_state_change: Callable[[str, CircuitState, CircuitState], None] | None = None,
    ) -> None:
        """Initialize circuit breaker.
        
        Args:
            name: Service name for identification
            config: Configuration overrides
            on_state_change: Callback for state changes
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._on_state_change = on_state_change
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_calls = 0
        
        self._stats = CircuitBreakerStats()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state

    @property
    def is_closed(self) -> bool:
        """Check if circuit is allowing all calls."""
        return self._state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is blocking calls."""
        return self._state == CircuitState.OPEN

    def _should_attempt_recovery(self) -> bool:
        """Check if open circuit should try half-open."""
        if self._state != CircuitState.OPEN:
            return False
        elapsed = time.time() - self._last_failure_time
        return elapsed >= self.config.open_timeout_seconds

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        if new_state == self._state:
            return
        
        old_state = self._state
        self._state = new_state
        self._stats.current_state = new_state
        self._stats.state_changes += 1
        
        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_calls = 0
            self._success_count = 0
        
        if self._on_state_change:
            try:
                self._on_state_change(self.name, old_state, new_state)
            except Exception:
                pass

    def _record_success(self) -> None:
        """Record a successful call."""
        self._stats.successful_calls += 1
        self._stats.last_success_time = time.time()
        self._success_count += 1
        
        if self._state == CircuitState.HALF_OPEN:
            if self._success_count >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)

    def _record_failure(self) -> None:
        """Record a failed call."""
        self._stats.failed_calls += 1
        self._stats.last_failure_time = time.time()
        self._last_failure_time = time.time()
        self._failure_count += 1
        
        if self._state == CircuitState.HALF_OPEN:
            # Immediate re-open on failure during half-open
            self._transition_to(CircuitState.OPEN)
        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    async def __aenter__(self) -> "CircuitBreaker":
        """Enter context manager - check if call is allowed."""
        async with self._lock:
            self._stats.total_calls += 1
            
            # Check for recovery attempt
            if self._should_attempt_recovery():
                self._transition_to(CircuitState.HALF_OPEN)
            
            # Check if circuit allows call
            if self._state == CircuitState.OPEN:
                self._stats.rejected_calls += 1
                raise CircuitBreakerError(
                    f"Circuit breaker '{self.name}' is open. "
                    f"Service unavailable after {self._failure_count} failures."
                )
            
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.config.max_half_open_calls:
                    self._stats.rejected_calls += 1
                    raise CircuitBreakerError(
                        f"Circuit breaker '{self.name}' is half-open. "
                        "Maximum concurrent test calls reached."
                    )
                self._half_open_calls += 1
        
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit context manager - record success or failure."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls = max(0, self._half_open_calls - 1)
            
            if exc_type is None:
                self._record_success()
            else:
                self._record_failure()
        
        return False  # Don't suppress exception

    async def call(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a call with circuit breaker protection.
        
        Args:
            func: Async function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Result from func
            
        Raises:
            CircuitBreakerError: If circuit is open
            asyncio.TimeoutError: If call times out
        """
        async with self:
            if asyncio.iscoroutinefunction(func):
                return await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=self.config.call_timeout_seconds,
                )
            else:
                return func(*args, **kwargs)

    def get_stats(self) -> CircuitBreakerStats:
        """Get current statistics."""
        return CircuitBreakerStats(
            total_calls=self._stats.total_calls,
            successful_calls=self._stats.successful_calls,
            failed_calls=self._stats.failed_calls,
            rejected_calls=self._stats.rejected_calls,
            last_failure_time=self._stats.last_failure_time,
            last_success_time=self._stats.last_success_time,
            current_state=self._state,
            state_changes=self._stats.state_changes,
        )

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._stats = CircuitBreakerStats()


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers.
    
    Provides centralized access to circuit breakers for different services.
    """
    
    def __init__(self) -> None:
        """Initialize registry."""
        self._breakers: dict[str, CircuitBreaker] = {}
        self._default_config = CircuitBreakerConfig()
        self._lock = asyncio.Lock()

    def get_or_create(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """Get or create a circuit breaker by name.
        
        Args:
            name: Service name
            config: Optional configuration (used only on creation)
            
        Returns:
            CircuitBreaker instance
        """
        if name not in self._breakers:
            self._breakers[name] = CircuitBreaker(
                name=name,
                config=config or self._default_config,
            )
        return self._breakers[name]

    def get(self, name: str) -> CircuitBreaker | None:
        """Get a circuit breaker by name (returns None if not found)."""
        return self._breakers.get(name)

    def list_breakers(self) -> list[str]:
        """List all registered circuit breaker names."""
        return list(self._breakers.keys())

    def get_all_stats(self) -> dict[str, CircuitBreakerStats]:
        """Get statistics for all circuit breakers."""
        return {name: breaker.get_stats() for name, breaker in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self._breakers.values():
            breaker.reset()


# Global registry singleton
_registry: CircuitBreakerRegistry | None = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    global _registry
    if _registry is None:
        _registry = CircuitBreakerRegistry()
    return _registry


def get_circuit_breaker(name: str, config: CircuitBreakerConfig | None = None) -> CircuitBreaker:
    """Get or create a circuit breaker by name."""
    return get_circuit_breaker_registry().get_or_create(name, config)


__all__ = [
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreakerStats",
    "CircuitBreakerError",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "get_circuit_breaker_registry",
    "get_circuit_breaker",
]
