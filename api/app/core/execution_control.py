"""Query execution control for cancellation support.

This module implements P1.9: Query Execution Feedback
- Clean pipeline cancellation via "Stop" button
- Partial result return on cancel
- Stage timing visibility
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


logger = logging.getLogger("autorag.execution_control")


class ExecutionState(str, Enum):
    """State of query execution."""
    
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class StageProgress:
    """Progress info for a single stage."""
    
    stage: str
    started_at: float
    completed_at: float | None = None
    tokens_used: int = 0
    items_processed: int = 0
    items_total: int = 0
    
    @property
    def duration_ms(self) -> float:
        end = self.completed_at or time.time()
        return (end - self.started_at) * 1000
    
    @property
    def is_complete(self) -> bool:
        return self.completed_at is not None
    
    @property
    def progress_percent(self) -> float:
        if self.items_total == 0:
            return 0.0
        return min(100.0, (self.items_processed / self.items_total) * 100)
    
    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "duration_ms": round(self.duration_ms),
            "tokens": self.tokens_used,
            "progress": round(self.progress_percent),
            "complete": self.is_complete,
        }


@dataclass
class ExecutionContext:
    """Context for tracking execution state.
    
    Passed through pipeline stages to:
    - Track progress
    - Check cancellation
    - Collect partial results
    """
    
    trace_id: str
    state: ExecutionState = ExecutionState.PENDING
    started_at: float = field(default_factory=time.time)
    cancelled_at: float | None = None
    completed_at: float | None = None
    stages: list[StageProgress] = field(default_factory=list)
    partial_result: dict[str, Any] = field(default_factory=dict)
    cancel_reason: str | None = None
    
    @property
    def is_cancelled(self) -> bool:
        return self.state == ExecutionState.CANCELLED
    
    @property
    def total_duration_ms(self) -> float:
        end = self.completed_at or self.cancelled_at or time.time()
        return (end - self.started_at) * 1000
    
    def start_stage(self, stage: str, items_total: int = 0) -> StageProgress:
        """Start tracking a new stage."""
        progress = StageProgress(
            stage=stage,
            started_at=time.time(),
            items_total=items_total,
        )
        self.stages.append(progress)
        return progress
    
    def complete_stage(self, stage: str, tokens_used: int = 0) -> None:
        """Mark a stage as complete."""
        for s in reversed(self.stages):
            if s.stage == stage and not s.is_complete:
                s.completed_at = time.time()
                s.tokens_used = tokens_used
                break
    
    def check_cancelled(self) -> None:
        """Check if cancelled and raise if so.
        
        Call this at safe points in long-running stages.
        """
        if self.is_cancelled:
            raise asyncio.CancelledError(f"Execution cancelled: {self.cancel_reason}")
    
    def cancel(self, reason: str = "User requested cancellation") -> None:
        """Cancel execution."""
        self.state = ExecutionState.CANCELLED
        self.cancelled_at = time.time()
        self.cancel_reason = reason
        logger.info(f"Execution {self.trace_id} cancelled: {reason}")
    
    def save_partial(self, key: str, value: Any) -> None:
        """Save partial result for return on cancel."""
        self.partial_result[key] = value
    
    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "state": self.state.value,
            "duration_ms": round(self.total_duration_ms),
            "stages": [s.to_dict() for s in self.stages],
            "cancel_reason": self.cancel_reason,
        }


class ExecutionController:
    """Manage active query executions.
    
    Provides:
    - Registration of new executions
    - Cancellation by trace_id
    - Progress tracking
    """
    
    def __init__(self) -> None:
        self._contexts: dict[str, ExecutionContext] = {}
        self._on_progress: dict[str, Callable[[dict], None]] = {}
    
    def register(self, trace_id: str) -> ExecutionContext:
        """Register a new execution."""
        ctx = ExecutionContext(trace_id=trace_id, state=ExecutionState.RUNNING)
        self._contexts[trace_id] = ctx
        logger.debug(f"Registered execution {trace_id}")
        return ctx
    
    def get(self, trace_id: str) -> ExecutionContext | None:
        """Get execution context by trace_id."""
        return self._contexts.get(trace_id)
    
    def cancel(self, trace_id: str, reason: str = "User cancelled") -> bool:
        """Cancel an execution by trace_id.
        
        Returns True if found and cancelled, False if not found.
        """
        ctx = self._contexts.get(trace_id)
        if ctx and ctx.state == ExecutionState.RUNNING:
            ctx.cancel(reason)
            return True
        return False
    
    def complete(self, trace_id: str) -> None:
        """Mark an execution as complete."""
        ctx = self._contexts.get(trace_id)
        if ctx:
            ctx.state = ExecutionState.COMPLETED
            ctx.completed_at = time.time()
    
    def fail(self, trace_id: str, error: str) -> None:
        """Mark an execution as failed."""
        ctx = self._contexts.get(trace_id)
        if ctx:
            ctx.state = ExecutionState.FAILED
            ctx.completed_at = time.time()
            ctx.cancel_reason = error
    
    def cleanup(self, trace_id: str) -> None:
        """Remove an execution from tracking."""
        self._contexts.pop(trace_id, None)
        self._on_progress.pop(trace_id, None)
    
    def get_active(self) -> list[str]:
        """Get list of active trace_ids."""
        return [
            tid for tid, ctx in self._contexts.items()
            if ctx.state == ExecutionState.RUNNING
        ]
    
    def set_progress_callback(
        self,
        trace_id: str,
        callback: Callable[[dict], None],
    ) -> None:
        """Set callback for progress updates."""
        self._on_progress[trace_id] = callback
    
    def emit_progress(self, trace_id: str, data: dict) -> None:
        """Emit progress update if callback registered."""
        callback = self._on_progress.get(trace_id)
        if callback:
            try:
                callback(data)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")


# Global instance
_controller: ExecutionController | None = None


def get_execution_controller() -> ExecutionController:
    """Get or create global execution controller."""
    global _controller
    if _controller is None:
        _controller = ExecutionController()
    return _controller


__all__ = [
    "ExecutionState",
    "StageProgress",
    "ExecutionContext",
    "ExecutionController",
    "get_execution_controller",
]
