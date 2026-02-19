"""Enhanced tracing for detailed pipeline observability.

This module provides:
- Detailed step-by-step traces
- Hierarchical span tracking
- Performance profiling
- Debug information capture
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class SpanStatus(str, Enum):
    """Status of a trace span."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class TraceSpan:
    """A single span in a trace."""
    id: str
    name: str
    parent_id: str | None
    status: SpanStatus = SpanStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def start(self) -> None:
        """Mark span as started."""
        self.status = SpanStatus.RUNNING
        self.started_at = datetime.now(UTC)

    def complete(self, error: str | None = None) -> None:
        """Mark span as completed."""
        self.completed_at = datetime.now(UTC)
        if self.started_at:
            delta = (self.completed_at - self.started_at).total_seconds()
            self.duration_ms = delta * 1000

        if error:
            self.status = SpanStatus.ERROR
            self.error = error
        else:
            self.status = SpanStatus.COMPLETED

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": datetime.now(UTC).isoformat(),
            "attributes": attributes or {},
        })

    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "parent_id": self.parent_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "events": self.events,
            "error": self.error,
        }


@dataclass
class DetailedTrace:
    """A complete trace with multiple spans."""
    id: str
    name: str
    started_at: datetime
    completed_at: datetime | None = None
    spans: list[TraceSpan] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Total trace duration."""
        if not self.completed_at:
            return 0.0
        delta = (self.completed_at - self.started_at).total_seconds()
        return delta * 1000

    @property
    def root_span(self) -> TraceSpan | None:
        """Get root span (no parent)."""
        for span in self.spans:
            if span.parent_id is None:
                return span
        return self.spans[0] if self.spans else None

    def get_children(self, span_id: str) -> list[TraceSpan]:
        """Get child spans."""
        return [s for s in self.spans if s.parent_id == span_id]

    def get_span(self, span_id: str) -> TraceSpan | None:
        """Get span by ID."""
        for span in self.spans:
            if span.id == span_id:
                return span
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
            "spans": [s.to_dict() for s in self.spans],
            "metadata": self.metadata,
        }

    def to_tree(self) -> dict[str, Any]:
        """Convert to hierarchical tree structure."""
        def build_node(span: TraceSpan) -> dict[str, Any]:
            children = self.get_children(span.id)
            return {
                "name": span.name,
                "duration_ms": span.duration_ms,
                "status": span.status.value,
                "attributes": span.attributes,
                "children": [build_node(c) for c in children],
            }

        root = self.root_span
        if not root:
            return {}

        return build_node(root)


class Tracer:
    """Creates and manages traces."""

    def __init__(self) -> None:
        self._current_trace: DetailedTrace | None = None
        self._current_span_id: str | None = None
        self._traces: list[DetailedTrace] = []
        self._max_traces = 100

    def start_trace(self, name: str, metadata: dict[str, Any] | None = None) -> DetailedTrace:
        """Start a new trace."""
        trace = DetailedTrace(
            id=str(uuid.uuid4()),
            name=name,
            started_at=datetime.now(UTC),
            metadata=metadata or {},
        )
        self._current_trace = trace
        return trace

    def end_trace(self) -> DetailedTrace | None:
        """End the current trace."""
        if not self._current_trace:
            return None

        self._current_trace.completed_at = datetime.now(UTC)
        self._traces.append(self._current_trace)

        # Trim old traces
        if len(self._traces) > self._max_traces:
            self._traces = self._traces[-self._max_traces:]

        trace = self._current_trace
        self._current_trace = None
        self._current_span_id = None
        return trace

    def start_span(self, name: str, parent_id: str | None = None) -> TraceSpan:
        """Start a new span in the current trace."""
        if not self._current_trace:
            self.start_trace("auto")

        span = TraceSpan(
            id=str(uuid.uuid4()),
            name=name,
            parent_id=parent_id or self._current_span_id,
        )
        span.start()

        self._current_trace.spans.append(span)
        self._current_span_id = span.id

        return span

    def end_span(self, span: TraceSpan, error: str | None = None) -> None:
        """End a span."""
        span.complete(error)

        # Reset current span to parent
        if span.parent_id:
            self._current_span_id = span.parent_id
        else:
            self._current_span_id = None

    @contextmanager
    def span(self, name: str, **attributes) -> Generator[TraceSpan, None, None]:
        """Context manager for spans."""
        span = self.start_span(name)
        for key, value in attributes.items():
            span.set_attribute(key, value)

        try:
            yield span
        except Exception as e:
            self.end_span(span, error=str(e))
            raise
        else:
            self.end_span(span)

    def get_current_trace(self) -> DetailedTrace | None:
        """Get the current trace."""
        return self._current_trace

    def get_trace(self, trace_id: str) -> DetailedTrace | None:
        """Get a trace by ID."""
        for trace in self._traces:
            if trace.id == trace_id:
                return trace
        return None

    def get_recent_traces(self, n: int = 10) -> list[DetailedTrace]:
        """Get recent traces."""
        return self._traces[-n:]

    def clear(self) -> None:
        """Clear all traces."""
        self._traces = []
        self._current_trace = None
        self._current_span_id = None


# Global tracer instance
_tracer = Tracer()


def get_tracer() -> Tracer:
    """Get the global tracer."""
    return _tracer


__all__ = [
    "SpanStatus",
    "TraceSpan",
    "DetailedTrace",
    "Tracer",
    "get_tracer",
]
