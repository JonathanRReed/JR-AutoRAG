"""Context metrics for improved coverage display.

This module implements P0.5: Sane Coverage Metrics
- Rename "coverage" to "context load ratio" when > 100%
- Add context overflow warnings
- Track and display dropped tokens
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ContextMetrics:
    """Comprehensive context size metrics.

    Provides clear labeling for context utilization:
    - Below 100%: "Context Utilization"
    - At or above 100%: "Context Load Ratio" with overflow warning
    """

    tokens_used: int
    max_context_tokens: int
    chunks_used: int
    chunks_dropped: int = 0
    tokens_dropped: int = 0

    @property
    def utilization_ratio(self) -> float:
        """Context utilization as a ratio (can exceed 1.0)."""
        if self.max_context_tokens <= 0:
            return 0.0
        return self.tokens_used / self.max_context_tokens

    @property
    def utilization_percent(self) -> float:
        """Context utilization as percentage."""
        return self.utilization_ratio * 100

    @property
    def is_overflow(self) -> bool:
        """True if context exceeds max."""
        return self.tokens_used > self.max_context_tokens

    @property
    def overflow_tokens(self) -> int:
        """Number of tokens over the limit."""
        if not self.is_overflow:
            return 0
        return self.tokens_used - self.max_context_tokens

    @property
    def label(self) -> str:
        """Human-readable label for the metric."""
        if self.is_overflow:
            return "Context Load Ratio"
        return "Context Utilization"

    @property
    def formatted_value(self) -> str:
        """Formatted display value."""
        pct = self.utilization_percent
        if pct > 100:
            return f"{pct:.0f}% (OVERFLOW)"
        return f"{pct:.0f}%"

    def get_warning(self) -> str | None:
        """Get context overflow warning if applicable."""
        if not self.is_overflow:
            return None

        overflow_pct = (self.overflow_tokens / self.max_context_tokens) * 100

        if overflow_pct > 50:
            return (
                f"Severe context overflow: {self.overflow_tokens:,} tokens over limit "
                f"({self.tokens_dropped:,} tokens dropped from {self.chunks_dropped} chunks)"
            )
        elif overflow_pct > 20:
            return (
                f"Context overflow: {self.overflow_tokens:,} tokens exceeded max. "
                f"Some evidence was truncated."
            )
        else:
            return (
                f"Context near capacity: {self.utilization_percent:.0f}% utilized"
            )

    def to_dict(self) -> dict:
        """Convert to API response format."""
        return {
            "label": self.label,
            "value": self.formatted_value,
            "tokens_used": self.tokens_used,
            "tokens_max": self.max_context_tokens,
            "tokens_dropped": self.tokens_dropped,
            "chunks_used": self.chunks_used,
            "chunks_dropped": self.chunks_dropped,
            "utilization_ratio": round(self.utilization_ratio, 3),
            "is_overflow": self.is_overflow,
            "warning": self.get_warning(),
        }


@dataclass
class GroundingInfo:
    """Grounding information for an answer.

    Implements P1.8: Grounding Visible in Answers
    Shows: grounded status, docs used, citations kept, chunks dropped
    """

    is_grounded: bool
    docs_used: int
    citations_kept: int
    chunks_total: int
    chunks_dropped: int
    mode: str = "grounded"
    evidence_spans: list[dict] | None = None  # Highlighted spans from evidence

    @property
    def grounding_label(self) -> str:
        """Human-readable grounding status."""
        if self.is_grounded:
            return f"Grounded ({self.docs_used} docs, {self.citations_kept} citations)"
        return "Partially Grounded" if self.docs_used > 0 else "Not Grounded"

    @property
    def dropout_warning(self) -> str | None:
        """Warning if significant chunks were dropped."""
        if self.chunks_total == 0:
            return None

        dropout_ratio = self.chunks_dropped / self.chunks_total
        if dropout_ratio > 0.5:
            return f"High evidence dropout: {self.chunks_dropped}/{self.chunks_total} chunks dropped"
        elif dropout_ratio > 0.2:
            return f"Some evidence dropped: {self.chunks_dropped} chunks"
        return None

    def to_dict(self) -> dict:
        """Convert to API response format."""
        return {
            "grounded": self.is_grounded,
            "label": self.grounding_label,
            "docs_used": self.docs_used,
            "citations_kept": self.citations_kept,
            "chunks_used": self.chunks_total - self.chunks_dropped,
            "chunks_dropped": self.chunks_dropped,
            "mode": self.mode,
            "dropout_warning": self.dropout_warning,
            "evidence_spans": self.evidence_spans,
        }


def compute_context_metrics(
    chunks: list,
    max_context_tokens: int,
    token_counter: callable = None,
) -> ContextMetrics:
    """Compute context metrics from chunks.

    Args:
        chunks: List of evidence chunks
        max_context_tokens: Max context window size
        token_counter: Optional function to count tokens per chunk

    Returns:
        ContextMetrics with utilization data
    """
    if token_counter is None:
        # Default: estimate tokens as words * 1.3
        def token_counter(c):
            return int(len(getattr(c, 'snippet', str(c)).split()) * 1.3)

    total_tokens = sum(token_counter(c) for c in chunks)

    return ContextMetrics(
        tokens_used=total_tokens,
        max_context_tokens=max_context_tokens,
        chunks_used=len(chunks),
    )


def compute_grounding_info(
    chunks: list,
    answer: str,
    citations_in_answer: int,
    chunks_dropped: int = 0,
    mode: str = "grounded",
) -> GroundingInfo:
    """Compute grounding info from answer and evidence.

    Args:
        chunks: Evidence chunks used
        answer: Generated answer
        citations_in_answer: Count of citations in answer
        chunks_dropped: Number of chunks dropped due to limits
        mode: Query mode ("grounded" or "open_domain")

    Returns:
        GroundingInfo with display data
    """
    import re

    # Count unique documents
    doc_ids = set()
    for c in chunks:
        doc_id = getattr(c, 'doc_id', None) or getattr(c, 'title', str(id(c)))
        doc_ids.add(doc_id)

    # Count citations in answer
    if citations_in_answer == 0:
        citation_matches = re.findall(r'\[(\d+)\]', answer)
        citations_in_answer = len(set(citation_matches))

    # Determine if grounded
    is_grounded = len(chunks) > 0 and citations_in_answer > 0

    return GroundingInfo(
        is_grounded=is_grounded,
        docs_used=len(doc_ids),
        citations_kept=citations_in_answer,
        chunks_total=len(chunks) + chunks_dropped,
        chunks_dropped=chunks_dropped,
        mode=mode,
    )


__all__ = [
    "ContextMetrics",
    "GroundingInfo",
    "compute_context_metrics",
    "compute_grounding_info",
]
