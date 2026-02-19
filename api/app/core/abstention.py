"""Abstention rules for evidence-insufficient scenarios.

This module provides configurable abstention behavior when:
- Retrieval quality is too low to support a reliable answer
- Evidence is missing for key claims
- Coverage ratio falls below acceptable thresholds

The abstention system helps prevent hallucinations by refusing to
generate answers when the evidence is insufficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .gatherer import EvidenceChunk
    from .retrieval_evaluator import RetrievalVerdict


class AbstentionReason(str, Enum):
    """Reasons for abstaining from providing an answer."""
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    LOW_COVERAGE = "low_coverage"
    NO_RELEVANT_DOCUMENTS = "no_relevant_documents"
    CONFLICTING_SOURCES = "conflicting_sources"
    OUT_OF_SCOPE = "out_of_scope"
    RETRIEVAL_FAILED = "retrieval_failed"


@dataclass
class AbstentionResult:
    """Result of abstention check."""
    should_abstain: bool
    reason: AbstentionReason | None = None
    confidence: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    suggested_response: str | None = None


@dataclass
class AbstentionConfig:
    """Configuration for abstention rules."""
    # Enable/disable abstention entirely
    # DISABLED: Coverage calculation has bugs causing false positives
    enabled: bool = False

    # Coverage thresholds (relaxed to avoid false positives)
    min_coverage_ratio: float = 0.15  # Was 0.3 - too aggressive
    min_plan_coverage_ratio: float = 0.2  # Was 0.4 - too aggressive

    # Chunk quality thresholds
    min_chunks_required: int = 1
    min_avg_chunk_score: float = 0.1  # Was 0.3 - scores can be low but still relevant

    # Retrieval verdict settings
    abstain_on_incorrect: bool = True
    abstain_on_low_coverage: bool = False  # Was True - too aggressive

    # Confidence thresholds
    min_verdict_confidence: float = 0.6  # Was 0.4 - require higher confidence to abstain

    # Response templates
    insufficient_evidence_response: str = (
        "I cannot provide a reliable answer to this question because the "
        "available documents do not contain sufficient relevant information. "
        "The knowledge base may not cover this topic, or more specific documents "
        "may be needed."
    )
    no_documents_response: str = (
        "No relevant documents were found for this query. Please ensure the "
        "knowledge base contains documents related to your question, or try "
        "rephrasing your query."
    )
    out_of_scope_response: str = (
        "This question appears to be outside the scope of the current knowledge "
        "base. The available documents do not contain information to answer this."
    )


class AbstentionRules:
    """Evaluate whether to abstain from answering based on retrieval quality.

    Implements explicit abstention behavior as per SOTA RAG best practices:
    - Check retrieval verdict quality
    - Verify coverage thresholds are met
    - Ensure minimum chunk quality and quantity
    - Provide clear explanations when abstaining
    """

    def __init__(self, config: AbstentionConfig | None = None) -> None:
        """Initialize abstention rules with configuration.

        Args:
            config: Optional configuration, defaults to standard thresholds
        """
        self.config = config or AbstentionConfig()

    def check(
        self,
        *,
        chunks: list[EvidenceChunk],
        retrieval_verdict: RetrievalVerdict | None = None,
        verdict_confidence: float = 0.5,
        coverage_ratio: float = 0.5,
        plan_coverage_ratio: float = 0.5,
        query: str = "",
    ) -> AbstentionResult:
        """Check if we should abstain from answering.

        Args:
            chunks: Retrieved evidence chunks
            retrieval_verdict: Verdict from CRAG evaluator
            verdict_confidence: Confidence in the verdict
            coverage_ratio: Query aspect coverage ratio (0-1)
            plan_coverage_ratio: Plan query coverage ratio (0-1)
            query: Original user query

        Returns:
            AbstentionResult with decision and explanation
        """
        if not self.config.enabled:
            return AbstentionResult(should_abstain=False)

        details: dict[str, Any] = {
            "chunks_count": len(chunks),
            "coverage_ratio": round(coverage_ratio, 3),
            "plan_coverage_ratio": round(plan_coverage_ratio, 3),
            "verdict": retrieval_verdict.value if retrieval_verdict else "unknown",
            "verdict_confidence": round(verdict_confidence, 3),
        }

        # Check 1: No chunks at all
        if len(chunks) < self.config.min_chunks_required:
            return AbstentionResult(
                should_abstain=True,
                reason=AbstentionReason.NO_RELEVANT_DOCUMENTS,
                confidence=0.95,
                details=details,
                suggested_response=self.config.no_documents_response,
            )

        # Check 2: Very low average chunk scores
        avg_score = sum(c.score for c in chunks) / len(chunks) if chunks else 0
        details["avg_chunk_score"] = round(avg_score, 3)

        if avg_score < self.config.min_avg_chunk_score:
            return AbstentionResult(
                should_abstain=True,
                reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
                confidence=0.85,
                details=details,
                suggested_response=self.config.insufficient_evidence_response,
            )

        # Check 3: Coverage too low
        if coverage_ratio < self.config.min_coverage_ratio:
            details["coverage_gap"] = round(
                self.config.min_coverage_ratio - coverage_ratio, 3
            )
            return AbstentionResult(
                should_abstain=True,
                reason=AbstentionReason.LOW_COVERAGE,
                confidence=0.8,
                details=details,
                suggested_response=self.config.insufficient_evidence_response,
            )

        # Check 4: Plan coverage too low
        if plan_coverage_ratio < self.config.min_plan_coverage_ratio:
            details["plan_coverage_gap"] = round(
                self.config.min_plan_coverage_ratio - plan_coverage_ratio, 3
            )
            return AbstentionResult(
                should_abstain=True,
                reason=AbstentionReason.LOW_COVERAGE,
                confidence=0.75,
                details=details,
                suggested_response=self.config.insufficient_evidence_response,
            )

        # Check 5: Retrieval verdict is INCORRECT with high confidence
        if retrieval_verdict:
            from .retrieval_evaluator import RetrievalVerdict

            if (
                self.config.abstain_on_incorrect
                and retrieval_verdict == RetrievalVerdict.INCORRECT
                and verdict_confidence >= self.config.min_verdict_confidence
            ):
                return AbstentionResult(
                    should_abstain=True,
                    reason=AbstentionReason.INSUFFICIENT_EVIDENCE,
                    confidence=verdict_confidence,
                    details=details,
                    suggested_response=self.config.insufficient_evidence_response,
                )

            # Check 6: LOW_COVERAGE verdict after all corrective actions
            if (
                self.config.abstain_on_low_coverage
                and retrieval_verdict == RetrievalVerdict.LOW_COVERAGE
                and verdict_confidence >= self.config.min_verdict_confidence
            ):
                return AbstentionResult(
                    should_abstain=True,
                    reason=AbstentionReason.LOW_COVERAGE,
                    confidence=verdict_confidence,
                    details=details,
                    suggested_response=self.config.insufficient_evidence_response,
                )

        # All checks passed - do not abstain
        return AbstentionResult(
            should_abstain=False,
            confidence=0.0,
            details=details,
        )

    def format_abstention_response(
        self,
        result: AbstentionResult,
        query: str,
        include_details: bool = False,
    ) -> str:
        """Format an abstention response for the user.

        Args:
            result: Abstention check result
            query: Original query for context
            include_details: Whether to include technical details

        Returns:
            Formatted response explaining the abstention
        """
        if not result.should_abstain:
            return ""

        response = result.suggested_response or self.config.insufficient_evidence_response

        if include_details and result.details:
            details_str = ", ".join(
                f"{k}={v}" for k, v in result.details.items()
                if k not in ("chunks_count", "verdict")
            )
            response += f"\n\n_Technical details: {details_str}_"

        return response


# Singleton for easy access
_abstention_rules: AbstentionRules | None = None


def get_abstention_rules(config: AbstentionConfig | None = None) -> AbstentionRules:
    """Get or create the abstention rules instance.

    Args:
        config: Optional configuration to use

    Returns:
        AbstentionRules instance
    """
    global _abstention_rules
    if _abstention_rules is None or config is not None:
        _abstention_rules = AbstentionRules(config)
    return _abstention_rules


__all__ = [
    "AbstentionReason",
    "AbstentionResult",
    "AbstentionConfig",
    "AbstentionRules",
    "get_abstention_rules",
]
