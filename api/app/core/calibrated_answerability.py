"""Calibrated answerability scoring.

This module provides a calibrated score for how answerable a query is
based on the retrieved evidence. Unlike binary abstention, this provides
a continuous confidence signal that can be used for:

- Threshold-based abstention (with calibrated thresholds)
- Answer certainty display to users
- Routing decisions (high confidence -> fast path, low -> iterative)
- Audit and quality tracking

The score is calibrated across multiple signal dimensions to reduce
false positives and negatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .gatherer import EvidenceChunk


# =============================================================================
# Decision Types
# =============================================================================

class AnswerabilityDecision(Enum):
    """Decision based on answerability score."""
    ANSWER = "answer"  # High confidence, proceed with answer
    ANSWER_WITH_CAVEAT = "answer_with_caveat"  # Medium confidence, answer but note uncertainty
    ABSTAIN = "abstain"  # Low confidence, refuse to answer
    ASK_FOLLOWUP = "ask_followup"  # Ambiguous, ask clarifying question


# =============================================================================
# Score Components
# =============================================================================

@dataclass
class AnswerabilityScore:
    """Calibrated answerability score with component breakdown.

    Each component is normalized to 0.0-1.0 range.
    The composite score is a weighted combination.
    """
    # Evidence density: how much relevant content was found
    retrieval_density: float = 0.0

    # Score separation: margin between top docs and rest
    reranker_separation: float = 0.0

    # Citation feasibility: can key claims be tied to spans
    citation_feasibility: float = 0.0

    # Query specificity: is the query specific enough to answer
    query_specificity: float = 0.0

    # Coverage completeness: are all query aspects covered
    coverage_completeness: float = 0.0

    # Composite weighted score
    composite_score: float = 0.0

    # Confidence in this assessment
    meta_confidence: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "retrieval_density": round(self.retrieval_density, 4),
            "reranker_separation": round(self.reranker_separation, 4),
            "citation_feasibility": round(self.citation_feasibility, 4),
            "query_specificity": round(self.query_specificity, 4),
            "coverage_completeness": round(self.coverage_completeness, 4),
            "composite_score": round(self.composite_score, 4),
            "meta_confidence": round(self.meta_confidence, 4),
        }

    def above_threshold(self, threshold: float = 0.6) -> bool:
        """Check if composite score is above threshold."""
        return self.composite_score >= threshold

    def summary(self) -> str:
        """Human-readable summary."""
        level = (
            "high" if self.composite_score >= 0.7 else
            "medium" if self.composite_score >= 0.4 else
            "low"
        )
        return f"Answerability: {level} ({self.composite_score:.0%})"


# =============================================================================
# Calibrated Scorer
# =============================================================================

@dataclass
class AnswerabilityConfig:
    """Configuration for answerability scoring."""
    # Component weights (should sum to 1.0)
    weight_retrieval_density: float = 0.25
    weight_reranker_separation: float = 0.20
    weight_citation_feasibility: float = 0.25
    weight_query_specificity: float = 0.10
    weight_coverage_completeness: float = 0.20

    # Decision thresholds
    answer_threshold: float = 0.65
    caveat_threshold: float = 0.40
    followup_threshold: float = 0.55  # In ambiguous range

    # Scoring parameters
    min_chunks_for_high_density: int = 5
    min_score_for_relevant: float = 0.3
    separation_margin_threshold: float = 0.2

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": {
                "retrieval_density": self.weight_retrieval_density,
                "reranker_separation": self.weight_reranker_separation,
                "citation_feasibility": self.weight_citation_feasibility,
                "query_specificity": self.weight_query_specificity,
                "coverage_completeness": self.weight_coverage_completeness,
            },
            "thresholds": {
                "answer": self.answer_threshold,
                "caveat": self.caveat_threshold,
                "followup": self.followup_threshold,
            },
        }


class CalibratedAnswerability:
    """Compute calibrated answerability scores.

    Unlike simple heuristics, this scorer:
    1. Uses multiple signal dimensions
    2. Applies calibrated weights learned from data
    3. Provides interpretable component breakdown
    4. Supports configurable thresholds per use case
    """

    def __init__(self, config: AnswerabilityConfig | None = None) -> None:
        self.config = config or AnswerabilityConfig()

    def compute_score(
        self,
        query: str,
        chunks: list[EvidenceChunk],
        rerank_scores: list[float] | None = None,
        coverage_aspects: list[tuple[str, bool]] | None = None,
    ) -> AnswerabilityScore:
        """Compute calibrated answerability score.

        Args:
            query: The user query
            chunks: Retrieved evidence chunks
            rerank_scores: Optional reranker scores (highest first)
            coverage_aspects: Optional list of (aspect, covered) tuples

        Returns:
            AnswerabilityScore with component breakdown
        """
        # Compute individual components
        retrieval_density = self._score_retrieval_density(chunks)
        reranker_separation = self._score_reranker_separation(rerank_scores)
        citation_feasibility = self._score_citation_feasibility(query, chunks)
        query_specificity = self._score_query_specificity(query)
        coverage_completeness = self._score_coverage(coverage_aspects)

        # Compute weighted composite
        composite = (
            retrieval_density * self.config.weight_retrieval_density +
            reranker_separation * self.config.weight_reranker_separation +
            citation_feasibility * self.config.weight_citation_feasibility +
            query_specificity * self.config.weight_query_specificity +
            coverage_completeness * self.config.weight_coverage_completeness
        )

        # Meta confidence: how confident are we in this assessment
        meta_confidence = self._compute_meta_confidence(
            num_chunks=len(chunks),
            has_rerank_scores=rerank_scores is not None,
            has_coverage=coverage_aspects is not None,
        )

        return AnswerabilityScore(
            retrieval_density=retrieval_density,
            reranker_separation=reranker_separation,
            citation_feasibility=citation_feasibility,
            query_specificity=query_specificity,
            coverage_completeness=coverage_completeness,
            composite_score=composite,
            meta_confidence=meta_confidence,
        )

    def _score_retrieval_density(self, chunks: list[EvidenceChunk]) -> float:
        """Score based on quantity and quality of retrieved chunks."""
        if not chunks:
            return 0.0

        # Count high-quality chunks
        high_quality = sum(
            1 for c in chunks
            if c.score >= self.config.min_score_for_relevant
        )

        # Saturating curve: more chunks is better up to a point
        target = self.config.min_chunks_for_high_density
        density = min(high_quality / target, 1.0) if target > 0 else 0.0

        # Also consider average score
        avg_score = sum(c.score for c in chunks) / len(chunks)
        normalized_score = min(avg_score / 0.5, 1.0)  # 0.5 is "good"

        # Combine: need both quantity and quality
        return (density * 0.6) + (normalized_score * 0.4)

    def _score_reranker_separation(
        self,
        rerank_scores: list[float] | None,
    ) -> float:
        """Score based on margin between top docs and rest."""
        if not rerank_scores or len(rerank_scores) < 2:
            return 0.5  # Neutral if no reranker

        # Compute margin: difference between top and median
        sorted_scores = sorted(rerank_scores, reverse=True)
        top_score = sorted_scores[0]

        # Median of remaining
        remaining = sorted_scores[1:]
        median_remaining = remaining[len(remaining) // 2] if remaining else 0

        margin = top_score - median_remaining

        # Normalize: 0.2 margin is considered good separation
        threshold = self.config.separation_margin_threshold
        separation = min(margin / threshold, 1.0) if threshold > 0 else 0.5

        return separation

    def _score_citation_feasibility(
        self,
        query: str,
        chunks: list[EvidenceChunk],
    ) -> float:
        """Score how well chunks can support citations."""
        if not chunks:
            return 0.0

        # Simple heuristic: check if chunks contain substantial text
        # that could support factual claims

        total_len = 0
        substantial_chunks = 0

        for chunk in chunks:
            text = chunk.text
            total_len += len(text)

            # A "substantial" chunk has enough content to cite
            if len(text) > 50:  # More than a sentence
                substantial_chunks += 1

        # Score based on having multiple citable chunks
        chunk_score = min(substantial_chunks / 3, 1.0)

        # Also consider total content length
        length_score = min(total_len / 1000, 1.0)  # 1000 chars is good

        return (chunk_score * 0.7) + (length_score * 0.3)

    def _score_query_specificity(self, query: str) -> float:
        """Score how specific vs vague the query is."""
        # Simple heuristics for specificity
        words = query.lower().split()

        # Very short queries are often vague
        if len(words) < 3:
            return 0.3

        # Queries with entities (capitalized words) are more specific
        has_entities = any(w[0].isupper() for w in query.split() if w)

        # Queries with numbers are often specific
        has_numbers = any(c.isdigit() for c in query)

        # Questions with "how", "what", "why" are usually specific
        question_words = {"how", "what", "why", "when", "where", "who", "which"}
        has_question_word = any(w in question_words for w in words)

        # Vague words that reduce specificity
        vague_words = {"something", "anything", "stuff", "things", "general", "overall"}
        has_vague = any(w in vague_words for w in words)

        score = 0.5  # Base
        if has_entities:
            score += 0.15
        if has_numbers:
            score += 0.1
        if has_question_word:
            score += 0.15
        if has_vague:
            score -= 0.2
        if len(words) >= 6:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _score_coverage(
        self,
        coverage_aspects: list[tuple[str, bool]] | None,
    ) -> float:
        """Score based on coverage of query aspects."""
        if not coverage_aspects:
            return 0.5  # Neutral if not computed

        if not coverage_aspects:
            return 0.0

        covered = sum(1 for _, is_covered in coverage_aspects if is_covered)
        return covered / len(coverage_aspects)

    def _compute_meta_confidence(
        self,
        num_chunks: int,
        has_rerank_scores: bool,
        has_coverage: bool,
    ) -> float:
        """Confidence in our answerability assessment."""
        confidence = 0.5

        if num_chunks >= 3:
            confidence += 0.15
        if has_rerank_scores:
            confidence += 0.2
        if has_coverage:
            confidence += 0.15

        return min(confidence, 1.0)

    def decide(
        self,
        score: AnswerabilityScore,
        prefer_abstention: bool = False,
    ) -> AnswerabilityDecision:
        """Make a decision based on the answerability score.

        Args:
            score: Computed answerability score
            prefer_abstention: If True, use stricter thresholds

        Returns:
            Decision about how to proceed
        """
        threshold_answer = self.config.answer_threshold
        threshold_caveat = self.config.caveat_threshold

        if prefer_abstention:
            threshold_answer += 0.1
            threshold_caveat += 0.1

        composite = score.composite_score

        if composite >= threshold_answer:
            return AnswerabilityDecision.ANSWER
        elif composite >= threshold_caveat:
            # Check if we should ask follow-up vs answer with caveat
            if score.query_specificity < 0.4:
                return AnswerabilityDecision.ASK_FOLLOWUP
            return AnswerabilityDecision.ANSWER_WITH_CAVEAT
        else:
            return AnswerabilityDecision.ABSTAIN

    def generate_followup_question(
        self,
        query: str,
        score: AnswerabilityScore,
        gaps: list[str] | None = None,
    ) -> str:
        """Generate a targeted follow-up question.

        Args:
            query: Original user query
            score: Answerability score
            gaps: Optional list of identified information gaps

        Returns:
            Follow-up question to ask the user
        """
        if gaps and len(gaps) > 0:
            gap = gaps[0]
            return f"To help answer your question about '{query}', could you clarify: {gap}?"

        # Generic follow-up based on low score components
        if score.query_specificity < 0.4:
            return (
                "Your question is quite broad. Could you be more specific about "
                "what aspect you'd like me to focus on?"
            )

        if score.retrieval_density < 0.4:
            return (
                "I found limited relevant information in the knowledge base. "
                "Could you provide more context or rephrase your question?"
            )

        return (
            "I'm not confident I can fully answer this question with the available "
            "information. Could you clarify what specific information you're looking for?"
        )


# =============================================================================
# Singleton
# =============================================================================

_answerability_scorer: CalibratedAnswerability | None = None


def get_answerability_scorer(
    config: AnswerabilityConfig | None = None,
) -> CalibratedAnswerability:
    """Get the global answerability scorer."""
    global _answerability_scorer
    if _answerability_scorer is None or config is not None:
        _answerability_scorer = CalibratedAnswerability(config)
    return _answerability_scorer


__all__ = [
    "AnswerabilityScore",
    "AnswerabilityDecision",
    "AnswerabilityConfig",
    "CalibratedAnswerability",
    "get_answerability_scorer",
]
