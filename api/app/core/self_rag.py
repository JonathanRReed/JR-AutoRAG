"""Self-RAG: Self-Reflective Retrieval-Augmented Generation.

Implements Self-RAG paper concepts:
- Retrieve tokens: Decide if retrieval is needed for each segment
- ISREL (Relevance): Evaluate if retrieved passages are relevant
- ISSUP (Support): Check if generated content is supported by evidence
- ISUSE (Utility): Score overall response quality

This module extends the basic SelfReflector with LLM-based critic passes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .gatherer import EvidenceChunk
    from .providers import LLMProvider


class RetrievalDecision(str, Enum):
    """Self-RAG retrieve token decision."""

    RETRIEVE = "retrieve"  # Need to retrieve for this segment
    NO_RETRIEVE = "no_retrieve"  # Can generate without retrieval
    CONTINUE = "continue"  # Continue with existing context


class RelevanceScore(str, Enum):
    """Self-RAG ISREL token."""

    RELEVANT = "relevant"
    PARTIALLY_RELEVANT = "partially_relevant"
    IRRELEVANT = "irrelevant"


class SupportScore(str, Enum):
    """Self-RAG ISSUP token for grounding."""

    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NO_SUPPORT = "no_support"
    CONTRADICTS = "contradicts"


class UtilityScore(int, Enum):
    """Self-RAG ISUSE utility score (1-5)."""

    EXCELLENT = 5
    GOOD = 4
    ADEQUATE = 3
    POOR = 2
    USELESS = 1


@dataclass
class CriticResult:
    """Result of Self-RAG critic evaluation."""

    relevance: RelevanceScore
    support: SupportScore
    utility: UtilityScore
    should_regenerate: bool
    critique: str
    suggestions: list[str] = field(default_factory=list)
    confidence: float = 0.8
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class SelfRAGConfig:
    """Configuration for Self-RAG critic."""

    # Thresholds for regeneration
    min_support_for_accept: SupportScore = SupportScore.PARTIALLY_SUPPORTED
    min_utility_for_accept: UtilityScore = UtilityScore.ADEQUATE
    max_regeneration_attempts: int = 2

    # Enable/disable specific checks
    check_relevance: bool = True
    check_support: bool = True
    check_utility: bool = True

    # LLM settings
    use_llm_critic: bool = True
    fallback_to_heuristic: bool = True


class SelfRAGCritic:
    """Self-RAG style critic for evaluating and improving responses.

    Implements reflection tokens from the Self-RAG paper:
    1. ISREL: Is the retrieved content relevant to the query?
    2. ISSUP: Is the generated response supported by the evidence?
    3. ISUSE: Is the response useful/high quality?

    When quality is insufficient, triggers regeneration with critique.
    """

    CRITIQUE_PROMPT = """You are a Self-RAG critic evaluating an AI response for quality and grounding.

## QUERY
{query}

## EVIDENCE CHUNKS
{evidence}

## GENERATED RESPONSE
{response}

## EVALUATION TASK
Evaluate the response on these dimensions:

1. RELEVANCE: Is the retrieved evidence relevant to answering the query?
   - RELEVANT: Evidence directly addresses the query
   - PARTIALLY_RELEVANT: Some relevant information, some tangential
   - IRRELEVANT: Evidence does not help answer the query

2. SUPPORT: Is each claim in the response grounded in the evidence?
   - FULLY_SUPPORTED: All claims have supporting evidence
   - PARTIALLY_SUPPORTED: Some claims supported, some unsupported
   - NO_SUPPORT: Claims made without evidence
   - CONTRADICTS: Response contradicts the evidence

3. UTILITY: How useful is this response? (1-5)
   - 5: Excellent - comprehensive, accurate, well-structured
   - 4: Good - answers the question well
   - 3: Adequate - basic but acceptable
   - 2: Poor - missing key information or has issues
   - 1: Useless - doesn't answer the question

4. SHOULD_REGENERATE: Should this response be regenerated? (yes/no)

5. CRITIQUE: Brief explanation of issues (if any)

6. SUGGESTIONS: How to improve the response (if regenerating)

## OUTPUT FORMAT
RELEVANCE: [RELEVANT/PARTIALLY_RELEVANT/IRRELEVANT]
SUPPORT: [FULLY_SUPPORTED/PARTIALLY_SUPPORTED/NO_SUPPORT/CONTRADICTS]
UTILITY: [1-5]
SHOULD_REGENERATE: [yes/no]
CRITIQUE: [brief explanation]
SUGGESTIONS: [comma-separated improvements]"""

    def __init__(self, config: SelfRAGConfig | None = None) -> None:
        """Initialize Self-RAG critic.

        Args:
            config: Optional configuration, uses defaults otherwise
        """
        self.config = config or SelfRAGConfig()

        # Heuristic patterns for fallback
        self._unsupported_patterns = [
            re.compile(r"\b(I think|I believe|probably|might be)\b", re.I),
            re.compile(r"\b(generally|typically|usually)\b", re.I),
            re.compile(r"\b(it is (possible|likely) that)\b", re.I),
        ]
        self._hedging_patterns = [
            re.compile(r"\b(however|although|but)\b", re.I),
            re.compile(r"\b(some|many|few)\b", re.I),
        ]

    async def critique(
        self,
        query: str,
        response: str,
        chunks: list[EvidenceChunk],
        provider: LLMProvider | None = None,
    ) -> CriticResult:
        """Evaluate a generated response using Self-RAG critic.

        Args:
            query: Original user query
            response: Generated response text
            chunks: Evidence chunks used for generation
            provider: Optional LLM provider for LLM-based critique

        Returns:
            CriticResult with evaluation scores and regeneration decision
        """
        if provider is not None and self.config.use_llm_critic:
            try:
                return await self._critique_llm(query, response, chunks, provider)
            except Exception as e:
                if self.config.fallback_to_heuristic:
                    result = self._critique_heuristic(query, response, chunks)
                    result.details["llm_error"] = str(e)
                    return result
                raise

        return self._critique_heuristic(query, response, chunks)

    async def _critique_llm(
        self,
        query: str,
        response: str,
        chunks: list[EvidenceChunk],
        provider: LLMProvider,
    ) -> CriticResult:
        """Use LLM for Self-RAG critique."""
        # Format evidence
        evidence_text = "\n\n".join(
            f"[{i + 1}] {c.title}: {c.snippet[:500]}" for i, c in enumerate(chunks[:8])
        )

        prompt = self.CRITIQUE_PROMPT.format(
            query=query,
            evidence=evidence_text or "(No evidence provided)",
            response=response[:2000],  # Truncate long responses
        )

        llm_response = await provider.chat(
            [
                {
                    "role": "system",
                    "content": "You are a precise quality evaluator for RAG systems.",
                },
                {"role": "user", "content": prompt},
            ]
        )

        return self._parse_critique_response(llm_response, query, response, chunks)

    def _parse_critique_response(
        self,
        llm_response: str,
        query: str,
        response: str,
        chunks: list[EvidenceChunk],
    ) -> CriticResult:
        """Parse LLM critique response into CriticResult."""
        # Extract relevance
        rel_match = re.search(
            r"RELEVANCE:\s*(RELEVANT|PARTIALLY_RELEVANT|IRRELEVANT)", llm_response, re.I
        )
        relevance = RelevanceScore.RELEVANT
        if rel_match:
            rel_str = rel_match.group(1).upper()
            relevance = RelevanceScore(rel_str.lower())

        # Extract support
        sup_match = re.search(
            r"SUPPORT:\s*(FULLY_SUPPORTED|PARTIALLY_SUPPORTED|NO_SUPPORT|CONTRADICTS)",
            llm_response,
            re.I,
        )
        support = SupportScore.PARTIALLY_SUPPORTED
        if sup_match:
            sup_str = sup_match.group(1).upper()
            support = SupportScore(sup_str.lower())

        # Extract utility
        util_match = re.search(r"UTILITY:\s*(\d)", llm_response)
        utility = UtilityScore.ADEQUATE
        if util_match:
            util_val = int(util_match.group(1))
            utility = UtilityScore(max(1, min(5, util_val)))

        # Extract regeneration decision
        regen_match = re.search(r"SHOULD_REGENERATE:\s*(yes|no)", llm_response, re.I)
        should_regenerate = False
        if regen_match:
            should_regenerate = regen_match.group(1).lower() == "yes"

        # Extract critique
        crit_match = re.search(
            r"CRITIQUE:\s*(.+?)(?=SUGGESTIONS:|$)", llm_response, re.I | re.S
        )
        critique = crit_match.group(1).strip() if crit_match else ""

        # Extract suggestions
        sugg_match = re.search(r"SUGGESTIONS:\s*(.+?)(?=$)", llm_response, re.I | re.S)
        suggestions = []
        if sugg_match:
            sugg_text = sugg_match.group(1).strip()
            suggestions = [s.strip() for s in sugg_text.split(",") if s.strip()]

        # Override regeneration decision based on thresholds
        if not should_regenerate:
            support_levels = {
                SupportScore.FULLY_SUPPORTED: 4,
                SupportScore.PARTIALLY_SUPPORTED: 3,
                SupportScore.NO_SUPPORT: 2,
                SupportScore.CONTRADICTS: 1,
            }
            if (
                support_levels[support]
                < support_levels[self.config.min_support_for_accept]
            ):
                should_regenerate = True
            if utility < self.config.min_utility_for_accept:
                should_regenerate = True

        return CriticResult(
            relevance=relevance,
            support=support,
            utility=utility,
            should_regenerate=should_regenerate,
            critique=critique,
            suggestions=suggestions,
            confidence=0.85,
            details={
                "method": "llm",
                "chunks_evaluated": len(chunks),
            },
        )

    def _critique_heuristic(
        self,
        query: str,
        response: str,
        chunks: list[EvidenceChunk],
    ) -> CriticResult:
        """Fallback heuristic-based critique."""
        issues = []
        suggestions = []

        # Check relevance by term overlap
        query_terms = {w.lower() for w in re.findall(r"\b\w{4,}\b", query)}
        response_terms = {w.lower() for w in re.findall(r"\b\w{4,}\b", response)}
        chunk_terms = set()
        for c in chunks:
            chunk_terms.update(w.lower() for w in re.findall(r"\b\w{4,}\b", c.snippet))

        query_chunk_overlap = len(query_terms & chunk_terms) / max(len(query_terms), 1)
        if query_chunk_overlap >= 0.6:
            relevance = RelevanceScore.RELEVANT
        elif query_chunk_overlap >= 0.3:
            relevance = RelevanceScore.PARTIALLY_RELEVANT
        else:
            relevance = RelevanceScore.IRRELEVANT
            issues.append("Retrieved evidence has low overlap with query")

        # Check support by citation presence and hedging
        citation_count = len(re.findall(r"\[\d+\]", response))
        unsupported_count = sum(
            1 for p in self._unsupported_patterns if p.search(response)
        )

        if citation_count >= 3 and unsupported_count == 0:
            support = SupportScore.FULLY_SUPPORTED
        elif citation_count >= 1 and unsupported_count <= 1:
            support = SupportScore.PARTIALLY_SUPPORTED
        else:
            support = SupportScore.NO_SUPPORT
            issues.append("Response lacks citations or contains unsupported claims")
            suggestions.append("Add more citations to claims")

        # Estimate utility
        response_len = len(response)
        query_response_overlap = len(query_terms & response_terms) / max(
            len(query_terms), 1
        )

        if response_len > 200 and query_response_overlap >= 0.5 and citation_count >= 2:
            utility = UtilityScore.GOOD
        elif response_len > 100 and query_response_overlap >= 0.3:
            utility = UtilityScore.ADEQUATE
        elif response_len < 50 or query_response_overlap < 0.2:
            utility = UtilityScore.POOR
            issues.append("Response is too short or doesn't address the query")
        else:
            utility = UtilityScore.ADEQUATE

        # Determine regeneration
        should_regenerate = (
            support == SupportScore.NO_SUPPORT
            or support == SupportScore.CONTRADICTS
            or utility.value < self.config.min_utility_for_accept.value
        )

        critique = "; ".join(issues) if issues else "Response appears adequate"

        return CriticResult(
            relevance=relevance,
            support=support,
            utility=utility,
            should_regenerate=should_regenerate,
            critique=critique,
            suggestions=suggestions,
            confidence=0.6,  # Lower confidence for heuristics
            details={
                "method": "heuristic",
                "citation_count": citation_count,
                "unsupported_patterns": unsupported_count,
                "query_response_overlap": round(query_response_overlap, 3),
            },
        )

    def should_regenerate(self, result: CriticResult) -> bool:
        """Check if regeneration is warranted based on critic result."""
        return result.should_regenerate

    def format_regeneration_prompt(
        self,
        original_query: str,
        original_response: str,
        critic_result: CriticResult,
        chunks: list[EvidenceChunk],
    ) -> str:
        """Format a prompt for regeneration based on critic feedback."""
        suggestions_text = "\n".join(f"- {s}" for s in critic_result.suggestions)

        return f"""Your previous response was evaluated and needs improvement.

## ORIGINAL QUESTION
{original_query}

## CRITIC FEEDBACK
- Relevance: {critic_result.relevance.value}
- Support: {critic_result.support.value}
- Utility: {critic_result.utility.value}/5
- Issues: {critic_result.critique}

## REQUIRED IMPROVEMENTS
{suggestions_text or "- Ensure all claims are supported by evidence with citations"}

## INSTRUCTIONS
Generate an improved response that addresses the feedback.
Ensure every claim has a citation [1], [2], etc.
Only use information from the provided evidence."""


# Singleton for easy access
_self_rag_critic: SelfRAGCritic | None = None


def get_self_rag_critic(config: SelfRAGConfig | None = None) -> SelfRAGCritic:
    """Get or create the Self-RAG critic instance."""
    global _self_rag_critic
    if _self_rag_critic is None or config is not None:
        _self_rag_critic = SelfRAGCritic(config)
    return _self_rag_critic


__all__ = [
    "RetrievalDecision",
    "RelevanceScore",
    "SupportScore",
    "UtilityScore",
    "CriticResult",
    "SelfRAGConfig",
    "SelfRAGCritic",
    "get_self_rag_critic",
]
