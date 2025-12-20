"""Self-reflection and answer evaluation for agentic RAG.

This module provides answer quality evaluation:
- Confidence scoring for generated answers
- Detection of low-quality or uncertain responses
- Triggering re-retrieval when quality is low
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gatherer import EvidenceChunk


class AnswerQuality(str, Enum):
    """Quality classification for answers."""
    HIGH = "high"           # Confident, well-supported
    MEDIUM = "medium"       # Reasonable but could be better
    LOW = "low"             # Uncertain or poorly supported
    INSUFFICIENT = "insufficient"  # Cannot answer from context


@dataclass
class ReflectionResult:
    """Result of self-reflection on an answer."""
    quality: AnswerQuality
    confidence: float  # 0-1
    issues: list[str]
    suggestions: list[str]
    should_retry: bool


class SelfReflector:
    """Evaluates answer quality and decides if re-retrieval is needed.
    
    Uses heuristics to detect:
    - Uncertainty language ("I'm not sure", "might", "perhaps")
    - Lack of citations or evidence
    - Very short or very long answers
    - Signs of hallucination
    """
    
    # Patterns indicating uncertainty
    UNCERTAINTY_PATTERNS = [
        r"\bi('m| am) not (sure|certain)\b",
        r"\bI don't (know|have|think)\b",
        r"\bperhaps\b",
        r"\bmaybe\b",
        r"\bmight be\b",
        r"\bcould be\b",
        r"\bpossibly\b",
        r"\bI think\b",
        r"\bI believe\b",
        r"\bnot enough information\b",
        r"\bcannot (find|determine|answer)\b",
    ]
    
    # Patterns indicating the model is refusing to answer
    REFUSAL_PATTERNS = [
        r"\bI cannot\b",
        r"\bI'm unable to\b",
        r"\bno information (about|on|regarding)\b",
        r"\bdoes not (contain|mention|include)\b",
        r"\bnot (found|mentioned|included) in\b",
    ]
    
    # Minimum answer length for "real" answers
    MIN_ANSWER_LENGTH = 50
    MAX_REASONABLE_LENGTH = 2000
    
    def __init__(
        self,
        min_confidence_threshold: float = 0.5,
        min_chunks_for_confidence: int = 2,
    ) -> None:
        self.min_confidence_threshold = min_confidence_threshold
        self.min_chunks_for_confidence = min_chunks_for_confidence
        self._uncertainty_re = [
            re.compile(p, re.IGNORECASE) for p in self.UNCERTAINTY_PATTERNS
        ]
        self._refusal_re = [
            re.compile(p, re.IGNORECASE) for p in self.REFUSAL_PATTERNS
        ]
    
    def _count_uncertainty(self, text: str) -> int:
        """Count uncertainty markers in text."""
        return sum(1 for p in self._uncertainty_re if p.search(text))
    
    def _count_refusals(self, text: str) -> int:
        """Count refusal markers in text."""
        return sum(1 for p in self._refusal_re if p.search(text))
    
    def _count_citations(self, text: str) -> int:
        """Count citation references like [1], [2]."""
        return len(re.findall(r'\[\d+\]', text))
    
    def reflect(
        self,
        answer: str,
        query: str,
        chunks: list["EvidenceChunk"],
        context_used: str = "",
    ) -> ReflectionResult:
        """Evaluate the quality of a generated answer.
        
        Args:
            answer: Generated answer text
            query: Original user query
            chunks: Evidence chunks used for generation
            context_used: Full context string sent to LLM
        
        Returns:
            ReflectionResult with quality assessment
        """
        issues: list[str] = []
        suggestions: list[str] = []
        confidence = 1.0
        
        # Check answer length
        answer_len = len(answer)
        if answer_len < self.MIN_ANSWER_LENGTH:
            issues.append("Answer is very short")
            confidence -= 0.3
            suggestions.append("Try retrieving more context")
        elif answer_len > self.MAX_REASONABLE_LENGTH:
            issues.append("Answer may be overly verbose")
            confidence -= 0.1
        
        # Check for uncertainty language
        uncertainty_count = self._count_uncertainty(answer)
        if uncertainty_count >= 3:
            issues.append(f"High uncertainty language ({uncertainty_count} markers)")
            confidence -= 0.3
            suggestions.append("Re-retrieve with expanded query")
        elif uncertainty_count >= 1:
            issues.append(f"Some uncertainty detected ({uncertainty_count} markers)")
            confidence -= 0.15
        
        # Check for refusal patterns
        refusal_count = self._count_refusals(answer)
        if refusal_count >= 2:
            issues.append("Answer indicates insufficient information")
            confidence -= 0.4
            suggestions.append("Try different search terms")
        
        # Check evidence support
        if len(chunks) < self.min_chunks_for_confidence:
            issues.append(f"Only {len(chunks)} evidence chunks (minimum: {self.min_chunks_for_confidence})")
            confidence -= 0.2
            suggestions.append("Increase retrieval depth")
        
        # Check for citations in answer
        citation_count = self._count_citations(answer)
        if citation_count == 0 and len(chunks) > 0:
            issues.append("No citations in answer despite having evidence")
            confidence -= 0.1
        
        # Check query terms appear in answer
        query_terms = set(re.findall(r'\b[a-z]{4,}\b', query.lower()))
        answer_terms = set(re.findall(r'\b[a-z]{4,}\b', answer.lower()))
        term_overlap = len(query_terms & answer_terms) / len(query_terms) if query_terms else 0
        if term_overlap < 0.3:
            issues.append("Low overlap between query and answer terms")
            confidence -= 0.15
        
        # Clamp confidence
        confidence = max(0.0, min(1.0, confidence))
        
        # Determine quality level
        if confidence >= 0.8:
            quality = AnswerQuality.HIGH
        elif confidence >= 0.6:
            quality = AnswerQuality.MEDIUM
        elif confidence >= 0.4:
            quality = AnswerQuality.LOW
        else:
            quality = AnswerQuality.INSUFFICIENT
        
        # Determine if retry is warranted
        should_retry = (
            quality in (AnswerQuality.LOW, AnswerQuality.INSUFFICIENT)
            and confidence < self.min_confidence_threshold
        )
        
        return ReflectionResult(
            quality=quality,
            confidence=confidence,
            issues=issues,
            suggestions=suggestions,
            should_retry=should_retry,
        )


__all__ = [
    "AnswerQuality",
    "ReflectionResult",
    "SelfReflector",
]
