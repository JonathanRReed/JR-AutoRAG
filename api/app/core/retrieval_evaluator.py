"""CRAG-style retrieval quality evaluation.

This module provides retrieval context assessment BEFORE generation:
- Verdict scoring (CORRECT/AMBIGUOUS/INCORRECT)
- Knowledge strip extraction for noisy contexts
- Query refinement suggestions
- Web search fallback trigger

Based on: Corrective Retrieval Augmented Generation (CRAG)
Paper: https://arxiv.org/abs/2401.15884
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .providers import LLMProvider
    from .gatherer import EvidenceChunk


class RetrievalVerdict(str, Enum):
    """Quality verdict for retrieved context."""
    CORRECT = "correct"          # Context is relevant and sufficient
    AMBIGUOUS = "ambiguous"      # Partially relevant, may need refinement
    INCORRECT = "incorrect"      # Context is irrelevant, need retry/fallback


@dataclass
class EvaluationResult:
    """Result of retrieval quality evaluation."""
    verdict: RetrievalVerdict
    confidence: float  # 0-1
    reasoning: str
    knowledge_strips: list[str] = field(default_factory=list)
    suggested_query: str | None = None
    should_fallback_web: bool = False


@dataclass 
class KnowledgeStrip:
    """A relevant portion extracted from a chunk."""
    text: str
    chunk_id: str
    relevance_score: float


class RetrievalEvaluator:
    """Evaluate retrieved context quality before generation.
    
    Implements CRAG-style corrective retrieval:
    1. Score context relevance to query
    2. Extract knowledge strips from partially relevant context
    3. Suggest query refinements for ambiguous results
    4. Trigger web fallback for incorrect retrievals
    """
    
    # Thresholds for verdict classification
    CORRECT_THRESHOLD = 0.7
    AMBIGUOUS_THRESHOLD = 0.4
    
    # Patterns indicating high relevance
    RELEVANCE_PATTERNS = [
        r'\b(definition|meaning|refers to|is defined as)\b',
        r'\b(according to|states that|explains)\b',
        r'\b(because|therefore|thus|hence)\b',
    ]
    
    # Patterns indicating low relevance (noise)
    NOISE_PATTERNS = [
        r'\b(unrelated|off-topic|different topic)\b',
        r'\b(however|but|although|despite)\b.*\b(not|no|none)\b',
    ]
    
    LLM_EVALUATION_PROMPT = """Evaluate if the retrieved context is relevant and sufficient to answer the query.

Query: {query}

Retrieved Context:
{context}

Evaluate and respond with EXACTLY this format:
VERDICT: [CORRECT/AMBIGUOUS/INCORRECT]
CONFIDENCE: [0.0-1.0]
REASONING: [one sentence explanation]
RELEVANT_PARTS: [list key relevant sentences, or "none" if incorrect]
SUGGESTED_QUERY: [refined query if ambiguous, or "none"]

Be strict: CORRECT means the context directly answers the query.
AMBIGUOUS means some relevant info exists but incomplete.
INCORRECT means the context is irrelevant to the query."""

    def __init__(
        self,
        correct_threshold: float = 0.7,
        ambiguous_threshold: float = 0.4,
    ) -> None:
        self.correct_threshold = correct_threshold
        self.ambiguous_threshold = ambiguous_threshold
        self._relevance_re = [re.compile(p, re.IGNORECASE) for p in self.RELEVANCE_PATTERNS]
        self._noise_re = [re.compile(p, re.IGNORECASE) for p in self.NOISE_PATTERNS]
    
    def _compute_term_overlap(self, query: str, text: str) -> float:
        """Compute term overlap between query and text."""
        query_terms = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
        text_terms = set(re.findall(r'\b[a-z]{3,}\b', text.lower()))
        if not query_terms:
            return 0.0
        return len(query_terms & text_terms) / len(query_terms)
    
    def _count_relevance_signals(self, text: str) -> int:
        """Count relevance pattern matches."""
        return sum(1 for p in self._relevance_re if p.search(text))
    
    def _count_noise_signals(self, text: str) -> int:
        """Count noise pattern matches."""
        return sum(1 for p in self._noise_re if p.search(text))
    
    def _heuristic_score(self, query: str, chunks: list["EvidenceChunk"]) -> float:
        """Compute heuristic relevance score."""
        if not chunks:
            return 0.0
        
        scores = []
        for chunk in chunks:
            text = chunk.snippet if hasattr(chunk, 'snippet') else str(chunk)
            
            # Term overlap (40% weight)
            overlap = self._compute_term_overlap(query, text)
            
            # Relevance signals (30% weight)
            relevance = min(1.0, self._count_relevance_signals(text) / 3)
            
            # Noise penalty (30% weight)
            noise = min(1.0, self._count_noise_signals(text) / 2)
            
            # Retrieval score boost (if available)
            retrieval_score = getattr(chunk, 'score', 0.5)
            
            # Combined score
            chunk_score = (
                0.4 * overlap +
                0.3 * relevance +
                0.3 * (1 - noise) +
                0.2 * retrieval_score  # Bonus
            )
            scores.append(min(1.0, chunk_score))
        
        # Return weighted average (top chunks matter more)
        if len(scores) == 1:
            return scores[0]
        
        weights = [1 / (i + 1) for i in range(len(scores))]
        total_weight = sum(weights)
        return sum(s * w for s, w in zip(scores, weights)) / total_weight
    
    def _classify_verdict(self, score: float) -> RetrievalVerdict:
        """Classify score into verdict."""
        if score >= self.correct_threshold:
            return RetrievalVerdict.CORRECT
        elif score >= self.ambiguous_threshold:
            return RetrievalVerdict.AMBIGUOUS
        else:
            return RetrievalVerdict.INCORRECT
    
    def _extract_knowledge_strips_heuristic(
        self,
        query: str,
        chunks: list["EvidenceChunk"],
    ) -> list[str]:
        """Extract relevant sentences from chunks."""
        query_terms = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
        strips = []
        
        for chunk in chunks:
            text = chunk.snippet if hasattr(chunk, 'snippet') else str(chunk)
            
            # Split into sentences
            sentences = re.split(r'[.!?]+', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue
                
                # Check term overlap
                sent_terms = set(re.findall(r'\b[a-z]{3,}\b', sentence.lower()))
                overlap = len(query_terms & sent_terms)
                
                # Keep sentences with 2+ query term matches
                if overlap >= 2:
                    strips.append(sentence)
        
        return strips[:10]  # Limit to top 10 strips
    
    def _suggest_query_refinement(
        self,
        query: str,
        chunks: list["EvidenceChunk"],
    ) -> str | None:
        """Suggest a refined query based on context."""
        if not chunks:
            return None
        
        # Extract key terms from top chunks that aren't in query
        query_terms = set(re.findall(r'\b[a-z]{4,}\b', query.lower()))
        chunk_terms: dict[str, int] = {}
        
        for chunk in chunks[:3]:
            text = chunk.snippet if hasattr(chunk, 'snippet') else str(chunk)
            for term in re.findall(r'\b[a-z]{4,}\b', text.lower()):
                if term not in query_terms:
                    chunk_terms[term] = chunk_terms.get(term, 0) + 1
        
        # Get most frequent new terms
        new_terms = sorted(chunk_terms.items(), key=lambda x: x[1], reverse=True)[:2]
        
        if new_terms:
            additions = ' '.join(t[0] for t in new_terms)
            return f"{query} {additions}"
        
        return None
    
    def evaluate_heuristic(
        self,
        query: str,
        chunks: list["EvidenceChunk"],
    ) -> EvaluationResult:
        """Evaluate context using heuristics (no LLM required)."""
        if not chunks:
            return EvaluationResult(
                verdict=RetrievalVerdict.INCORRECT,
                confidence=1.0,
                reasoning="No chunks retrieved",
                should_fallback_web=True,
            )
        
        score = self._heuristic_score(query, chunks)
        verdict = self._classify_verdict(score)
        
        result = EvaluationResult(
            verdict=verdict,
            confidence=score,
            reasoning=f"Heuristic score: {score:.2f}",
        )
        
        if verdict == RetrievalVerdict.AMBIGUOUS:
            result.knowledge_strips = self._extract_knowledge_strips_heuristic(query, chunks)
            result.suggested_query = self._suggest_query_refinement(query, chunks)
        elif verdict == RetrievalVerdict.INCORRECT:
            result.should_fallback_web = True
            result.suggested_query = self._suggest_query_refinement(query, chunks)
        
        return result
    
    async def evaluate_llm(
        self,
        query: str,
        chunks: list["EvidenceChunk"],
        provider: "LLMProvider",
    ) -> EvaluationResult:
        """Evaluate context using LLM for higher accuracy."""
        if not chunks:
            return EvaluationResult(
                verdict=RetrievalVerdict.INCORRECT,
                confidence=1.0,
                reasoning="No chunks retrieved",
                should_fallback_web=True,
            )
        
        # Build context string
        context_parts = []
        for i, chunk in enumerate(chunks[:5]):  # Limit to top 5
            text = chunk.snippet if hasattr(chunk, 'snippet') else str(chunk)
            context_parts.append(f"[{i+1}] {text[:500]}")
        context = "\n\n".join(context_parts)
        
        prompt = self.LLM_EVALUATION_PROMPT.format(query=query, context=context)
        
        try:
            response = await provider.chat([
                {"role": "system", "content": "You are a retrieval quality evaluator."},
                {"role": "user", "content": prompt},
            ])
            return self._parse_llm_response(response, query, chunks)
        except Exception:
            # Fallback to heuristic
            return self.evaluate_heuristic(query, chunks)
    
    def _parse_llm_response(
        self,
        response: str,
        query: str,
        chunks: list["EvidenceChunk"],
    ) -> EvaluationResult:
        """Parse LLM evaluation response."""
        # Extract verdict
        verdict_match = re.search(r'VERDICT:\s*(CORRECT|AMBIGUOUS|INCORRECT)', response, re.IGNORECASE)
        verdict_str = verdict_match.group(1).upper() if verdict_match else "AMBIGUOUS"
        verdict = RetrievalVerdict(verdict_str.lower())
        
        # Extract confidence
        confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.5
        confidence = max(0.0, min(1.0, confidence))
        
        # Extract reasoning
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?=\n|$)', response)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "LLM evaluation"
        
        # Extract relevant parts as knowledge strips
        strips_match = re.search(r'RELEVANT_PARTS:\s*(.+?)(?=SUGGESTED_QUERY|$)', response, re.DOTALL)
        strips_text = strips_match.group(1).strip() if strips_match else ""
        knowledge_strips = []
        if strips_text.lower() != "none":
            # Split by common delimiters
            for strip in re.split(r'[\n•\-\d\.]+', strips_text):
                strip = strip.strip()
                if len(strip) > 20:
                    knowledge_strips.append(strip)
        
        # Extract suggested query
        suggested_match = re.search(r'SUGGESTED_QUERY:\s*(.+?)(?=\n|$)', response)
        suggested_query = None
        if suggested_match:
            suggested = suggested_match.group(1).strip()
            if suggested.lower() != "none":
                suggested_query = suggested
        
        return EvaluationResult(
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            knowledge_strips=knowledge_strips,
            suggested_query=suggested_query,
            should_fallback_web=(verdict == RetrievalVerdict.INCORRECT),
        )
    
    async def evaluate(
        self,
        query: str,
        chunks: list["EvidenceChunk"],
        provider: "LLMProvider | None" = None,
    ) -> EvaluationResult:
        """Evaluate retrieval quality using LLM if available, else heuristics."""
        if provider is not None:
            return await self.evaluate_llm(query, chunks, provider)
        return self.evaluate_heuristic(query, chunks)
    
    def extract_knowledge_strips(
        self,
        chunks: list["EvidenceChunk"],
        query: str,
    ) -> list[KnowledgeStrip]:
        """Extract only relevant portions from each chunk.
        
        This is the "knowledge refinement" step from CRAG that filters
        out irrelevant parts of partially-relevant chunks.
        """
        query_terms = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
        strips: list[KnowledgeStrip] = []
        
        for chunk in chunks:
            chunk_id = getattr(chunk, 'id', str(id(chunk)))
            text = chunk.snippet if hasattr(chunk, 'snippet') else str(chunk)
            
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 30:
                    continue
                
                # Score sentence
                sent_terms = set(re.findall(r'\b[a-z]{3,}\b', sentence.lower()))
                if not query_terms:
                    continue
                    
                overlap = len(query_terms & sent_terms) / len(query_terms)
                relevance_boost = 0.1 * self._count_relevance_signals(sentence)
                score = min(1.0, overlap + relevance_boost)
                
                if score >= 0.3:  # Minimum relevance threshold
                    strips.append(KnowledgeStrip(
                        text=sentence,
                        chunk_id=chunk_id,
                        relevance_score=score,
                    ))
        
        # Sort by relevance and deduplicate
        strips.sort(key=lambda s: s.relevance_score, reverse=True)
        seen_texts: set[str] = set()
        unique_strips: list[KnowledgeStrip] = []
        
        for strip in strips:
            # Normalize for dedup
            normalized = strip.text.lower()[:50]
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                unique_strips.append(strip)
        
        return unique_strips[:15]  # Limit to top 15


__all__ = [
    "RetrievalVerdict",
    "EvaluationResult",
    "KnowledgeStrip",
    "RetrievalEvaluator",
]
