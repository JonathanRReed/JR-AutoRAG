"""Adaptive retrieval gating for intelligent retrieval decisions.

This module implements Self-RAG style retrieval gating:
- Assess if retrieval is needed at all (LLM may already know)
- Classify query complexity for appropriate retrieval strategy
- Support cost-aware decisions (skip retrieval for simple queries)

Based on: Self-RAG: Learning to Retrieve, Generate, and Critique
Paper: https://arxiv.org/abs/2310.11511
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .providers import LLMProvider


class GateDecision(str, Enum):
    """Retrieval gating decision."""
    NO_RETRIEVAL = "no_retrieval"      # LLM knows answer, skip retrieval
    SINGLE_RETRIEVAL = "single"        # Simple query, one retrieval pass
    ITERATIVE_RETRIEVAL = "iterative"  # Complex query, multiple passes
    CLARIFY_FIRST = "clarify"          # Ambiguous query, ask for clarification


@dataclass
class GateResult:
    """Result of adaptive gating decision."""
    decision: GateDecision
    confidence: float  # 0-1
    reasoning: str
    suggested_iterations: int = 1
    clarification_question: str | None = None


class AdaptiveGate:
    """Self-RAG style retrieval gating.
    
    Decides whether and how to retrieve based on:
    - Query type and complexity
    - LLM's self-assessed knowledge
    - Cost/latency considerations
    """
    
    # Patterns that suggest no retrieval needed
    NO_RETRIEVAL_PATTERNS = [
        r'^(hi|hello|hey|thanks|thank you|bye|goodbye)\b',
        r'^(what can you do|who are you|help me)\b',
        r'\b(calculate|compute|math|arithmetic)\b.*\b(\d+)\b',
        r'^(what is|what\'s)\s+\d+\s*[\+\-\*\/]\s*\d+',
        r'\b(today|current date|current time|what time)\b',
    ]
    
    # Patterns suggesting simple single retrieval
    SIMPLE_PATTERNS = [
        r'^(what is|what are|who is|who are|define)\b',
        r'^(when did|where is|where was)\b',
        r'\b(definition of|meaning of)\b',
    ]
    
    # Patterns suggesting complex iterative retrieval
    COMPLEX_PATTERNS = [
        r'\b(compare|contrast|difference between|similarities)\b',
        r'\b(explain how|explain why|analyze|evaluate)\b',
        r'\b(pros and cons|advantages and disadvantages)\b',
        r'\b(relationship between|connection between)\b',
        r'\band\b.*\band\b',  # Multiple "and"s suggest multi-part
        r'\?.*\?',  # Multiple questions
    ]
    
    # Patterns suggesting ambiguity
    AMBIGUOUS_PATTERNS = [
        r'^(it|this|that|they|them)\b',  # Pronouns without context
        r'\b(the thing|the one|the stuff)\b',
        r'^[\w\s]{1,10}$',  # Very short queries
    ]
    
    GATING_PROMPT = """Analyze this query and decide the best retrieval strategy.

Query: {query}

Options:
1. NO_RETRIEVAL: You can answer this directly without external documents (e.g., greetings, calculations, common knowledge)
2. SINGLE: Simple factual query that needs one retrieval pass
3. ITERATIVE: Complex query requiring multiple retrieval passes (comparisons, analysis, multi-part questions)
4. CLARIFY: Query is ambiguous and needs clarification before retrieval

Respond with EXACTLY this format:
DECISION: [NO_RETRIEVAL/SINGLE/ITERATIVE/CLARIFY]
CONFIDENCE: [0.0-1.0]
REASONING: [one sentence explanation]
ITERATIONS: [1-5, only if ITERATIVE]
CLARIFICATION: [question to ask user, only if CLARIFY, else "none"]"""

    def __init__(self) -> None:
        self._no_retrieval_re = [re.compile(p, re.IGNORECASE) for p in self.NO_RETRIEVAL_PATTERNS]
        self._simple_re = [re.compile(p, re.IGNORECASE) for p in self.SIMPLE_PATTERNS]
        self._complex_re = [re.compile(p, re.IGNORECASE) for p in self.COMPLEX_PATTERNS]
        self._ambiguous_re = [re.compile(p, re.IGNORECASE) for p in self.AMBIGUOUS_PATTERNS]
    
    def _count_matches(self, patterns: list, text: str) -> int:
        """Count pattern matches."""
        return sum(1 for p in patterns if p.search(text))
    
    def _estimate_complexity(self, query: str) -> float:
        """Estimate query complexity (0-1)."""
        score = 0.5  # Base
        
        # Length factor
        word_count = len(query.split())
        if word_count > 20:
            score += 0.2
        elif word_count < 5:
            score -= 0.2
        
        # Complexity patterns
        complex_matches = self._count_matches(self._complex_re, query)
        score += 0.15 * complex_matches
        
        # Simple patterns reduce complexity
        simple_matches = self._count_matches(self._simple_re, query)
        score -= 0.1 * simple_matches
        
        return max(0.0, min(1.0, score))
    
    def gate_heuristic(self, query: str) -> GateResult:
        """Make gating decision using heuristics (no LLM)."""
        # Check no-retrieval patterns first
        if any(p.match(query) for p in self._no_retrieval_re):
            return GateResult(
                decision=GateDecision.NO_RETRIEVAL,
                confidence=0.9,
                reasoning="Query matches no-retrieval pattern (greeting, calculation, etc.)",
            )
        
        # Check for ambiguity
        if any(p.match(query) for p in self._ambiguous_re):
            return GateResult(
                decision=GateDecision.CLARIFY_FIRST,
                confidence=0.7,
                reasoning="Query appears ambiguous or too short",
                clarification_question="Could you provide more context or details about what you're looking for?",
            )
        
        # Estimate complexity
        complexity = self._estimate_complexity(query)
        
        if complexity >= 0.7:
            iterations = 2 if complexity < 0.85 else 3
            return GateResult(
                decision=GateDecision.ITERATIVE_RETRIEVAL,
                confidence=0.8,
                reasoning=f"Complex query (score: {complexity:.2f}) requires iterative retrieval",
                suggested_iterations=iterations,
            )
        
        return GateResult(
            decision=GateDecision.SINGLE_RETRIEVAL,
            confidence=0.8,
            reasoning=f"Standard query (complexity: {complexity:.2f})",
            suggested_iterations=1,
        )
    
    async def gate_llm(
        self,
        query: str,
        provider: "LLMProvider",
    ) -> GateResult:
        """Make gating decision using LLM for higher accuracy."""
        prompt = self.GATING_PROMPT.format(query=query)
        
        try:
            response = await provider.chat([
                {"role": "system", "content": "You are a query analyzer that determines optimal retrieval strategy."},
                {"role": "user", "content": prompt},
            ])
            return self._parse_llm_response(response)
        except Exception:
            # Fallback to heuristic
            return self.gate_heuristic(query)
    
    def _parse_llm_response(self, response: str) -> GateResult:
        """Parse LLM gating response."""
        # Extract decision
        decision_match = re.search(
            r'DECISION:\s*(NO_RETRIEVAL|SINGLE|ITERATIVE|CLARIFY)',
            response,
            re.IGNORECASE
        )
        decision_str = decision_match.group(1).upper() if decision_match else "SINGLE"
        
        decision_map = {
            "NO_RETRIEVAL": GateDecision.NO_RETRIEVAL,
            "SINGLE": GateDecision.SINGLE_RETRIEVAL,
            "ITERATIVE": GateDecision.ITERATIVE_RETRIEVAL,
            "CLARIFY": GateDecision.CLARIFY_FIRST,
        }
        decision = decision_map.get(decision_str, GateDecision.SINGLE_RETRIEVAL)
        
        # Extract confidence
        confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.7
        confidence = max(0.0, min(1.0, confidence))
        
        # Extract reasoning
        reasoning_match = re.search(r'REASONING:\s*(.+?)(?=\n|$)', response)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "LLM gating decision"
        
        # Extract iterations (for iterative)
        iterations = 1
        if decision == GateDecision.ITERATIVE_RETRIEVAL:
            iter_match = re.search(r'ITERATIONS:\s*(\d+)', response)
            if iter_match:
                iterations = max(1, min(5, int(iter_match.group(1))))
        
        # Extract clarification (for clarify)
        clarification = None
        if decision == GateDecision.CLARIFY_FIRST:
            clarify_match = re.search(r'CLARIFICATION:\s*(.+?)(?=\n|$)', response)
            if clarify_match:
                clarify_text = clarify_match.group(1).strip()
                if clarify_text.lower() != "none":
                    clarification = clarify_text
        
        return GateResult(
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            suggested_iterations=iterations,
            clarification_question=clarification,
        )
    
    async def should_retrieve(
        self,
        query: str,
        provider: "LLMProvider | None" = None,
    ) -> GateResult:
        """Assess if retrieval is needed and what type.
        
        Uses LLM if available, otherwise falls back to heuristics.
        """
        if provider is not None:
            return await self.gate_llm(query, provider)
        return self.gate_heuristic(query)


__all__ = [
    "GateDecision",
    "GateResult",
    "AdaptiveGate",
]
