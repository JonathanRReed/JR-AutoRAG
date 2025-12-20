"""LLM-as-judge automated evaluation.

This module provides automated answer quality assessment:
- Faithfulness evaluation (grounded in context)
- Relevance scoring (addresses the query)
- Completeness checking
- Coherence analysis
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .providers import LLMProvider


class EvaluationCriterion(str, Enum):
    """Criteria for evaluation."""
    FAITHFULNESS = "faithfulness"
    RELEVANCE = "relevance"
    COMPLETENESS = "completeness"
    COHERENCE = "coherence"
    CITATION_ACCURACY = "citation_accuracy"


@dataclass
class JudgmentResult:
    """Result of a single judgment."""
    criterion: EvaluationCriterion
    score: float  # 0-1
    explanation: str
    evidence: list[str]


@dataclass
class EvaluationScore:
    """Complete evaluation scores."""
    faithfulness: float
    relevance: float
    completeness: float
    coherence: float
    citation_accuracy: float
    overall: float
    explanations: dict[str, str]
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "faithfulness": self.faithfulness,
            "relevance": self.relevance,
            "completeness": self.completeness,
            "coherence": self.coherence,
            "citation_accuracy": self.citation_accuracy,
            "overall": self.overall,
            "explanations": self.explanations,
        }


class HeuristicEvaluator:
    """Heuristic-based evaluation (no LLM required)."""
    
    def evaluate_faithfulness(
        self,
        answer: str,
        context: str,
    ) -> tuple[float, str]:
        """Check if answer is grounded in context."""
        # Extract key statements from answer
        answer_sentences = re.split(r'[.!?]', answer)
        answer_words = set(re.findall(r'\b[a-z]{4,}\b', answer.lower()))
        context_words = set(re.findall(r'\b[a-z]{4,}\b', context.lower()))
        
        if not answer_words:
            return 1.0, "Empty answer"
        
        overlap = len(answer_words & context_words) / len(answer_words)
        
        if overlap > 0.7:
            return 0.9, "High overlap with context"
        elif overlap > 0.5:
            return 0.7, "Moderate overlap with context"
        elif overlap > 0.3:
            return 0.5, "Low overlap - may include external knowledge"
        else:
            return 0.3, "Very low overlap - possibly hallucinated"
    
    def evaluate_relevance(
        self,
        answer: str,
        query: str,
    ) -> tuple[float, str]:
        """Check if answer is relevant to query."""
        query_words = set(re.findall(r'\b[a-z]{4,}\b', query.lower()))
        answer_words = set(re.findall(r'\b[a-z]{4,}\b', answer.lower()))
        
        if not query_words:
            return 1.0, "Empty query"
        
        overlap = len(query_words & answer_words) / len(query_words)
        
        if overlap > 0.6:
            return 0.9, "Answer addresses query terms"
        elif overlap > 0.4:
            return 0.7, "Answer partially addresses query"
        elif overlap > 0.2:
            return 0.5, "Answer has limited relevance"
        else:
            return 0.3, "Answer may not address the query"
    
    def evaluate_completeness(
        self,
        answer: str,
        query: str,
    ) -> tuple[float, str]:
        """Check if answer fully addresses the query."""
        # Check answer length
        word_count = len(answer.split())
        
        # Check for question words addressed
        question_words = re.findall(r'\b(what|why|how|when|where|who|which)\b', query.lower())
        
        if word_count < 20:
            return 0.4, "Answer may be too brief"
        elif word_count > 500:
            return 0.8, "Comprehensive answer"
        else:
            return 0.7, "Reasonable answer length"
    
    def evaluate_coherence(
        self,
        answer: str,
    ) -> tuple[float, str]:
        """Check if answer is well-structured."""
        sentences = re.split(r'[.!?]', answer)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0.5, "Could not parse sentences"
        
        # Check basic structure
        has_structure = bool(re.search(r'\n|^\d+\.|^-\s', answer, re.MULTILINE))
        avg_sentence_len = sum(len(s.split()) for s in sentences) / len(sentences)
        
        score = 0.6
        explanation_parts = []
        
        if has_structure:
            score += 0.2
            explanation_parts.append("has structure")
        
        if 10 <= avg_sentence_len <= 30:
            score += 0.2
            explanation_parts.append("good sentence length")
        
        return min(score, 1.0), ", ".join(explanation_parts) or "basic coherence"
    
    def evaluate_citations(
        self,
        answer: str,
        num_sources: int,
    ) -> tuple[float, str]:
        """Check citation usage."""
        citations = re.findall(r'\[(\d+)\]', answer)
        unique_citations = set(int(c) for c in citations)
        
        if num_sources == 0:
            return 1.0, "No sources to cite"
        
        if not citations:
            return 0.3, "No citations used"
        
        # Check if citations are valid (within source range)
        valid = all(1 <= c <= num_sources for c in unique_citations)
        coverage = len(unique_citations) / num_sources
        
        if not valid:
            return 0.4, "Some citations reference non-existent sources"
        elif coverage > 0.5:
            return 0.9, "Good citation coverage"
        else:
            return 0.7, "Some sources cited"
    
    def evaluate(
        self,
        answer: str,
        query: str,
        context: str,
        num_sources: int = 0,
    ) -> EvaluationScore:
        """Full heuristic evaluation."""
        faith_score, faith_exp = self.evaluate_faithfulness(answer, context)
        rel_score, rel_exp = self.evaluate_relevance(answer, query)
        comp_score, comp_exp = self.evaluate_completeness(answer, query)
        coh_score, coh_exp = self.evaluate_coherence(answer)
        cit_score, cit_exp = self.evaluate_citations(answer, num_sources)
        
        # Weighted overall score
        overall = (
            faith_score * 0.3 +
            rel_score * 0.3 +
            comp_score * 0.2 +
            coh_score * 0.1 +
            cit_score * 0.1
        )
        
        return EvaluationScore(
            faithfulness=faith_score,
            relevance=rel_score,
            completeness=comp_score,
            coherence=coh_score,
            citation_accuracy=cit_score,
            overall=overall,
            explanations={
                "faithfulness": faith_exp,
                "relevance": rel_exp,
                "completeness": comp_exp,
                "coherence": coh_exp,
                "citation_accuracy": cit_exp,
            },
        )


class LLMJudge:
    """LLM-based evaluation (when provider available)."""
    
    EVALUATION_PROMPT = """Evaluate the following answer based on the given criteria.

Query: {query}

Context (source material):
{context}

Answer to evaluate:
{answer}

Rate the answer on these criteria (0-10 scale):
1. FAITHFULNESS: Is the answer grounded in the provided context?
2. RELEVANCE: Does the answer address the query?
3. COMPLETENESS: Is the answer thorough and complete?
4. COHERENCE: Is the answer well-structured and clear?
5. CITATION_ACCURACY: Are citations used correctly?

Respond in this exact format:
FAITHFULNESS: [score]/10 - [brief explanation]
RELEVANCE: [score]/10 - [brief explanation]
COMPLETENESS: [score]/10 - [brief explanation]
COHERENCE: [score]/10 - [brief explanation]
CITATION_ACCURACY: [score]/10 - [brief explanation]
OVERALL: [score]/10"""

    def __init__(self, provider: "LLMProvider | None" = None) -> None:
        self._provider = provider
        self._heuristic = HeuristicEvaluator()
    
    def set_provider(self, provider: "LLMProvider") -> None:
        """Set or update the LLM provider."""
        self._provider = provider
    
    async def evaluate_async(
        self,
        answer: str,
        query: str,
        context: str,
        num_sources: int = 0,
    ) -> EvaluationScore:
        """Evaluate using LLM if available, else heuristics."""
        if not self._provider:
            return self._heuristic.evaluate(answer, query, context, num_sources)
        
        prompt = self.EVALUATION_PROMPT.format(
            query=query,
            context=context[:2000],  # Limit context length
            answer=answer,
        )
        
        try:
            response = await self._provider.chat([
                {"role": "system", "content": "You are an expert evaluator of RAG system answers."},
                {"role": "user", "content": prompt},
            ])
            return self._parse_response(response)
        except Exception:
            # Fallback to heuristics
            return self._heuristic.evaluate(answer, query, context, num_sources)
    
    def evaluate(
        self,
        answer: str,
        query: str,
        context: str,
        num_sources: int = 0,
    ) -> EvaluationScore:
        """Synchronous evaluation using heuristics only."""
        return self._heuristic.evaluate(answer, query, context, num_sources)
    
    def _parse_response(self, response: str) -> EvaluationScore:
        """Parse LLM evaluation response."""
        scores = {}
        explanations = {}
        
        patterns = {
            "faithfulness": r"FAITHFULNESS:\s*(\d+(?:\.\d+)?)/10\s*-?\s*(.+?)(?:\n|$)",
            "relevance": r"RELEVANCE:\s*(\d+(?:\.\d+)?)/10\s*-?\s*(.+?)(?:\n|$)",
            "completeness": r"COMPLETENESS:\s*(\d+(?:\.\d+)?)/10\s*-?\s*(.+?)(?:\n|$)",
            "coherence": r"COHERENCE:\s*(\d+(?:\.\d+)?)/10\s*-?\s*(.+?)(?:\n|$)",
            "citation_accuracy": r"CITATION_ACCURACY:\s*(\d+(?:\.\d+)?)/10\s*-?\s*(.+?)(?:\n|$)",
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                scores[key] = float(match.group(1)) / 10
                explanations[key] = match.group(2).strip()
            else:
                scores[key] = 0.5
                explanations[key] = "Could not parse"
        
        # Parse overall
        overall_match = re.search(r"OVERALL:\s*(\d+(?:\.\d+)?)/10", response, re.IGNORECASE)
        overall = float(overall_match.group(1)) / 10 if overall_match else sum(scores.values()) / len(scores)
        
        return EvaluationScore(
            faithfulness=scores["faithfulness"],
            relevance=scores["relevance"],
            completeness=scores["completeness"],
            coherence=scores["coherence"],
            citation_accuracy=scores["citation_accuracy"],
            overall=overall,
            explanations=explanations,
        )


__all__ = [
    "EvaluationCriterion",
    "JudgmentResult",
    "EvaluationScore",
    "HeuristicEvaluator",
    "LLMJudge",
]
