"""RAGAS-style reference-free RAG evaluation metrics.

Implements evaluation metrics from RAGAS (Retrieval Augmented Generation Assessment)
without requiring gold-standard reference answers:
- Faithfulness: Are claims supported by retrieved context?
- Answer Relevance: Does answer address the query?
- Context Precision: Is retrieved context relevant to query?
- Context Recall: Does context cover required information?

Paper: https://arxiv.org/abs/2309.15217
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .gatherer import EvidenceChunk
    from .providers import LLMProvider


@dataclass
class RAGASMetrics:
    """Complete RAGAS evaluation metrics."""
    faithfulness: float  # 0-1: claims supported by context
    answer_relevance: float  # 0-1: answer addresses query
    context_precision: float  # 0-1: context relevance to query
    context_recall: float  # 0-1: context completeness
    overall_score: float  # Weighted average
    details: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "faithfulness": round(self.faithfulness, 3),
            "answer_relevance": round(self.answer_relevance, 3),
            "context_precision": round(self.context_precision, 3),
            "context_recall": round(self.context_recall, 3),
            "overall_score": round(self.overall_score, 3),
        }


@dataclass
class InvocationMetrics:
    """Metrics for retrieval invocation correctness."""
    should_have_retrieved: bool
    did_retrieve: bool
    correct_invocation: bool
    retrieval_helped: bool  # Did retrieval improve answer?
    unnecessary_retrieval: bool  # Retrieved when not needed?
    missed_retrieval: bool  # Didn't retrieve when needed?
    
    def to_dict(self) -> dict[str, bool]:
        return {
            "should_have_retrieved": self.should_have_retrieved,
            "did_retrieve": self.did_retrieve,
            "correct_invocation": self.correct_invocation,
            "retrieval_helped": self.retrieval_helped,
            "unnecessary_retrieval": self.unnecessary_retrieval,
            "missed_retrieval": self.missed_retrieval,
        }


class RAGASEvaluator:
    """Reference-free RAG evaluation using RAGAS methodology.
    
    Evaluates RAG outputs without gold-standard answers by:
    1. Extracting claims from generated answer
    2. Checking claim support in retrieved context
    3. Measuring query-answer alignment
    4. Assessing context quality
    """
    
    # Stopwords for overlap calculations
    STOPWORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'and', 'but', 'or',
        'if', 'then', 'else', 'when', 'where', 'why', 'how', 'all', 'each',
        'for', 'to', 'of', 'in', 'on', 'at', 'by', 'with', 'from', 'as',
        'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them',
    }
    
    def __init__(
        self,
        faithfulness_weight: float = 0.3,
        relevance_weight: float = 0.3,
        precision_weight: float = 0.2,
        recall_weight: float = 0.2,
    ):
        """Initialize evaluator with metric weights."""
        self.weights = {
            "faithfulness": faithfulness_weight,
            "relevance": relevance_weight,
            "precision": precision_weight,
            "recall": recall_weight,
        }
    
    def _tokenize(self, text: str) -> set[str]:
        """Tokenize and filter stopwords."""
        words = re.findall(r'\b[a-z]+\b', text.lower())
        return {w for w in words if w not in self.STOPWORDS and len(w) > 2}
    
    def _extract_sentences(self, text: str) -> list[str]:
        """Extract sentences from text."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.split()) >= 4]
    
    def _extract_claims(self, answer: str) -> list[str]:
        """Extract factual claims from answer."""
        sentences = self._extract_sentences(answer)
        claims = []
        
        # Filter to likely factual claims
        claim_patterns = [
            r'\b(is|are|was|were|has|have|will|can)\b',
            r'\d+',  # Contains numbers
            r'\b(because|therefore|thus|since)\b',
        ]
        
        for sentence in sentences:
            # Skip meta sentences
            if sentence.lower().startswith(('note:', 'sources:', '##', '[', 'references')):
                continue
            if any(re.search(p, sentence, re.I) for p in claim_patterns):
                claims.append(sentence)
        
        return claims
    
    def evaluate_faithfulness(
        self,
        answer: str,
        chunks: list["EvidenceChunk"],
    ) -> tuple[float, dict]:
        """Evaluate if answer claims are supported by context.
        
        Returns:
            Tuple of (score, details)
        """
        claims = self._extract_claims(answer)
        if not claims:
            return 1.0, {"claims": 0, "supported": 0}
        
        # Build context text
        context_text = " ".join(c.snippet.lower() for c in chunks)
        context_terms = self._tokenize(context_text)
        
        supported_count = 0
        claim_details = []
        
        for claim in claims:
            claim_terms = self._tokenize(claim)
            if not claim_terms:
                continue
            
            # Check overlap with context
            overlap = len(claim_terms & context_terms) / len(claim_terms)
            
            # Check for citation markers
            has_citation = bool(re.search(r'\[\d+\]|\(Doc:|ChunkID:', claim))
            
            is_supported = overlap > 0.4 or has_citation
            if is_supported:
                supported_count += 1
            
            claim_details.append({
                "claim": claim[:80] + "..." if len(claim) > 80 else claim,
                "overlap": round(overlap, 3),
                "has_citation": has_citation,
                "supported": is_supported,
            })
        
        score = supported_count / len(claims) if claims else 1.0
        return score, {"claims": len(claims), "supported": supported_count, "details": claim_details[:5]}
    
    def evaluate_answer_relevance(
        self,
        query: str,
        answer: str,
    ) -> tuple[float, dict]:
        """Evaluate if answer addresses the query.
        
        Returns:
            Tuple of (score, details)
        """
        query_terms = self._tokenize(query)
        answer_terms = self._tokenize(answer)
        
        if not query_terms:
            return 1.0, {"query_terms": 0}
        
        # Calculate term coverage
        query_coverage = len(query_terms & answer_terms) / len(query_terms)
        
        # Check if answer addresses question type
        wh_patterns = {
            'what': r'\b(is|are|refers to|means|definition)\b',
            'how': r'\b(by|through|using|steps?|process)\b',
            'why': r'\b(because|reason|due to|since|therefore)\b',
            'when': r'\b(in \d{4}|\d+ (days?|years?|months?))\b',
            'where': r'\b(in|at|located|place|location)\b',
        }
        
        question_type = None
        for wh, pattern in wh_patterns.items():
            if query.lower().startswith(wh):
                question_type = wh
                break
        
        type_addressed = 1.0
        if question_type and question_type in wh_patterns:
            if re.search(wh_patterns[question_type], answer, re.I):
                type_addressed = 1.0
            else:
                type_addressed = 0.5
        
        # Combined score
        score = 0.6 * query_coverage + 0.4 * type_addressed
        
        return score, {
            "query_terms": len(query_terms),
            "answer_terms": len(answer_terms),
            "coverage": round(query_coverage, 3),
            "question_type": question_type,
            "type_addressed": type_addressed,
        }
    
    def evaluate_context_precision(
        self,
        query: str,
        chunks: list["EvidenceChunk"],
    ) -> tuple[float, dict]:
        """Evaluate if retrieved context is relevant to query.
        
        Returns:
            Tuple of (score, details)
        """
        if not chunks:
            return 0.0, {"chunks": 0}
        
        query_terms = self._tokenize(query)
        if not query_terms:
            return 1.0, {"chunks": len(chunks)}
        
        # Score each chunk's relevance
        chunk_scores = []
        for chunk in chunks:
            chunk_terms = self._tokenize(chunk.snippet)
            if chunk_terms:
                overlap = len(query_terms & chunk_terms) / len(query_terms)
            else:
                overlap = 0.0
            chunk_scores.append({
                "id": chunk.id,
                "retrieval_score": chunk.score,
                "term_overlap": round(overlap, 3),
            })
        
        # Weight by position (earlier chunks matter more)
        weighted_sum = 0.0
        weight_total = 0.0
        for i, cs in enumerate(chunk_scores):
            weight = 1.0 / (i + 1)  # Inverse position weight
            weighted_sum += cs["term_overlap"] * weight
            weight_total += weight
        
        precision = weighted_sum / weight_total if weight_total > 0 else 0.0
        
        return precision, {
            "chunks": len(chunks),
            "chunk_scores": chunk_scores[:5],
        }
    
    def evaluate_context_recall(
        self,
        query: str,
        answer: str,
        chunks: list["EvidenceChunk"],
    ) -> tuple[float, dict]:
        """Evaluate if context covers information needed for answer.
        
        Approximates recall by checking if answer terms appear in context.
        """
        answer_terms = self._tokenize(answer)
        if not answer_terms:
            return 1.0, {"answer_terms": 0}
        
        context_text = " ".join(c.snippet.lower() for c in chunks)
        context_terms = self._tokenize(context_text)
        
        # Terms in answer that came from context
        covered = len(answer_terms & context_terms)
        recall = covered / len(answer_terms)
        
        # Penalty for answer content not in context (potential hallucination)
        uncovered_terms = answer_terms - context_terms
        
        return recall, {
            "answer_terms": len(answer_terms),
            "covered_terms": covered,
            "uncovered_terms": len(uncovered_terms),
        }
    
    def evaluate(
        self,
        query: str,
        answer: str,
        chunks: list["EvidenceChunk"],
    ) -> RAGASMetrics:
        """Run complete RAGAS evaluation.
        
        Args:
            query: User query
            answer: Generated answer
            chunks: Retrieved context chunks
            
        Returns:
            RAGASMetrics with all scores
        """
        # Evaluate each metric
        faithfulness, faith_details = self.evaluate_faithfulness(answer, chunks)
        relevance, rel_details = self.evaluate_answer_relevance(query, answer)
        precision, prec_details = self.evaluate_context_precision(query, chunks)
        recall, recall_details = self.evaluate_context_recall(query, answer, chunks)
        
        # Weighted overall score
        overall = (
            self.weights["faithfulness"] * faithfulness +
            self.weights["relevance"] * relevance +
            self.weights["precision"] * precision +
            self.weights["recall"] * recall
        )
        
        return RAGASMetrics(
            faithfulness=faithfulness,
            answer_relevance=relevance,
            context_precision=precision,
            context_recall=recall,
            overall_score=overall,
            details={
                "faithfulness": faith_details,
                "relevance": rel_details,
                "precision": prec_details,
                "recall": recall_details,
            },
        )


class InvocationEvaluator:
    """Evaluate retrieval invocation correctness.
    
    Determines if retrieval was necessary and helpful,
    detecting both unnecessary retrieval and missed retrieval cases.
    """
    
    # Queries that typically don't need retrieval
    NO_RETRIEVAL_PATTERNS = [
        r'^(hi|hello|hey|thanks?|bye|goodbye)\b',
        r'^what can you (do|help)',
        r'^who are you\b',
        r'\b(calculate|compute|math)\b.*\d+',
    ]
    
    def __init__(self):
        self._patterns = [re.compile(p, re.I) for p in self.NO_RETRIEVAL_PATTERNS]
    
    def _should_have_retrieved(self, query: str, answer: str) -> bool:
        """Determine if query required retrieval."""
        # Check if it's a simple query
        for pattern in self._patterns:
            if pattern.search(query):
                return False
        
        # Check if answer suggests retrieval was needed
        needs_retrieval_signals = [
            r'\b(according to|based on|the document|source)\b',
            r'\[\d+\]',  # Citations
            r'\b(states|mentions|describes|explains)\b',
        ]
        
        for pattern in needs_retrieval_signals:
            if re.search(pattern, answer, re.I):
                return True
        
        # If answer is substantive, likely needed retrieval
        return len(answer.split()) > 30
    
    def evaluate(
        self,
        query: str,
        answer: str,
        did_retrieve: bool,
        chunks_used: int,
        answer_quality: float,
    ) -> InvocationMetrics:
        """Evaluate if retrieval invocation was correct.
        
        Args:
            query: User query
            answer: Generated answer
            did_retrieve: Whether retrieval was performed
            chunks_used: Number of chunks used
            answer_quality: Quality score of answer (0-1)
            
        Returns:
            InvocationMetrics with correctness assessment
        """
        should_retrieve = self._should_have_retrieved(query, answer)
        
        correct = (should_retrieve == did_retrieve)
        
        # Retrieval helped if quality is good and we did retrieve
        retrieval_helped = did_retrieve and answer_quality > 0.6 and chunks_used > 0
        
        # Unnecessary if we didn't need to but did
        unnecessary = did_retrieve and not should_retrieve
        
        # Missed if we should have but didn't
        missed = should_retrieve and not did_retrieve
        
        return InvocationMetrics(
            should_have_retrieved=should_retrieve,
            did_retrieve=did_retrieve,
            correct_invocation=correct,
            retrieval_helped=retrieval_helped,
            unnecessary_retrieval=unnecessary,
            missed_retrieval=missed,
        )


__all__ = [
    "RAGASMetrics",
    "InvocationMetrics",
    "RAGASEvaluator",
    "InvocationEvaluator",
]
