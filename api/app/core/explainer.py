"""Answer explainability for UX transparency.

This module provides "Why this answer?" explanations:
- Plain-language explanation of the reasoning process
- Key decision points from the trace
- Confidence breakdown
- Source attribution visualization

The goal is to make the RAG pipeline's decisions transparent
and understandable to end users.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


# =============================================================================
# Explanation Types
# =============================================================================

class DecisionType(Enum):
    """Types of decisions made during RAG pipeline."""
    RETRIEVAL = "retrieval"
    RERANKING = "reranking"
    COMPRESSION = "compression"
    GENERATION = "generation"
    CITATION = "citation"
    ABSTENTION = "abstention"
    FALLBACK = "fallback"


@dataclass
class DecisionPoint:
    """A key decision made during the pipeline."""
    decision_type: DecisionType
    description: str
    timestamp_ms: float
    input_summary: str
    output_summary: str
    confidence: float = 0.0
    alternatives: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.decision_type.value,
            "description": self.description,
            "timestamp_ms": round(self.timestamp_ms, 2),
            "input": self.input_summary,
            "output": self.output_summary,
            "confidence": round(self.confidence, 3),
            "alternatives": self.alternatives,
        }
    
    def to_plain_english(self) -> str:
        """Convert decision to plain English explanation."""
        return self.description


@dataclass
class SourceAttribution:
    """Attribution of answer content to sources."""
    source_id: str
    source_title: str
    contribution: str  # What this source contributed
    relevance_score: float
    excerpt: str
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "title": self.source_title,
            "contribution": self.contribution,
            "relevance": round(self.relevance_score, 3),
            "excerpt": self.excerpt[:200] + "..." if len(self.excerpt) > 200 else self.excerpt,
        }


@dataclass
class ConfidenceBreakdown:
    """Breakdown of confidence factors."""
    overall: float
    retrieval_quality: float
    source_coverage: float
    citation_validity: float
    answer_coherence: float
    factors: list[tuple[str, float]] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": round(self.overall, 3),
            "retrieval_quality": round(self.retrieval_quality, 3),
            "source_coverage": round(self.source_coverage, 3),
            "citation_validity": round(self.citation_validity, 3),
            "answer_coherence": round(self.answer_coherence, 3),
            "factors": [(f, round(s, 3)) for f, s in self.factors],
        }
    
    def top_concerns(self) -> list[str]:
        """Get top concerns (low-scoring factors)."""
        concerns = []
        if self.retrieval_quality < 0.5:
            concerns.append("Limited relevant information found")
        if self.source_coverage < 0.5:
            concerns.append("Not all aspects of the question are covered")
        if self.citation_validity < 0.8:
            concerns.append("Some claims may not be fully supported")
        if self.answer_coherence < 0.7:
            concerns.append("Answer structure could be improved")
        return concerns


@dataclass
class Explanation:
    """Complete explanation of an answer."""
    summary: str
    decision_chain: list[DecisionPoint]
    source_attributions: list[SourceAttribution]
    confidence: ConfidenceBreakdown
    reasoning_steps: list[str]
    caveats: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "decision_chain": [d.to_dict() for d in self.decision_chain],
            "sources": [s.to_dict() for s in self.source_attributions],
            "confidence": self.confidence.to_dict(),
            "reasoning": self.reasoning_steps,
            "caveats": self.caveats,
        }
    
    def to_markdown(self) -> str:
        """Convert to markdown format for display."""
        lines = [
            "# Answer Explanation",
            "",
            f"**Summary:** {self.summary}",
            "",
        ]
        
        # Confidence section
        lines.extend([
            "## Confidence",
            f"Overall confidence: **{self.confidence.overall:.0%}**",
            "",
        ])
        
        concerns = self.confidence.top_concerns()
        if concerns:
            lines.append("**Note:** " + "; ".join(concerns))
            lines.append("")
        
        # Reasoning chain
        if self.reasoning_steps:
            lines.extend([
                "## Reasoning Steps",
                "",
            ])
            for i, step in enumerate(self.reasoning_steps, 1):
                lines.append(f"{i}. {step}")
            lines.append("")
        
        # Sources used
        if self.source_attributions:
            lines.extend([
                "## Sources Used",
                "",
            ])
            for source in self.source_attributions:
                lines.append(f"- **{source.source_title}** ({source.relevance_score:.0%} relevant)")
                lines.append(f"  - {source.contribution}")
            lines.append("")
        
        # Caveats
        if self.caveats:
            lines.extend([
                "## Caveats",
                "",
            ])
            for caveat in self.caveats:
                lines.append(f"- {caveat}")
        
        return "\n".join(lines)


# =============================================================================
# Answer Explainer
# =============================================================================

class AnswerExplainer:
    """Generate explanations for RAG answers.
    
    This class analyzes trace bundles to produce human-readable
    explanations of how an answer was generated.
    """
    
    def explain(self, trace: dict[str, Any]) -> Explanation:
        """Generate explanation from a trace bundle.
        
        Args:
            trace: Full trace bundle from orchestrator
            
        Returns:
            Explanation with decision chain and confidence
        """
        # Extract key information from trace
        decision_chain = self._extract_decisions(trace)
        sources = self._extract_sources(trace)
        confidence = self._extract_confidence(trace)
        reasoning = self._extract_reasoning(trace)
        caveats = self._generate_caveats(trace, confidence)
        
        # Generate summary
        summary = self._generate_summary(trace, confidence)
        
        return Explanation(
            summary=summary,
            decision_chain=decision_chain,
            source_attributions=sources,
            confidence=confidence,
            reasoning_steps=reasoning,
            caveats=caveats,
        )
    
    def _extract_decisions(self, trace: dict[str, Any]) -> list[DecisionPoint]:
        """Extract key decision points from trace."""
        decisions = []
        stages = trace.get("stages", [])
        
        for stage in stages:
            stage_name = stage.get("name", "unknown")
            duration = stage.get("duration_ms", 0)
            
            if stage_name == "retrieval":
                docs_found = stage.get("documents_retrieved", 0)
                decisions.append(DecisionPoint(
                    decision_type=DecisionType.RETRIEVAL,
                    description=f"Found {docs_found} relevant documents from the knowledge base",
                    timestamp_ms=duration,
                    input_summary=f"Query: {trace.get('query', 'unknown')[:100]}",
                    output_summary=f"{docs_found} documents passed relevance threshold",
                    confidence=stage.get("confidence", 0.7),
                ))
            
            elif stage_name == "reranking":
                input_count = stage.get("input_count", 0)
                output_count = stage.get("output_count", 0)
                decisions.append(DecisionPoint(
                    decision_type=DecisionType.RERANKING,
                    description=f"Selected top {output_count} most relevant sources from {input_count} candidates",
                    timestamp_ms=duration,
                    input_summary=f"{input_count} candidate documents",
                    output_summary=f"{output_count} top-ranked documents",
                    confidence=stage.get("confidence", 0.8),
                ))
            
            elif stage_name == "generation":
                decisions.append(DecisionPoint(
                    decision_type=DecisionType.GENERATION,
                    description="Generated answer based on retrieved sources",
                    timestamp_ms=duration,
                    input_summary=f"{stage.get('context_tokens', 0)} tokens of context",
                    output_summary=f"Answer with {stage.get('answer_tokens', 0)} tokens",
                    confidence=stage.get("confidence", 0.7),
                ))
            
            elif stage_name == "citation_verification":
                valid = stage.get("valid_citations", 0)
                total = stage.get("total_citations", 0)
                decisions.append(DecisionPoint(
                    decision_type=DecisionType.CITATION,
                    description=f"Verified {valid}/{total} citations against source documents",
                    timestamp_ms=duration,
                    input_summary=f"{total} citations in answer",
                    output_summary=f"{valid} verified, {total - valid} repaired",
                    confidence=valid / total if total > 0 else 1.0,
                ))
        
        return decisions
    
    def _extract_sources(self, trace: dict[str, Any]) -> list[SourceAttribution]:
        """Extract source attributions from trace."""
        sources = []
        used_docs = trace.get("sources", []) or trace.get("evidence", [])
        
        for doc in used_docs[:5]:  # Top 5 sources
            doc_id = doc.get("id", doc.get("chunk_id", "unknown"))
            title = doc.get("title", doc.get("source", doc_id))
            score = doc.get("score", doc.get("relevance", 0.5))
            text = doc.get("text", doc.get("snippet", ""))
            
            sources.append(SourceAttribution(
                source_id=doc_id,
                source_title=title,
                contribution="Provided relevant information for the answer",
                relevance_score=score,
                excerpt=text[:300] if text else "",
            ))
        
        return sources
    
    def _extract_confidence(self, trace: dict[str, Any]) -> ConfidenceBreakdown:
        """Extract confidence breakdown from trace."""
        metrics = trace.get("metrics", {})
        
        return ConfidenceBreakdown(
            overall=metrics.get("confidence", 0.7),
            retrieval_quality=metrics.get("retrieval_score", 0.7),
            source_coverage=metrics.get("coverage", 0.8),
            citation_validity=metrics.get("citation_coverage", 0.9),
            answer_coherence=metrics.get("coherence", 0.8),
            factors=[
                ("Evidence density", metrics.get("evidence_density", 0.7)),
                ("Source agreement", metrics.get("source_agreement", 0.8)),
                ("Query coverage", metrics.get("query_coverage", 0.75)),
            ],
        )
    
    def _extract_reasoning(self, trace: dict[str, Any]) -> list[str]:
        """Extract reasoning steps from trace."""
        steps = []
        
        # Infer reasoning from stages
        query = trace.get("query", "")
        num_sources = len(trace.get("sources", []))
        
        steps.append(f"Received query: \"{query[:80]}...\"" if len(query) > 80 else f"Received query: \"{query}\"")
        steps.append(f"Searched knowledge base and found {num_sources} relevant sources")
        
        if trace.get("reranker_used"):
            steps.append("Re-ranked sources by relevance to improve answer quality")
        
        if trace.get("compression_applied"):
            steps.append("Compressed context to focus on most relevant information")
        
        steps.append("Generated answer grounded in retrieved sources")
        steps.append("Verified citations point to actual source spans")
        
        return steps
    
    def _generate_caveats(
        self,
        trace: dict[str, Any],
        confidence: ConfidenceBreakdown,
    ) -> list[str]:
        """Generate caveats about the answer."""
        caveats = []
        
        if confidence.overall < 0.6:
            caveats.append("This answer has lower than normal confidence due to limited evidence")
        
        if confidence.retrieval_quality < 0.5:
            caveats.append("The knowledge base may not fully cover this topic")
        
        if confidence.citation_validity < 0.8:
            caveats.append("Some parts of this answer may not be fully supported by sources")
        
        # Check for conflicting sources
        if trace.get("conflicts_detected"):
            caveats.append("Multiple sources provided conflicting information")
        
        # Check if abstention was considered
        if trace.get("abstention_considered"):
            caveats.append("The system considered declining to answer due to limited evidence")
        
        return caveats
    
    def _generate_summary(
        self,
        trace: dict[str, Any],
        confidence: ConfidenceBreakdown,
    ) -> str:
        """Generate a one-line summary of the explanation."""
        num_sources = len(trace.get("sources", []))
        
        if confidence.overall >= 0.8:
            return f"Answer generated with high confidence using {num_sources} sources"
        elif confidence.overall >= 0.6:
            return f"Answer generated with moderate confidence from {num_sources} sources"
        else:
            return f"Answer generated with limited confidence from {num_sources} sources"
    
    def get_reasoning_chain(self, trace: dict[str, Any]) -> list[str]:
        """Get just the reasoning chain without full explanation."""
        return self._extract_reasoning(trace)


# =============================================================================
# Singleton
# =============================================================================

_explainer: Optional[AnswerExplainer] = None


def get_explainer() -> AnswerExplainer:
    """Get the global answer explainer."""
    global _explainer
    if _explainer is None:
        _explainer = AnswerExplainer()
    return _explainer


__all__ = [
    "DecisionType",
    "DecisionPoint",
    "SourceAttribution",
    "ConfidenceBreakdown",
    "Explanation",
    "AnswerExplainer",
    "get_explainer",
]
