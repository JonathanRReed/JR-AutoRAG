"""Query router for intelligent retrieval strategy selection.

This module implements agentic query routing that:
- Classifies queries to determine the best retrieval approach
- Routes to different retrieval strategies based on query type
- Supports fallback chains for improved recall
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .smart_planner import QueryType


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""
    HYBRID = "hybrid"           # Dense + BM25 (default)
    DENSE_ONLY = "dense_only"   # Semantic search for conceptual queries
    SPARSE_ONLY = "sparse_only" # Keyword search for exact terms
    MULTI_STEP = "multi_step"   # Decompose and iterate
    DIRECT = "direct"           # Skip retrieval (for greetings, etc.)


@dataclass
class RoutingDecision:
    """Result of query routing."""
    strategy: RetrievalStrategy
    confidence: float  # 0-1 confidence in decision
    reasoning: str
    parameters: dict[str, Any]  # Strategy-specific parameters


class QueryRouter:
    """Intelligent query router for agentic RAG.

    Routes queries to the most appropriate retrieval strategy based on:
    - Query classification (from SmartPlanner)
    - Query characteristics (length, keywords, etc.)
    - Document collection characteristics
    """

    # Patterns for routing decisions
    DIRECT_PATTERNS = [
        r'^(hi|hello|hey|thanks|thank you|bye|goodbye)\b',
        r'^what can you (do|help)',
        r'^who are you\b',
    ]

    KEYWORD_HEAVY_PATTERNS = [
        r'\b(error|exception|traceback|stacktrace)\b',
        r'\b(version|v\d+\.\d+)\b',
        r'\b(api|sdk|cli|ui|ux)\b',
        r'[A-Z]{2,}',  # Acronyms
    ]

    CONCEPTUAL_PATTERNS = [
        r'\b(concept|idea|theory|approach|philosophy)\b',
        r'\b(explain|describe|overview|introduction)\b',
        r'\b(how does|what is the purpose)\b',
    ]

    def __init__(self) -> None:
        import re
        self._re = re
        self._direct_re = [re.compile(p, re.IGNORECASE) for p in self.DIRECT_PATTERNS]
        self._keyword_re = [re.compile(p, re.IGNORECASE) for p in self.KEYWORD_HEAVY_PATTERNS]
        self._conceptual_re = [re.compile(p, re.IGNORECASE) for p in self.CONCEPTUAL_PATTERNS]

    def _count_matches(self, patterns: list, text: str) -> int:
        """Count pattern matches in text."""
        return sum(1 for p in patterns if p.search(text))

    def _is_direct_query(self, query: str) -> bool:
        """Check if query should skip retrieval."""
        return any(p.match(query) for p in self._direct_re)

    def route(
        self,
        query: str,
        query_type: QueryType | None = None,
        document_count: int = 0,
    ) -> RoutingDecision:
        """Route a query to the appropriate retrieval strategy.

        Args:
            query: The user query
            query_type: Optional classification from SmartPlanner
            document_count: Number of documents in the collection

        Returns:
            RoutingDecision with strategy and parameters
        """
        # Check for direct response (no retrieval needed)
        if self._is_direct_query(query):
            return RoutingDecision(
                strategy=RetrievalStrategy.DIRECT,
                confidence=0.9,
                reasoning="Query appears to be a greeting or meta-question",
                parameters={"skip_retrieval": True},
            )

        # No documents - can't retrieve
        if document_count == 0:
            return RoutingDecision(
                strategy=RetrievalStrategy.DIRECT,
                confidence=1.0,
                reasoning="No documents in collection",
                parameters={"skip_retrieval": True},
            )

        # Count pattern matches
        keyword_score = self._count_matches(self._keyword_re, query)
        conceptual_score = self._count_matches(self._conceptual_re, query)

        # Route based on query characteristics
        if keyword_score > conceptual_score and keyword_score >= 2:
            return RoutingDecision(
                strategy=RetrievalStrategy.SPARSE_ONLY,
                confidence=0.7,
                reasoning=f"Query contains {keyword_score} technical/keyword patterns",
                parameters={"boost_exact_match": True},
            )

        if conceptual_score > keyword_score and conceptual_score >= 2:
            return RoutingDecision(
                strategy=RetrievalStrategy.DENSE_ONLY,
                confidence=0.7,
                reasoning=f"Query is conceptual ({conceptual_score} patterns matched)",
                parameters={"expand_context": True},
            )

        # Use query type if available
        if query_type:
            from .smart_planner import QueryType

            if query_type == QueryType.COMPARATIVE:
                return RoutingDecision(
                    strategy=RetrievalStrategy.MULTI_STEP,
                    confidence=0.8,
                    reasoning="Comparative query benefits from multi-step retrieval",
                    parameters={"max_steps": 3},
                )

            if query_type == QueryType.FACTUAL:
                return RoutingDecision(
                    strategy=RetrievalStrategy.HYBRID,
                    confidence=0.8,
                    reasoning="Factual query - hybrid retrieval for precision",
                    parameters={"rerank": True},
                )

        # Default to hybrid
        return RoutingDecision(
            strategy=RetrievalStrategy.HYBRID,
            confidence=0.6,
            reasoning="Default hybrid strategy for balanced retrieval",
            parameters={"rerank": True},
        )


__all__ = [
    "RetrievalStrategy",
    "RoutingDecision",
    "QueryRouter",
]
