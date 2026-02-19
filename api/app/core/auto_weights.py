"""Auto Hybrid Weights: Adaptive dense/sparse weight optimization.

Implements automatic adjustment of hybrid retrieval weights based on:
1. Query characteristics (keyword-heavy vs semantic)
2. Historical performance feedback
3. Corpus characteristics

This provides self-tuning retrieval without manual configuration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .smart_planner import QueryType


@dataclass
class WeightProfile:
    """Optimal weights for a query profile."""
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    rerank_enabled: bool = True
    confidence: float = 0.5


@dataclass
class AutoWeightConfig:
    """Configuration for auto weight optimization."""
    # Default weights
    default_dense_weight: float = 0.6
    default_sparse_weight: float = 0.4

    # Adjustment ranges
    min_dense_weight: float = 0.2
    max_dense_weight: float = 0.9

    # Learning rate for feedback-based adjustment
    learning_rate: float = 0.1

    # History window for averaging
    history_window: int = 100


class AutoHybridWeights:
    """Self-tuning hybrid weight optimizer.

    Analyzes query characteristics to select optimal dense/sparse balance:
    - Keyword-heavy queries → higher sparse weight
    - Semantic/conceptual queries → higher dense weight
    - Entity/name queries → balanced with exact matching

    Also learns from retrieval quality feedback over time.
    """

    # Query patterns indicating keyword-focused retrieval
    KEYWORD_PATTERNS = [
        r'\b(error|exception|code|version|v\d+)\b',
        r'\b([A-Z]{2,}[0-9]+|0x[0-9a-fA-F]+)\b',  # Codes like ABC123, 0x1234
        r'\b(\d{4}-\d{2}-\d{2}|\d+\.\d+\.\d+)\b',  # Dates, versions
        r'"[^"]+"',  # Quoted exact terms
        r"'[^']+'",
    ]

    # Query patterns indicating semantic retrieval
    SEMANTIC_PATTERNS = [
        r'\b(how|why|what|explain|describe|understand)\b',
        r'\b(similar|like|related|analogous)\b',
        r'\b(concept|idea|approach|method)\b',
        r'\b(best|optimal|effective|efficient)\b',
    ]

    # Query patterns indicating entity focus
    ENTITY_PATTERNS = [
        r'\b(who|where|when)\b',
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b',  # Proper nouns
    ]

    def __init__(self, config: AutoWeightConfig | None = None) -> None:
        """Initialize auto weight optimizer."""
        self.config = config or AutoWeightConfig()

        # Compile patterns
        self._keyword_re = [re.compile(p, re.I) for p in self.KEYWORD_PATTERNS]
        self._semantic_re = [re.compile(p, re.I) for p in self.SEMANTIC_PATTERNS]
        self._entity_re = [re.compile(p, re.I) for p in self.ENTITY_PATTERNS]

        # Performance history for learning
        self._history: list[dict[str, Any]] = []

        # Learned adjustments per query type
        self._type_adjustments: dict[str, float] = {}

    def analyze_query(self, query: str) -> dict[str, float]:
        """Analyze query to determine optimal retrieval approach.

        Returns dict with signal strengths for different retrieval modes.
        """
        keyword_score = sum(1 for p in self._keyword_re if p.search(query)) / max(len(self._keyword_re), 1)
        semantic_score = sum(1 for p in self._semantic_re if p.search(query)) / max(len(self._semantic_re), 1)
        entity_score = sum(1 for p in self._entity_re if p.search(query)) / max(len(self._entity_re), 1)

        # Adjust based on query length (longer = more semantic)
        word_count = len(query.split())
        if word_count > 15:
            semantic_score += 0.2
        elif word_count < 5:
            keyword_score += 0.2

        # Check for quoted terms (exact match preference)
        if '"' in query or "'" in query:
            keyword_score += 0.3

        return {
            "keyword": min(1.0, keyword_score),
            "semantic": min(1.0, semantic_score),
            "entity": min(1.0, entity_score),
        }

    def compute_weights(
        self,
        query: str,
        query_type: QueryType | None = None,
    ) -> WeightProfile:
        """Compute optimal dense/sparse weights for a query.

        Args:
            query: User query text
            query_type: Optional query classification

        Returns:
            WeightProfile with recommended weights
        """
        analysis = self.analyze_query(query)

        # Start with defaults
        dense_weight = self.config.default_dense_weight
        sparse_weight = self.config.default_sparse_weight

        # Adjust based on analysis
        keyword_signal = analysis["keyword"]
        semantic_signal = analysis["semantic"]

        if keyword_signal > semantic_signal:
            # Keyword-heavy: increase sparse weight
            adjustment = (keyword_signal - semantic_signal) * 0.3
            sparse_weight = min(0.7, sparse_weight + adjustment)
            dense_weight = 1.0 - sparse_weight
        elif semantic_signal > keyword_signal:
            # Semantic-heavy: increase dense weight
            adjustment = (semantic_signal - keyword_signal) * 0.3
            dense_weight = min(0.85, dense_weight + adjustment)
            sparse_weight = 1.0 - dense_weight

        # Apply learned adjustments from feedback
        if query_type and query_type.value in self._type_adjustments:
            learned = self._type_adjustments[query_type.value]
            dense_weight = max(
                self.config.min_dense_weight,
                min(self.config.max_dense_weight, dense_weight + learned)
            )
            sparse_weight = 1.0 - dense_weight

        # Determine reranking (generally good for mixed queries)
        rerank_enabled = True

        # Entity queries benefit from exact matching
        if analysis["entity"] > 0.5:
            rerank_enabled = True  # Reranking helps disambiguate entities

        confidence = 0.5 + (abs(keyword_signal - semantic_signal) * 0.3)

        return WeightProfile(
            dense_weight=round(dense_weight, 3),
            sparse_weight=round(sparse_weight, 3),
            rerank_enabled=rerank_enabled,
            confidence=round(confidence, 3),
        )

    def record_feedback(
        self,
        query: str,
        query_type: QueryType | None,
        weights_used: WeightProfile,
        quality_score: float,
    ) -> None:
        """Record retrieval quality feedback for learning.

        Args:
            query: Original query
            query_type: Query classification
            weights_used: Weights that were used
            quality_score: Quality score achieved (0-1)
        """
        self._history.append({
            "query": query[:100],
            "query_type": query_type.value if query_type else "unknown",
            "dense_weight": weights_used.dense_weight,
            "quality": quality_score,
        })

        # Trim history to window
        if len(self._history) > self.config.history_window:
            self._history = self._history[-self.config.history_window:]

        # Update learned adjustments
        if query_type:
            self._update_type_adjustment(query_type.value, weights_used.dense_weight, quality_score)

    def _update_type_adjustment(
        self,
        query_type: str,
        dense_weight: float,
        quality: float,
    ) -> None:
        """Update learned adjustment for a query type.

        Uses simple gradient: if quality was good with high dense, nudge higher.
        """
        current = self._type_adjustments.get(query_type, 0.0)

        # Quality above 0.7 is good, below is bad
        if quality > 0.7:
            # Good result - nudge toward used weight
            adjustment = self.config.learning_rate * 0.1 if dense_weight > 0.6 else -self.config.learning_rate * 0.1
        else:
            # Bad result - nudge away from used weight
            adjustment = -self.config.learning_rate * 0.1 if dense_weight > 0.6 else self.config.learning_rate * 0.1

        self._type_adjustments[query_type] = max(-0.2, min(0.2, current + adjustment))

    def get_stats(self) -> dict[str, Any]:
        """Get optimizer statistics."""
        return {
            "history_size": len(self._history),
            "type_adjustments": dict(self._type_adjustments),
            "avg_quality": (
                sum(h["quality"] for h in self._history) / len(self._history)
                if self._history else 0.0
            ),
        }


# Singleton for easy access
_auto_weights: AutoHybridWeights | None = None


def get_auto_weights(config: AutoWeightConfig | None = None) -> AutoHybridWeights:
    """Get or create auto weights optimizer."""
    global _auto_weights
    if _auto_weights is None or config is not None:
        _auto_weights = AutoHybridWeights(config)
    return _auto_weights


__all__ = [
    "WeightProfile",
    "AutoWeightConfig",
    "AutoHybridWeights",
    "get_auto_weights",
]
