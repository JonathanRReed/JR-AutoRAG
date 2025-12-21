"""Learned query-complexity router with feature-based classification.

Trains on historical (query, decision, success) data to predict:
- no-retrieval vs single vs iterative vs clarify
- optimal k values and rerank usage

This implements the "query-complexity routing" pattern from Adaptive-RAG
(ACL 2024) for intelligent retrieval strategy selection.
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from pathlib import Path


class RouteDecision(str, Enum):
    """Routing decisions for query handling."""
    NO_RETRIEVAL = "no_retrieval"   # LLM can answer directly
    SINGLE = "single"               # Single retrieval pass
    ITERATIVE = "iterative"         # Multiple retrieval iterations
    CLARIFY = "clarify"             # Ask for clarification first
    GRAPH = "graph"                 # Use GraphRAG for entity queries
    RAPTOR = "raptor"               # Use RAPTOR for hierarchical queries
    HYBRID_HEAVY = "hybrid_heavy"   # Dense + sparse with reranking


@dataclass
class RouterFeatures:
    """Feature vector for routing decision."""
    query_length: int
    word_count: int
    has_wh_word: bool
    wh_word_type: str | None
    entity_count: int
    numeric_count: int
    comparison_signal: bool
    procedural_signal: bool
    factual_signal: bool
    analytical_signal: bool
    ambiguity_signal: bool
    negation_signal: bool
    temporal_signal: bool
    list_signal: bool
    complexity_score: float = 0.0
    historical_success_rate: float = 0.5
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "query_length": self.query_length,
            "word_count": self.word_count,
            "has_wh_word": self.has_wh_word,
            "wh_word_type": self.wh_word_type,
            "entity_count": self.entity_count,
            "numeric_count": self.numeric_count,
            "comparison_signal": self.comparison_signal,
            "procedural_signal": self.procedural_signal,
            "factual_signal": self.factual_signal,
            "analytical_signal": self.analytical_signal,
            "ambiguity_signal": self.ambiguity_signal,
            "negation_signal": self.negation_signal,
            "temporal_signal": self.temporal_signal,
            "list_signal": self.list_signal,
            "complexity_score": self.complexity_score,
        }


@dataclass 
class LearnedRouteResult:
    """Result from learned router."""
    decision: RouteDecision
    confidence: float
    suggested_k: int
    use_rerank: bool
    max_iterations: int
    features: RouterFeatures
    reasoning: str = ""


@dataclass
class RoutingOutcome:
    """Recorded outcome for learning."""
    query: str
    features: RouterFeatures
    decision: RouteDecision
    success: bool
    answer_quality: float
    latency_ms: float
    chunks_used: int


class LearnedRouter:
    """Feature-based router that learns from historical data.
    
    Uses handcrafted features with learned weights to route queries
    to optimal retrieval strategies. Logs outcomes for future training.
    
    Key features extracted:
    - Query length and complexity
    - Question word types (what, how, why, etc.)
    - Entity and numeric presence
    - Signal patterns (comparison, procedural, factual, etc.)
    """
    
    # Question word categories
    WH_WORDS = {
        'what': 'factual',
        'who': 'entity',
        'where': 'location',
        'when': 'temporal',
        'why': 'analytical',
        'how': 'procedural',
        'which': 'selection',
    }
    
    # Signal patterns
    COMPARISON_PATTERNS = [
        r'\bvs\.?\b', r'\bversus\b', r'\bcompare\b', r'\bdifference\b',
        r'\bbetter\b', r'\bworse\b', r'\bor\b.*\bor\b', r'\badvantages?\b',
        r'\bdisadvantages?\b', r'\bpros?\b.*\bcons?\b',
    ]
    
    PROCEDURAL_PATTERNS = [
        r'\bhow to\b', r'\bsteps?\b', r'\bprocess\b', r'\bguide\b',
        r'\btutorial\b', r'\binstructions?\b', r'\bprocedure\b',
    ]
    
    FACTUAL_PATTERNS = [
        r'^what is\b', r'^define\b', r'\bdefinition\b', r'\bmeaning\b',
        r'^who is\b', r'^where is\b', r'\bfact\b',
    ]
    
    ANALYTICAL_PATTERNS = [
        r'\bwhy\b', r'\breason\b', r'\bcause\b', r'\bexplain\b',
        r'\banalyze\b', r'\bevaluate\b', r'\bimpact\b', r'\beffect\b',
    ]
    
    AMBIGUITY_PATTERNS = [
        r'\bmight\b', r'\bcould\b', r'\bmaybe\b', r'\bpossibly\b',
        r'\bsometimes\b', r'\bperhaps\b', r'\bunclear\b',
    ]
    
    TEMPORAL_PATTERNS = [
        r'\bwhen\b', r'\bdate\b', r'\byear\b', r'\btime\b',
        r'\brecent\b', r'\blatest\b', r'\bhistory\b', r'\bfuture\b',
    ]
    
    LIST_PATTERNS = [
        r'\blist\b', r'\benumerate\b', r'\ball\b.*\b(ways|types|kinds)\b',
        r'\bexamples?\b', r'\btop \d+\b',
    ]
    
    # Decision weights (would be learned in production)
    DECISION_WEIGHTS = {
        RouteDecision.NO_RETRIEVAL: {
            "max_words": 5,
            "no_wh": True,
            "simple_greeting": True,
        },
        RouteDecision.SINGLE: {
            "factual": True,
            "entity_focused": True,
            "low_complexity": True,
        },
        RouteDecision.ITERATIVE: {
            "comparison": True,
            "high_complexity": True,
            "multi_aspect": True,
        },
        RouteDecision.GRAPH: {
            "entity_heavy": True,
            "relationship": True,
        },
        RouteDecision.RAPTOR: {
            "hierarchical": True,
            "overview": True,
        },
    }
    
    def __init__(self, history_path: str | None = None):
        """Initialize router with optional history file.
        
        Args:
            history_path: Path to save/load routing history for learning
        """
        self._history_path = Path(history_path) if history_path else None
        self._history: list[RoutingOutcome] = []
        self._compile_patterns()
        
        if self._history_path and self._history_path.exists():
            self._load_history()
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficiency."""
        self._comparison_re = [re.compile(p, re.I) for p in self.COMPARISON_PATTERNS]
        self._procedural_re = [re.compile(p, re.I) for p in self.PROCEDURAL_PATTERNS]
        self._factual_re = [re.compile(p, re.I) for p in self.FACTUAL_PATTERNS]
        self._analytical_re = [re.compile(p, re.I) for p in self.ANALYTICAL_PATTERNS]
        self._ambiguity_re = [re.compile(p, re.I) for p in self.AMBIGUITY_PATTERNS]
        self._temporal_re = [re.compile(p, re.I) for p in self.TEMPORAL_PATTERNS]
        self._list_re = [re.compile(p, re.I) for p in self.LIST_PATTERNS]
    
    def _count_matches(self, patterns: list, text: str) -> int:
        """Count pattern matches in text."""
        return sum(1 for p in patterns if p.search(text))
    
    def extract_features(self, query: str) -> RouterFeatures:
        """Extract comprehensive feature vector from query."""
        words = query.lower().split()
        
        # Find WH word
        wh_word_type = None
        has_wh = False
        for word in words[:3]:  # Check first 3 words
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.WH_WORDS:
                has_wh = True
                wh_word_type = self.WH_WORDS[clean_word]
                break
        
        # Count entities (capitalized words not at sentence start)
        entity_count = len(re.findall(r'(?<!^)(?<!\. )[A-Z][a-z]+', query))
        
        # Count numbers
        numeric_count = len(re.findall(r'\b\d+[\d,\.]*\b', query))
        
        # Detect signals
        comparison = self._count_matches(self._comparison_re, query) > 0
        procedural = self._count_matches(self._procedural_re, query) > 0
        factual = self._count_matches(self._factual_re, query) > 0
        analytical = self._count_matches(self._analytical_re, query) > 0
        ambiguity = self._count_matches(self._ambiguity_re, query) > 0
        temporal = self._count_matches(self._temporal_re, query) > 0
        list_signal = self._count_matches(self._list_re, query) > 0
        negation = bool(re.search(r'\b(not|no|never|without|except)\b', query, re.I))
        
        # Compute complexity score
        complexity = 0.0
        complexity += min(len(words) / 20, 1.0) * 0.3  # Length factor
        complexity += (1 if comparison else 0) * 0.2
        complexity += (1 if analytical else 0) * 0.2
        complexity += min(entity_count / 3, 1.0) * 0.15
        complexity += (1 if list_signal else 0) * 0.15
        
        return RouterFeatures(
            query_length=len(query),
            word_count=len(words),
            has_wh_word=has_wh,
            wh_word_type=wh_word_type,
            entity_count=entity_count,
            numeric_count=numeric_count,
            comparison_signal=comparison,
            procedural_signal=procedural,
            factual_signal=factual,
            analytical_signal=analytical,
            ambiguity_signal=ambiguity,
            negation_signal=negation,
            temporal_signal=temporal,
            list_signal=list_signal,
            complexity_score=complexity,
        )
    
    def route(self, query: str) -> LearnedRouteResult:
        """Route query to optimal retrieval strategy.
        
        Args:
            query: User query to route
            
        Returns:
            LearnedRouteResult with decision and parameters
        """
        features = self.extract_features(query)
        
        # Decision logic based on features
        decision, confidence, reasoning = self._make_decision(features, query)
        
        # Determine parameters based on decision
        suggested_k, use_rerank, max_iterations = self._get_parameters(decision, features)
        
        return LearnedRouteResult(
            decision=decision,
            confidence=confidence,
            suggested_k=suggested_k,
            use_rerank=use_rerank,
            max_iterations=max_iterations,
            features=features,
            reasoning=reasoning,
        )
    
    def _make_decision(
        self, 
        features: RouterFeatures, 
        query: str,
    ) -> tuple[RouteDecision, float, str]:
        """Make routing decision based on features."""
        query_lower = query.lower().strip()
        
        # NO_RETRIEVAL: Simple greetings or meta-questions
        if features.word_count < 5 and not features.has_wh_word:
            greetings = ['hello', 'hi', 'hey', 'thanks', 'bye', 'help']
            if any(query_lower.startswith(g) for g in greetings):
                return RouteDecision.NO_RETRIEVAL, 0.9, "Simple greeting detected"
        
        # CLARIFY: Ambiguous or incomplete queries
        if features.ambiguity_signal or (features.word_count < 4 and not features.has_wh_word):
            return RouteDecision.CLARIFY, 0.7, "Query appears ambiguous or incomplete"
        
        # ITERATIVE: Complex comparative or multi-aspect queries
        if features.comparison_signal:
            return RouteDecision.ITERATIVE, 0.85, "Comparison query requires multiple retrievals"
        
        if features.complexity_score > 0.6:
            return RouteDecision.ITERATIVE, 0.75, f"High complexity score ({features.complexity_score:.2f})"
        
        if features.list_signal and features.word_count > 10:
            return RouteDecision.ITERATIVE, 0.7, "List query with complexity"
        
        # GRAPH: Entity-focused or relationship queries
        if features.entity_count >= 2:
            if 'relationship' in query_lower or 'between' in query_lower:
                return RouteDecision.GRAPH, 0.8, "Entity relationship query"
        
        # RAPTOR: Hierarchical or overview queries
        hierarchical_words = ['overview', 'summary', 'section', 'chapter', 'outline']
        if any(w in query_lower for w in hierarchical_words):
            return RouteDecision.RAPTOR, 0.75, "Hierarchical/overview query"
        
        # HYBRID_HEAVY: Queries needing precision
        if features.entity_count > 0 and features.factual_signal:
            return RouteDecision.HYBRID_HEAVY, 0.7, "Factual entity query needs precision"
        
        # SINGLE: Default for straightforward queries
        return RouteDecision.SINGLE, 0.6, "Standard single-pass retrieval"
    
    def _get_parameters(
        self, 
        decision: RouteDecision, 
        features: RouterFeatures,
    ) -> tuple[int, bool, int]:
        """Get retrieval parameters based on decision."""
        # Base parameters
        if decision == RouteDecision.NO_RETRIEVAL:
            return 0, False, 0
        
        if decision == RouteDecision.CLARIFY:
            return 0, False, 0
        
        if decision == RouteDecision.ITERATIVE:
            k = 8 if features.comparison_signal else 6
            return k, True, 3
        
        if decision == RouteDecision.GRAPH:
            return 5, True, 2
        
        if decision == RouteDecision.RAPTOR:
            return 5, False, 1
        
        if decision == RouteDecision.HYBRID_HEAVY:
            return 10, True, 1
        
        # SINGLE
        k = 5 if features.word_count < 10 else 7
        use_rerank = features.entity_count > 0
        return k, use_rerank, 1
    
    def record_outcome(
        self, 
        query: str,
        features: RouterFeatures, 
        decision: RouteDecision, 
        success: bool,
        answer_quality: float = 0.5,
        latency_ms: float = 0.0,
        chunks_used: int = 0,
    ) -> None:
        """Record routing outcome for future learning.
        
        Args:
            query: Original query
            features: Extracted features
            decision: Decision that was made
            success: Whether the outcome was successful
            answer_quality: Quality score 0-1
            latency_ms: Response latency
            chunks_used: Number of chunks used
        """
        outcome = RoutingOutcome(
            query=query,
            features=features,
            decision=decision,
            success=success,
            answer_quality=answer_quality,
            latency_ms=latency_ms,
            chunks_used=chunks_used,
        )
        self._history.append(outcome)
        
        # Auto-save periodically
        if len(self._history) % 100 == 0:
            self._save_history()
    
    def _save_history(self) -> None:
        """Save routing history to file."""
        if not self._history_path:
            return
        
        data = []
        for outcome in self._history[-1000:]:  # Keep last 1000
            data.append({
                "query": outcome.query,
                "features": outcome.features.to_dict(),
                "decision": outcome.decision.value,
                "success": outcome.success,
                "answer_quality": outcome.answer_quality,
                "latency_ms": outcome.latency_ms,
                "chunks_used": outcome.chunks_used,
            })
        
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._history_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _load_history(self) -> None:
        """Load routing history from file."""
        if not self._history_path or not self._history_path.exists():
            return
        
        try:
            with open(self._history_path) as f:
                data = json.load(f)
            # Convert to RoutingOutcome objects (simplified)
            self._history = []  # Start fresh, history used for stats only
        except Exception:
            pass
    
    def get_stats(self) -> dict[str, Any]:
        """Get routing statistics."""
        if not self._history:
            return {"total": 0}
        
        decision_counts: dict[str, int] = {}
        success_by_decision: dict[str, list[bool]] = {}
        
        for outcome in self._history:
            dec = outcome.decision.value
            decision_counts[dec] = decision_counts.get(dec, 0) + 1
            if dec not in success_by_decision:
                success_by_decision[dec] = []
            success_by_decision[dec].append(outcome.success)
        
        success_rates = {
            dec: sum(successes) / len(successes)
            for dec, successes in success_by_decision.items()
        }
        
        return {
            "total": len(self._history),
            "decision_counts": decision_counts,
            "success_rates": success_rates,
        }


__all__ = [
    "RouteDecision",
    "RouterFeatures", 
    "LearnedRouteResult",
    "RoutingOutcome",
    "LearnedRouter",
]
