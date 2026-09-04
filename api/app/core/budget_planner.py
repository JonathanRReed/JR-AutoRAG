"""Budget-aware retrieval planning.

Adjusts retrieval parameters based on latency/cost constraints.
Supports tiered budget classes for different use cases.

This module implements cost-aware RAG planning:
- MINIMAL: Fast responses with basic retrieval
- STANDARD: Balanced performance with reranking
- PREMIUM: All features enabled for best quality
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BudgetClass(str, Enum):
    """Budget tiers for retrieval."""

    MINIMAL = "minimal"  # Fast, cheap, ~2s target
    STANDARD = "standard"  # Balanced, ~5s target
    PREMIUM = "premium"  # High quality, ~15s target


@dataclass
class BudgetConstraints:
    """Constraints for budget-aware planning."""

    max_latency_ms: float = 5000
    max_retrieval_calls: int = 5
    use_rerank: bool = True
    use_colbert: bool = False
    use_graph: bool = False
    use_raptor: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "max_latency_ms": self.max_latency_ms,
            "max_retrieval_calls": self.max_retrieval_calls,
            "use_rerank": self.use_rerank,
            "use_colbert": self.use_colbert,
            "use_graph": self.use_graph,
            "use_raptor": self.use_raptor,
        }


@dataclass
class BudgetPlan:
    """Budget-adjusted retrieval plan."""

    suggested_k: int
    max_iterations: int
    use_rerank: bool
    use_colbert: bool
    use_graph: bool
    use_raptor: bool
    estimated_latency_ms: float
    budget_class: BudgetClass = BudgetClass.STANDARD

    def to_dict(self) -> dict:
        """Convert to dictionary for logging."""
        return {
            "suggested_k": self.suggested_k,
            "max_iterations": self.max_iterations,
            "use_rerank": self.use_rerank,
            "use_colbert": self.use_colbert,
            "use_graph": self.use_graph,
            "use_raptor": self.use_raptor,
            "estimated_latency_ms": self.estimated_latency_ms,
            "budget_class": self.budget_class.value,
        }


class BudgetPlanner:
    """Plan retrieval parameters based on budget constraints.

    Estimates latencies for each component and adjusts parameters
    to fit within the target budget while maximizing quality.

    Key features:
    - Three preset budget tiers (minimal, standard, premium)
    - Dynamic adjustment based on query complexity
    - Latency estimation for planning
    - Feature toggles based on available budget
    """

    # Estimated latencies (ms) - conservative estimates
    LATENCY_ESTIMATES = {
        "embedding": 50,  # Query embedding
        "dense_search": 10,  # FAISS search
        "sparse_search": 5,  # BM25 search
        "rrf_fusion": 2,  # RRF combination
        "cross_encoder_per_doc": 15,  # Cross-encoder rerank per doc
        "colbert_per_doc": 25,  # ColBERT rerank per doc
        "graph_query": 100,  # Graph traversal
        "raptor_expansion": 50,  # Hierarchy expansion
        "llm_generation": 2000,  # LLM response generation
        "conflict_detection": 20,  # Conflict analysis
        "claim_verification": 30,  # Hallucination check
    }

    def __init__(self):
        """Initialize budget planner with preset configurations."""
        self.presets = {
            BudgetClass.MINIMAL: BudgetConstraints(
                max_latency_ms=2000,
                max_retrieval_calls=1,
                use_rerank=False,
                use_colbert=False,
                use_graph=False,
                use_raptor=False,
            ),
            BudgetClass.STANDARD: BudgetConstraints(
                max_latency_ms=5000,
                max_retrieval_calls=3,
                use_rerank=True,
                use_colbert=False,
                use_graph=False,
                use_raptor=True,
            ),
            BudgetClass.PREMIUM: BudgetConstraints(
                max_latency_ms=15000,
                max_retrieval_calls=5,
                use_rerank=True,
                use_colbert=True,
                use_graph=True,
                use_raptor=True,
            ),
        }

    def _estimate_base_latency(self) -> float:
        """Estimate base latency for any query."""
        return (
            self.LATENCY_ESTIMATES["embedding"]
            + self.LATENCY_ESTIMATES["dense_search"]
            + self.LATENCY_ESTIMATES["sparse_search"]
            + self.LATENCY_ESTIMATES["rrf_fusion"]
            + self.LATENCY_ESTIMATES["llm_generation"]
        )

    def _estimate_per_doc_latency(self, use_rerank: bool, use_colbert: bool) -> float:
        """Estimate per-document processing latency."""
        latency = 2  # Base per-doc overhead
        if use_rerank:
            latency += self.LATENCY_ESTIMATES["cross_encoder_per_doc"]
        if use_colbert:
            latency += self.LATENCY_ESTIMATES["colbert_per_doc"]
        return latency

    def plan(
        self,
        budget_class: BudgetClass = BudgetClass.STANDARD,
        query_complexity: float = 0.5,  # 0-1 scale
        constraints: BudgetConstraints | None = None,
    ) -> BudgetPlan:
        """Generate budget-aware retrieval plan.

        Args:
            budget_class: Preset budget tier to use
            query_complexity: Estimated query complexity (0-1)
            constraints: Optional custom constraints override

        Returns:
            BudgetPlan with optimized parameters
        """
        c = constraints or self.presets[budget_class]

        # Calculate base latency
        base_latency = self._estimate_base_latency()
        remaining_budget = c.max_latency_ms - base_latency

        if remaining_budget < 0:
            # Budget too tight, use absolute minimum
            return BudgetPlan(
                suggested_k=3,
                max_iterations=1,
                use_rerank=False,
                use_colbert=False,
                use_graph=False,
                use_raptor=False,
                estimated_latency_ms=base_latency,
                budget_class=budget_class,
            )

        # Calculate per-document cost
        per_doc_cost = self._estimate_per_doc_latency(c.use_rerank, c.use_colbert)

        # Reserve budget for optional features
        feature_budget = 0
        if c.use_graph:
            feature_budget += self.LATENCY_ESTIMATES["graph_query"]
        if c.use_raptor:
            feature_budget += self.LATENCY_ESTIMATES["raptor_expansion"]

        remaining_for_retrieval = remaining_budget - feature_budget

        # Determine k based on remaining budget
        # Leave 50% headroom for uncertainty
        max_k = int(remaining_for_retrieval / per_doc_cost / 2)
        suggested_k = max(3, min(20, max_k))

        # Adjust iterations based on budget
        iteration_cost = (
            self.LATENCY_ESTIMATES["dense_search"]
            + self.LATENCY_ESTIMATES["sparse_search"]
            + (per_doc_cost * suggested_k)
        )
        max_iterations = max(
            1, min(c.max_retrieval_calls, int(remaining_for_retrieval / iteration_cost))
        )

        # Scale by query complexity
        if query_complexity > 0.7:
            # Complex query: more docs, more iterations
            suggested_k = min(suggested_k + 5, 25)
            max_iterations = min(max_iterations + 1, c.max_retrieval_calls)
        elif query_complexity < 0.3:
            # Simple query: fewer docs, single pass
            suggested_k = max(suggested_k - 2, 3)
            max_iterations = 1

        # Determine which features to enable based on remaining budget
        actual_raptor = c.use_raptor and remaining_for_retrieval > 300
        actual_graph = c.use_graph and remaining_for_retrieval > 500
        actual_colbert = c.use_colbert and remaining_for_retrieval > 1000

        # Estimate final latency
        estimated = (
            base_latency
            + (per_doc_cost * suggested_k * max_iterations)
            + (self.LATENCY_ESTIMATES["graph_query"] if actual_graph else 0)
            + (self.LATENCY_ESTIMATES["raptor_expansion"] if actual_raptor else 0)
        )

        return BudgetPlan(
            suggested_k=suggested_k,
            max_iterations=max_iterations,
            use_rerank=c.use_rerank,
            use_colbert=actual_colbert,
            use_graph=actual_graph,
            use_raptor=actual_raptor,
            estimated_latency_ms=estimated,
            budget_class=budget_class,
        )

    def estimate_query_complexity(self, query: str) -> float:
        """Estimate query complexity on 0-1 scale.

        Args:
            query: User query string

        Returns:
            Complexity score from 0 (simple) to 1 (complex)
        """
        import re

        score = 0.5  # Base score

        # Query length factor
        words = query.split()
        if len(words) > 20:
            score += 0.2
        elif len(words) < 5:
            score -= 0.2

        # Complexity indicators
        if any(
            w in query.lower()
            for w in ["compare", "versus", "vs", "difference", "similarities"]
        ):
            score += 0.2
        if any(w in query.lower() for w in ["how", "why", "explain", "describe"]):
            score += 0.1
        if re.search(r"\d+", query):  # Contains numbers
            score += 0.05
        if query.count("?") > 1:  # Multiple questions
            score += 0.15

        # Simplicity indicators
        if query.lower().startswith(("what is", "define", "who is")):
            score -= 0.1
        if len(words) < 3:
            score -= 0.2

        return max(0.0, min(1.0, score))


__all__ = ["BudgetClass", "BudgetConstraints", "BudgetPlan", "BudgetPlanner"]
