"""Smart planner with LLM-based query decomposition and routing.

This module provides advanced query planning capabilities:
- Query classification (factual, comparative, summary, etc.)
- Query decomposition into sub-queries for complex questions
- Query expansion with synonyms and related terms
- Dynamic retrieval parameter selection based on query type
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .providers import LLMProvider

from ..schemas.config import AppConfig


class QueryType(str, Enum):
    """Classification of query types for routing."""
    FACTUAL = "factual"           # Simple fact lookup
    COMPARATIVE = "comparative"   # Comparing multiple items
    ANALYTICAL = "analytical"     # Requires reasoning
    SUMMARY = "summary"           # Needs broad coverage
    PROCEDURAL = "procedural"     # How-to questions
    CONVERSATIONAL = "conversational"  # Follow-up/clarification
    # New query types for adaptive routing
    MULTI_HOP = "multi_hop"       # Requires multiple retrieval steps
    LOCATOR = "locator"           # "Where is X in my docs?" - keyword heavy
    EXPLORATORY = "exploratory"   # Broad research, high diversity


@dataclass
class PlanStep:
    """A single retrieval step in the plan."""
    query: str
    dense_k: int
    sparse_k: int
    rerank_pool: int
    compression: bool
    priority: int = 1  # Higher = more important


@dataclass
class RetrievalPlan:
    """Complete retrieval plan with multiple steps."""
    steps: list[PlanStep]
    target_tokens: int
    coverage_target: float
    original_query: str = ""
    query_type: QueryType = QueryType.FACTUAL
    decomposed: bool = False
    expanded_terms: list[str] = field(default_factory=list)
    # Iterative retrieval settings
    iterative: bool = False
    max_iterations: int = 1
    stop_threshold: float = 0.05  # Stop when marginal gain < this
    routing_params: dict = field(default_factory=dict)  # Full routing strategy


@dataclass
class QueryAnalysis:
    """Result of analyzing a query."""
    query_type: QueryType
    sub_queries: list[str]
    expanded_terms: list[str]
    complexity_score: float  # 0-1, higher = more complex


def compute_marginal_gain(
    existing_chunks: list[dict],
    new_chunks: list[dict],
) -> float:
    """Compute marginal evidence gain from new retrieval iteration.
    
    Returns a score 0-1 indicating how much unique information the new
    chunks add. Used for stop criteria in iterative retrieval.
    
    Args:
        existing_chunks: Previously retrieved chunks (each with 'text' and 'score')
        new_chunks: Newly retrieved chunks from this iteration
    
    Returns:
        Marginal gain score (0 = all redundant, 1 = all new)
    """
    if not new_chunks:
        return 0.0
    
    if not existing_chunks:
        return 1.0
    
    # Get existing text content for deduplication
    existing_texts = set()
    for chunk in existing_chunks:
        text = chunk.get("text", chunk.get("chunk_text", "")).lower()
        # Use content hash for comparison (first 100 chars + length)
        key = f"{text[:100]}_{len(text)}"
        existing_texts.add(key)
    
    # Count unique new chunks
    unique_count = 0
    score_improvement = 0.0
    
    for chunk in new_chunks:
        text = chunk.get("text", chunk.get("chunk_text", "")).lower()
        key = f"{text[:100]}_{len(text)}"
        
        if key not in existing_texts:
            unique_count += 1
            # Weight by score
            score_improvement += chunk.get("score", 0.5)
    
    if unique_count == 0:
        return 0.0
    
    # Combine uniqueness ratio with score-weighted value
    uniqueness = unique_count / len(new_chunks)
    avg_new_score = score_improvement / unique_count
    
    # Weighted combination: 70% uniqueness, 30% score
    return 0.7 * uniqueness + 0.3 * avg_new_score


class SmartPlanner:
    """LLM-enhanced planner with query decomposition and routing.
    
    When an LLM provider is available, uses it to:
    1. Classify the query type
    2. Decompose complex queries into sub-queries
    3. Expand queries with related terms
    4. Select optimal retrieval parameters
    
    Falls back to heuristic-based planning when no LLM is available.
    """
    
    # Heuristic patterns for query classification
    COMPARATIVE_PATTERNS = [
        r'\bvs\.?\b', r'\bversus\b', r'\bcompare\b', r'\bdifference\b',
        r'\bbetter\b', r'\bworse\b', r'\bor\b.*\bor\b'
    ]
    PROCEDURAL_PATTERNS = [
        r'^how\b', r'\bsteps?\b', r'\bprocess\b', r'\bprocedure\b',
        r'\bguide\b', r'\btutorial\b', r'\binstructions?\b'
    ]
    SUMMARY_PATTERNS = [
        r'\bsummar\w+\b', r'\boverview\b', r'\bexplain\b', r'\bdescribe\b',
        r'\bwhat is\b', r'\bwhat are\b'
    ]
    ANALYTICAL_PATTERNS = [
        r'\bwhy\b', r'\bcause\b', r'\beffect\b', r'\bimpact\b',
        r'\banalyze\b', r'\bevaluate\b', r'\bassess\b'
    ]
    # New patterns for adaptive routing
    LOCATOR_PATTERNS = [
        r'\bwhere\b.*\b(find|located|mention)\b', r'\bwhich (document|file|section)\b',
        r'\bfind\b.*\bin\b', r'\blocate\b', r'\bsearch for\b'
    ]
    MULTI_HOP_PATTERNS = [
        r'\band\b.*\bthen\b', r'\bfirst\b.*\bthen\b', r'\bafter\b.*\bwhat\b',
        r'\bbased on\b.*\bwhat\b', r'\busing\b.*\bcalculate\b'
    ]
    EXPLORATORY_PATTERNS = [
        r'\btell me (about|everything)\b', r'\bexplore\b', r'\bresearch\b',
        r'\ball\b.*\b(information|details)\b', r'\bcomprehensive\b'
    ]
    
    # Routing strategies per query type
    ROUTING_STRATEGIES: dict[str, dict] = {
        QueryType.FACTUAL: {
            "dense_k": 5,
            "sparse_k": 3,
            "rerank_pool": 10,
            "compression": "tight",
            "raptor": False,
            "diversity": 0.0,
            "description": "Precise fact lookup with minimal k",
        },
        QueryType.COMPARATIVE: {
            "dense_k": 10,
            "sparse_k": 6,
            "rerank_pool": 20,
            "compression": "moderate",
            "raptor": False,
            "diversity": 0.3,
            "description": "Gather multiple items for comparison",
        },
        QueryType.MULTI_HOP: {
            "dense_k": 8,
            "sparse_k": 5,
            "rerank_pool": 15,
            "compression": "moderate",
            "decompose": True,
            "iterative": True,
            "max_hops": 3,
            "description": "Multi-step reasoning with iteration",
        },
        QueryType.SUMMARY: {
            "dense_k": 12,
            "sparse_k": 8,
            "rerank_pool": 25,
            "compression": "light",
            "raptor": True,
            "diversity": 0.2,
            "description": "Broad coverage for summarization",
        },
        QueryType.EXPLORATORY: {
            "dense_k": 15,
            "sparse_k": 10,
            "rerank_pool": 30,
            "compression": "light",
            "raptor": True,
            "diversity": 0.4,
            "description": "Maximum coverage for research",
        },
        QueryType.LOCATOR: {
            "dense_k": 5,
            "sparse_k": 10,  # Favor keyword matching
            "sparse_weight": 0.7,
            "rerank_pool": 15,
            "compression": "none",
            "title_boost": 2.0,
            "heading_boost": 1.5,
            "description": "Keyword-heavy for document location",
        },
        QueryType.PROCEDURAL: {
            "dense_k": 8,
            "sparse_k": 5,
            "rerank_pool": 15,
            "compression": "moderate",
            "raptor": False,
            "description": "Step-by-step instructions",
        },
        QueryType.ANALYTICAL: {
            "dense_k": 10,
            "sparse_k": 6,
            "rerank_pool": 20,
            "compression": "moderate",
            "raptor": True,
            "description": "Evidence for reasoning",
        },
        QueryType.CONVERSATIONAL: {
            "dense_k": 5,
            "sparse_k": 3,
            "rerank_pool": 8,
            "compression": "tight",
            "description": "Quick follow-up retrieval",
        },
    }
    
    def __init__(self, config: AppConfig, provider: "LLMProvider | None" = None) -> None:
        self._config = config
        self._provider = provider
        self._last_planner_mode = "heuristic"
    
    def rebuild(self, config: AppConfig) -> None:
        self._config = config
    
    def set_provider(self, provider: "LLMProvider | None") -> None:
        """Set or update the LLM provider for smart planning."""
        self._provider = provider
    
    def _classify_query_heuristic(self, query: str) -> QueryType:
        """Classify query using pattern matching."""
        query_lower = query.lower()
        
        # Check new specialized types first
        for pattern in self.LOCATOR_PATTERNS:
            if re.search(pattern, query_lower):
                return QueryType.LOCATOR
        
        for pattern in self.MULTI_HOP_PATTERNS:
            if re.search(pattern, query_lower):
                return QueryType.MULTI_HOP
        
        for pattern in self.EXPLORATORY_PATTERNS:
            if re.search(pattern, query_lower):
                return QueryType.EXPLORATORY
        
        # Existing patterns
        for pattern in self.COMPARATIVE_PATTERNS:
            if re.search(pattern, query_lower):
                return QueryType.COMPARATIVE
        
        for pattern in self.PROCEDURAL_PATTERNS:
            if re.search(pattern, query_lower):
                return QueryType.PROCEDURAL
        
        for pattern in self.SUMMARY_PATTERNS:
            if re.search(pattern, query_lower):
                return QueryType.SUMMARY
        
        for pattern in self.ANALYTICAL_PATTERNS:
            if re.search(pattern, query_lower):
                return QueryType.ANALYTICAL
        
        return QueryType.FACTUAL
    
    def _decompose_query_heuristic(self, query: str, query_type: QueryType) -> list[str]:
        """Decompose query into sub-queries using heuristics."""
        sub_queries = [query]  # Always include original
        
        if query_type == QueryType.COMPARATIVE:
            # Try to extract items being compared
            parts = re.split(r'\bvs\.?\b|\bversus\b|\bor\b', query, flags=re.IGNORECASE)
            if len(parts) >= 2:
                for part in parts:
                    part = part.strip()
                    if part and len(part) > 5:
                        sub_queries.append(f"What is {part}?")
        
        elif query_type == QueryType.ANALYTICAL:
            # Add related factual query
            sub_queries.append(query.replace("why", "what").replace("Why", "What"))
        
        elif query_type == QueryType.PROCEDURAL:
            # Add overview query
            sub_queries.append(query.replace("how to", "what is").replace("How to", "What is"))
        
        return sub_queries[:3]  # Limit to 3 sub-queries
    
    def _expand_query_heuristic(self, query: str) -> list[str]:
        """Extract key terms for query expansion."""
        # Remove common words and extract likely important terms
        stop_words = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'what', 'how', 'why', 'when', 'where', 'who', 'which',
            'can', 'could', 'would', 'should', 'do', 'does', 'did',
            'i', 'you', 'we', 'they', 'it', 'this', 'that', 'these', 'those',
            'and', 'or', 'but', 'if', 'then', 'so', 'because',
            'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'about'
        }
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        key_terms = [w for w in words if w not in stop_words]
        
        return key_terms[:5]  # Return top 5 key terms
    
    def _estimate_complexity(self, query: str, query_type: QueryType) -> float:
        """Estimate query complexity (0-1)."""
        score = 0.0
        
        # Length factor
        word_count = len(query.split())
        if word_count > 20:
            score += 0.3
        elif word_count > 10:
            score += 0.15
        
        # Query type factor
        type_complexity = {
            QueryType.FACTUAL: 0.0,
            QueryType.CONVERSATIONAL: 0.1,
            QueryType.PROCEDURAL: 0.2,
            QueryType.SUMMARY: 0.3,
            QueryType.ANALYTICAL: 0.4,
            QueryType.COMPARATIVE: 0.5,
        }
        score += type_complexity.get(query_type, 0.2)
        
        # Special indicators
        if '?' in query and query.count('?') > 1:
            score += 0.2  # Multiple questions
        if re.search(r'\b(all|every|each)\b', query.lower()):
            score += 0.1  # Comprehensive request
        
        return min(1.0, score)
    
    def _compute_dynamic_k(
        self,
        base_k: int,
        corpus_size: int,
        query_type: QueryType,
    ) -> int:
        """Compute dynamic k based on corpus size and query type.
        
        Scales k up for larger corpora to maintain recall,
        scales down for small corpora to avoid noise.
        """
        if corpus_size == 0:
            return base_k
        
        # Scaling factors based on corpus size
        # Small corpus (<100): use base or less
        # Medium corpus (100-1000): use base
        # Large corpus (>1000): scale up
        if corpus_size < 50:
            scale = 0.6
        elif corpus_size < 100:
            scale = 0.8
        elif corpus_size < 500:
            scale = 1.0
        elif corpus_size < 1000:
            scale = 1.2
        elif corpus_size < 5000:
            scale = 1.4
        else:
            scale = 1.6
        
        # Exploratory/Summary queries scale more aggressively
        if query_type in (QueryType.EXPLORATORY, QueryType.SUMMARY):
            scale *= 1.2
        
        # Factual queries should stay tight
        if query_type == QueryType.FACTUAL:
            scale *= 0.8
        
        return max(3, min(50, int(base_k * scale)))
    
    def _get_retrieval_params(
        self,
        query_type: QueryType,
        complexity: float,
        corpus_size: int = 0,
    ) -> dict:
        """Get retrieval parameters based on query analysis and corpus size.
        
        Uses ROUTING_STRATEGIES for base parameters, then adjusts for
        complexity and corpus size.
        """
        defaults = self._config.retrieval
        
        # Get strategy for this query type (or fall back to FACTUAL)
        strategy = self.ROUTING_STRATEGIES.get(
            query_type,
            self.ROUTING_STRATEGIES[QueryType.FACTUAL]
        )
        
        # Base parameters from strategy
        base_dense_k = strategy.get("dense_k", defaults.dense_k)
        base_sparse_k = strategy.get("sparse_k", defaults.sparse_k)
        base_rerank_pool = strategy.get("rerank_pool", defaults.rerank_pool)
        
        # Apply dynamic k based on corpus size
        dense_k = self._compute_dynamic_k(base_dense_k, corpus_size, query_type)
        sparse_k = self._compute_dynamic_k(base_sparse_k, corpus_size, query_type)
        rerank_pool = self._compute_dynamic_k(base_rerank_pool, corpus_size, query_type)
        
        # Compression setting
        compression_map = {"tight": True, "moderate": True, "light": True, "none": False}
        compression = compression_map.get(strategy.get("compression", "moderate"), defaults.compression)
        
        # Build params dict including strategy extras
        params = {
            'dense_k': dense_k,
            'sparse_k': sparse_k,
            'rerank_pool': rerank_pool,
            'compression': compression,
            'raptor': strategy.get('raptor', False),
            'diversity': strategy.get('diversity', 0.0),
            'iterative': strategy.get('iterative', False),
            'max_hops': strategy.get('max_hops', 1),
            'sparse_weight': strategy.get('sparse_weight', 0.4),
        }
        
        # Adjust for high complexity
        if complexity > 0.7:
            params['dense_k'] = min(40, int(params['dense_k'] * 1.3))
            params['rerank_pool'] = min(50, int(params['rerank_pool'] * 1.3))
        
        return params
    
    async def analyze_query_llm(self, query: str) -> QueryAnalysis | None:
        """Use LLM to analyze the query (async)."""
        if not self._provider:
            return None
        
        prompt = f"""Analyze this query and respond in the exact format below.

Guidelines:
- Keep sub-queries short and information-seeking.
- Prefer terms that will surface documents in a knowledge base.
- Use 1 sub-query for simple factual questions, up to 15 for complex or comparative ones.
- If the query is already specific, keep sub-queries close to the original intent.

Query: {query}

Respond with:
TYPE: [factual|comparative|analytical|summary|procedural|conversational]
SUB_QUERIES: [semicolon-separated list of 1-15 sub-queries, or "none"]
KEY_TERMS: [comma-separated list of 3-5 key terms]
COMPLEXITY: [low|medium|high]

Example response:
TYPE: comparative
SUB_QUERIES: What is Python?; What is JavaScript?; How do they compare?
KEY_TERMS: python, javascript, programming, comparison
COMPLEXITY: medium"""

        try:
            response = await self._provider.chat([
                {
                    "role": "system",
                    "content": (
                        "You are a high-precision retrieval planner for a RAG system. "
                        "Your job is to classify the query, propose 1–3 focused sub-queries, "
                        "and list key terms that will improve retrieval. "
                        "Be concise, deterministic, and follow the format exactly."
                    ),
                },
                {"role": "user", "content": prompt}
            ])
            self._last_planner_mode = "llm"
            return self._parse_llm_analysis(response, query)
        except Exception as e:
            print(f"LLM query analysis failed: {e}")
            return None
    
    def _parse_llm_analysis(self, response: str, original_query: str) -> QueryAnalysis:
        """Parse LLM response into QueryAnalysis."""
        lines = response.strip().split('\n')
        
        query_type = QueryType.FACTUAL
        sub_queries = [original_query]
        expanded_terms = []
        complexity = 0.5
        
        for line in lines:
            line = line.strip()
            if line.startswith('TYPE:'):
                type_str = line.split(':', 1)[1].strip().lower()
                try:
                    query_type = QueryType(type_str)
                except ValueError:
                    pass
            elif line.startswith('SUB_QUERIES:'):
                queries_str = line.split(':', 1)[1].strip()
                if queries_str.lower() != 'none':
                    sub_queries = [q.strip() for q in queries_str.split(';') if q.strip()]
                    if not sub_queries:
                        sub_queries = [original_query]
            elif line.startswith('KEY_TERMS:'):
                terms_str = line.split(':', 1)[1].strip()
                expanded_terms = [t.strip() for t in terms_str.split(',') if t.strip()]
            elif line.startswith('COMPLEXITY:'):
                comp_str = line.split(':', 1)[1].strip().lower()
                complexity = {'low': 0.3, 'medium': 0.5, 'high': 0.8}.get(comp_str, 0.5)
        
        return QueryAnalysis(
            query_type=query_type,
            sub_queries=sub_queries,
            expanded_terms=expanded_terms,
            complexity_score=complexity,
        )
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze query using heuristics (sync fallback)."""
        self._last_planner_mode = "heuristic"
        query_type = self._classify_query_heuristic(query)
        sub_queries = self._decompose_query_heuristic(query, query_type)
        expanded_terms = self._expand_query_heuristic(query)
        complexity = self._estimate_complexity(query, query_type)
        
        return QueryAnalysis(
            query_type=query_type,
            sub_queries=sub_queries,
            expanded_terms=expanded_terms,
            complexity_score=complexity,
        )
    
    def plan(self, query: str) -> RetrievalPlan:
        """Create a retrieval plan for the given query (sync)."""
        analysis = self.analyze_query(query)
        return self._build_plan(query, analysis)
    
    async def plan_async(self, query: str) -> RetrievalPlan:
        """Create a retrieval plan using LLM if available (async)."""
        # Try LLM analysis first
        analysis = await self.analyze_query_llm(query)
        if not analysis:
            analysis = self.analyze_query(query)
        
        return self._build_plan(query, analysis)
    
    def _build_plan(self, query: str, analysis: QueryAnalysis) -> RetrievalPlan:
        """Build retrieval plan from analysis."""
        defaults = self._config.retrieval
        params = self._get_retrieval_params(analysis.query_type, analysis.complexity_score)
        
        steps = []
        for i, sub_query in enumerate(analysis.sub_queries):
            step = PlanStep(
                query=sub_query,
                dense_k=params['dense_k'],
                sparse_k=params['sparse_k'],
                rerank_pool=params['rerank_pool'],
                compression=params['compression'],
                priority=len(analysis.sub_queries) - i,  # First query highest priority
            )
            steps.append(step)
        
        # Determine iterative settings
        iterative = params.get('iterative', False)
        max_iterations = params.get('max_hops', 1) if iterative else 1
        
        return RetrievalPlan(
            steps=steps,
            target_tokens=defaults.target_tokens,
            coverage_target=defaults.coverage_target,
            original_query=query,
            query_type=analysis.query_type,
            decomposed=len(analysis.sub_queries) > 1,
            expanded_terms=analysis.expanded_terms,
            iterative=iterative,
            max_iterations=max_iterations,
            routing_params=params,
        )


# Backward compatibility alias
Planner = SmartPlanner

__all__ = [
    "QueryType",
    "PlanStep", 
    "RetrievalPlan",
    "QueryAnalysis",
    "SmartPlanner",
    "Planner",
    "compute_marginal_gain",
]
