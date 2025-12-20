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


@dataclass
class QueryAnalysis:
    """Result of analyzing a query."""
    query_type: QueryType
    sub_queries: list[str]
    expanded_terms: list[str]
    complexity_score: float  # 0-1, higher = more complex


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
    
    def _get_retrieval_params(self, query_type: QueryType, complexity: float) -> dict:
        """Get retrieval parameters based on query analysis."""
        defaults = self._config.retrieval
        
        # Base parameters
        params = {
            'dense_k': defaults.dense_k,
            'sparse_k': defaults.sparse_k,
            'rerank_pool': defaults.rerank_pool,
            'compression': defaults.compression,
        }
        
        # Adjust based on query type
        if query_type == QueryType.SUMMARY:
            params['dense_k'] = min(15, params['dense_k'] * 2)
            params['rerank_pool'] = min(30, params['rerank_pool'] * 2)
        elif query_type == QueryType.COMPARATIVE:
            params['dense_k'] = min(12, int(params['dense_k'] * 1.5))
        elif query_type == QueryType.FACTUAL:
            params['dense_k'] = max(3, params['dense_k'] - 2)
            params['compression'] = False  # Keep full context for facts
        
        # Adjust for complexity
        if complexity > 0.7:
            params['dense_k'] = min(20, int(params['dense_k'] * 1.5))
            params['rerank_pool'] = min(40, int(params['rerank_pool'] * 1.5))
        
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
        
        return RetrievalPlan(
            steps=steps,
            target_tokens=defaults.target_tokens,
            coverage_target=defaults.coverage_target,
            original_query=query,
            query_type=analysis.query_type,
            decomposed=len(analysis.sub_queries) > 1,
            expanded_terms=analysis.expanded_terms,
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
]
