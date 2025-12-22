"""Retrieval Cascade: Staged retrieval with progressive stop criteria.

Implements a cascade approach to retrieval:
1. Start with fast, sparse retrieval
2. Add dense retrieval if coverage insufficient
3. Add reranking if quality low
4. Escalate to advanced modes (RAPTOR/Graph) if needed

Stop early when coverage and quality thresholds are met.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from .gatherer import EvidenceChunk, GatherResult
    from .hybrid_retrieval import HybridRetrievalEngine


class CascadeStage(str, Enum):
    """Stages in the retrieval cascade."""
    SPARSE_ONLY = "sparse_only"
    DENSE_SPARSE = "dense_sparse"
    RERANKED = "reranked"
    COLBERT = "colbert"
    RAPTOR = "raptor"
    GRAPH = "graph"
    FULL = "full"


@dataclass
class CascadeConfig:
    """Configuration for retrieval cascade."""
    # Coverage threshold to stop early
    coverage_threshold: float = 0.7
    
    # Minimum chunk score to consider "good quality"
    quality_threshold: float = 0.4
    
    # Maximum stages to run (1 = single stage, run full cascade)
    max_stages: int = 4
    
    # Enable/disable specific cascade stages
    enable_sparse_first: bool = True
    enable_reranking: bool = True
    enable_colbert: bool = False
    enable_raptor: bool = True
    enable_graph: bool = True
    
    # Top-k for each stage
    sparse_k: int = 20
    dense_k: int = 10
    rerank_k: int = 15
    colbert_k: int = 12


@dataclass
class CascadeResult:
    """Result of cascade retrieval."""
    chunks: list["EvidenceChunk"]
    stages_run: list[CascadeStage]
    stopped_early: bool
    final_coverage: float
    final_quality: float
    details: dict[str, Any] = field(default_factory=dict)


class RetrievalCascade:
    """Cascade retrieval system with progressive escalation.
    
    Runs retrieval in stages, stopping early when quality is sufficient:
    
    Stage 1: Sparse retrieval (BM25) - fast, keyword-focused
    Stage 2: Dense + Sparse hybrid - semantic understanding
    Stage 3: Reranked results - quality filtering
    Stage 4: ColBERT late interaction (if enabled)
    Stage 5: RAPTOR hierarchical (if enabled and available)
    Stage 6: GraphRAG (if enabled and available)
    
    This reduces latency for simple queries while maintaining quality
    for complex queries.
    """
    
    def __init__(self, config: CascadeConfig | None = None) -> None:
        """Initialize cascade with configuration."""
        self.config = config or CascadeConfig()
    
    def estimate_coverage(
        self,
        query: str,
        chunks: list["EvidenceChunk"],
    ) -> float:
        """Estimate query coverage based on chunk content.
        
        Returns a score 0-1 indicating how well chunks cover the query.
        """
        if not chunks:
            return 0.0
        
        # Simple term overlap coverage
        import re
        query_terms = set(w.lower() for w in re.findall(r'\b\w{4,}\b', query))
        if not query_terms:
            return 0.5  # Can't estimate
        
        chunk_text = " ".join(c.snippet.lower() for c in chunks if hasattr(c, 'snippet'))
        covered = sum(1 for t in query_terms if t in chunk_text)
        
        return covered / len(query_terms)
    
    def estimate_quality(
        self,
        chunks: list["EvidenceChunk"],
    ) -> float:
        """Estimate retrieval quality based on chunk scores."""
        if not chunks:
            return 0.0
        
        avg_score = sum(c.score for c in chunks) / len(chunks)
        return avg_score
    
    async def cascade_retrieve(
        self,
        query: str,
        retriever: "HybridRetrievalEngine",
        document_ids: list[str] | None = None,
        on_progress: Callable[[str, int, int], None] | None = None,
        raptor_available: bool = False,
        graph_available: bool = False,
    ) -> CascadeResult:
        """Run cascade retrieval with progressive stages.
        
        Args:
            query: User query
            retriever: Hybrid retrieval engine
            document_ids: Optional document filter
            on_progress: Progress callback (stage_name, current, total)
            raptor_available: Whether RAPTOR trees are built
            graph_available: Whether GraphRAG is built
            
        Returns:
            CascadeResult with merged chunks from all run stages
        """
        stages_run: list[CascadeStage] = []
        all_chunks: list["EvidenceChunk"] = []
        seen_ids: set[str] = set()
        details: dict[str, Any] = {}
        
        def add_chunks(new_chunks: list["EvidenceChunk"], stage: str) -> int:
            """Add unique chunks and return count added."""
            added = 0
            for chunk in new_chunks:
                if chunk.id not in seen_ids:
                    seen_ids.add(chunk.id)
                    all_chunks.append(chunk)
                    added += 1
            details[f"{stage}_added"] = added
            return added
        
        current_stage = 0
        total_stages = self.config.max_stages
        
        # Stage 1: Sparse-only retrieval (BM25)
        if self.config.enable_sparse_first:
            if on_progress:
                on_progress("sparse_retrieval", current_stage, total_stages)
            
            try:
                sparse_chunks = await retriever.search_sparse(
                    query,
                    top_k=self.config.sparse_k,
                    document_ids=document_ids,
                )
                add_chunks(sparse_chunks, "sparse")
                stages_run.append(CascadeStage.SPARSE_ONLY)
                
                # Check stop criteria
                coverage = self.estimate_coverage(query, all_chunks)
                quality = self.estimate_quality(all_chunks)
                
                if coverage >= self.config.coverage_threshold and quality >= self.config.quality_threshold:
                    return CascadeResult(
                        chunks=all_chunks,
                        stages_run=stages_run,
                        stopped_early=True,
                        final_coverage=coverage,
                        final_quality=quality,
                        details={**details, "stop_reason": "sparse_sufficient"},
                    )
            except Exception as e:
                details["sparse_error"] = str(e)
            
            current_stage += 1
        
        # Stage 2: Dense + Sparse hybrid
        if current_stage < total_stages:
            if on_progress:
                on_progress("dense_retrieval", current_stage, total_stages)
            
            try:
                dense_chunks = await retriever.search_dense(
                    query,
                    top_k=self.config.dense_k,
                    document_ids=document_ids,
                )
                add_chunks(dense_chunks, "dense")
                stages_run.append(CascadeStage.DENSE_SPARSE)
                
                coverage = self.estimate_coverage(query, all_chunks)
                quality = self.estimate_quality(all_chunks)
                
                if coverage >= self.config.coverage_threshold and quality >= self.config.quality_threshold:
                    return CascadeResult(
                        chunks=all_chunks,
                        stages_run=stages_run,
                        stopped_early=True,
                        final_coverage=coverage,
                        final_quality=quality,
                        details={**details, "stop_reason": "dense_sufficient"},
                    )
            except Exception as e:
                details["dense_error"] = str(e)
            
            current_stage += 1
        
        # Stage 3: Reranking
        if current_stage < total_stages and self.config.enable_reranking:
            if on_progress:
                on_progress("reranking", current_stage, total_stages)
            
            try:
                reranked = retriever.rerank(query, all_chunks, top_k=self.config.rerank_k)
                all_chunks = reranked  # Replace with reranked set
                stages_run.append(CascadeStage.RERANKED)
                
                coverage = self.estimate_coverage(query, all_chunks)
                quality = self.estimate_quality(all_chunks)
                
                if coverage >= self.config.coverage_threshold:
                    return CascadeResult(
                        chunks=all_chunks,
                        stages_run=stages_run,
                        stopped_early=True,
                        final_coverage=coverage,
                        final_quality=quality,
                        details={**details, "stop_reason": "rerank_sufficient"},
                    )
            except Exception as e:
                details["rerank_error"] = str(e)
            
            current_stage += 1
        
        # Stage 4: ColBERT (if enabled)
        if current_stage < total_stages and self.config.enable_colbert:
            if on_progress:
                on_progress("colbert", current_stage, total_stages)
            
            try:
                if hasattr(retriever, 'search_colbert'):
                    colbert_chunks = await retriever.search_colbert(
                        query,
                        top_k=self.config.colbert_k,
                        document_ids=document_ids,
                    )
                    add_chunks(colbert_chunks, "colbert")
                    stages_run.append(CascadeStage.COLBERT)
            except Exception as e:
                details["colbert_error"] = str(e)
            
            current_stage += 1
        
        # Final metrics
        final_coverage = self.estimate_coverage(query, all_chunks)
        final_quality = self.estimate_quality(all_chunks)
        
        # Sort by score
        all_chunks.sort(key=lambda c: c.score, reverse=True)
        
        return CascadeResult(
            chunks=all_chunks,
            stages_run=stages_run,
            stopped_early=False,
            final_coverage=final_coverage,
            final_quality=final_quality,
            details=details,
        )
    
    def should_escalate_to_raptor(
        self,
        coverage: float,
        query: str,
    ) -> bool:
        """Check if RAPTOR escalation is warranted."""
        if not self.config.enable_raptor:
            return False
        
        # Low coverage on a query that looks like it needs overview
        overview_patterns = ['overview', 'summarize', 'main', 'key', 'all']
        is_overview_query = any(p in query.lower() for p in overview_patterns)
        
        return coverage < self.config.coverage_threshold or is_overview_query
    
    def should_escalate_to_graph(
        self,
        coverage: float,
        query: str,
    ) -> bool:
        """Check if GraphRAG escalation is warranted."""
        if not self.config.enable_graph:
            return False
        
        # Low coverage on entity or relationship queries
        entity_patterns = ['related', 'connection', 'between', 'relationship']
        is_entity_query = any(p in query.lower() for p in entity_patterns)
        
        return coverage < self.config.coverage_threshold or is_entity_query


# Singleton for easy access
_cascade: RetrievalCascade | None = None


def get_retrieval_cascade(config: CascadeConfig | None = None) -> RetrievalCascade:
    """Get or create cascade instance."""
    global _cascade
    if _cascade is None or config is not None:
        _cascade = RetrievalCascade(config)
    return _cascade


__all__ = [
    "CascadeStage",
    "CascadeConfig",
    "CascadeResult",
    "RetrievalCascade",
    "get_retrieval_cascade",
]
