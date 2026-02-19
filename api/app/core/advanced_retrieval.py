"""Advanced retrieval strategies integrating RAPTOR and GraphRAG.

This module provides unified access to SOTA retrieval approaches:
- RAPTOR: Hierarchical overview + leaf chunk retrieval
- GraphRAG: Entity graph traversal and community summaries
- Multi-resolution: Parent-child chunk expansion
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .gatherer import EvidenceChunk
    from .graph_rag import GraphRAG
    from .hierarchy import DocumentTree, HierarchyBuilder
    from .hybrid_retrieval import HybridRetrievalEngine


class AdvancedRetrievalMode(str, Enum):
    """Available advanced retrieval modes."""
    STANDARD = "standard"           # Normal hybrid retrieval
    RAPTOR = "raptor"               # Hierarchical overview + leaves
    GRAPH = "graph"                 # Entity graph traversal
    MULTI_RESOLUTION = "multi_res"  # Parent-child expansion
    COMBINED = "combined"           # All strategies together


@dataclass
class AdvancedRetrievalResult:
    """Result from advanced retrieval strategies."""
    chunks: list[EvidenceChunk]
    mode_used: AdvancedRetrievalMode
    overview_summaries: list[str] = field(default_factory=list)
    community_summaries: list[str] = field(default_factory=list)
    graph_context: dict[str, Any] = field(default_factory=dict)
    hierarchy_context: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class AdvancedRetriever:
    """Unified advanced retrieval combining RAPTOR, GraphRAG, and multi-resolution.

    Key features:
    - RAPTOR: Retrieve overview summaries from tree roots, then drill to leaves
    - GraphRAG: Query entity graph, use community summaries for global questions
    - Multi-resolution: Expand retrieved chunks with parent/sibling context
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetrievalEngine | None = None,
    ) -> None:
        self._hybrid = hybrid_retriever
        self._document_trees: dict[str, DocumentTree] = {}
        self._graph_rag: GraphRAG | None = None
        self._hierarchy_builder: HierarchyBuilder | None = None

    def set_hybrid_retriever(self, retriever: HybridRetrievalEngine) -> None:
        """Set the base hybrid retriever."""
        self._hybrid = retriever

    def set_graph_rag(self, graph: GraphRAG) -> None:
        """Set the GraphRAG instance."""
        self._graph_rag = graph

    def add_document_tree(self, doc_id: str, tree: DocumentTree) -> None:
        """Add a RAPTOR document tree."""
        self._document_trees[doc_id] = tree

    def set_hierarchy_builder(self, builder: HierarchyBuilder) -> None:
        """Set hierarchy builder for on-the-fly tree construction."""
        self._hierarchy_builder = builder

    def _detect_optimal_mode(self, query: str) -> AdvancedRetrievalMode:
        """Detect optimal retrieval mode based on query characteristics."""
        query_lower = query.lower()

        # Global sensemaking queries benefit from GraphRAG
        global_patterns = [
            'overview', 'summarize', 'all', 'entire', 'whole',
            'main themes', 'key points', 'generally', 'overall'
        ]
        if any(p in query_lower for p in global_patterns):
            if self._graph_rag and self._graph_rag.graph:
                return AdvancedRetrievalMode.GRAPH
            elif self._document_trees:
                return AdvancedRetrievalMode.RAPTOR

        # Multi-hop or entity-focused queries benefit from GraphRAG
        entity_patterns = ['relationship', 'connected', 'related to', 'between']
        if any(p in query_lower for p in entity_patterns) and self._graph_rag:
            return AdvancedRetrievalMode.GRAPH

        # Hierarchical questions benefit from RAPTOR
        hierarchical_patterns = ['section', 'chapter', 'part', 'introduction', 'conclusion']
        if any(p in query_lower for p in hierarchical_patterns) and self._document_trees:
            return AdvancedRetrievalMode.RAPTOR

        # Long, complex queries may benefit from multi-resolution
        if len(query.split()) > 15:
            return AdvancedRetrievalMode.MULTI_RESOLUTION

        return AdvancedRetrievalMode.STANDARD

    async def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
        mode: AdvancedRetrievalMode | None = None,
    ) -> AdvancedRetrievalResult:
        """Execute advanced retrieval with the optimal or specified mode.

        Args:
            query: User query
            top_k: Maximum chunks to return
            document_ids: Optional document filter
            mode: Force specific mode, or auto-detect if None

        Returns:
            AdvancedRetrievalResult with chunks and context
        """
        # Auto-detect mode if not specified
        if mode is None:
            mode = self._detect_optimal_mode(query)

        result = AdvancedRetrievalResult(
            chunks=[],
            mode_used=mode,
        )

        # Always get base chunks from hybrid retriever
        if self._hybrid:
            hybrid_results = await self._hybrid.query(query, top_k=top_k, document_ids=document_ids)

            # Convert to EvidenceChunk format
            from .gatherer import EvidenceChunk
            for hr in hybrid_results:
                chunk = EvidenceChunk(
                    id=hr.chunk_id or f"{hr.document.id}_{hr.start_char}",
                    title=hr.document.title,
                    snippet=hr.chunk_text,
                    score=hr.score,
                )
                result.chunks.append(chunk)

        # Apply mode-specific enhancements
        if mode == AdvancedRetrievalMode.RAPTOR:
            self._apply_raptor_retrieval(result, query, document_ids)
        elif mode == AdvancedRetrievalMode.GRAPH:
            self._apply_graph_retrieval(result, query, document_ids)
        elif mode == AdvancedRetrievalMode.MULTI_RESOLUTION:
            self._apply_multi_resolution(result, query, document_ids)
        elif mode == AdvancedRetrievalMode.COMBINED:
            self._apply_raptor_retrieval(result, query, document_ids)
            self._apply_graph_retrieval(result, query, document_ids)

        result.details["total_chunks"] = len(result.chunks)
        return result

    def _apply_raptor_retrieval(
        self,
        result: AdvancedRetrievalResult,
        query: str,
        document_ids: list[str] | None,
    ) -> None:
        """Apply RAPTOR hierarchical retrieval."""
        from .gatherer import EvidenceChunk
        from .hierarchy import HierarchicalRetriever

        for doc_id, tree in self._document_trees.items():
            if document_ids and doc_id not in document_ids:
                continue

            retriever = HierarchicalRetriever(tree)
            root = tree.get_node(tree.root_id)

            # Add root overview summary
            if root and root.summary:
                result.overview_summaries.append(f"[{doc_id}] {root.summary}")
                # Also add as a pseudo-chunk for context
                result.chunks.insert(0, EvidenceChunk(
                    id=f"{doc_id}_overview",
                    title=f"{root.title} (Overview)",
                    snippet=root.summary,
                    score=0.95,  # High score for overview
                ))

            # Get hierarchical context for retrieved chunks
            for chunk in result.chunks[:5]:  # Limit to top 5
                chunk_id_parts = chunk.id.split("_")
                if len(chunk_id_parts) > 1:
                    potential_chunk_id = chunk_id_parts[-1]
                    context_chain = retriever.get_context_chain(potential_chunk_id)
                    if context_chain:
                        result.hierarchy_context.extend(context_chain[:3])

        result.details["raptor_trees_used"] = len(self._document_trees)
        result.details["overview_count"] = len(result.overview_summaries)

    def _apply_graph_retrieval(
        self,
        result: AdvancedRetrievalResult,
        query: str,
        document_ids: list[str] | None,
    ) -> None:
        """Apply GraphRAG entity and community retrieval."""
        from .gatherer import EvidenceChunk

        if not self._graph_rag or not self._graph_rag.graph:
            return

        # Query relevant entities
        relevant_entities = self._graph_rag.query_entities(query, top_k=5)
        result.graph_context["entities"] = [
            {"name": e.name, "type": e.type.value, "score": s}
            for e, s in relevant_entities
        ]

        # Get community summaries for relevant entities
        entity_names = {e.name for e, _ in relevant_entities}
        for community in self._graph_rag.communities:
            if any(ent in community.entities for ent in entity_names) and community.summary:
                result.community_summaries.append(community.summary)
                # Add as pseudo-chunk
                result.chunks.insert(0, EvidenceChunk(
                    id=f"community_{community.id}",
                    title=f"Community {community.id} Summary",
                    snippet=community.summary,
                    score=0.85,
                ))

        # Multi-hop traversal for related chunks
        related_chunk_ids = self._graph_rag.multi_hop_query(query, hops=2, top_k=5)
        result.graph_context["multi_hop_chunks"] = related_chunk_ids

        result.details["entities_found"] = len(relevant_entities)
        result.details["communities_used"] = len(result.community_summaries)

    def _apply_multi_resolution(
        self,
        result: AdvancedRetrievalResult,
        query: str,
        document_ids: list[str] | None,
    ) -> None:
        """Apply multi-resolution parent-child expansion."""
        from .gatherer import EvidenceChunk
        from .hierarchy import HierarchicalRetriever

        expanded_chunks: list[EvidenceChunk] = []
        seen_ids: set[str] = {c.id for c in result.chunks}

        for chunk in result.chunks[:5]:  # Expand top 5 chunks
            # Find containing tree and get siblings/parents
            for doc_id, tree in self._document_trees.items():
                if document_ids and doc_id not in document_ids:
                    continue

                retriever = HierarchicalRetriever(tree)

                # Find node containing this chunk
                for node_id, node in tree.nodes.items():
                    chunk_id_match = chunk.id.split("_")[-1] if "_" in chunk.id else chunk.id
                    if chunk_id_match in node.chunk_ids:
                        # Add sibling summaries
                        siblings = retriever.expand_with_siblings(node_id)
                        for i, sibling_summary in enumerate(siblings[:2]):
                            sib_id = f"{doc_id}_sibling_{node_id}_{i}"
                            if sib_id not in seen_ids:
                                expanded_chunks.append(EvidenceChunk(
                                    id=sib_id,
                                    title="Related Section",
                                    snippet=sibling_summary,
                                    score=chunk.score * 0.8,  # Slightly lower score
                                ))
                                seen_ids.add(sib_id)

                        # Add parent context
                        if node.parent_id:
                            parent = tree.get_node(node.parent_id)
                            if parent and parent.summary:
                                parent_id = f"{doc_id}_parent_{node.parent_id}"
                                if parent_id not in seen_ids:
                                    expanded_chunks.append(EvidenceChunk(
                                        id=parent_id,
                                        title=f"{parent.title} (Context)",
                                        snippet=parent.summary,
                                        score=chunk.score * 0.9,
                                    ))
                                    seen_ids.add(parent_id)
                        break

        # Merge expanded chunks into result
        result.chunks.extend(expanded_chunks)
        # Re-sort by score
        result.chunks.sort(key=lambda c: c.score, reverse=True)

        result.details["chunks_expanded"] = len(expanded_chunks)


__all__ = [
    "AdvancedRetrievalMode",
    "AdvancedRetrievalResult",
    "AdvancedRetriever",
]
