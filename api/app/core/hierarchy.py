"""Document hierarchy for RAPTOR-style chunk relationships.

This module provides hierarchical document indexing:
- Parent-child chunk relationships via markdown headers
- RAPTOR-style clustering using embeddings
- Abstractive summarization per cluster via LLM
- Multi-level retrieval (summary → detail)

RAPTOR Paper: https://arxiv.org/abs/2401.18059
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

    from .chunking import Chunk
    from .providers import LLMProvider


@dataclass
class HierarchyNode:
    """A node in the document hierarchy tree."""

    id: str
    level: int  # 0 = root, 1 = section, 2 = subsection, etc.
    title: str
    text: str
    summary: str  # Compressed representation for tree retrieval
    parent_id: str | None = None
    children: list[str] = field(default_factory=list)  # Child node IDs
    chunk_ids: list[str] = field(default_factory=list)  # Associated chunk IDs
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentTree:
    """Complete document hierarchy tree."""

    root_id: str
    nodes: dict[str, HierarchyNode]
    document_id: str

    def get_node(self, node_id: str) -> HierarchyNode | None:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_children(self, node_id: str) -> list[HierarchyNode]:
        """Get all children of a node."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[cid] for cid in node.children if cid in self.nodes]

    def get_ancestors(self, node_id: str) -> list[HierarchyNode]:
        """Get all ancestors from node to root."""
        ancestors = []
        current = self.nodes.get(node_id)

        while current and current.parent_id:
            parent = self.nodes.get(current.parent_id)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break

        return ancestors

    def get_siblings(self, node_id: str) -> list[HierarchyNode]:
        """Get sibling nodes (same parent)."""
        node = self.nodes.get(node_id)
        if not node or not node.parent_id:
            return []

        parent = self.nodes.get(node.parent_id)
        if not parent:
            return []

        return [
            self.nodes[cid]
            for cid in parent.children
            if cid in self.nodes and cid != node_id
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert tree to dictionary for serialization."""
        return {
            "root_id": self.root_id,
            "document_id": self.document_id,
            "nodes": {
                nid: {
                    "id": n.id,
                    "level": n.level,
                    "title": n.title,
                    "summary": n.summary,
                    "parent_id": n.parent_id,
                    "children": n.children,
                    "chunk_ids": n.chunk_ids,
                }
                for nid, n in self.nodes.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentTree:
        """Create a tree from a dictionary."""
        nodes = {}
        for nid, n_data in data.get("nodes", {}).items():
            nodes[nid] = HierarchyNode(
                id=n_data["id"],
                level=n_data["level"],
                title=n_data["title"],
                text=n_data.get("text", ""),  # Might be empty in some serializations
                summary=n_data["summary"],
                parent_id=n_data.get("parent_id"),
                children=n_data.get("children", []),
                chunk_ids=n_data.get("chunk_ids", []),
            )
        return cls(
            root_id=data["root_id"],
            nodes=nodes,
            document_id=data["document_id"],
        )


class HierarchyBuilder:
    """Builds document hierarchy trees from text.

    Uses markdown headers to determine structure.
    Creates summaries at each level for RAPTOR-style retrieval.
    """

    # Header pattern for markdown
    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def __init__(self, summary_max_length: int = 200) -> None:
        self.summary_max_length = summary_max_length

    def _generate_summary(self, text: str) -> str:
        """Generate a summary for a node."""
        # Simple extractive summary - take first sentences
        clean = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?])\s+", clean)

        summary_parts = []
        current_len = 0

        for sentence in sentences[:3]:
            if current_len + len(sentence) > self.summary_max_length:
                break
            summary_parts.append(sentence)
            current_len += len(sentence)

        return (
            " ".join(summary_parts)
            if summary_parts
            else clean[: self.summary_max_length]
        )

    def _find_sections(self, text: str) -> list[tuple[int, str, int, int]]:
        """Find all sections with their boundaries.

        Returns:
            List of (level, title, start_pos, end_pos)
        """
        matches = list(self.HEADER_PATTERN.finditer(text))
        sections = []

        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()
            start_pos = match.end()

            # End position is start of next header or end of text
            end_pos = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            sections.append((level, title, start_pos, end_pos))

        return sections

    def build(
        self, text: str, document_id: str, title: str = "Document"
    ) -> DocumentTree:
        """Build a hierarchy tree from document text.

        Args:
            text: Full document text
            document_id: ID of the source document
            title: Document title for root node

        Returns:
            DocumentTree with hierarchical structure
        """
        nodes: dict[str, HierarchyNode] = {}

        # Create root node
        root_id = str(uuid.uuid4())
        root_node = HierarchyNode(
            id=root_id,
            level=0,
            title=title,
            text=text,
            summary=self._generate_summary(text),
            parent_id=None,
        )
        nodes[root_id] = root_node

        # Find sections
        sections = self._find_sections(text)

        if not sections:
            # No headers found - document is flat
            return DocumentTree(
                root_id=root_id,
                nodes=nodes,
                document_id=document_id,
            )

        # Build hierarchy from sections
        # Track the most recent node at each level
        level_stack: dict[int, str] = {0: root_id}

        for level, section_title, start, end in sections:
            section_text = text[start:end].strip()

            # Create node for this section
            node_id = str(uuid.uuid4())

            # Find parent - closest ancestor at lower level
            parent_id = root_id
            for lvl in range(level - 1, -1, -1):
                if lvl in level_stack:
                    parent_id = level_stack[lvl]
                    break

            node = HierarchyNode(
                id=node_id,
                level=level,
                title=section_title,
                text=section_text,
                summary=self._generate_summary(section_text),
                parent_id=parent_id,
            )

            nodes[node_id] = node

            # Add as child to parent
            if parent_id in nodes:
                nodes[parent_id].children.append(node_id)

            # Update level stack
            level_stack[level] = node_id

            # Clear deeper levels (new section at this level)
            to_remove = [lvl for lvl in level_stack if lvl > level]
            for lvl in to_remove:
                del level_stack[lvl]

        return DocumentTree(
            root_id=root_id,
            nodes=nodes,
            document_id=document_id,
        )

    # =========================================================================
    # RAPTOR-style clustering and abstractive summarization
    # =========================================================================

    async def build_raptor(
        self,
        chunks: list[Chunk],
        document_id: str,
        embedder: SentenceTransformer,
        provider: LLMProvider | None = None,
        min_cluster_size: int = 2,
        max_levels: int = 4,
    ) -> DocumentTree:
        """Build RAPTOR-style tree via clustering + summarization.

        RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)
        builds a hierarchical tree by:
        1. Embedding all chunks
        2. Clustering similar chunks together
        3. Generating abstractive summary per cluster
        4. Recursively clustering summaries until single root

        Args:
            chunks: List of document chunks
            document_id: ID of the source document
            embedder: Sentence transformer for embeddings
            provider: LLM provider for abstractive summaries (optional)
            min_cluster_size: Minimum chunks per cluster
            max_levels: Maximum tree depth

        Returns:
            DocumentTree with hierarchical structure
        """
        if not chunks:
            # Return empty tree
            root_id = str(uuid.uuid4())
            return DocumentTree(
                root_id=root_id,
                nodes={
                    root_id: HierarchyNode(
                        id=root_id, level=0, title="Empty", text="", summary=""
                    )
                },
                document_id=document_id,
            )

        nodes: dict[str, HierarchyNode] = {}
        current_level_texts: list[
            tuple[str, str, list[str]]
        ] = []  # (node_id, text, chunk_ids)

        # Level 0: Create leaf nodes from chunks
        for chunk in chunks:
            node_id = str(uuid.uuid4())
            text = (
                chunk.text if hasattr(chunk, "text") else getattr(chunk, "snippet", "")
            )
            chunk_id = str(chunk.index) if hasattr(chunk, "index") else str(id(chunk))

            node = HierarchyNode(
                id=node_id,
                level=0,
                title=f"Chunk {chunk_id}",
                text=text,
                summary=text[: self.summary_max_length]
                if len(text) > self.summary_max_length
                else text,
                chunk_ids=[chunk_id],
            )
            nodes[node_id] = node
            current_level_texts.append((node_id, text, [chunk_id]))

        level = 1

        # Recursively cluster until single root or max levels
        while len(current_level_texts) > 1 and level <= max_levels:
            # Get texts for clustering
            texts = [t[1] for t in current_level_texts]

            # Compute embeddings
            embeddings = embedder.encode(texts, convert_to_numpy=True)

            # Cluster
            clusters = self._cluster_chunks_kmeans(embeddings, min_cluster_size)

            if len(clusters) >= len(current_level_texts):
                # No further clustering possible
                break

            next_level_texts: list[tuple[str, str, list[str]]] = []

            for cluster_indices in clusters:
                if not cluster_indices:
                    continue

                # Gather cluster members
                cluster_node_ids = [current_level_texts[i][0] for i in cluster_indices]
                cluster_chunk_ids = []
                for i in cluster_indices:
                    cluster_chunk_ids.extend(current_level_texts[i][2])

                # Combine texts for summary
                cluster_texts = [current_level_texts[i][1] for i in cluster_indices]
                combined_text = "\n\n".join(cluster_texts)

                # Generate summary
                if provider is not None:
                    summary = await self._summarize_cluster_llm(cluster_texts, provider)
                else:
                    summary = self._summarize_cluster_extractive(cluster_texts)

                # Create parent node
                parent_id = str(uuid.uuid4())
                parent_node = HierarchyNode(
                    id=parent_id,
                    level=level,
                    title=f"Cluster L{level}-{len(next_level_texts)}",
                    text=combined_text[:1000],  # Truncate for storage
                    summary=summary,
                    children=cluster_node_ids,
                    chunk_ids=cluster_chunk_ids,
                )
                nodes[parent_id] = parent_node

                # Update children's parent reference
                for child_id in cluster_node_ids:
                    if child_id in nodes:
                        nodes[child_id].parent_id = parent_id

                next_level_texts.append((parent_id, summary, cluster_chunk_ids))

            current_level_texts = next_level_texts
            level += 1

        # Create root node if needed
        if len(current_level_texts) == 1:
            root_id = current_level_texts[0][0]
        else:
            # Create a final root combining remaining nodes
            root_id = str(uuid.uuid4())
            root_children = [t[0] for t in current_level_texts]
            all_chunk_ids = []
            for t in current_level_texts:
                all_chunk_ids.extend(t[2])

            root_texts = [t[1] for t in current_level_texts]
            if provider is not None:
                root_summary = await self._summarize_cluster_llm(root_texts, provider)
            else:
                root_summary = self._summarize_cluster_extractive(root_texts)

            root_node = HierarchyNode(
                id=root_id,
                level=level,
                title="Document Root",
                text="",
                summary=root_summary,
                children=root_children,
                chunk_ids=all_chunk_ids,
            )
            nodes[root_id] = root_node

            # Update children's parent
            for child_id in root_children:
                if child_id in nodes:
                    nodes[child_id].parent_id = root_id

        return DocumentTree(
            root_id=root_id,
            nodes=nodes,
            document_id=document_id,
        )

    def _cluster_chunks_kmeans(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int = 2,
    ) -> list[list[int]]:
        """Cluster embeddings using k-means.

        Uses simple k-means clustering. Falls back to grouping by pairs
        if sklearn is not available.
        """
        n_samples = len(embeddings)

        if n_samples <= min_cluster_size:
            return [list(range(n_samples))]

        # Determine number of clusters (reduce by ~half each level)
        n_clusters = max(1, n_samples // 2)
        n_clusters = min(n_clusters, n_samples // min_cluster_size)

        try:
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)

            # Group indices by cluster
            clusters: dict[int, list[int]] = {}
            for idx, label in enumerate(labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(idx)

            return list(clusters.values())

        except ImportError:
            # Fallback: simple pairing
            clusters = []
            for i in range(0, n_samples, 2):
                if i + 1 < n_samples:
                    clusters.append([i, i + 1])
                else:
                    clusters.append([i])
            return clusters

    def _summarize_cluster_extractive(self, texts: list[str]) -> str:
        """Generate extractive summary for a cluster (no LLM)."""
        # Take first sentence from each text
        summaries = []
        for text in texts[:3]:  # Limit to 3 texts
            sentences = re.split(r"(?<=[.!?])\s+", text.strip())
            if sentences:
                summaries.append(sentences[0][:150])

        combined = " ".join(summaries)
        if len(combined) > self.summary_max_length:
            combined = combined[: self.summary_max_length] + "..."
        return combined

    async def _summarize_cluster_llm(
        self,
        texts: list[str],
        provider: LLMProvider,
    ) -> str:
        """Generate abstractive summary using LLM."""
        # Truncate texts to fit context
        truncated_texts = []
        total_chars = 0
        for text in texts:
            if total_chars > 3000:
                break
            truncated = text[:800] if len(text) > 800 else text
            truncated_texts.append(truncated)
            total_chars += len(truncated)

        combined = "\n\n---\n\n".join(truncated_texts)

        prompt = f"""Summarize the following document sections into a single coherent paragraph.
Focus on the key information and main themes. Be concise but comprehensive.

Sections:
{combined}

Summary:"""

        try:
            response = await provider.chat(
                [
                    {
                        "role": "system",
                        "content": "You are a precise document summarizer.",
                    },
                    {"role": "user", "content": prompt},
                ]
            )
            return response.strip()[: self.summary_max_length]
        except Exception:
            # Fallback to extractive
            return self._summarize_cluster_extractive(texts)

    def associate_chunks(
        self,
        tree: DocumentTree,
        chunks: list[Chunk],
        text: str,
    ) -> DocumentTree:
        """Associate chunks with their hierarchy nodes.

        Args:
            tree: Existing document tree
            chunks: List of chunks with start_char positions
            text: Original document text

        Returns:
            Updated tree with chunk associations
        """
        # Build position map for nodes
        node_positions: list[tuple[int, int, str]] = []

        for node_id, node in tree.nodes.items():
            if node.level == 0:
                continue

            # Find position of node's title in text
            title_match = re.search(
                r"^#{" + str(node.level) + r"}\s+" + re.escape(node.title[:50]),
                text,
                re.MULTILINE,
            )
            if title_match:
                start = title_match.start()
                # Estimate end position
                end = start + len(node.text) + 100
                node_positions.append((start, end, node_id))

        # Sort by position
        node_positions.sort()

        # Associate each chunk with the appropriate node
        for chunk in chunks:
            chunk_pos = chunk.start_char

            # Find the node that contains this chunk
            best_node_id = tree.root_id
            for start, end, node_id in node_positions:
                if start <= chunk_pos < end:
                    best_node_id = node_id
                    break
                elif chunk_pos < start:
                    break

            # Add chunk to node
            if best_node_id in tree.nodes:
                tree.nodes[best_node_id].chunk_ids.append(
                    f"{chunk.index}"  # Use chunk index as ID
                )

        return tree


class HierarchicalRetriever:
    """Retrieves chunks using hierarchy information.

    Supports RAPTOR-style tree traversal for context expansion.
    """

    def __init__(self, tree: DocumentTree) -> None:
        self.tree = tree

    def get_context_chain(self, chunk_id: str) -> list[str]:
        """Get hierarchical context for a chunk.

        Returns summaries from leaf to root for context.
        """
        # Find node containing this chunk
        containing_node = None
        for node in self.tree.nodes.values():
            if chunk_id in node.chunk_ids:
                containing_node = node
                break

        if not containing_node:
            return []

        # Build context chain from ancestors
        context = []
        ancestors = self.tree.get_ancestors(containing_node.id)

        # Add from root down
        for ancestor in reversed(ancestors):
            context.append(f"{ancestor.title}: {ancestor.summary}")

        # Add current node
        context.append(f"{containing_node.title}: {containing_node.summary}")

        return context

    def expand_with_siblings(self, node_id: str) -> list[str]:
        """Get sibling summaries for broader context."""
        siblings = self.tree.get_siblings(node_id)
        return [f"{s.title}: {s.summary}" for s in siblings]

    def get_subtree_chunks(self, node_id: str) -> list[str]:
        """Get all chunk IDs in a subtree."""
        chunks = []

        def _collect(nid: str) -> None:
            node = self.tree.get_node(nid)
            if node:
                chunks.extend(node.chunk_ids)
                for child_id in node.children:
                    _collect(child_id)

        _collect(node_id)
        return chunks


__all__ = [
    "HierarchyNode",
    "DocumentTree",
    "HierarchyBuilder",
    "HierarchicalRetriever",
]
