"""Document hierarchy for RAPTOR-style chunk relationships.

This module provides hierarchical document indexing:
- Parent-child chunk relationships
- Section-based grouping
- Multi-level summaries for tree retrieval
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .chunking import Chunk


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
            }
        }


class HierarchyBuilder:
    """Builds document hierarchy trees from text.
    
    Uses markdown headers to determine structure.
    Creates summaries at each level for RAPTOR-style retrieval.
    """
    
    # Header pattern for markdown
    HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    def __init__(self, summary_max_length: int = 200) -> None:
        self.summary_max_length = summary_max_length
    
    def _generate_summary(self, text: str) -> str:
        """Generate a summary for a node."""
        # Simple extractive summary - take first sentences
        clean = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?])\s+', clean)
        
        summary_parts = []
        current_len = 0
        
        for sentence in sentences[:3]:
            if current_len + len(sentence) > self.summary_max_length:
                break
            summary_parts.append(sentence)
            current_len += len(sentence)
        
        return ' '.join(summary_parts) if summary_parts else clean[:self.summary_max_length]
    
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
            if i + 1 < len(matches):
                end_pos = matches[i + 1].start()
            else:
                end_pos = len(text)
            
            sections.append((level, title, start_pos, end_pos))
        
        return sections
    
    def build(self, text: str, document_id: str, title: str = "Document") -> DocumentTree:
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
            for l in range(level - 1, -1, -1):
                if l in level_stack:
                    parent_id = level_stack[l]
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
            to_remove = [l for l in level_stack if l > level]
            for l in to_remove:
                del level_stack[l]
        
        return DocumentTree(
            root_id=root_id,
            nodes=nodes,
            document_id=document_id,
        )
    
    def associate_chunks(
        self,
        tree: DocumentTree,
        chunks: list["Chunk"],
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
                r'^#{' + str(node.level) + r'}\s+' + re.escape(node.title[:50]),
                text,
                re.MULTILINE
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
