"""GraphRAG: Knowledge graph construction and graph-guided retrieval.

This module provides:
- Entity extraction from documents using LLM
- Relationship extraction between entities
- Knowledge graph construction via NetworkX
- Community detection for thematic clustering
- Graph-guided retrieval for multi-hop reasoning

Based on: From Local to Global: A GraphRAG Approach to Query-Focused Summarization
Paper: https://arxiv.org/abs/2404.16130
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .gatherer import EvidenceChunk
    from .providers import LLMProvider


class EntityType(str, Enum):
    """Standard entity types for knowledge graphs."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    PRODUCT = "product"
    TECHNOLOGY = "technology"
    OTHER = "other"


@dataclass
class Entity:
    """An entity extracted from documents."""
    name: str
    type: EntityType
    description: str = ""
    mentions: list[str] = field(default_factory=list)  # chunk_ids where mentioned
    embedding: list[float] | None = None

    def __hash__(self) -> int:
        return hash(self.name.lower())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return False
        return self.name.lower() == other.name.lower()


@dataclass
class Relationship:
    """A relationship between two entities."""
    source: str  # entity name
    target: str  # entity name
    relation: str  # e.g., "works_for", "related_to", "part_of"
    weight: float = 1.0
    chunk_ids: list[str] = field(default_factory=list)  # evidence chunks
    description: str = ""


@dataclass
class Community:
    """A thematic community of related entities."""
    id: int
    entities: list[str]  # entity names
    summary: str = ""
    level: int = 0  # Hierarchy level (for Leiden algorithm)


class GraphRAG:
    """Knowledge graph construction and graph-guided retrieval.

    Implements the GraphRAG approach:
    1. Extract entities and relationships from chunks
    2. Build a knowledge graph
    3. Detect communities for thematic clustering
    4. Generate community summaries
    5. Use graph structure for multi-hop reasoning
    """

    KNOWLEDGE_EXTRACTION_PROMPT = """Extract a knowledge graph from the following text.
Identify all high-value entities and the relationships between them.

Entity types: PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT, PRODUCT, TECHNOLOGY, OTHER

Text:
{text}

Respond in VALID JSON format with this structure:
{{
  "entities": [
    {{"name": "Entity Name", "type": "TYPE", "description": "concise description"}}
  ],
  "relationships": [
    {{"source": "Entity Name", "target": "Entity Name", "relation": "type_of_link", "description": "how they are linked"}}
  ]
}}

STRICT FORMATTING RULES:
1. Return ONLY the JSON object.
2. Do NOT use markdown code blocks (```json).
3. Do NOT add preamble or explanation.
4. Escape quotes within strings.
"""

    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.relationships: list[Relationship] = []
        self.communities: list[Community] = []
        self.chunk_documents: dict[str, str] = {}
        self._graph: Any = None  # NetworkX graph

    @property
    def graph(self) -> Any:
        """Public accessor for the NetworkX graph."""
        return self._graph

    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for consistent matching."""
        return name.strip().lower()

    async def extract_knowledge_from_chunk(
        self,
        chunk_text: str,
        chunk_id: str,
        provider: LLMProvider,
    ) -> tuple[list[Entity], list[Relationship]]:
        """Extract both entities and relationships in a single pass with retries."""
        import asyncio
        import json
        import re

        prompt = self.KNOWLEDGE_EXTRACTION_PROMPT.format(text=chunk_text[:2500])

        # Retry loop for reliability with local models
        for attempt in range(3):
            try:
                response = await provider.chat([
                    {"role": "system", "content": "You are a precise knowledge graph extractor. Always respond in valid JSON. No Markdown."},
                    {"role": "user", "content": prompt},
                ])

                clean_response = response.strip()

                # Robust JSON extraction: Handle markdown code blocks common in local models
                # Try to find content inside ```json ... ``` or just { ... }
                json_block_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', clean_response, re.IGNORECASE)
                if json_block_match:
                    clean_response = json_block_match.group(1)
                else:
                    # Fallback: Find first { and last }
                    json_match = re.search(r'(\{[\s\S]*\})', clean_response)
                    if json_match:
                        clean_response = json_match.group(1)

                # Simple repair: remove trailing commas before closing braces/brackets
                clean_response = re.sub(r',\s*([}\]])', r'\1', clean_response)

                try:
                    data = json.loads(clean_response)
                except json.JSONDecodeError:
                    # Fallback: try to fix unescaped quotes if safe
                    try:
                        clean_response_fixed = clean_response.replace("'", '"')
                        data = json.loads(clean_response_fixed)
                    except Exception:
                        if attempt < 2:
                            continue  # Retry on parse error
                        raise ValueError(f"Failed to parse JSON: {clean_response[:100]}...")

                entities = []
                for e_data in data.get("entities", []):
                    name = e_data.get("name", "").strip()
                    if not name:
                        continue

                    type_str = e_data.get("type", "OTHER").upper()
                    try:
                        entity_type = EntityType(type_str.lower())
                    except ValueError:
                        entity_type = EntityType.OTHER

                    entities.append(Entity(
                        name=name,
                        type=entity_type,
                        description=e_data.get("description", ""),
                        mentions=[chunk_id],
                    ))

                relationships = []
                for r_data in data.get("relationships", []):
                    source = r_data.get("source", "").strip()
                    target = r_data.get("target", "").strip()
                    relation = r_data.get("relation", "").strip()
                    if not (source and target and relation):
                        continue

                    relationships.append(Relationship(
                        source=source,
                        target=target,
                        relation=relation,
                        chunk_ids=[chunk_id],
                        description=r_data.get("description", ""),
                    ))

                return entities, relationships

            except Exception as e:
                # Backoff slightly on failure
                await asyncio.sleep(0.5 * (attempt + 1))
                if attempt == 2:
                    # Provide more descriptive error details for debugging
                    error_msg = str(e) or "Empty error message"
                    if "Provider request failed" in error_msg:
                        print(f"GraphRAG Error: LLM Provider failed to process chunk {chunk_id} after 3 retries. Error: {error_msg}")
                    else:
                        print(f"GraphRAG Warning: Failed to parse extraction JSON for chunk {chunk_id}: {error_msg}")

        return [], []

    async def extract_entities_from_chunk(
        self,
        chunk_text: str,
        chunk_id: str,
        provider: LLMProvider,
    ) -> list[Entity]:
        """Deprecated: Use extract_knowledge_from_chunk for better performance."""
        entities, _ = await self.extract_knowledge_from_chunk(chunk_text, chunk_id, provider)
        return entities

    async def extract_relationships_from_chunk(
        self,
        chunk_text: str,
        chunk_id: str,
        entities: list[Entity],
        provider: LLMProvider,
    ) -> list[Relationship]:
        """Deprecated: Use extract_knowledge_from_chunk for better performance."""
        # This is now effectively a no-op if called after extraction,
        # but kept for interface compatibility.
        return []

    async def build_from_chunks(
        self,
        chunks: list[EvidenceChunk],
        provider: LLMProvider,
        on_progress: Callable[[str, int, int, str | None], None] | None = None,
    ) -> None:
        """Build knowledge graph from document chunks.

        This is the main entry point for graph construction.
        Extracts entities and relationships from all chunks.
        """
        import asyncio

        all_entities: list[Entity] = []
        all_relationships: list[Relationship] = []

        # Limit concurrent LLM calls to prevent local provider overload (e.g. Ollama queue full)
        semaphore = asyncio.Semaphore(3)
        total_chunks = len(chunks)
        processed_chunks = 0

        async def process_chunk(chunk: EvidenceChunk) -> None:
            nonlocal processed_chunks
            chunk_id = getattr(chunk, 'id', str(id(chunk)))
            chunk_doc_id = getattr(chunk, 'doc_id', None)
            if chunk_doc_id is not None:
                self.chunk_documents[chunk_id] = chunk_doc_id
            chunk_text = getattr(chunk, 'snippet', '') or getattr(chunk, 'text', '')

            if not chunk_text:
                processed_chunks += 1
                return

            async with semaphore:
                # Optimized: Extract both in one call
                chunk_entities, chunk_relationships = await self.extract_knowledge_from_chunk(
                    chunk_text, chunk_id, provider
                )

            all_entities.extend(chunk_entities)
            all_relationships.extend(chunk_relationships)

            processed_chunks += 1
            if on_progress:
                detail = f"Found {len(chunk_entities)} entities, {len(chunk_relationships)} relations"
                on_progress("extracting_graph", processed_chunks, total_chunks, detail)

        if on_progress:
            on_progress("extracting_graph", 0, total_chunks)

        await asyncio.gather(*(process_chunk(c) for c in chunks))

        # Merge entities with same name
        for entity in all_entities:
            norm_name = self._normalize_name(entity.name)
            if norm_name in self.entities:
                # Merge mentions
                self.entities[norm_name].mentions.extend(entity.mentions)
            else:
                self.entities[norm_name] = entity

        # Add relationships
        self.relationships.extend(all_relationships)

        # Build NetworkX graph
        self._build_networkx_graph()

    def _build_networkx_graph(self) -> None:
        """Construct NetworkX graph from entities and relationships."""
        try:
            import networkx as nx

            self._graph = nx.Graph()

            # Add entity nodes
            for name, entity in self.entities.items():
                self._graph.add_node(
                    name,
                    type=entity.type.value,
                    description=entity.description,
                    mentions=len(entity.mentions),
                )

            # Add relationship edges
            for rel in self.relationships:
                source_norm = self._normalize_name(rel.source)
                target_norm = self._normalize_name(rel.target)

                if source_norm in self.entities and target_norm in self.entities:
                    if self._graph.has_edge(source_norm, target_norm):
                        # Increase weight for existing edge
                        self._graph[source_norm][target_norm]['weight'] += rel.weight
                    else:
                        self._graph.add_edge(
                            source_norm,
                            target_norm,
                            relation=rel.relation,
                            weight=rel.weight,
                            description=rel.description,
                        )
        except ImportError:
            self._graph = None

    def detect_communities(self) -> list[Community]:
        """Detect communities in the knowledge graph.

        Uses Louvain algorithm for community detection.
        Falls back to connected components if not available.
        """
        if self._graph is None:
            return []

        try:
            import networkx as nx

            # Try Louvain algorithm first
            try:
                from networkx.algorithms.community import louvain_communities
                communities_set = louvain_communities(self._graph, seed=42)
            except (ImportError, AttributeError):
                # Fall back to connected components
                communities_set = list(nx.connected_components(self._graph))

            self.communities = []
            for i, members in enumerate(communities_set):
                community = Community(
                    id=i,
                    entities=list(members),
                )
                self.communities.append(community)

            return self.communities
        except Exception:
            return []

    async def summarize_communities(
        self,
        provider: LLMProvider,
        on_progress: Callable[[str, int, int, str | None], None] | None = None,
    ) -> dict[int, str]:
        """Generate summaries for each community."""
        import asyncio
        summaries = {}

        total_communities = len(self.communities)
        processed_communities = 0
        semaphore = asyncio.Semaphore(5)  # Limit concurrent summaries

        async def summarize_community(community: Community) -> None:
            nonlocal processed_communities

            if not community.entities:
                summary = "Empty community"
                community.summary = summary
                summaries[community.id] = summary
                processed_communities += 1
                if on_progress:
                    on_progress("summarizing_communities", processed_communities, total_communities)
                return
            if len(community.entities) < 2:
                summary = f"Single entity: {community.entities[0]}"
                community.summary = summary
                summaries[community.id] = summary
                processed_communities += 1
                if on_progress:
                    on_progress("summarizing_communities", processed_communities, total_communities)
                return

            # Build entity descriptions
            entity_info = []
            for name in community.entities[:10]:  # Limit to 10 entities
                if name in self.entities:
                    entity = self.entities[name]
                    entity_info.append(f"- {entity.name} ({entity.type.value}): {entity.description}")

            # Get relationships within community
            community_rels = []
            community_set = set(community.entities)
            for rel in self.relationships[:20]:  # Limit relationships
                src_norm = self._normalize_name(rel.source)
                tgt_norm = self._normalize_name(rel.target)
                if src_norm in community_set and tgt_norm in community_set:
                    community_rels.append(f"- {rel.source} {rel.relation} {rel.target}")

            prompt = f"""Summarize this thematic cluster of entities and their relationships.

Entities:
{chr(10).join(entity_info)}

Relationships:
{chr(10).join(community_rels) if community_rels else '(No explicit relationships)'}

Write a 1-2 sentence summary describing the main theme or topic of this cluster."""

            try:
                async with semaphore:
                    response = await provider.chat([
                        {"role": "system", "content": "You are a knowledge graph summarizer."},
                        {"role": "user", "content": prompt},
                    ])
                community.summary = response.strip()[:300]
                summaries[community.id] = community.summary
            except Exception:
                community.summary = f"Cluster of {len(community.entities)} related entities"
                summaries[community.id] = community.summary

            processed_communities += 1
            if on_progress:
                on_progress("summarizing_communities", processed_communities, total_communities)

        if on_progress:
            on_progress("summarizing_communities", 0, total_communities)

        await asyncio.gather(*(summarize_community(c) for c in self.communities))
        return summaries

    def query_entities(self, query: str, top_k: int = 5) -> list[Entity]:
        """Find entities relevant to a query via simple matching."""
        query_terms = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))

        scored_entities = []
        for name, entity in self.entities.items():
            # Score by term overlap with name and description
            entity_terms = set(re.findall(r'\b[a-z]{3,}\b', name.lower()))
            entity_terms.update(re.findall(r'\b[a-z]{3,}\b', entity.description.lower()))

            overlap = len(query_terms & entity_terms)
            if overlap > 0:
                scored_entities.append((entity, overlap))

        # Sort by score and return top_k
        scored_entities.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored_entities[:top_k]]

    def multi_hop_query(
        self,
        query: str,
        hops: int = 2,
        top_k: int = 5,
    ) -> list[str]:
        """Follow relationships for multi-hop reasoning.

        Returns chunk_ids that may contain relevant information
        based on graph traversal.
        """
        if self._graph is None:
            return []

        try:

            # Find starting entities
            start_entities = self.query_entities(query, top_k=3)
            if not start_entities:
                return []

            # BFS traversal up to N hops
            visited_chunks: set[str] = set()
            visited_entities: set[str] = set()

            queue = [(self._normalize_name(e.name), 0) for e in start_entities]

            while queue:
                entity_name, depth = queue.pop(0)

                if entity_name in visited_entities:
                    continue
                visited_entities.add(entity_name)

                # Add chunks where this entity is mentioned
                if entity_name in self.entities:
                    visited_chunks.update(self.entities[entity_name].mentions)

                # Add neighbors if within hop limit
                if depth < hops and entity_name in self._graph:
                    for neighbor in self._graph.neighbors(entity_name):
                        if neighbor not in visited_entities:
                            queue.append((neighbor, depth + 1))

            return list(visited_chunks)[:top_k * hops]
        except Exception:
            return []

    def get_entity_context(self, entity_name: str) -> dict[str, Any]:
        """Get full context for an entity including neighbors and relationships."""
        norm_name = self._normalize_name(entity_name)

        if norm_name not in self.entities:
            return {}

        entity = self.entities[norm_name]

        context = {
            "entity": {
                "name": entity.name,
                "type": entity.type.value,
                "description": entity.description,
                "mentions": entity.mentions,
            },
            "relationships": [],
            "neighbors": [],
        }

        # Get relationships involving this entity
        for rel in self.relationships:
            src_norm = self._normalize_name(rel.source)
            tgt_norm = self._normalize_name(rel.target)

            if src_norm == norm_name or tgt_norm == norm_name:
                context["relationships"].append({
                    "source": rel.source,
                    "relation": rel.relation,
                    "target": rel.target,
                    "description": rel.description,
                })

        # Get neighbor entities from graph
        if self._graph and norm_name in self._graph:
            for neighbor in self._graph.neighbors(norm_name):
                if neighbor in self.entities:
                    context["neighbors"].append({
                        "name": self.entities[neighbor].name,
                        "type": self.entities[neighbor].type.value,
                    })

        return context

    def to_dict(self) -> dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "entities": {
                name: {
                    "name": e.name,
                    "type": e.type.value,
                    "description": e.description,
                    "mentions": e.mentions,
                }
                for name, e in self.entities.items()
            },
            "relationships": [
                {
                    "source": r.source,
                    "target": r.target,
                    "relation": r.relation,
                    "weight": r.weight,
                    "chunk_ids": r.chunk_ids,
                }
                for r in self.relationships
            ],
            "chunk_documents": self.chunk_documents,
            "communities": [
                {
                    "id": c.id,
                    "entities": c.entities,
                    "summary": c.summary,
                }
                for c in self.communities
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphRAG:
        """Deserialize graph from dictionary."""
        graph = cls()

        for name, e_data in data.get("entities", {}).items():
            entity = Entity(
                name=e_data["name"],
                type=EntityType(e_data["type"]),
                description=e_data.get("description", ""),
                mentions=e_data.get("mentions", []),
            )
            graph.entities[name] = entity

        graph.chunk_documents = data.get("chunk_documents", {})

        for r_data in data.get("relationships", []):
            relationship = Relationship(
                source=r_data["source"],
                target=r_data["target"],
                relation=r_data["relation"],
                weight=r_data.get("weight", 1.0),
                chunk_ids=r_data.get("chunk_ids", []),
            )
            graph.relationships.append(relationship)

        for c_data in data.get("communities", []):
            community = Community(
                id=c_data["id"],
                entities=c_data["entities"],
                summary=c_data.get("summary", ""),
            )
            graph.communities.append(community)

        graph._build_networkx_graph()
        return graph


__all__ = [
    "EntityType",
    "Entity",
    "Relationship",
    "Community",
    "GraphRAG",
]
