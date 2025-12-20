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
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .providers import LLMProvider
    from .gatherer import EvidenceChunk


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
    
    ENTITY_EXTRACTION_PROMPT = """Extract entities from the following text.
For each entity, identify its type and a brief description.

Entity types: PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT, PRODUCT, TECHNOLOGY, OTHER

Text:
{text}

Respond in this exact format (one entity per line):
ENTITY: [name] | TYPE: [type] | DESCRIPTION: [brief description]

Only extract clearly mentioned entities. Be precise and concise."""

    RELATIONSHIP_EXTRACTION_PROMPT = """Given these entities extracted from a document, identify relationships between them.

Entities:
{entities}

Text:
{text}

For each relationship, specify:
RELATIONSHIP: [source entity] | [relationship type] | [target entity] | [description]

Relationship types examples: works_for, part_of, located_in, related_to, caused_by, used_for, created_by

Only extract relationships that are clearly stated or strongly implied in the text."""

    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.relationships: list[Relationship] = []
        self.communities: list[Community] = []
        self._graph: Any = None  # NetworkX graph
    
    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for consistent matching."""
        return name.strip().lower()
    
    async def extract_entities_from_chunk(
        self,
        chunk_text: str,
        chunk_id: str,
        provider: "LLMProvider",
    ) -> list[Entity]:
        """Extract entities from a single chunk."""
        prompt = self.ENTITY_EXTRACTION_PROMPT.format(text=chunk_text[:2000])
        
        try:
            response = await provider.chat([
                {"role": "system", "content": "You are a precise entity extractor."},
                {"role": "user", "content": prompt},
            ])
            return self._parse_entities(response, chunk_id)
        except Exception:
            return []
    
    def _parse_entities(self, response: str, chunk_id: str) -> list[Entity]:
        """Parse LLM entity extraction response."""
        entities = []
        
        for line in response.strip().split('\n'):
            if not line.startswith('ENTITY:'):
                continue
            
            try:
                # Parse: ENTITY: [name] | TYPE: [type] | DESCRIPTION: [description]
                parts = line.split('|')
                if len(parts) < 2:
                    continue
                
                name = parts[0].replace('ENTITY:', '').strip()
                type_str = parts[1].replace('TYPE:', '').strip().upper()
                description = parts[2].replace('DESCRIPTION:', '').strip() if len(parts) > 2 else ""
                
                # Map to EntityType
                try:
                    entity_type = EntityType(type_str.lower())
                except ValueError:
                    entity_type = EntityType.OTHER
                
                entity = Entity(
                    name=name,
                    type=entity_type,
                    description=description,
                    mentions=[chunk_id],
                )
                entities.append(entity)
            except Exception:
                continue
        
        return entities
    
    async def extract_relationships_from_chunk(
        self,
        chunk_text: str,
        chunk_id: str,
        entities: list[Entity],
        provider: "LLMProvider",
    ) -> list[Relationship]:
        """Extract relationships between entities in a chunk."""
        if len(entities) < 2:
            return []
        
        entity_list = "\n".join([f"- {e.name} ({e.type.value})" for e in entities])
        prompt = self.RELATIONSHIP_EXTRACTION_PROMPT.format(
            entities=entity_list,
            text=chunk_text[:2000],
        )
        
        try:
            response = await provider.chat([
                {"role": "system", "content": "You are a precise relationship extractor."},
                {"role": "user", "content": prompt},
            ])
            return self._parse_relationships(response, chunk_id)
        except Exception:
            return []
    
    def _parse_relationships(self, response: str, chunk_id: str) -> list[Relationship]:
        """Parse LLM relationship extraction response."""
        relationships = []
        
        for line in response.strip().split('\n'):
            if not line.startswith('RELATIONSHIP:'):
                continue
            
            try:
                # Parse: RELATIONSHIP: [source] | [relation] | [target] | [description]
                content = line.replace('RELATIONSHIP:', '').strip()
                parts = [p.strip() for p in content.split('|')]
                if len(parts) < 3:
                    continue
                
                source = parts[0]
                relation = parts[1]
                target = parts[2]
                description = parts[3] if len(parts) > 3 else ""
                
                relationship = Relationship(
                    source=source,
                    target=target,
                    relation=relation,
                    chunk_ids=[chunk_id],
                    description=description,
                )
                relationships.append(relationship)
            except Exception:
                continue
        
        return relationships
    
    async def build_from_chunks(
        self,
        chunks: list["EvidenceChunk"],
        provider: "LLMProvider",
    ) -> None:
        """Build knowledge graph from document chunks.
        
        This is the main entry point for graph construction.
        Extracts entities and relationships from all chunks.
        """
        all_entities: list[Entity] = []
        all_relationships: list[Relationship] = []
        
        for chunk in chunks:
            chunk_id = getattr(chunk, 'id', str(id(chunk)))
            chunk_text = getattr(chunk, 'snippet', '') or getattr(chunk, 'text', '')
            
            if not chunk_text:
                continue
            
            # Extract entities
            chunk_entities = await self.extract_entities_from_chunk(
                chunk_text, chunk_id, provider
            )
            all_entities.extend(chunk_entities)
            
            # Extract relationships
            if chunk_entities:
                chunk_relationships = await self.extract_relationships_from_chunk(
                    chunk_text, chunk_id, chunk_entities, provider
                )
                all_relationships.extend(chunk_relationships)
        
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
        provider: "LLMProvider",
    ) -> dict[int, str]:
        """Generate summaries for each community."""
        summaries = {}
        
        for community in self.communities:
            if len(community.entities) < 2:
                community.summary = f"Single entity: {community.entities[0]}"
                summaries[community.id] = community.summary
                continue
            
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
                response = await provider.chat([
                    {"role": "system", "content": "You are a knowledge graph summarizer."},
                    {"role": "user", "content": prompt},
                ])
                community.summary = response.strip()[:300]
                summaries[community.id] = community.summary
            except Exception:
                community.summary = f"Cluster of {len(community.entities)} related entities"
                summaries[community.id] = community.summary
        
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
            import networkx as nx
            
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
    def from_dict(cls, data: dict[str, Any]) -> "GraphRAG":
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
