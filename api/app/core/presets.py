"""Retrieval preset configurations for speed/accuracy tradeoffs.

This module provides predefined presets that configure the RAG pipeline
for different use cases:
- TURBO: Fastest responses, minimal retrieval
- FAST: Quick answers with reranking
- BALANCED: Good balance of speed and accuracy
- THOROUGH: Deep research with hierarchy
- ULTRA_ACCURATE: Maximum accuracy, all SOTA features

Each preset controls retrieval parameters, SOTA feature toggles,
and target latency expectations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PresetLevel(str, Enum):
    """Available preset levels."""
    TURBO = "turbo"
    FAST = "fast"
    BALANCED = "balanced"
    THOROUGH = "thorough"
    ULTRA_ACCURATE = "ultra_accurate"


@dataclass
class RetrievalPreset:
    """Complete preset configuration for the RAG pipeline.
    
    Attributes:
        name: Human-readable preset name
        level: Preset level enum
        description: Short description of the preset
        
        # Retrieval parameters
        top_k: Number of chunks to retrieve (final)
        dense_k: Dense retrieval candidates
        sparse_k: BM25 retrieval candidates
        rerank_pool: Number of candidates to rerank
        target_tokens: Target context tokens
        coverage: Target coverage ratio (0-1)
        
        # SOTA feature toggles
        rerank: Enable cross-encoder reranking
        raptor: Enable RAPTOR hierarchical retrieval
        graph: Enable GraphRAG entity retrieval
        flare: Enable FLARE mid-generation retrieval
        iterative: Enable iterative retrieval
        max_iterations: Max iterations for iterative retrieval
        hallucination_check: Enable hallucination firewall
        evidence_contract: Enable evidence contract verification
        citation_verify: Enable citation verification
        
        # Performance targets
        target_latency_ms: Expected latency in milliseconds
    """
    name: str
    level: PresetLevel
    description: str
    
    # Retrieval parameters
    top_k: int = 5
    dense_k: int = 10
    sparse_k: int = 5
    rerank_pool: int = 20
    target_tokens: int = 1600
    coverage: float = 0.7
    
    # SOTA feature toggles
    rerank: bool = True
    raptor: bool = False
    graph: bool = False
    flare: bool = False
    iterative: bool = False
    max_iterations: int = 1
    hallucination_check: bool = False
    evidence_contract: bool = False
    citation_verify: bool = False
    
    # Routing parameters
    diversity: float = 0.0
    sparse_weight: float = 0.4
    
    # Performance targets
    target_latency_ms: int = 5000
    
    def to_dict(self) -> dict[str, Any]:
        """Convert preset to dictionary for API responses."""
        return {
            "name": self.name,
            "level": self.level.value,
            "description": self.description,
            "top_k": self.top_k,
            "dense_k": self.dense_k,
            "sparse_k": self.sparse_k,
            "rerank_pool": self.rerank_pool,
            "target_tokens": self.target_tokens,
            "coverage": self.coverage,
            "rerank": self.rerank,
            "raptor": self.raptor,
            "graph": self.graph,
            "flare": self.flare,
            "iterative": self.iterative,
            "max_iterations": self.max_iterations,
            "hallucination_check": self.hallucination_check,
            "evidence_contract": self.evidence_contract,
            "citation_verify": self.citation_verify,
            "diversity": self.diversity,
            "sparse_weight": self.sparse_weight,
            "target_latency_ms": self.target_latency_ms,
        }


# =============================================================================
# Preset Definitions
# =============================================================================

TURBO_PRESET = RetrievalPreset(
    name="Turbo",
    level=PresetLevel.TURBO,
    description="Fastest responses with basic retrieval. Best for simple questions.",
    top_k=2,
    dense_k=5,
    sparse_k=3,
    rerank_pool=8,
    target_tokens=400,
    coverage=0.3,
    rerank=False,
    raptor=False,
    graph=False,
    flare=False,
    iterative=False,
    max_iterations=1,
    hallucination_check=False,
    evidence_contract=False,
    citation_verify=False,
    diversity=0.0,
    sparse_weight=0.4,
    target_latency_ms=500,
)

FAST_PRESET = RetrievalPreset(
    name="Fast",
    level=PresetLevel.FAST,
    description="Quick answers with reranking. Good for straightforward queries.",
    top_k=3,
    dense_k=8,
    sparse_k=4,
    rerank_pool=12,
    target_tokens=800,
    coverage=0.5,
    rerank=True,
    raptor=False,
    graph=False,
    flare=False,
    iterative=False,
    max_iterations=1,
    hallucination_check=False,
    evidence_contract=False,
    citation_verify=False,
    diversity=0.0,
    sparse_weight=0.4,
    target_latency_ms=2000,
)

BALANCED_PRESET = RetrievalPreset(
    name="Balanced",
    level=PresetLevel.BALANCED,
    description="Good balance of speed and accuracy. Recommended for most queries.",
    top_k=5,
    dense_k=12,
    sparse_k=6,
    rerank_pool=20,
    target_tokens=1600,
    coverage=0.7,
    rerank=True,
    raptor=False,
    graph=False,
    flare=False,
    iterative=False,
    max_iterations=1,
    hallucination_check=False,
    evidence_contract=False,
    citation_verify=False,
    diversity=0.1,
    sparse_weight=0.4,
    target_latency_ms=5000,
)

THOROUGH_PRESET = RetrievalPreset(
    name="Thorough",
    level=PresetLevel.THOROUGH,
    description="Deep research with hierarchical context. Use for complex topics.",
    top_k=10,
    dense_k=20,
    sparse_k=10,
    rerank_pool=30,
    target_tokens=3000,
    coverage=0.9,
    rerank=True,
    raptor=True,
    graph=False,
    flare=True,
    iterative=True,
    max_iterations=2,
    hallucination_check=False,
    evidence_contract=False,
    citation_verify=False,
    diversity=0.2,
    sparse_weight=0.4,
    target_latency_ms=15000,
)

ULTRA_ACCURATE_PRESET = RetrievalPreset(
    name="Ultra Accurate",
    level=PresetLevel.ULTRA_ACCURATE,
    description="Maximum accuracy with all SOTA features. For critical research.",
    top_k=20,
    dense_k=30,
    sparse_k=15,
    rerank_pool=50,
    target_tokens=5000,
    coverage=0.95,
    rerank=True,
    raptor=True,
    graph=True,
    flare=True,
    iterative=True,
    max_iterations=3,
    hallucination_check=True,
    evidence_contract=True,
    citation_verify=True,
    diversity=0.3,
    sparse_weight=0.4,
    target_latency_ms=60000,
)

# Preset registry
PRESETS: dict[str, RetrievalPreset] = {
    PresetLevel.TURBO.value: TURBO_PRESET,
    PresetLevel.FAST.value: FAST_PRESET,
    PresetLevel.BALANCED.value: BALANCED_PRESET,
    PresetLevel.THOROUGH.value: THOROUGH_PRESET,
    PresetLevel.ULTRA_ACCURATE.value: ULTRA_ACCURATE_PRESET,
}

# Ordered list for UI display
PRESET_ORDER = [
    PresetLevel.TURBO,
    PresetLevel.FAST,
    PresetLevel.BALANCED,
    PresetLevel.THOROUGH,
    PresetLevel.ULTRA_ACCURATE,
]


def get_preset(level: str | PresetLevel) -> RetrievalPreset:
    """Get a preset by level name.
    
    Args:
        level: Preset level name or enum
        
    Returns:
        RetrievalPreset configuration
        
    Raises:
        ValueError: If preset level is unknown
    """
    if isinstance(level, PresetLevel):
        level = level.value
    
    level = level.lower()
    if level not in PRESETS:
        raise ValueError(f"Unknown preset level: {level}. Available: {list(PRESETS.keys())}")
    
    return PRESETS[level]


def get_all_presets() -> list[dict]:
    """Get all presets as dictionaries for API responses."""
    return [PRESETS[level.value].to_dict() for level in PRESET_ORDER]


def suggest_preset_for_query(
    query: str,
    query_type: str | None = None,
    complexity: float | None = None,
) -> PresetLevel:
    """Suggest a preset based on query characteristics.
    
    Args:
        query: The user query
        query_type: Optional query type from SmartPlanner
        complexity: Optional complexity score (0-1)
        
    Returns:
        Suggested preset level
    """
    # Simple heuristic-based suggestion
    word_count = len(query.split())
    
    # Very short queries -> Fast
    if word_count <= 5:
        return PresetLevel.FAST
    
    # Complex query types -> Thorough or Ultra
    if query_type in ("multi_hop", "analytical", "comparative"):
        if complexity and complexity > 0.7:
            return PresetLevel.ULTRA_ACCURATE
        return PresetLevel.THOROUGH
    
    # Summary/exploratory -> Thorough
    if query_type in ("summary", "exploratory"):
        return PresetLevel.THOROUGH
    
    # Long queries might need more context
    if word_count > 20:
        return PresetLevel.THOROUGH
    
    # Default to Balanced
    return PresetLevel.BALANCED


__all__ = [
    "PresetLevel",
    "RetrievalPreset",
    "PRESETS",
    "PRESET_ORDER",
    "TURBO_PRESET",
    "FAST_PRESET",
    "BALANCED_PRESET",
    "THOROUGH_PRESET",
    "ULTRA_ACCURATE_PRESET",
    "get_preset",
    "get_all_presets",
    "suggest_preset_for_query",
]
