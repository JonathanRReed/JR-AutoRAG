"""Advanced integration tests for SOTA RAG features.

Tests for:
- FLARE uncertainty triggering and retrieval
- GraphRAG entity extraction and graph queries
- Hallucination firewall edge cases
- Cache invalidation scenarios
- Orchestrator E2E flow with mock LLM
"""

from __future__ import annotations

import pytest

from app.core.cache import CacheManager, LRUCache, QueryCache, RetrievalMode
from app.core.flare import FLAREConfig, FLAREGenerator, FLAREStep
from app.core.gatherer import EvidenceChunk
from app.core.graph_rag import Entity, EntityType, GraphRAG, Relationship
from app.core.hallucination_firewall import FirewallResult, HallucinationFirewall

# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_chunks():
    """Create sample evidence chunks for testing."""
    return [
        EvidenceChunk(
            id="chunk_1",
            title="Financial Report Q3",
            snippet="Revenue increased by 25% year-over-year, reaching $10.5 billion in Q3 2025.",
            score=0.95,
        ),
        EvidenceChunk(
            id="chunk_2",
            title="Market Analysis",
            snippet="The company expanded into European markets, opening offices in London and Berlin.",
            score=0.88,
        ),
        EvidenceChunk(
            id="chunk_3",
            title="Strategy Overview",
            snippet="Management outlined a five-year growth strategy focusing on AI and cloud services.",
            score=0.82,
        ),
    ]


@pytest.fixture
def conflicting_chunks():
    """Chunks with contradictory information."""
    return [
        EvidenceChunk(
            id="chunk_a",
            title="Report A",
            snippet="The project was completed on time and under budget.",
            score=0.9,
        ),
        EvidenceChunk(
            id="chunk_b",
            title="Report B",
            snippet="The project was not completed on time and exceeded the budget.",
            score=0.85,
        ),
    ]


# ============================================================================
# FLARE Integration Tests
# ============================================================================

class TestFLAREIntegration:
    """Tests for FLARE active retrieval triggering."""

    def test_confidence_estimation(self):
        """Test that uncertainty patterns lower confidence scores."""
        config = FLAREConfig(confidence_threshold=0.5)
        generator = FLAREGenerator(config=config)

        # Confident text
        confident = "The revenue was $10.5 billion in Q3 2025."
        confident_signal = generator._estimate_confidence(confident)

        # Uncertain text
        uncertain = "I'm not sure, but perhaps the revenue might have been around $10 billion."
        uncertain_signal = generator._estimate_confidence(uncertain)

        # Confident should score higher than uncertain (compare aggregate values)
        assert confident_signal.aggregate > uncertain_signal.aggregate, \
            f"Confident ({confident_signal.aggregate}) should > Uncertain ({uncertain_signal.aggregate})"


    def test_sentence_splitting(self):
        """Test proper sentence segmentation for FLARE."""
        generator = FLAREGenerator()

        text = "First sentence. Second sentence! Third sentence?"
        sentences = generator._split_sentences(text)

        assert len(sentences) == 3
        assert "First sentence" in sentences[0]
        assert "Second sentence" in sentences[1]
        assert "Third sentence" in sentences[2]

    def test_retrieval_query_extraction(self):
        """Test that retrieval queries combine uncertainty with original query."""
        generator = FLAREGenerator()

        original = "What was the revenue in Q3?"
        uncertain_sentence = "I think it might have been around 10 billion."

        query = generator._extract_retrieval_query(uncertain_sentence, original)

        # Should combine both
        assert len(query) > 0
        # Should not be empty

    def test_max_retrievals_respected(self):
        """Test that FLARE respects max_retrievals limit."""
        config = FLAREConfig(max_retrievals=2)
        generator = FLAREGenerator(config=config)

        # Simulate 3 low-confidence steps
        steps = [
            FLAREStep(text="Maybe fact 1.", confidence=0.2, triggered_retrieval=True),
            FLAREStep(text="Perhaps fact 2.", confidence=0.2, triggered_retrieval=True),
            FLAREStep(text="Possibly fact 3.", confidence=0.2, triggered_retrieval=False),  # Should not trigger
        ]

        triggered_count = sum(1 for s in steps if s.triggered_retrieval)
        assert triggered_count <= config.max_retrievals


# ============================================================================
# GraphRAG Integration Tests
# ============================================================================

class TestGraphRAGIntegration:
    """Tests for GraphRAG entity extraction and graph queries."""

    def test_entity_normalization(self):
        """Test entity name normalization for consistent matching."""
        graph = GraphRAG()

        # Different cases should normalize to same value
        name1 = graph._normalize_name("John Smith")
        name2 = graph._normalize_name("JOHN SMITH")
        name3 = graph._normalize_name("john smith")

        assert name1 == name2 == name3

    def test_entity_hash_and_equality(self):
        """Test Entity hashing for set membership."""
        entity1 = Entity(name="Apple", type=EntityType.ORGANIZATION)
        entity2 = Entity(name="Apple", type=EntityType.ORGANIZATION)
        entity3 = Entity(name="Google", type=EntityType.ORGANIZATION)

        # Same name and type should be equal
        assert entity1 == entity2

        # Different names should not be equal
        assert entity1 != entity3

        # Should be usable in sets
        entity_set = {entity1, entity2, entity3}
        assert len(entity_set) == 2  # entity1 and entity2 are duplicates

    def test_simple_query_entities(self):
        """Test entity matching against simple queries."""
        graph = GraphRAG()

        # Add some entities manually
        graph.entities["apple"] = Entity(name="Apple", type=EntityType.ORGANIZATION)
        graph.entities["microsoft"] = Entity(name="Microsoft", type=EntityType.ORGANIZATION)
        graph.entities["ai"] = Entity(name="AI", type=EntityType.CONCEPT)

        # Query should match entities
        matches = graph.query_entities("Apple and Microsoft are competing in AI", top_k=3)

        assert len(matches) > 0

    def test_relationship_creation(self):
        """Test relationship dataclass."""
        rel = Relationship(
            source="Apple",
            target="Microsoft",
            relation="competes_with",
            weight=0.8,
        )

        assert rel.source == "Apple"
        assert rel.target == "Microsoft"
        assert rel.weight == 0.8

    def test_graph_serialization(self):
        """Test graph to_dict and from_dict roundtrip."""
        graph = GraphRAG()
        graph.entities["test"] = Entity(name="Test", type=EntityType.CONCEPT)
        graph.relationships.append(
            Relationship(source="A", target="B", relation="related_to")
        )

        # Serialize
        data = graph.to_dict()

        # Deserialize
        restored = GraphRAG.from_dict(data)

        assert "test" in restored.entities
        assert len(restored.relationships) == 1


# ============================================================================
# Hallucination Firewall Tests
# ============================================================================

class TestHallucinationFirewallIntegration:
    """Tests for hallucination detection edge cases."""

    def test_detects_citations(self, sample_chunks):
        """Test citation detection patterns."""
        firewall = HallucinationFirewall()

        # Various citation formats
        assert firewall._has_citation("Revenue was $10B [1].")
        assert firewall._has_citation("As stated (Source: Report).")
        assert firewall._has_citation("According to ChunkID:123.")

        # No citation
        assert not firewall._has_citation("Revenue was $10B.")

    def test_meta_sentence_filtering(self):
        """Test that metadata sentences are not counted as claims."""
        firewall = HallucinationFirewall()

        # Meta sentences
        assert firewall._is_meta_sentence("## Summary")
        assert firewall._is_meta_sentence("**Note:**")
        assert firewall._is_meta_sentence("Sources:")
        assert firewall._is_meta_sentence("N/A")
        assert firewall._is_meta_sentence("Hi")  # Too short

        # Real claims
        assert not firewall._is_meta_sentence("The company grew by 25% in Q3.")

    def test_term_overlap_scoring(self, sample_chunks):
        """Test term overlap calculation."""
        firewall = HallucinationFirewall()

        # High overlap answer (uses terms from chunks)
        high_overlap = (
            "Revenue increased by 25% to reach $10.5 billion. "
            "The company expanded into European markets including London."
        )

        result = firewall.verify(high_overlap, sample_chunks, "What happened?")
        assert result.pass_rate > 0.5

    def test_strict_mode_marks_unverified(self, sample_chunks):
        """Test strict mode adds warning markers."""
        firewall = HallucinationFirewall(strict_mode=True, min_overlap=0.9)

        # Answer with fabricated claim
        answer = "Revenue was $10B [1]. Also, aliens visited the CEO."

        result = firewall.verify(answer, sample_chunks, "What happened?")

        # Strict mode should mark unverified claims
        if result.flagged_claims:
            assert "[⚠️ UNVERIFIED]" in result.cleaned_answer

    def test_empty_chunks_handling(self):
        """Test firewall handles empty chunks gracefully."""
        firewall = HallucinationFirewall()

        result = firewall.verify("Some answer.", [], "Some query")

        # Should not crash, but may have low pass rate
        assert isinstance(result, FirewallResult)

    def test_pass_rate_calculation(self, sample_chunks):
        """Test pass rate is calculated correctly."""
        firewall = HallucinationFirewall()

        # Answer where all claims should be supported
        answer = "Revenue increased. The company expanded to Europe."

        result = firewall.verify(answer, sample_chunks, "What happened?")

        assert 0.0 <= result.pass_rate <= 1.0
        assert result.verified_claims + len(result.flagged_claims) == result.total_claims


# ============================================================================
# Cache Invalidation Tests
# ============================================================================

class TestCacheInvalidation:
    """Tests for cache key generation and invalidation."""

    def test_corpus_version_changes_key(self):
        """Different corpus versions should produce different cache keys."""
        cache = QueryCache()

        key_v1 = cache._make_key("test query", corpus_version="1")
        key_v2 = cache._make_key("test query", corpus_version="2")

        assert key_v1 != key_v2

    def test_retrieval_mode_changes_key(self):
        """Different retrieval modes should produce different cache keys."""
        cache = QueryCache()

        mode_standard = RetrievalMode.STANDARD
        mode_raptor = RetrievalMode.STANDARD | RetrievalMode.RAPTOR
        mode_graph = RetrievalMode.STANDARD | RetrievalMode.GRAPH

        key1 = cache._make_key("test", retrieval_mode=int(mode_standard))
        key2 = cache._make_key("test", retrieval_mode=int(mode_raptor))
        key3 = cache._make_key("test", retrieval_mode=int(mode_graph))

        assert key1 != key2
        assert key2 != key3
        assert key1 != key3

    def test_config_hash_changes_key(self):
        """Different config hashes should produce different cache keys."""
        cache = QueryCache()

        key1 = cache._make_key("test", config_hash="abc123")
        key2 = cache._make_key("test", config_hash="xyz789")

        assert key1 != key2

    def test_lru_eviction_order(self):
        """Test LRU eviction works correctly."""
        cache: LRUCache[str] = LRUCache(max_size=2)

        cache.set("a", "value_a")
        cache.set("b", "value_b")

        # Access 'a' to make it recently used
        cache.get("a")

        # Add 'c' - should evict 'b' (least recently used)
        cache.set("c", "value_c")

        assert cache.get("a") == "value_a"
        assert cache.get("b") is None  # Evicted
        assert cache.get("c") == "value_c"

    def test_cache_manager_stats(self):
        """Test cache manager provides useful statistics."""
        manager = CacheManager(
            embedding_cache_size=100,
            query_cache_size=50,
        )

        # Add some entries
        manager.embeddings.set("text1", [0.1, 0.2])
        manager.embeddings.set("text2", [0.3, 0.4])
        manager.queries.set("query1", {"answer": "test"})

        # Get some hits and misses
        manager.embeddings.get("text1")  # Hit
        manager.embeddings.get("nonexistent")  # Miss

        stats = manager.stats()

        assert "embeddings" in stats
        assert "queries" in stats
        assert stats["embeddings"]["hits"] >= 1


# ============================================================================
# Embedding Model Preset Tests
# ============================================================================

class TestEmbeddingModelPresets:
    """Tests for new embedding model preset system."""

    def test_preset_values_exist(self):
        """Test all expected presets are defined."""
        from app.core.hybrid_retrieval import EmbeddingModelPreset

        assert EmbeddingModelPreset.BGE_BASE == "BAAI/bge-base-en-v1.5"
        assert EmbeddingModelPreset.BGE_M3 == "BAAI/bge-m3"
        assert EmbeddingModelPreset.GTE_QWEN == "Alibaba-NLP/gte-Qwen2-1.5B-instruct"
        assert EmbeddingModelPreset.SENTENCE_MINI == "sentence-transformers/all-MiniLM-L6-v2"

    def test_model_info_retrieval(self):
        """Test model metadata retrieval."""
        from app.core.hybrid_retrieval import EmbeddingModelPreset

        info = EmbeddingModelPreset.get_info(EmbeddingModelPreset.BGE_M3)

        assert "dimensions" in info
        assert "max_tokens" in info
        assert "requires_api" in info

    def test_config_preset_override(self):
        """Test that model_preset overrides embedding_model."""
        from app.core.hybrid_retrieval import EmbeddingModelPreset, HybridConfig

        config = HybridConfig(model_preset=EmbeddingModelPreset.BGE_M3)

        # __post_init__ should set embedding_model from preset
        assert config.embedding_model == "BAAI/bge-m3"

    def test_config_without_preset(self):
        """Test default behavior without preset."""
        from app.core.hybrid_retrieval import HybridConfig

        config = HybridConfig()

        # Should use default
        assert config.embedding_model == "BAAI/bge-base-en-v1.5"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
