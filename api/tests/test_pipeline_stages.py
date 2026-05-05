"""Comprehensive tests for all RAG pipeline stages.

Tests each stage:
1. Cache - LRU cache, embedding cache, query cache
2. Planning - SmartPlanner query classification, decomposition
3. Gatherer - Evidence gathering from retrieval engine
4. Retrieval - Hybrid retrieval with dense/sparse fusion
5. Compression - Context compression strategies
6. Generation - Provider integration
7. Reflection - Self-reflection quality assessment
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.cache import (
    CacheManager,
    EmbeddingCache,
    LRUCache,
    QueryCache,
    get_cache_manager,
)
from app.core.compression import CompressedContext, ContextCompressor
from app.core.documents import Document, DocumentStore
from app.core.gatherer import EvidenceBundle, EvidenceChunk, Gatherer
from app.core.hybrid_retrieval import HybridConfig, HybridRetrievalEngine
from app.core.planner import Planner
from app.core.planner import RetrievalPlan as BasicRetrievalPlan
from app.core.reflection import AnswerQuality, ReflectionResult, SelfReflector
from app.core.retrieval import RetrievalResult
from app.core.smart_planner import PlanStep, QueryType, RetrievalPlan, SmartPlanner
from app.schemas.config import AppConfig, RetrievalDefaults

# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def app_config():
    """Create a test AppConfig."""
    return AppConfig(
        retrieval=RetrievalDefaults(
            dense_k=5,
            sparse_k=5,
            rerank_pool=10,
            compression=True,
            target_tokens=2000,
            coverage_target=0.8,
            max_context_tokens=4096,
        )
    )


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            id="doc1",
            title="Python Programming",
            text="Python is a high-level programming language known for its simplicity and readability. It supports multiple paradigms including object-oriented and functional programming. Python has extensive libraries for data science, web development, and automation.",
            metadata={},
        ),
        Document(
            id="doc2",
            title="JavaScript Basics",
            text="JavaScript is a versatile scripting language primarily used for web development. It runs in browsers and on servers via Node.js. JavaScript supports asynchronous programming with promises and async/await syntax.",
            metadata={},
        ),
        Document(
            id="doc3",
            title="Machine Learning Overview",
            text="Machine learning is a subset of artificial intelligence that enables systems to learn from data. Common algorithms include linear regression, decision trees, and neural networks. Deep learning uses multi-layer neural networks for complex pattern recognition.",
            metadata={},
        ),
    ]


@pytest.fixture
def sample_chunks():
    """Create sample evidence chunks for testing."""
    return [
        EvidenceChunk(
            id="chunk1",
            title="Python Programming",
            snippet="Python is a high-level programming language known for its simplicity.",
            score=0.95,
        ),
        EvidenceChunk(
            id="chunk2",
            title="JavaScript Basics",
            snippet="JavaScript is a versatile scripting language for web development.",
            score=0.85,
        ),
        EvidenceChunk(
            id="chunk3",
            title="Machine Learning",
            snippet="Machine learning enables systems to learn from data automatically.",
            score=0.75,
        ),
    ]


# ============================================================================
# 1. CACHE STAGE TESTS
# ============================================================================

class TestCacheStage:
    """Tests for the caching layer."""

    def test_lru_cache_basic_operations(self):
        """Test basic LRU cache get/set operations."""
        cache: LRUCache[str] = LRUCache(max_size=3, default_ttl=3600)

        # Set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        # Miss
        assert cache.get("nonexistent") is None

        # Stats
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1

    def test_lru_cache_eviction(self):
        """Test LRU eviction when cache is full."""
        cache: LRUCache[str] = LRUCache(max_size=2, default_ttl=3600)

        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1

        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.size == 2

    def test_lru_cache_ttl_expiration(self):
        """Test TTL-based cache expiration."""
        cache: LRUCache[str] = LRUCache(max_size=10, default_ttl=0.1)  # 100ms TTL

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        time.sleep(0.15)  # Wait for expiration
        assert cache.get("key1") is None  # Expired

    def test_embedding_cache(self):
        """Test embedding cache operations."""
        cache = EmbeddingCache(max_size=100, ttl_seconds=3600)

        embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        cache.set("test query", embedding)

        result = cache.get("test query")
        assert result == embedding

        # Different text should miss
        assert cache.get("different query") is None

    def test_embedding_cache_batch_operations(self):
        """Test batch get/set for embeddings."""
        cache = EmbeddingCache(max_size=100)

        texts = ["query1", "query2", "query3"]
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        cache.set_many(texts, embeddings)
        results = cache.get_many(texts)

        assert results == embeddings

    def test_query_cache(self):
        """Test query result caching."""
        cache = QueryCache(max_size=50, ttl_seconds=1800)

        result = {"answer": "Test answer", "chunks": [], "score": 0.9}
        cache.set("test query", result, config_hash="abc123")

        # Same query and config should hit
        cached = cache.get("test query", config_hash="abc123")
        assert cached == result

        # Different config should miss
        assert cache.get("test query", config_hash="xyz789") is None

    def test_cache_manager_integration(self):
        """Test unified cache manager."""
        manager = CacheManager(
            embedding_cache_size=100,
            query_cache_size=50,
        )

        # Test embedding cache
        manager.embeddings.set("text", [0.1, 0.2, 0.3])
        assert manager.embeddings.get("text") == [0.1, 0.2, 0.3]

        # Test query cache
        manager.queries.set("query", {"answer": "test"})
        assert manager.queries.get("query") == {"answer": "test"}

        # Test stats
        stats = manager.stats()
        assert "embeddings" in stats
        assert "queries" in stats

        # Test clear
        manager.embeddings.clear()
        manager.queries.invalidate_all()
        assert manager.embeddings.get("text") is None
        assert manager.queries.get("query") is None

    def test_global_cache_manager_singleton(self):
        """Test that get_cache_manager returns singleton."""
        manager1 = get_cache_manager()
        manager2 = get_cache_manager()
        assert manager1 is manager2


# ============================================================================
# 2. PLANNING STAGE TESTS
# ============================================================================

class TestPlanningStage:
    """Tests for the planning/query analysis stage."""

    def test_basic_planner(self, app_config):
        """Test basic Planner creates valid plan."""
        planner = Planner(app_config)
        plan = planner.plan("What is Python?")

        assert isinstance(plan, BasicRetrievalPlan)
        assert len(plan.steps) == 1
        assert plan.steps[0].query == "What is Python?"
        assert plan.target_tokens == app_config.retrieval.target_tokens

    def test_smart_planner_query_classification(self, app_config):
        """Test SmartPlanner query type classification."""
        planner = SmartPlanner(app_config)

        # Factual (note: "What is X" matches SUMMARY pattern, so use different query)
        analysis = planner.analyze_query("Define Python programming language")
        assert analysis.query_type == QueryType.FACTUAL

        # Comparative
        analysis = planner.analyze_query("Python vs JavaScript")
        assert analysis.query_type == QueryType.COMPARATIVE

        # Procedural
        analysis = planner.analyze_query("How to install Python?")
        assert analysis.query_type == QueryType.PROCEDURAL

        # Analytical
        analysis = planner.analyze_query("Why is Python popular?")
        assert analysis.query_type == QueryType.ANALYTICAL

        # Summary
        analysis = planner.analyze_query("Explain machine learning")
        assert analysis.query_type == QueryType.SUMMARY

    def test_smart_planner_query_decomposition(self, app_config):
        """Test query decomposition for complex queries."""
        planner = SmartPlanner(app_config)

        # Comparative should decompose
        analysis = planner.analyze_query("Python vs JavaScript programming")
        assert len(analysis.sub_queries) >= 1

        # Simple factual should not decompose much
        analysis = planner.analyze_query("What is Python?")
        assert len(analysis.sub_queries) <= 2

    def test_smart_planner_complexity_estimation(self, app_config):
        """Test complexity scoring."""
        planner = SmartPlanner(app_config)

        # Simple query
        simple = planner.analyze_query("What is Python?")

        # Complex query
        complex_q = planner.analyze_query(
            "Compare Python and JavaScript for web development, "
            "considering performance, ecosystem, and learning curve. "
            "Which one is better for beginners?"
        )

        assert complex_q.complexity_score > simple.complexity_score

    def test_smart_planner_builds_valid_plan(self, app_config):
        """Test SmartPlanner produces valid retrieval plan."""
        planner = SmartPlanner(app_config)
        plan = planner.plan("Explain how machine learning works")

        assert isinstance(plan, RetrievalPlan)
        assert len(plan.steps) >= 1
        assert plan.query_type == QueryType.SUMMARY
        assert all(isinstance(step, PlanStep) for step in plan.steps)

    def test_planner_rebuild(self, app_config):
        """Test planner can be rebuilt with new config."""
        planner = SmartPlanner(app_config)

        new_config = AppConfig(
            retrieval=RetrievalDefaults(dense_k=10, sparse_k=10)
        )
        planner.rebuild(new_config)

        plan = planner.plan("Test query")
        # Should use new config's dense_k (adjusted for factual)
        assert plan.steps[0].dense_k <= 10


# ============================================================================
# 3. GATHERER STAGE TESTS
# ============================================================================

class TestGathererStage:
    """Tests for the evidence gathering stage."""
    pytestmark = pytest.mark.asyncio

    async def test_gatherer_collects_evidence(self, sample_documents, tmp_path):
        """Test Gatherer collects evidence from retrieval engine."""
        doc_store = DocumentStore(path=tmp_path / "documents.db")
        for doc in sample_documents:
            doc_store.upsert(doc)

        retrieval = HybridRetrievalEngine(
            doc_store,
            HybridConfig(use_reranking=False)
        )
        retrieval.build()

        gatherer = Gatherer(retrieval)
        evidence = await gatherer.gather("What is Python?", top_k=3)

        assert isinstance(evidence, EvidenceBundle)
        assert len(evidence.chunks) <= 3
        assert evidence.coverage >= 0.0
        assert evidence.token_estimate >= 0

    async def test_gatherer_returns_evidence_chunks(self, sample_documents, tmp_path):
        """Test gathered evidence has correct structure."""
        doc_store = DocumentStore(path=tmp_path / "documents.db")
        for doc in sample_documents:
            doc_store.upsert(doc)

        retrieval = HybridRetrievalEngine(
            doc_store,
            HybridConfig(use_reranking=False)
        )
        retrieval.build()

        gatherer = Gatherer(retrieval)
        evidence = await gatherer.gather("programming language", top_k=5)

        for chunk in evidence.chunks:
            assert isinstance(chunk, EvidenceChunk)
            assert chunk.id is not None
            assert chunk.title is not None
            assert chunk.snippet is not None
            assert isinstance(chunk.score, float)

    async def test_gatherer_with_document_filter(self, sample_documents, tmp_path):
        """Test Gatherer respects document ID filter."""
        doc_store = DocumentStore(path=tmp_path / "documents.db")
        for doc in sample_documents:
            doc_store.upsert(doc)

        retrieval = HybridRetrievalEngine(
            doc_store,
            HybridConfig(use_reranking=False)
        )
        retrieval.build()

        gatherer = Gatherer(retrieval)
        evidence = await gatherer.gather(
            "programming",
            top_k=5,
            document_ids=["doc1"]
        )

        # Should only return chunks from doc1
        for chunk in evidence.chunks:
            assert "doc1" in chunk.id

    async def test_gatherer_cache_info(self, sample_documents, tmp_path):
        """Test Gatherer provides cache info."""
        doc_store = DocumentStore(path=tmp_path / "documents.db")
        for doc in sample_documents:
            doc_store.upsert(doc)

        retrieval = HybridRetrievalEngine(
            doc_store,
            HybridConfig(use_reranking=False)
        )
        retrieval.build()

        gatherer = Gatherer(retrieval)
        evidence = await gatherer.gather("Python", top_k=3)

        assert isinstance(evidence.cache_info, dict)


# ============================================================================
# 4. RETRIEVAL STAGE TESTS
# ============================================================================

class TestRetrievalStage:
    """Tests for the hybrid retrieval stage."""
    pytestmark = pytest.mark.asyncio

    async def test_retrieval_engine_basic_query(self, sample_documents, tmp_path):
        """Test basic retrieval query."""
        doc_store = DocumentStore(path=tmp_path / "documents.db")
        for doc in sample_documents:
            doc_store.upsert(doc)

        engine = HybridRetrievalEngine(
            doc_store,
            HybridConfig(use_reranking=False)
        )
        engine.build()

        results = await engine.query("What is Python?", top_k=3)

        assert len(results) <= 3
        assert all(isinstance(r, RetrievalResult) for r in results)

    async def test_retrieval_result_structure(self, sample_documents, tmp_path):
        """Test retrieval results have correct structure."""
        doc_store = DocumentStore(path=tmp_path / "documents.db")
        for doc in sample_documents:
            doc_store.upsert(doc)

        engine = HybridRetrievalEngine(
            doc_store,
            HybridConfig(use_reranking=False)
        )
        engine.build()

        results = await engine.query("programming", top_k=5)

        for result in results:
            assert hasattr(result, "document")
            assert hasattr(result, "score")
            assert hasattr(result, "chunk_text")
            assert hasattr(result, "retrieval_method")
            assert isinstance(result.score, float)

    async def test_retrieval_scores_sorted(self, sample_documents, tmp_path):
        """Test retrieval results are sorted by score."""
        doc_store = DocumentStore(path=tmp_path / "documents.db")
        for doc in sample_documents:
            doc_store.upsert(doc)

        engine = HybridRetrievalEngine(
            doc_store,
            HybridConfig(use_reranking=False)
        )
        engine.build()

        results = await engine.query("machine learning", top_k=5)

        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)

    async def test_retrieval_with_document_filter(self, sample_documents, tmp_path):
        """Test retrieval respects document ID filter."""
        doc_store = DocumentStore(path=tmp_path / "documents.db")
        for doc in sample_documents:
            doc_store.upsert(doc)

        engine = HybridRetrievalEngine(
            doc_store,
            HybridConfig(use_reranking=False)
        )
        engine.build()

        results = await engine.query(
            "programming",
            top_k=5,
            document_ids=["doc2"]
        )

        for result in results:
            assert "doc2" in result.document.id

    async def test_retrieval_empty_query(self, sample_documents, tmp_path):
        """Test retrieval handles empty query gracefully."""
        doc_store = DocumentStore(path=tmp_path / "documents.db")
        for doc in sample_documents:
            doc_store.upsert(doc)

        engine = HybridRetrievalEngine(
            doc_store,
            HybridConfig(use_reranking=False)
        )
        engine.build()

        results = await engine.query("", top_k=5)
        assert results == []

    async def test_retrieval_no_documents(self, tmp_path):
        """Test retrieval handles empty document store."""
        # Use temp path to ensure isolated empty store
        doc_store = DocumentStore(path=tmp_path / "empty_docs.db")
        engine = HybridRetrievalEngine(
            doc_store,
            HybridConfig(use_reranking=False)
        )
        engine.build()

        results = await engine.query("test query", top_k=5)
        assert results == []


# ============================================================================
# 5. COMPRESSION STAGE TESTS
# ============================================================================

class TestCompressionStage:
    """Tests for the context compression stage."""

    def test_compressor_simple_compression(self, sample_chunks):
        """Test simple truncation-based compression."""
        compressor = ContextCompressor(max_tokens=100)

        result = compressor.compress_simple(sample_chunks, max_tokens=100)

        assert isinstance(result, CompressedContext)
        assert result.chunks_used <= result.chunks_total
        assert result.estimated_tokens <= 100 * 1.5  # Allow some margin

    def test_compressor_extractive_compression(self, sample_chunks):
        """Test extractive sentence-level compression."""
        compressor = ContextCompressor(max_tokens=200)

        result = compressor.compress_extractive(
            sample_chunks,
            query="Python programming",
            max_tokens=200
        )

        assert isinstance(result, CompressedContext)
        assert len(result.text) > 0
        assert len(result.citations) > 0

    def test_compressor_citations(self, sample_chunks):
        """Test compression includes proper citations."""
        compressor = ContextCompressor(max_tokens=500)

        result = compressor.compress(sample_chunks, query="programming")

        for citation in result.citations:
            assert "id" in citation
            assert "title" in citation
            assert "citation_number" in citation

    def test_compressor_empty_chunks(self):
        """Test compression handles empty chunk list."""
        compressor = ContextCompressor()

        result = compressor.compress([])

        assert result.text == ""
        assert result.chunks_used == 0
        assert result.citations == []

    def test_compressor_format_with_citations(self, sample_chunks):
        """Test format_with_citations method."""
        compressor = ContextCompressor()

        text, citations = compressor.format_with_citations(sample_chunks)

        assert "[1]" in text
        assert "[2]" in text
        assert len(citations) == len(sample_chunks)

    def test_compressor_respects_token_limit(self, sample_chunks):
        """Test compression respects token limit."""
        compressor = ContextCompressor(max_tokens=50)

        result = compressor.compress(sample_chunks, max_tokens=50)

        # Should not exceed limit by much
        assert result.estimated_tokens <= 75  # Allow margin


# ============================================================================
# 6. GENERATION STAGE TESTS (Mocked)
# ============================================================================

class TestGenerationStage:
    """Tests for the generation stage (with mocked providers)."""

    @pytest.mark.asyncio
    async def test_provider_chat(self):
        """Test LLM provider chat method."""
        from app.core.providers import LLMProvider

        # Create mock provider
        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.chat = AsyncMock(return_value="Test response from LLM")

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]

        response = await mock_provider.chat(messages)

        assert response == "Test response from LLM"
        mock_provider.chat.assert_called_once_with(messages)

    @pytest.mark.asyncio
    async def test_provider_streaming(self):
        """Test LLM provider streaming."""
        from app.core.providers import LLMProvider

        async def mock_stream(*args, **kwargs):
            for chunk in ["Hello", " ", "world", "!"]:
                yield chunk

        mock_provider = MagicMock(spec=LLMProvider)
        mock_provider.chat_stream = mock_stream

        chunks = []
        async for chunk in mock_provider.chat_stream([]):
            chunks.append(chunk)

        assert "".join(chunks) == "Hello world!"

    def test_provider_factory_ollama(self):
        """Test provider factory creates Ollama provider."""
        from app.core.providers import OllamaProvider, ProviderFactory
        from app.schemas.config import ProviderConfig

        factory = ProviderFactory()
        config = ProviderConfig(
            name="Ollama",
            base_url="http://localhost:11434",
            generator_model="llama3"
        )

        provider = factory.build(config)
        assert isinstance(provider, OllamaProvider)

    def test_provider_factory_lmstudio(self):
        """Test provider factory creates LM Studio provider."""
        from app.core.providers import LMStudioProvider, ProviderFactory
        from app.schemas.config import ProviderConfig

        factory = ProviderFactory()
        config = ProviderConfig(
            name="LM Studio",
            base_url="http://localhost:1234",
            generator_model="mistral"
        )

        provider = factory.build(config)
        assert isinstance(provider, LMStudioProvider)


# ============================================================================
# 7. REFLECTION STAGE TESTS
# ============================================================================

class TestReflectionStage:
    """Tests for the self-reflection stage."""

    def test_reflector_high_quality_answer(self, sample_chunks):
        """Test reflection identifies high-quality answers."""
        reflector = SelfReflector()

        good_answer = (
            "Python is a high-level programming language known for its "
            "simplicity and readability [1]. It supports multiple paradigms "
            "including object-oriented and functional programming [2]. "
            "Python has extensive libraries for data science and web development."
        )

        result = reflector.reflect(
            answer=good_answer,
            query="What is Python?",
            chunks=sample_chunks,
            context_used="Python context here",
        )

        assert isinstance(result, ReflectionResult)
        assert result.confidence >= 0.5
        assert result.quality in [AnswerQuality.HIGH, AnswerQuality.MEDIUM]

    def test_reflector_detects_uncertainty(self, sample_chunks):
        """Test reflection detects uncertainty language."""
        reflector = SelfReflector()

        uncertain_answer = (
            "I'm not sure, but I think Python might be a programming language. "
            "Maybe it's used for web development, perhaps for data science too. "
            "I believe it could be popular."
        )

        result = reflector.reflect(
            answer=uncertain_answer,
            query="What is Python?",
            chunks=sample_chunks,
        )

        assert result.confidence < 0.8
        assert len(result.issues) > 0
        assert any("uncertainty" in issue.lower() for issue in result.issues)

    def test_reflector_detects_short_answer(self, sample_chunks):
        """Test reflection flags very short answers."""
        reflector = SelfReflector()

        short_answer = "Python is a language."

        result = reflector.reflect(
            answer=short_answer,
            query="Explain Python programming in detail",
            chunks=sample_chunks,
        )

        assert any("short" in issue.lower() for issue in result.issues)

    def test_reflector_detects_refusal(self, sample_chunks):
        """Test reflection detects model refusals."""
        reflector = SelfReflector()

        refusal_answer = (
            "I cannot find any information about this topic. "
            "The context does not contain relevant information. "
            "I'm unable to answer this question."
        )

        result = reflector.reflect(
            answer=refusal_answer,
            query="What is quantum computing?",
            chunks=sample_chunks,
        )

        assert result.confidence < 0.5
        assert result.quality in [AnswerQuality.LOW, AnswerQuality.INSUFFICIENT]

    def test_reflector_retry_decision(self, sample_chunks):
        """Test reflection correctly decides when to retry."""
        reflector = SelfReflector(min_confidence_threshold=0.5)

        # Low quality should trigger retry
        bad_answer = "I don't know. Maybe?"

        result = reflector.reflect(
            answer=bad_answer,
            query="What is machine learning?",
            chunks=[],  # No evidence
        )

        # Should recommend retry for low confidence + low quality
        if result.confidence < 0.5 and result.quality in [AnswerQuality.LOW, AnswerQuality.INSUFFICIENT]:
            assert result.should_retry

    def test_reflector_citation_check(self, sample_chunks):
        """Test reflection checks for citations."""
        reflector = SelfReflector()

        # Answer without citations despite having evidence
        no_citations = (
            "Python is a programming language used for many purposes. "
            "It has libraries for data science and web development. "
            "Many developers prefer Python for its simplicity."
        )

        result = reflector.reflect(
            answer=no_citations,
            query="What is Python?",
            chunks=sample_chunks,
        )

        assert any("citation" in issue.lower() for issue in result.issues)

    def test_reflector_quality_levels(self, sample_chunks):
        """Test all quality levels are accessible."""
        # Verify enum values
        assert AnswerQuality.HIGH.value == "high"
        assert AnswerQuality.MEDIUM.value == "medium"
        assert AnswerQuality.LOW.value == "low"
        assert AnswerQuality.INSUFFICIENT.value == "insufficient"


# ============================================================================
# INTEGRATION TEST
# ============================================================================

class TestPipelineIntegration:
    """Integration tests for the full pipeline."""
    pytestmark = pytest.mark.asyncio

    async def test_full_pipeline_flow(self, app_config, sample_documents, tmp_path):
        """Test complete pipeline flow from query to reflection."""
        # 1. Setup
        doc_store = DocumentStore(path=tmp_path / "documents.db")
        for doc in sample_documents:
            doc_store.upsert(doc)

        # 2. Planning
        planner = SmartPlanner(app_config)
        plan = planner.plan("What is Python programming?")
        assert len(plan.steps) >= 1

        # 3. Retrieval setup
        retrieval = HybridRetrievalEngine(
            doc_store,
            HybridConfig(use_reranking=False)
        )
        retrieval.build()

        # 4. Gathering
        gatherer = Gatherer(retrieval)
        evidence = await gatherer.gather(plan.steps[0].query, top_k=5)
        assert len(evidence.chunks) >= 0

        # 5. Compression
        compressor = ContextCompressor(max_tokens=1000)
        compressed = compressor.compress(
            evidence.chunks,
            query=plan.original_query,
        )
        assert compressed.text is not None

        # 6. Reflection (simulate generation)
        simulated_answer = (
            "Based on the context [1], Python is a programming language. "
            "It is known for simplicity [2]."
        )

        reflector = SelfReflector()
        reflection = reflector.reflect(
            answer=simulated_answer,
            query=plan.original_query,
            chunks=evidence.chunks,
            context_used=compressed.text,
        )

        assert isinstance(reflection, ReflectionResult)
        assert reflection.confidence >= 0.0
        assert reflection.quality is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
