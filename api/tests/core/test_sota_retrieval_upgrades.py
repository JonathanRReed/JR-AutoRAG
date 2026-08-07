"""Tests for SOTA retrieval upgrades: AutoHybridWeights, LateChunker, MMR."""

from app.core.chunking import ChunkingStrategy, LateChunker, get_chunker
from app.core.hybrid_retrieval import AutoHybridWeights, HybridConfig


class TestAutoHybridWeights:
    """Per-query hybrid weight computation."""

    def test_short_keyword_query_gets_more_sparse(self):
        dense, sparse = AutoHybridWeights.compute_weights("python asyncio", 0.6, 0.4)
        # Short query with 2 tokens -> sparse boost
        assert sparse > 0.4, f"Expected sparse > 0.4, got {sparse}"

    def test_question_query_gets_more_dense(self):
        dense, sparse = AutoHybridWeights.compute_weights(
            "What is the difference between async and sync programming?", 0.6, 0.4
        )
        # Has question word "what" and "difference" -> dense boost
        assert dense > 0.6, f"Expected dense > 0.6, got {dense}"

    def test_quoted_query_gets_more_sparse(self):
        dense, sparse = AutoHybridWeights.compute_weights(
            'find documents mentioning "exact phrase match"', 0.6, 0.4
        )
        # Has quotes -> sparse boost
        assert sparse > 0.4, f"Expected sparse > 0.4, got {sparse}"

    def test_weights_sum_to_one(self):
        for query in ["hello", "what is the meaning of life?", "python 3.12", "a b c d e f g h i j k"]:
            dense, sparse = AutoHybridWeights.compute_weights(query, 0.6, 0.4)
            assert abs(dense + sparse - 1.0) < 1e-6, f"Weights don't sum to 1: {dense} + {sparse}"

    def test_empty_query_returns_base(self):
        dense, sparse = AutoHybridWeights.compute_weights("", 0.6, 0.4)
        assert dense == 0.6
        assert sparse == 0.4

    def test_numbers_boost_sparse(self):
        dense, sparse = AutoHybridWeights.compute_weights("error code 500", 0.6, 0.4)
        # Has number -> sparse boost
        assert sparse > 0.4, f"Expected sparse > 0.4 for numeric query, got {sparse}"


class TestLateChunker:
    """Late chunking strategy."""

    def test_late_chunker_produces_chunks(self):
        chunker = LateChunker(target_size=100)
        text = "This is a test. " * 50
        chunks = chunker.chunk(text)
        assert len(chunks) > 1
        for c in chunks:
            assert c.text.strip()
            assert c.start_char < c.end_char

    def test_late_chunker_marks_metadata(self):
        chunker = LateChunker(target_size=100)
        text = "This is a test sentence. " * 20
        chunks = chunker.chunk(text)
        for c in chunks:
            assert c.metadata is not None
            assert c.metadata.get("late_chunking") == "true"

    def test_late_chunker_no_overlap(self):
        chunker = LateChunker(target_size=50)
        text = "Word " * 100
        chunks = chunker.chunk(text)
        # Late chunking should not have overlapping content
        for i in range(1, len(chunks)):
            # End of previous should be <= start of current (no overlap)
            assert chunks[i - 1].end_char <= chunks[i].start_char + 2  # small tolerance for stripping

    def test_get_chunker_returns_late_chunker(self):
        chunker = get_chunker(strategy=ChunkingStrategy.LATE, target_size=200)
        assert isinstance(chunker, LateChunker)

    def test_get_chunker_by_string(self):
        chunker = get_chunker(strategy="late", target_size=200)
        assert isinstance(chunker, LateChunker)

    def test_empty_text_returns_single_chunk(self):
        chunker = LateChunker(target_size=100)
        chunks = chunker.chunk("")
        assert len(chunks) == 1
        assert chunks[0].text == ""


class TestHybridConfigMatryoshka:
    """Matryoshka dimension support in HybridConfig."""

    def test_default_matryoshka_disabled(self):
        config = HybridConfig()
        assert config.matryoshka_dim == 0

    def test_matryoshka_dim_settable(self):
        config = HybridConfig(matryoshka_dim=256)
        assert config.matryoshka_dim == 256

    def test_diversity_field_exists(self):
        config = HybridConfig(diversity=0.5)
        assert config.diversity == 0.5
