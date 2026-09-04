import pytest
from app.core.chunking import (
    ChunkingStrategy,
    FixedChunker,
    RecursiveChunker,
    SemanticChunker,
    get_chunker,
)


def test_get_chunker_default():
    """Test that default strategy returns FixedChunker."""
    chunker = get_chunker()
    assert isinstance(chunker, FixedChunker)
    assert chunker._target_size == 400
    assert chunker._overlap == 50


def test_get_chunker_fixed():
    """Test get_chunker with fixed strategy."""
    chunker = get_chunker(strategy=ChunkingStrategy.FIXED)
    assert isinstance(chunker, FixedChunker)

    # Test with string literal
    chunker_str = get_chunker(strategy="fixed", target_size=500, overlap=100)
    assert isinstance(chunker_str, FixedChunker)
    assert chunker_str._target_size == 500
    assert chunker_str._overlap == 100


def test_get_chunker_recursive():
    """Test get_chunker with recursive strategy."""
    chunker = get_chunker(
        strategy=ChunkingStrategy.RECURSIVE, target_size=600, overlap=20
    )
    assert isinstance(chunker, RecursiveChunker)
    assert chunker._target_size == 600
    assert chunker._overlap == 20

    # Test with string literal
    chunker_str = get_chunker(strategy="recursive")
    assert isinstance(chunker_str, RecursiveChunker)


def test_get_chunker_semantic():
    """Test get_chunker with semantic strategy."""
    chunker = get_chunker(
        strategy=ChunkingStrategy.SEMANTIC, target_size=800, overlap=100
    )
    assert isinstance(chunker, SemanticChunker)
    assert chunker._target_size == 800
    # overlap_sentences = max(1, overlap // 50) -> 100 // 50 = 2
    assert chunker._overlap_sentences == 2

    # Test with string literal and custom semantic params
    chunker_str = get_chunker(
        strategy="semantic", min_size=50, similarity_threshold=0.8
    )
    assert isinstance(chunker_str, SemanticChunker)
    assert chunker_str._min_size == 50
    assert chunker_str._similarity_threshold == 0.8


def test_get_chunker_invalid_strategy():
    """Test get_chunker with an invalid strategy string."""
    with pytest.raises(ValueError):
        get_chunker(strategy="invalid_strategy")
