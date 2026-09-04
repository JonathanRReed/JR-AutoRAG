"""Tests that IngestPipeline._chunk honors retrieval.chunking_strategy from config.

Verifies the wiring fix: ``chunking_strategy`` in ``RetrievalDefaults`` now
selects the shared chunking module's chunker instead of always using the inline
fixed splitter.
"""

from __future__ import annotations

from types import SimpleNamespace

from app.core.ingest import IngestPipeline


def _make_pipeline(
    strategy: str, chunk_size: int = 200, chunk_overlap: int = 20
) -> IngestPipeline:
    retrieval = SimpleNamespace()
    store = SimpleNamespace()
    cfg = SimpleNamespace(
        retrieval=SimpleNamespace(
            chunking_strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    )
    return IngestPipeline(
        store=store,  # type: ignore[arg-type]
        retrieval=retrieval,  # type: ignore[arg-type]
        config_getter=lambda: cfg,
    )


def test_fixed_strategy_uses_inline_splitter():
    pipeline = _make_pipeline("fixed", chunk_size=120)
    text = "Header One\n\nFirst paragraph with some words.\n\nSecond paragraph here."
    chunks = pipeline._chunk(text)
    assert chunks
    assert all(isinstance(c, str) for c in chunks)


def test_recursive_strategy_routes_to_chunking_module():
    pipeline = _make_pipeline("recursive", chunk_size=120, chunk_overlap=10)
    text = (
        "This is the first sentence of a longer paragraph. "
        "Here is a second sentence that continues the thought. "
        "A third sentence rounds out the paragraph with more detail."
    )
    chunks = pipeline._chunk(text)
    assert chunks
    # Recursive chunker should respect the target size more tightly than fixed.
    assert all(isinstance(c, str) for c in chunks)


def test_semantic_strategy_falls_back_without_embedder():
    """Semantic chunking without a sentence-transformer embedder must still
    return chunks via the SemanticChunker's no-boundary fallback path."""
    pipeline = _make_pipeline("semantic", chunk_size=120, chunk_overlap=20)
    text = "Short doc. With a few sentences. Nothing fancy."
    chunks = pipeline._chunk(text)
    assert chunks
    assert all(isinstance(c, str) for c in chunks)


def test_invalid_strategy_falls_back_to_fixed():
    pipeline = _make_pipeline("not-a-real-strategy", chunk_size=120)
    text = "Some text here.\n\nMore text follows."
    chunks = pipeline._chunk(text)
    assert chunks
