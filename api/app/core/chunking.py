"""Semantic chunking strategies for document processing.

Provides multiple chunking algorithms from simple fixed-size to
intelligent semantic-boundary detection using embeddings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

import numpy as np


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""
    FIXED = "fixed"           # Paragraph-based (original behavior)
    SEMANTIC = "semantic"     # Sentence-transformer boundary detection
    RECURSIVE = "recursive"   # Recursive character splitting
    LATE = "late"             # Non-overlapping windows; token-level late pooling is not yet wired


@dataclass
class Chunk:
    """A document chunk with metadata."""
    text: str
    index: int
    start_char: int
    end_char: int
    metadata: dict[str, str] | None = None


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Trim boundary whitespace without changing coordinates in the source."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _segment_spans(text: str, separator: str) -> list[tuple[int, int]]:
    """Locate segments directly; searching reconstructed text loses repeats/CRLF."""
    spans = []
    start = 0
    for match in re.finditer(separator, text):
        left, right = _trim_span(text, start, match.start())
        if left < right:
            spans.append((left, right))
        start = match.end()
    left, right = _trim_span(text, start, len(text))
    if left < right:
        spans.append((left, right))
    return spans


def _chunks_from_spans(
    text: str, spans: list[tuple[int, int]], metadata: dict[str, str] | None = None
) -> list[Chunk]:
    chunks = []
    for start, end in spans:
        start, end = _trim_span(text, start, end)
        if start < end:
            # Trimming an overlap window that ends in whitespace can leave
            # only text already covered by its predecessor. Keep useful
            # overlap, but do not index a wholly contained duplicate.
            if chunks and chunks[-1].start_char <= start and end <= chunks[-1].end_char:
                continue
            chunks.append(Chunk(
                text=text[start:end], index=len(chunks), start_char=start,
                end_char=end, metadata=dict(metadata) if metadata else None,
            ))
    return chunks or [Chunk(text="", index=0, start_char=0, end_char=0)]


class FixedChunker:
    """Paragraph-based chunking with exact source spans (backward compatible)."""

    def __init__(self, target_size: int = 800, overlap: int = 0) -> None:
        if target_size <= 0:
            raise ValueError("target_size must be positive")
        self._target_size = target_size
        self._overlap = overlap

    def chunk(self, text: str) -> list[Chunk]:
        paragraphs = _segment_spans(text, r"\r?\n[ \t]*\r?\n")
        spans = []
        current: tuple[int, int] | None = None
        for start, end in paragraphs:
            if current is not None and end - current[0] > self._target_size:
                spans.append(current)
                current = None
            current = (current[0] if current else start, end)
        if current is not None:
            spans.append(current)
        return _chunks_from_spans(text, spans)


class SemanticChunker:
    """Split documents at semantic boundaries using embedding similarity.

    Uses sentence embeddings to detect topic shifts, creating chunks
    that are semantically coherent rather than arbitrarily split.
    """

    def __init__(
        self,
        embedder: SentenceTransformer | None = None,
        target_size: int = 400,
        min_size: int = 100,
        overlap_sentences: int = 1,
        similarity_threshold: float = 0.5,
    ) -> None:
        if target_size <= 0:
            raise ValueError("target_size must be positive")
        self._embedder = embedder
        self._target_size = target_size
        self._min_size = min_size
        self._overlap_sentences = overlap_sentences
        self._similarity_threshold = similarity_threshold

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitter - handles common cases
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _find_semantic_boundaries(self, sentences: list[str]) -> list[int]:
        """Find indices where semantic shifts occur."""
        if (
            not self._embedder
            or getattr(self._embedder, "supports_semantic_chunking", True) is False
            or len(sentences) < 3
        ):
            return []

        # Embed all sentences
        embeddings = self._embedder.encode(sentences, convert_to_numpy=True)

        # Compute cosine similarities between adjacent sentences
        boundaries: list[int] = []
        for i in range(1, len(embeddings)):
            sim = np.dot(embeddings[i-1], embeddings[i]) / (
                np.linalg.norm(embeddings[i-1]) * np.linalg.norm(embeddings[i])
            )
            # Low similarity indicates a topic shift
            if sim < self._similarity_threshold:
                boundaries.append(i)

        return boundaries

    def chunk(self, text: str) -> list[Chunk]:
        sentence_spans = _segment_spans(text, r"(?<=[.!?])\s+(?=[A-Z])")
        if not sentence_spans:
            return _chunks_from_spans(text, [])
        sentences = [text[start:end] for start, end in sentence_spans]
        boundaries = set(self._find_semantic_boundaries(sentences))
        spans = []
        current: list[tuple[int, int]] = []
        for i, sentence_span in enumerate(sentence_spans):
            current_size = current[-1][1] - current[0][0] if current else 0
            would_exceed = bool(current and sentence_span[1] - current[0][0] > self._target_size)
            if current and (i in boundaries or would_exceed) and current_size >= self._min_size:
                spans.append((current[0][0], current[-1][1]))
                current = current[-self._overlap_sentences:] if self._overlap_sentences > 0 else []
            current.append(sentence_span)
        if current:
            spans.append((current[0][0], current[-1][1]))
        return _chunks_from_spans(text, spans)


class RecursiveChunker:
    """Recursive character splitting with hierarchy of separators.

    Tries to split on paragraph breaks first, then sentences,
    then words, ensuring chunks stay within size limits.
    """

    def __init__(
        self,
        target_size: int = 500,
        overlap: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        if target_size <= 0:
            raise ValueError("target_size must be positive")
        if overlap < 0 or overlap >= target_size:
            raise ValueError("overlap must be nonnegative and smaller than target_size")
        self._target_size = target_size
        self._overlap = overlap
        self._separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def _split_spans(
        self, text: str, start: int, end: int, separators: list[str]
    ) -> list[tuple[int, int]]:
        start, end = _trim_span(text, start, end)
        if start >= end:
            return []
        if end - start <= self._target_size:
            return [(start, end)]
        if not separators or not separators[0]:
            step = self._target_size - self._overlap
            spans = []
            for i in range(start, end, step):
                right = min(i + self._target_size, end)
                spans.append((i, right))
                if right == end:
                    break
            return spans

        separator, remaining = separators[0], separators[1:]
        matches = list(re.finditer(re.escape(separator), text[start:end]))
        if not matches:
            return self._split_spans(text, start, end, remaining)
        parts = []
        cursor = start
        for match in matches:
            boundary = start + match.end()
            parts.append((cursor, boundary))
            cursor = boundary
        if cursor < end:
            parts.append((cursor, end))

        spans = []
        current: tuple[int, int] | None = None
        for left, right in parts:
            left, right = _trim_span(text, left, right)
            if left >= right:
                continue
            if current is not None and right - current[0] > self._target_size:
                spans.append(current)
                current = None
            if right - left > self._target_size:
                spans.extend(self._split_spans(text, left, right, remaining))
            else:
                current = (current[0] if current else left, right)
        if current is not None:
            spans.append(current)
        return spans

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Compatibility helper; the main path retains source coordinates."""
        return [text[start:end] for start, end in self._split_spans(text, 0, len(text), separators)]

    def chunk(self, text: str) -> list[Chunk]:
        spans = self._split_spans(text, 0, len(text), self._separators)
        return _chunks_from_spans(text, spans)


class LateChunker(RecursiveChunker):
    """Non-overlapping source windows for a future late-pooling embedder.

    This implements the splitting stage only. The current retrieval engine
    embeds windows independently; it does not perform contextual token pooling
    from the full document required by arXiv 2409.04701.
    """

    def __init__(
        self, target_size: int = 500, overlap: int = 0,
        separators: list[str] | None = None,
    ) -> None:
        super().__init__(target_size=target_size, overlap=0, separators=separators)

    def chunk(self, text: str) -> list[Chunk]:
        spans = self._split_spans(text, 0, len(text), self._separators)
        return _chunks_from_spans(text, spans, metadata={"late_chunking": "true"})


def get_chunker(
    strategy: ChunkingStrategy | str = ChunkingStrategy.FIXED,
    embedder: SentenceTransformer | None = None,
    target_size: int = 400,
    overlap: int = 50,
    **kwargs,
) -> FixedChunker | SemanticChunker | RecursiveChunker | LateChunker:
    """Factory function to get a chunker based on strategy.

    Args:
        strategy: Chunking strategy to use
        embedder: Optional sentence transformer for semantic chunking
        target_size: Target chunk size in characters
        overlap: Overlap between chunks (interpretation varies by strategy)
        **kwargs: Additional strategy-specific parameters
    """
    strategy = ChunkingStrategy(strategy) if isinstance(strategy, str) else strategy

    if strategy == ChunkingStrategy.SEMANTIC:
        # SemanticChunker uses overlap_sentences (number of sentences)
        overlap_sentences = max(1, overlap // 50)  # Rough heuristic
        return SemanticChunker(
            embedder=embedder,
            target_size=target_size,
            overlap_sentences=overlap_sentences,
            **{k: v for k, v in kwargs.items() if k in ('min_size', 'similarity_threshold')}
        )
    elif strategy == ChunkingStrategy.RECURSIVE:
        return RecursiveChunker(target_size=target_size, overlap=overlap)
    elif strategy == ChunkingStrategy.LATE:
        # Late windows are non-overlapping; the embedder currently encodes each independently
        return LateChunker(target_size=target_size, overlap=0)
    else:
        return FixedChunker(target_size=target_size, overlap=overlap)
