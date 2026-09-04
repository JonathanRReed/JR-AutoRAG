"""Semantic chunking strategies for document processing.

Provides multiple chunking algorithms from simple fixed-size to
intelligent semantic-boundary detection using embeddings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

import numpy as np


class ChunkingStrategy(str, Enum):
    """Available chunking strategies."""

    FIXED = "fixed"  # Paragraph-based (original behavior)
    SEMANTIC = "semantic"  # Sentence-transformer boundary detection
    RECURSIVE = "recursive"  # Recursive character splitting
    LATE = "late"  # Late chunking: embed full doc, pool per chunk window


@dataclass
class Chunk:
    """A document chunk with metadata."""

    text: str
    index: int
    start_char: int
    end_char: int
    metadata: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "index": self.index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(
            text=data["text"],
            index=data["index"],
            start_char=data["start_char"],
            end_char=data["end_char"],
            metadata=data.get("metadata"),
        )


class FixedChunker:
    """Original paragraph-based chunking (backward compatible)."""

    def __init__(self, target_size: int = 800, overlap: int = 0) -> None:
        self._target_size = target_size
        self._overlap = overlap

    def chunk(self, text: str) -> list[Chunk]:
        clean = text.replace("\r", "")
        paragraphs = [p.strip() for p in clean.split("\n\n") if p.strip()]

        chunks: list[Chunk] = []
        current_texts: list[str] = []
        current_len = 0
        char_pos = 0

        for para in paragraphs:
            if current_len + len(para) > self._target_size and current_texts:
                chunk_text = "\n".join(current_texts)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        index=len(chunks),
                        start_char=char_pos - len(chunk_text),
                        end_char=char_pos,
                    )
                )
                current_texts = []
                current_len = 0

            current_texts.append(para)
            current_len += len(para)
            char_pos += len(para) + 2  # +2 for \n\n

        if current_texts:
            chunk_text = "\n".join(current_texts)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=len(chunks),
                    start_char=char_pos - len(chunk_text),
                    end_char=char_pos,
                )
            )

        return (
            chunks
            if chunks
            else [Chunk(text=text.strip(), index=0, start_char=0, end_char=len(text))]
        )


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
        self._embedder = embedder
        self._target_size = target_size
        self._min_size = min_size
        self._overlap_sentences = overlap_sentences
        self._similarity_threshold = similarity_threshold

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        # Simple sentence splitter - handles common cases
        pattern = r"(?<=[.!?])\s+(?=[A-Z])"
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
            sim = np.dot(embeddings[i - 1], embeddings[i]) / (
                np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i])
            )
            # Low similarity indicates a topic shift
            if sim < self._similarity_threshold:
                boundaries.append(i)

        return boundaries

    def chunk(self, text: str) -> list[Chunk]:
        sentences = self._split_sentences(text)
        if not sentences:
            return [Chunk(text=text.strip(), index=0, start_char=0, end_char=len(text))]

        # Find semantic boundaries
        boundaries = self._find_semantic_boundaries(sentences)

        # Create chunks respecting boundaries and size constraints
        chunks: list[Chunk] = []
        current_sentences: list[str] = []
        current_len = 0
        char_pos = 0

        for i, sentence in enumerate(sentences):
            is_boundary = i in boundaries
            would_exceed = current_len + len(sentence) > self._target_size

            # Create chunk if at boundary or size exceeded (and minimum met)
            if (is_boundary or would_exceed) and current_len >= self._min_size:
                chunk_text = " ".join(current_sentences)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        index=len(chunks),
                        start_char=char_pos - len(chunk_text),
                        end_char=char_pos,
                    )
                )
                # Keep overlap sentences
                if self._overlap_sentences > 0:
                    current_sentences = current_sentences[-self._overlap_sentences :]
                    current_len = sum(len(s) for s in current_sentences)
                else:
                    current_sentences = []
                    current_len = 0

            current_sentences.append(sentence)
            current_len += len(sentence)
            char_pos += len(sentence) + 1

        # Final chunk
        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    index=len(chunks),
                    start_char=char_pos - len(chunk_text),
                    end_char=char_pos,
                )
            )

        return (
            chunks
            if chunks
            else [Chunk(text=text.strip(), index=0, start_char=0, end_char=len(text))]
        )


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
        self._target_size = target_size
        self._overlap = overlap
        self._separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        """Recursively split text using hierarchy of separators."""
        if not text.strip():
            return []

        if len(text) <= self._target_size:
            return [text]

        if not separators:
            # No more separators - force split at target size
            return [
                text[i : i + self._target_size]
                for i in range(0, len(text), self._target_size - self._overlap)
            ]

        sep = separators[0]
        remaining_seps = separators[1:]

        if sep not in text:
            return self._split_recursive(text, remaining_seps)

        splits = text.split(sep)
        result: list[str] = []
        current = ""

        for split in splits:
            if len(current) + len(split) + len(sep) <= self._target_size:
                current += (sep if current else "") + split
            else:
                if current:
                    result.append(current)
                if len(split) > self._target_size:
                    result.extend(self._split_recursive(split, remaining_seps))
                else:
                    current = split

        if current:
            result.append(current)

        return result

    def chunk(self, text: str) -> list[Chunk]:
        raw_chunks = self._split_recursive(text.replace("\r", ""), self._separators)

        chunks: list[Chunk] = []
        char_pos = 0

        for _i, raw in enumerate(raw_chunks):
            clean = raw.strip()
            if clean:
                chunks.append(
                    Chunk(
                        text=clean,
                        index=len(chunks),
                        start_char=char_pos,
                        end_char=char_pos + len(clean),
                    )
                )
            char_pos += len(raw)

        return (
            chunks
            if chunks
            else [Chunk(text=text.strip(), index=0, start_char=0, end_char=len(text))]
        )


class LateChunker:
    """Late chunking: split into windows for post-embedding pooling.

    Implements the splitting phase of late chunking (arXiv 2409.04701).
    The full document is split into token-window chunks. At embedding time,
    the retrieval engine should embed the full document with a long-context
    model (e.g. BGE-M3, Jina v3) and pool token-level representations over
    these chunk windows, preserving cross-chunk context.

    This chunker produces clean, non-overlapping windows that align with
    token boundaries. If no long-context model is available at embedding
    time, the retrieval engine falls back to per-chunk embedding.
    """

    def __init__(
        self,
        target_size: int = 500,
        overlap: int = 0,  # Late chunking typically uses no overlap
        separators: list[str] | None = None,
    ) -> None:
        self._target_size = target_size
        self._overlap = 0  # No overlap for late chunking; pooling handles context
        self._separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, text: str) -> list[Chunk]:
        clean = text.replace("\r", "")
        # Use recursive splitting to get boundary-respecting windows
        raw_chunks: list[str] = []
        self._split_recursive(clean, self._separators, raw_chunks)

        chunks: list[Chunk] = []
        char_pos = 0
        for raw in raw_chunks:
            stripped = raw.strip()
            if stripped:
                chunks.append(
                    Chunk(
                        text=stripped,
                        index=len(chunks),
                        start_char=char_pos,
                        end_char=char_pos + len(stripped),
                        metadata={"late_chunking": "true"},
                    )
                )
            char_pos += len(raw)

        return (
            chunks
            if chunks
            else [Chunk(text=text.strip(), index=0, start_char=0, end_char=len(text))]
        )

    def _split_recursive(
        self, text: str, separators: list[str], out: list[str]
    ) -> None:
        """Recursively split text, appending results to out."""
        if not text.strip():
            return
        if len(text) <= self._target_size:
            out.append(text)
            return
        if not separators:
            # Force split at target size
            for i in range(0, len(text), self._target_size):
                out.append(text[i : i + self._target_size])
            return
        sep = separators[0]
        remaining = separators[1:]
        if sep not in text:
            self._split_recursive(text, remaining, out)
            return
        splits = text.split(sep)
        current = ""
        for split in splits:
            if len(current) + len(split) + len(sep) <= self._target_size:
                current += (sep if current else "") + split
            else:
                if current:
                    out.append(current)
                if len(split) > self._target_size:
                    self._split_recursive(split, remaining, out)
                current = split
        if current:
            out.append(current)


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
            **{
                k: v
                for k, v in kwargs.items()
                if k in ("min_size", "similarity_threshold")
            },
        )
    elif strategy == ChunkingStrategy.RECURSIVE:
        return RecursiveChunker(target_size=target_size, overlap=overlap)
    elif strategy == ChunkingStrategy.LATE:
        # Late chunking uses no overlap; pooling handles cross-chunk context
        return LateChunker(target_size=target_size, overlap=0)
    else:
        return FixedChunker(target_size=target_size, overlap=overlap)
