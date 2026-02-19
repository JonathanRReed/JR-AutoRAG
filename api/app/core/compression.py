"""Context compression for efficient LLM usage.

This module provides intelligent context trimming to:
- Fit retrieved content within LLM token limits
- Preserve the most relevant information
- Maintain coherence and readability
- Support citation tracking
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gatherer import EvidenceChunk


@dataclass
class CompressedContext:
    """Result of context compression."""
    text: str
    chunks_used: int
    chunks_total: int
    estimated_tokens: int
    citations: list[dict]  # [{id, title, snippet_preview}]


@dataclass
class CitedPassage:
    """A passage with its citation information."""
    text: str
    chunk_id: str
    chunk_title: str
    relevance_score: float


class ContextCompressor:
    """Compress retrieved context to fit LLM token limits.

    Strategies:
    1. Sentence-level extraction: Keep most relevant sentences
    2. Truncation: Simply cut at token limit
    3. Summarization: Use LLM to summarize (when available)
    """

    # Rough estimate: 1 token ≈ 4 characters for English
    CHARS_PER_TOKEN = 4

    def __init__(
        self,
        max_tokens: int = 4096,
        preserve_first_n: int = 2,  # Always keep first N chunks
        min_chunk_tokens: int = 50,  # Minimum tokens per chunk
    ) -> None:
        self.max_tokens = max_tokens
        self.preserve_first_n = preserve_first_n
        self.min_chunk_tokens = min_chunk_tokens

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text."""
        return len(text) // self.CHARS_PER_TOKEN

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        pattern = r'(?<=[.!?])\s+'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _score_sentence(self, sentence: str, query_terms: set[str]) -> float:
        """Score a sentence by relevance to query terms."""
        words = set(re.findall(r'\b[a-z]+\b', sentence.lower()))
        if not words:
            return 0.0

        overlap = len(words & query_terms)
        return overlap / len(query_terms) if query_terms else 0.0

    def compress_simple(
        self,
        chunks: list[EvidenceChunk],
        max_tokens: int | None = None,
    ) -> CompressedContext:
        """Simple truncation-based compression."""
        max_tokens = max_tokens or self.max_tokens
        max_chars = max_tokens * self.CHARS_PER_TOKEN

        texts = []
        citations = []
        current_chars = 0

        for i, chunk in enumerate(chunks):
            chunk_chars = len(chunk.snippet)

            if current_chars + chunk_chars > max_chars:
                # Truncate this chunk
                remaining = max_chars - current_chars
                if remaining > 100:  # Only add if meaningful
                    truncated = chunk.snippet[:remaining] + "..."
                    texts.append(f"[{i+1}] {truncated}")
                    citations.append({
                        "id": chunk.id,
                        "title": chunk.title,
                        "snippet_preview": chunk.snippet[:100],
                        "citation_number": i + 1,
                    })
                break

            texts.append(f"[{i+1}] {chunk.snippet}")
            citations.append({
                "id": chunk.id,
                "title": chunk.title,
                "snippet_preview": chunk.snippet[:100],
                "citation_number": i + 1,
            })
            current_chars += chunk_chars

        return CompressedContext(
            text="\n\n".join(texts),
            chunks_used=len(texts),
            chunks_total=len(chunks),
            estimated_tokens=self._estimate_tokens("\n\n".join(texts)),
            citations=citations,
        )

    def compress_extractive(
        self,
        chunks: list[EvidenceChunk],
        query: str,
        max_tokens: int | None = None,
    ) -> CompressedContext:
        """Extractive compression: keep most relevant sentences."""
        max_tokens = max_tokens or self.max_tokens

        # Extract query terms for scoring
        query_terms = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))

        # Score and collect all sentences with their sources
        scored_passages: list[CitedPassage] = []

        for chunk in chunks:
            sentences = self._split_sentences(chunk.snippet)
            for sentence in sentences:
                score = self._score_sentence(sentence, query_terms)
                # Boost sentences from higher-scored chunks
                adjusted_score = score + (chunk.score * 0.3)

                scored_passages.append(CitedPassage(
                    text=sentence,
                    chunk_id=chunk.id,
                    chunk_title=chunk.title,
                    relevance_score=adjusted_score,
                ))

        # Sort by relevance
        scored_passages.sort(key=lambda p: p.relevance_score, reverse=True)

        # Build compressed context respecting token limit
        max_chars = max_tokens * self.CHARS_PER_TOKEN
        selected: list[CitedPassage] = []
        current_chars = 0
        used_chunk_ids = set()

        for passage in scored_passages:
            passage_chars = len(passage.text)
            if current_chars + passage_chars > max_chars:
                continue

            selected.append(passage)
            used_chunk_ids.add(passage.chunk_id)
            current_chars += passage_chars

        # Group by source for coherent output
        by_source: dict[str, list[str]] = {}
        for passage in selected:
            if passage.chunk_id not in by_source:
                by_source[passage.chunk_id] = []
            by_source[passage.chunk_id].append(passage.text)

        # Build output with citations
        texts = []
        citations = []

        for i, (chunk_id, sentences) in enumerate(by_source.items(), 1):
            # Find original chunk info
            chunk_title = next(
                (p.chunk_title for p in selected if p.chunk_id == chunk_id),
                "Unknown"
            )

            combined = " ".join(sentences)
            texts.append(f"[{i}] {combined}")
            citations.append({
                "id": chunk_id,
                "title": chunk_title,
                "snippet_preview": combined[:100],
                "citation_number": i,
            })

        return CompressedContext(
            text="\n\n".join(texts),
            chunks_used=len(by_source),
            chunks_total=len(chunks),
            estimated_tokens=self._estimate_tokens("\n\n".join(texts)),
            citations=citations,
        )

    def compress(
        self,
        chunks: list[EvidenceChunk],
        query: str = "",
        max_tokens: int | None = None,
        strategy: str = "extractive",
    ) -> CompressedContext:
        """Compress context using specified strategy.

        Args:
            chunks: Retrieved evidence chunks
            query: Original query (for relevance scoring)
            max_tokens: Maximum tokens in output
            strategy: "simple" (truncation) or "extractive" (sentence-level)
        """
        if not chunks:
            return CompressedContext(
                text="",
                chunks_used=0,
                chunks_total=0,
                estimated_tokens=0,
                citations=[],
            )

        if strategy == "simple" or not query:
            return self.compress_simple(chunks, max_tokens)
        else:
            return self.compress_extractive(chunks, query, max_tokens)

    def format_with_citations(
        self,
        chunks: list[EvidenceChunk],
        include_numbers: bool = True,
        rich_format: bool = True,
    ) -> tuple[str, list[dict]]:
        """Format chunks with inline citation numbers and rich metadata.

        Args:
            chunks: List of evidence chunks
            include_numbers: Include [1], [2] markers
            rich_format: Include document title and chunk ID headers

        Returns:
            Tuple of (formatted_text, citations_list)
        """
        texts = []
        citations = []

        for i, chunk in enumerate(chunks, 1):
            if rich_format:
                # Rich format with document title and chunk ID for precise locators
                header = f"[{i}] Source: {chunk.title} | ChunkID: {chunk.id}"
                body = f'"{chunk.snippet[:200]}..."' if len(chunk.snippet) > 200 else f'"{chunk.snippet}"'
                texts.append(f"{header}\n{body}")
            elif include_numbers:
                texts.append(f"[{i}] {chunk.snippet}")
            else:
                texts.append(chunk.snippet)

            # Extract a key quote (first sentence or first 100 chars)
            key_quote = chunk.snippet.split('.')[0][:100] if '.' in chunk.snippet else chunk.snippet[:100]

            citations.append({
                "id": chunk.id,
                "title": chunk.title,
                "snippet_preview": chunk.snippet[:150] + "..." if len(chunk.snippet) > 150 else chunk.snippet,
                "key_quote": key_quote,
                "citation_number": i,
                "score": chunk.score,
                "document_title": chunk.title,  # Alias for clarity
                "chunk_id": chunk.id,  # Alias for clarity
            })

        return "\n\n".join(texts), citations


__all__ = [
    "CompressedContext",
    "CitedPassage",
    "ContextCompressor",
]
