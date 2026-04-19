"""Contextual Chunk Enrichment: Add document context to chunks during ingestion.

Implements the Anthropic Contextual Retrieval approach:
- Prepends document-level context (title, section) to each chunk
- Generates short chunk summaries using LLM
- Adds semantic headers and metadata

This improves retrieval by ensuring chunks contain enough context
to be understood and matched independently.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .chunking import Chunk
    from .providers import LLMProvider


@dataclass
class EnrichmentContext:
    """Context added to each chunk."""
    document_title: str = ""
    document_summary: str = ""
    section_header: str = ""
    chunk_summary: str = ""
    preceding_context: str = ""
    following_context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EnrichedChunk:
    """A chunk with contextual enrichment."""
    original_text: str
    enriched_text: str
    context: EnrichmentContext
    index: int
    start_char: int
    end_char: int
    content_hash: str

    @property
    def text(self) -> str:
        """Alias for enriched_text for compatibility."""
        return self.enriched_text


@dataclass
class EnrichmentConfig:
    """Configuration for contextual enrichment."""
    # Enable/disable specific enrichment types
    add_document_title: bool = True
    add_section_header: bool = True
    add_chunk_summary: bool = True
    add_context_window: bool = True

    # Context window settings
    preceding_chars: int = 100
    following_chars: int = 100

    # LLM summary settings
    summary_max_tokens: int = 50
    use_llm_for_summary: bool = True

    # Fallback settings
    fallback_to_heuristic: bool = True


class ContextualEnricher:
    """Enrich chunks with document context for better retrieval.

    Implements the contextual retrieval pattern where each chunk
    is prepended with relevant document context:

    ```
    Document: <title>
    Section: <section_header>
    Context: <chunk_summary>
    ---
    <original_chunk_text>
    ```

    This allows chunks to be understood in isolation.
    """

    SUMMARY_PROMPT = """Summarize the following text chunk in 1-2 sentences.
Focus on the key information that would help identify what this chunk is about.

Text:
{chunk_text}

Summary:"""

    def __init__(self, config: EnrichmentConfig | None = None) -> None:
        """Initialize enricher with configuration."""
        self.config = config or EnrichmentConfig()
        self._cache: dict[str, str] = {}  # Content hash -> summary

    def extract_document_title(self, document_text: str, filename: str = "") -> str:
        """Extract document title from text or filename.

        Tries to find a title from:
        1. First heading in markdown/text
        2. Filename without extension
        """
        # Try to find markdown heading
        heading_match = re.search(r'^#\s+(.+?)$', document_text, re.MULTILINE)
        if heading_match:
            return heading_match.group(1).strip()

        # Try first line if it looks like a title
        lines = document_text.split('\n')
        if lines:
            first_line = lines[0].strip()
            if len(first_line) < 100 and first_line:
                return first_line

        # Fall back to filename
        if filename:
            return re.sub(r'\.[^.]+$', '', filename).replace('_', ' ').replace('-', ' ')

        return "Untitled Document"

    def extract_section_header(self, chunk_text: str, document_text: str, chunk_start: int) -> str:
        """Find the section header for a chunk.

        Looks backwards from chunk position to find the nearest heading.
        """
        # Get text before chunk
        preceding_text = document_text[:chunk_start]

        # Find all headings (markdown or all-caps lines)
        headings = list(re.finditer(r'^#+\s+(.+?)$|^([A-Z][A-Z\s]{5,})$',
                                     preceding_text, re.MULTILINE))

        if headings:
            last_heading = headings[-1]
            return (last_heading.group(1) or last_heading.group(2)).strip()

        return ""

    async def generate_chunk_summary(
        self,
        chunk_text: str,
        provider: LLMProvider | None = None,
    ) -> str:
        """Generate a brief summary of the chunk content.

        Uses LLM if available, otherwise falls back to heuristics.
        """
        # Check cache first
        content_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
        if content_hash in self._cache:
            return self._cache[content_hash]

        summary = ""

        if provider is not None and self.config.use_llm_for_summary:
            try:
                prompt = self.SUMMARY_PROMPT.format(chunk_text=chunk_text[:1000])
                response = await provider.chat([
                    {"role": "user", "content": prompt},
                ])
                summary = response.strip()[:200]
            except Exception:
                if self.config.fallback_to_heuristic:
                    summary = self._heuristic_summary(chunk_text)
        else:
            summary = self._heuristic_summary(chunk_text)

        self._cache[content_hash] = summary
        return summary

    def _heuristic_summary(self, chunk_text: str) -> str:
        """Generate a simple heuristic summary.

        Takes the first complete sentence and key terms.
        """
        # Get first sentence
        sentences = re.split(r'[.!?]+', chunk_text.strip())
        first_sentence = sentences[0].strip() if sentences else ""

        if len(first_sentence) > 100:
            first_sentence = first_sentence[:97] + "..."

        # Extract key terms (capitalized words, numbers with context)
        key_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', chunk_text)
        unique_terms = list(dict.fromkeys(key_terms))[:3]

        if unique_terms and first_sentence:
            return f"{first_sentence}. Key topics: {', '.join(unique_terms)}"
        elif first_sentence:
            return first_sentence

        return chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text

    def get_context_window(
        self,
        chunk_index: int,
        all_chunks: list[Chunk],
        document_text: str,
    ) -> tuple[str, str]:
        """Get preceding and following context for a chunk."""
        preceding = ""
        following = ""

        if self.config.add_context_window:
            # Get preceding context from previous chunk
            if chunk_index > 0:
                prev_chunk = all_chunks[chunk_index - 1]
                preceding = prev_chunk.text[-self.config.preceding_chars:]

            # Get following context from next chunk
            if chunk_index < len(all_chunks) - 1:
                next_chunk = all_chunks[chunk_index + 1]
                following = next_chunk.text[:self.config.following_chars]

        return preceding, following

    def format_enriched_text(
        self,
        original_text: str,
        context: EnrichmentContext,
    ) -> str:
        """Format the enriched chunk text with context prepended."""
        parts = []

        if context.document_title and self.config.add_document_title:
            parts.append(f"Document: {context.document_title}")

        if context.section_header and self.config.add_section_header:
            parts.append(f"Section: {context.section_header}")

        if context.chunk_summary and self.config.add_chunk_summary:
            parts.append(f"Context: {context.chunk_summary}")

        if parts:
            header = "\n".join(parts)
            return f"{header}\n---\n{original_text}"

        return original_text

    async def enrich_chunk(
        self,
        chunk: Chunk,
        chunk_index: int,
        all_chunks: list[Chunk],
        document_text: str,
        document_title: str = "",
        document_summary: str = "",
        provider: LLMProvider | None = None,
    ) -> EnrichedChunk:
        """Enrich a single chunk with contextual information.

        Args:
            chunk: The chunk to enrich
            chunk_index: Index in the chunk list
            all_chunks: All chunks from the document
            document_text: Full document text
            document_title: Pre-extracted document title
            document_summary: Pre-extracted document summary
            provider: Optional LLM provider for summaries

        Returns:
            EnrichedChunk with context added
        """
        # Extract section header
        section_header = self.extract_section_header(
            chunk.text, document_text, chunk.start_char
        )

        # Generate chunk summary
        chunk_summary = ""
        if self.config.add_chunk_summary:
            chunk_summary = await self.generate_chunk_summary(chunk.text, provider)

        # Get context window
        preceding, following = self.get_context_window(
            chunk_index, all_chunks, document_text
        )

        # Build context object
        context = EnrichmentContext(
            document_title=document_title,
            document_summary=document_summary,
            section_header=section_header,
            chunk_summary=chunk_summary,
            preceding_context=preceding,
            following_context=following,
            metadata=chunk.metadata.copy() if chunk.metadata else {},
        )

        # Add enrichment metadata
        context.metadata["enriched"] = True
        context.metadata["section"] = section_header

        # Format enriched text
        enriched_text = self.format_enriched_text(chunk.text, context)

        # Compute content hash
        content_hash = hashlib.sha256(chunk.text.encode()).hexdigest()

        return EnrichedChunk(
            original_text=chunk.text,
            enriched_text=enriched_text,
            context=context,
            index=chunk_index,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            content_hash=content_hash,
        )

    async def enrich_chunks(
        self,
        chunks: list[Chunk],
        document_text: str,
        filename: str = "",
        document_summary: str = "",
        provider: LLMProvider | None = None,
    ) -> list[EnrichedChunk]:
        """Enrich all chunks with contextual information.

        Args:
            chunks: List of chunks to enrich
            document_text: Full document text
            filename: Document filename for title extraction
            document_summary: Optional pre-generated document summary
            provider: Optional LLM provider

        Returns:
            List of EnrichedChunks
        """
        # Extract document title
        document_title = self.extract_document_title(document_text, filename)

        enriched = []
        for i, chunk in enumerate(chunks):
            enriched_chunk = await self.enrich_chunk(
                chunk=chunk,
                chunk_index=i,
                all_chunks=chunks,
                document_text=document_text,
                document_title=document_title,
                document_summary=document_summary,
                provider=provider,
            )
            enriched.append(enriched_chunk)

        return enriched

    def clear_cache(self) -> None:
        """Clear the summary cache."""
        self._cache.clear()


# Singleton for easy access
_enricher: ContextualEnricher | None = None


def get_contextual_enricher(config: EnrichmentConfig | None = None) -> ContextualEnricher:
    """Get or create the contextual enricher instance."""
    global _enricher
    if _enricher is None or config is not None:
        _enricher = ContextualEnricher(config)
    return _enricher


__all__ = [
    "EnrichmentContext",
    "EnrichedChunk",
    "EnrichmentConfig",
    "ContextualEnricher",
    "get_contextual_enricher",
]
