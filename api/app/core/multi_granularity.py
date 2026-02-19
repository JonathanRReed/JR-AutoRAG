"""Multi-Granularity Storage: Store chunks at multiple levels for better retrieval.

Implements multi-granularity indexing where documents are stored at:
1. Sentence level - for precise matching
2. Paragraph level - for context
3. Section level - for broad topics
4. Document level - for summaries

This allows retrieval at the optimal granularity for each query.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .chunking import Chunk


class GranularityLevel(str, Enum):
    """Storage granularity levels."""
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SECTION = "section"
    DOCUMENT = "document"


@dataclass
class GranularChunk:
    """A chunk with granularity metadata."""
    text: str
    level: GranularityLevel
    parent_id: str | None = None  # Reference to parent chunk
    children_ids: list[str] = field(default_factory=list)  # References to child chunks
    chunk_id: str = ""
    start_char: int = 0
    end_char: int = 0
    section_path: list[str] = field(default_factory=list)  # Hierarchical section path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiGranularityConfig:
    """Configuration for multi-granularity storage."""
    # Enable/disable specific levels
    enable_sentence: bool = True
    enable_paragraph: bool = True
    enable_section: bool = True
    enable_document: bool = True

    # Size thresholds
    min_sentence_chars: int = 20
    max_sentence_chars: int = 500
    min_paragraph_chars: int = 100
    max_paragraph_chars: int = 2000
    min_section_chars: int = 500
    max_section_chars: int = 10000

    # Document summary settings
    document_summary_max_chars: int = 1000


class MultiGranularityIndexer:
    """Index documents at multiple granularity levels.

    Creates a hierarchy of chunks:
    - Document summary
    - Section summaries (based on headers)
    - Paragraph chunks (natural paragraph breaks)
    - Sentence chunks (for precise matching)

    Each chunk links to its parent and children for traversal.
    """

    def __init__(self, config: MultiGranularityConfig | None = None) -> None:
        """Initialize indexer with configuration."""
        self.config = config or MultiGranularityConfig()
        self._chunk_counter = 0

    def _generate_chunk_id(self, level: GranularityLevel) -> str:
        """Generate a unique chunk ID."""
        self._chunk_counter += 1
        return f"{level.value}_{self._chunk_counter}"

    def _split_sentences(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into sentences with positions.

        Returns list of (sentence, start, end) tuples.
        """
        import re
        sentences = []
        for match in re.finditer(r'[^.!?]+[.!?]+', text):
            sentence = match.group().strip()
            if self.config.min_sentence_chars <= len(sentence) <= self.config.max_sentence_chars:
                sentences.append((sentence, match.start(), match.end()))
        return sentences

    def _split_paragraphs(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into paragraphs with positions.

        Returns list of (paragraph, start, end) tuples.
        """
        import re
        paragraphs = []
        for match in re.finditer(r'(?:^|\n\n)(.+?)(?=\n\n|$)', text, re.DOTALL):
            paragraph = match.group(1).strip()
            if self.config.min_paragraph_chars <= len(paragraph) <= self.config.max_paragraph_chars:
                paragraphs.append((paragraph, match.start(), match.end()))
        return paragraphs

    def _split_sections(self, text: str) -> list[tuple[str, int, int, str]]:
        """Split text into sections based on headers.

        Returns list of (section_text, start, end, header) tuples.
        """
        import re
        sections = []

        # Find all headers
        headers = list(re.finditer(r'^(#{1,6})\s+(.+?)$', text, re.MULTILINE))

        if not headers:
            # No headers - treat entire text as one section
            if len(text) >= self.config.min_section_chars:
                sections.append((text, 0, len(text), ""))
            return sections

        # Create sections between headers
        for i, header_match in enumerate(headers):
            header_text = header_match.group(2).strip()
            section_start = header_match.start()

            # Section ends at next header or end of text
            section_end = headers[i + 1].start() if i < len(headers) - 1 else len(text)

            section_text = text[section_start:section_end].strip()

            if self.config.min_section_chars <= len(section_text) <= self.config.max_section_chars:
                sections.append((section_text, section_start, section_end, header_text))

        return sections

    def _create_document_summary(self, text: str, title: str = "") -> GranularChunk:
        """Create a document-level summary chunk.

        Takes the beginning of the document as a summary.
        """
        summary_text = text[:self.config.document_summary_max_chars]
        if len(text) > self.config.document_summary_max_chars:
            # Try to end at a sentence boundary
            last_period = summary_text.rfind('.')
            if last_period > self.config.document_summary_max_chars // 2:
                summary_text = summary_text[:last_period + 1]

        # Prepend title if available
        if title:
            summary_text = f"Document: {title}\n\n{summary_text}"

        return GranularChunk(
            text=summary_text,
            level=GranularityLevel.DOCUMENT,
            chunk_id=self._generate_chunk_id(GranularityLevel.DOCUMENT),
            start_char=0,
            end_char=len(summary_text),
            metadata={"title": title, "is_summary": True},
        )

    def index_document(
        self,
        text: str,
        title: str = "",
        existing_chunks: list[Chunk] | None = None,
    ) -> list[GranularChunk]:
        """Create multi-granularity index for a document.

        Args:
            text: Full document text
            title: Optional document title
            existing_chunks: Optional pre-chunked paragraphs

        Returns:
            List of GranularChunks at all enabled levels
        """
        all_chunks: list[GranularChunk] = []

        # Document level
        doc_chunk = None
        if self.config.enable_document:
            doc_chunk = self._create_document_summary(text, title)
            all_chunks.append(doc_chunk)

        # Section level
        section_chunks: list[GranularChunk] = []
        if self.config.enable_section:
            sections = self._split_sections(text)
            for section_text, start, end, header in sections:
                chunk = GranularChunk(
                    text=section_text,
                    level=GranularityLevel.SECTION,
                    parent_id=doc_chunk.chunk_id if doc_chunk else None,
                    chunk_id=self._generate_chunk_id(GranularityLevel.SECTION),
                    start_char=start,
                    end_char=end,
                    section_path=[header] if header else [],
                    metadata={"header": header},
                )
                section_chunks.append(chunk)
                all_chunks.append(chunk)

            # Link document to sections
            if doc_chunk:
                doc_chunk.children_ids = [s.chunk_id for s in section_chunks]

        # Paragraph level
        paragraph_chunks: list[GranularChunk] = []
        if self.config.enable_paragraph:
            paragraphs = self._split_paragraphs(text)
            for para_text, start, end in paragraphs:
                # Find parent section
                parent_section = None
                for section in section_chunks:
                    if section.start_char <= start < section.end_char:
                        parent_section = section
                        break

                chunk = GranularChunk(
                    text=para_text,
                    level=GranularityLevel.PARAGRAPH,
                    parent_id=parent_section.chunk_id if parent_section else (doc_chunk.chunk_id if doc_chunk else None),
                    chunk_id=self._generate_chunk_id(GranularityLevel.PARAGRAPH),
                    start_char=start,
                    end_char=end,
                    section_path=parent_section.section_path if parent_section else [],
                )
                paragraph_chunks.append(chunk)
                all_chunks.append(chunk)

                # Link parent to child
                if parent_section:
                    parent_section.children_ids.append(chunk.chunk_id)

        # Sentence level
        if self.config.enable_sentence:
            sentences = self._split_sentences(text)
            for sent_text, start, end in sentences:
                # Find parent paragraph
                parent_para = None
                for para in paragraph_chunks:
                    if para.start_char <= start < para.end_char:
                        parent_para = para
                        break

                chunk = GranularChunk(
                    text=sent_text,
                    level=GranularityLevel.SENTENCE,
                    parent_id=parent_para.chunk_id if parent_para else None,
                    chunk_id=self._generate_chunk_id(GranularityLevel.SENTENCE),
                    start_char=start,
                    end_char=end,
                    section_path=parent_para.section_path if parent_para else [],
                )
                all_chunks.append(chunk)

                # Link parent to child
                if parent_para:
                    parent_para.children_ids.append(chunk.chunk_id)

        return all_chunks

    def get_chunks_at_level(
        self,
        chunks: list[GranularChunk],
        level: GranularityLevel,
    ) -> list[GranularChunk]:
        """Filter chunks to a specific granularity level."""
        return [c for c in chunks if c.level == level]

    def get_parent_chain(
        self,
        chunk: GranularChunk,
        all_chunks: list[GranularChunk],
    ) -> list[GranularChunk]:
        """Get the chain of parent chunks up to document level."""
        chain = []
        chunk_map = {c.chunk_id: c for c in all_chunks}

        current = chunk
        while current.parent_id and current.parent_id in chunk_map:
            parent = chunk_map[current.parent_id]
            chain.append(parent)
            current = parent

        return chain


# Singleton for easy access
_indexer: MultiGranularityIndexer | None = None


def get_multi_granularity_indexer(config: MultiGranularityConfig | None = None) -> MultiGranularityIndexer:
    """Get or create the multi-granularity indexer."""
    global _indexer
    if _indexer is None or config is not None:
        _indexer = MultiGranularityIndexer(config)
    return _indexer


__all__ = [
    "GranularityLevel",
    "GranularChunk",
    "MultiGranularityConfig",
    "MultiGranularityIndexer",
    "get_multi_granularity_indexer",
]
