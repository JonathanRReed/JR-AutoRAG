"""Metadata enrichment for automatic document annotation.

This module provides:
- Automatic title extraction/generation
- Summary generation
- Tag/keyword extraction
- Entity extraction
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentMetadata:
    """Enriched metadata for a document."""
    title: str
    summary: str
    keywords: list[str]
    entities: list[str]
    word_count: int
    estimated_reading_time_minutes: float
    language: str
    content_type: str  # "technical", "narrative", "reference", etc.
    headers: list[str]
    extra: dict[str, Any] = field(default_factory=dict)


class MetadataEnricher:
    """Enriches documents with automatically extracted metadata.

    Uses heuristics for fast, offline processing without requiring LLM.
    """

    # Words per minute for reading time estimation
    WORDS_PER_MINUTE = 200

    # Common words to exclude from keywords
    STOP_WORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'can', 'shall',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
        'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom',
        'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'why',
        'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'up',
        'about', 'into', 'over', 'after', 'before', 'between', 'under',
    }

    # Technical indicators
    TECHNICAL_PATTERNS = [
        r'\b(function|class|method|api|sdk|http|json|xml)\b',
        r'\b(install|configure|deploy|build|compile)\b',
        r'```\w*\n',  # Code blocks
        r'\bdef\s+\w+\s*\(',  # Python functions
        r'\b(import|require|include)\b',
    ]

    # Reference document indicators
    REFERENCE_PATTERNS = [
        r'\btable of contents\b',
        r'\bglossary\b',
        r'\bappendix\b',
        r'\bindex\b',
        r'\breference\b',
    ]

    def __init__(self) -> None:
        self._technical_re = [
            re.compile(p, re.IGNORECASE) for p in self.TECHNICAL_PATTERNS
        ]
        self._reference_re = [
            re.compile(p, re.IGNORECASE) for p in self.REFERENCE_PATTERNS
        ]

    def extract_title(self, text: str, fallback: str = "Untitled") -> str:
        """Extract or generate a title from text."""
        # Try first markdown header
        h1_match = re.search(r'^#\s+(.+)$', text, re.MULTILINE)
        if h1_match:
            return h1_match.group(1).strip()[:100]

        # Try first line if it's short
        first_line = text.strip().split('\n')[0].strip()
        if first_line and len(first_line) < 100 and not first_line.startswith('```'):
            return first_line

        # Generate from first sentence
        sentences = re.split(r'[.!?]', text[:500])
        if sentences and sentences[0].strip():
            title = sentences[0].strip()[:80]
            if len(title) > 60:
                title = title[:60] + "..."
            return title

        return fallback

    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate a summary from the first paragraph(s)."""
        # Clean text
        clean = re.sub(r'\s+', ' ', text[:1000]).strip()

        # Get first few sentences
        sentences = re.split(r'(?<=[.!?])\s+', clean)
        summary_parts = []
        current_length = 0

        for sentence in sentences[:5]:  # Max 5 sentences
            if current_length + len(sentence) > max_length:
                break
            summary_parts.append(sentence)
            current_length += len(sentence)

        return ' '.join(summary_parts) if summary_parts else clean[:max_length]

    def extract_keywords(self, text: str, max_keywords: int = 10) -> list[str]:
        """Extract keywords using TF-based scoring."""
        # Tokenize
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

        # Filter stop words
        filtered = [w for w in words if w not in self.STOP_WORDS]

        # Count frequencies
        freq: dict[str, int] = {}
        for word in filtered:
            freq[word] = freq.get(word, 0) + 1

        # Sort by frequency
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)

        # Return top keywords
        return [word for word, _ in sorted_words[:max_keywords]]

    def extract_entities(self, text: str) -> list[str]:
        """Extract potential named entities using capitalization heuristics."""
        entities = set()

        # Find capitalized words not at sentence start
        sentences = re.split(r'[.!?]\s+', text)
        for sentence in sentences:
            words = sentence.split()
            for i, word in enumerate(words):
                # Skip first word of sentence
                if i == 0:
                    continue
                # Check for capitalized word
                if word and word[0].isupper() and len(word) > 2:
                    clean = re.sub(r'[^\w]', '', word)
                    if clean and clean.lower() not in self.STOP_WORDS:
                        entities.add(clean)

        # Find multi-word entities (consecutive capitals)
        multi_pattern = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')
        for match in multi_pattern.finditer(text):
            entities.add(match.group(1))

        return sorted(entities)[:20]

    def detect_content_type(self, text: str) -> str:
        """Detect the type of content."""
        tech_score = sum(1 for p in self._technical_re if p.search(text))
        ref_score = sum(1 for p in self._reference_re if p.search(text))

        if tech_score >= 3:
            return "technical"
        if ref_score >= 2:
            return "reference"
        if re.search(r'\bchapter\s+\d+\b', text, re.IGNORECASE):
            return "narrative"
        if len(re.findall(r'^\s*[-*]\s+', text, re.MULTILINE)) > 5:
            return "list-based"

        return "general"

    def detect_language(self, text: str) -> str:
        """Simple language detection based on common words."""
        # This is a simplified version - production would use a library
        text_lower = text.lower()

        # English indicators
        english_words = ['the', 'is', 'are', 'and', 'or', 'of', 'to', 'in']
        english_count = sum(1 for w in english_words if f' {w} ' in text_lower)

        if english_count >= 3:
            return "en"

        return "unknown"

    def extract_headers(self, text: str) -> list[str]:
        """Extract all headers from the document."""
        headers = []

        # Markdown headers
        for match in re.finditer(r'^#{1,6}\s+(.+)$', text, re.MULTILINE):
            headers.append(match.group(1).strip())

        return headers[:20]  # Limit to 20 headers

    def enrich(self, text: str, existing_metadata: dict[str, Any] | None = None) -> DocumentMetadata:
        """Enrich document with full metadata.

        Args:
            text: Document text content
            existing_metadata: Optional existing metadata to merge

        Returns:
            DocumentMetadata with all extracted information
        """
        existing = existing_metadata or {}

        word_count = len(text.split())

        return DocumentMetadata(
            title=existing.get('title') or self.extract_title(text),
            summary=existing.get('summary') or self.generate_summary(text),
            keywords=existing.get('keywords') or self.extract_keywords(text),
            entities=self.extract_entities(text),
            word_count=word_count,
            estimated_reading_time_minutes=round(word_count / self.WORDS_PER_MINUTE, 1),
            language=self.detect_language(text),
            content_type=self.detect_content_type(text),
            headers=self.extract_headers(text),
            extra=existing,
        )


__all__ = [
    "DocumentMetadata",
    "MetadataEnricher",
]
