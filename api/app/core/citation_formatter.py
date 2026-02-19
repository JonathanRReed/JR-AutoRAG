"""Citation formatting utilities for strict citation fidelity.

This module provides:
- Rich context formatting with document titles and chunk IDs
- Reference section generation for answer footers
- Citation validation against retrieved sources
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .gatherer import EvidenceChunk


@dataclass
class RichCitation:
    """A citation with full metadata for verifiable references."""
    citation_number: int
    document_title: str
    chunk_id: str
    key_quote: str
    full_snippet: str
    relevance_score: float
    section: str = ""  # Optional section/page info

    def to_reference_line(self) -> str:
        """Format as a reference list entry."""
        quote = self.key_quote[:80] + "..." if len(self.key_quote) > 80 else self.key_quote
        return f'[{self.citation_number}] {self.document_title} (ChunkID: {self.chunk_id}) - "{quote}"'


def format_rich_context(
    chunks: list[EvidenceChunk],
    max_quote_length: int = 300,
) -> tuple[str, list[RichCitation]]:
    """Format chunks with full metadata for strict citation requirements.

    Each chunk is formatted as:
    [N] Document: Title | ChunkID: xxx
    "Key quote from the passage..."
    Additional context sentences...

    Returns:
        Tuple of (formatted_context, list of RichCitation objects)
    """
    formatted_parts = []
    citations: list[RichCitation] = []

    for i, chunk in enumerate(chunks, 1):
        # Extract key quote (first complete sentence)
        sentences = re.split(r'(?<=[.!?])\s+', chunk.snippet)
        key_quote = sentences[0] if sentences else chunk.snippet[:100]

        # Format the context block
        header = f"[{i}] Document: {chunk.title} | ChunkID: {chunk.id}"

        # Truncate snippet if too long
        display_snippet = chunk.snippet
        if len(display_snippet) > max_quote_length:
            display_snippet = display_snippet[:max_quote_length] + "..."

        formatted_parts.append(f"{header}\n\"{display_snippet}\"")

        citations.append(RichCitation(
            citation_number=i,
            document_title=chunk.title,
            chunk_id=chunk.id,
            key_quote=key_quote,
            full_snippet=chunk.snippet,
            relevance_score=chunk.score,
        ))

    return "\n\n".join(formatted_parts), citations


def generate_reference_section(citations: list[RichCitation]) -> str:
    """Generate a formatted References section for answer footer.

    Returns a markdown-formatted reference list:
    ## References
    [1] DocTitle (ChunkID: xxx) - "key quote"
    [2] DocTitle (ChunkID: xxx) - "key quote"
    """
    if not citations:
        return ""

    lines = ["## References", ""]
    for citation in citations:
        lines.append(citation.to_reference_line())

    return "\n".join(lines)


def validate_answer_citations(
    answer: str,
    available_citations: list[RichCitation],
) -> dict[str, Any]:
    """Validate that an answer only cites available sources.

    Returns:
        {
            "valid": bool,
            "cited_numbers": list of citation numbers used,
            "invalid_citations": list of citation numbers not in sources,
            "uncited_sources": list of available sources not cited,
            "has_assumptions": bool,
            "assumption_count": int,
        }
    """
    # Extract citation numbers from answer
    cited_numbers = [int(n) for n in re.findall(r'\[(\d+)\]', answer)]
    unique_cited = set(cited_numbers)

    # Check for assumption markers
    assumption_markers = re.findall(r'\[ASSUMPTION[^\]]*\]', answer, re.IGNORECASE)

    # Available citation numbers
    available_numbers = {c.citation_number for c in available_citations}

    # Find invalid citations (cited but not in sources)
    invalid = unique_cited - available_numbers

    # Find uncited sources (available but not used)
    uncited = available_numbers - unique_cited

    return {
        "valid": len(invalid) == 0,
        "cited_numbers": sorted(unique_cited),
        "invalid_citations": sorted(invalid),
        "uncited_sources": sorted(uncited),
        "has_assumptions": len(assumption_markers) > 0,
        "assumption_count": len(assumption_markers),
    }


def extract_inline_quotes(answer: str) -> list[dict]:
    """Extract inline quotes from an answer.

    Looks for patterns like:
    - "quoted text" [1]
    - According to X, "quoted text" [1]

    Returns:
        List of {quote, citation_number, context}
    """
    quotes = []

    # Pattern: "quoted text" [N]
    pattern = r'"([^"]+)"\s*\[(\d+)\]'
    matches = re.findall(pattern, answer)

    for quote_text, citation_num in matches:
        quotes.append({
            "quote": quote_text,
            "citation_number": int(citation_num),
            "context": "",
        })

    return quotes


def verify_quote_in_source(
    quote: str,
    citation: RichCitation,
    min_overlap: float = 0.6,
) -> bool:
    """Verify that a quoted passage exists in the source.

    Uses fuzzy matching to handle minor variations.
    """
    # Normalize for comparison
    quote_normalized = quote.lower().strip()
    source_normalized = citation.full_snippet.lower()

    # Direct containment check
    if quote_normalized in source_normalized:
        return True

    # Word overlap check for paraphrasing
    quote_words = set(re.findall(r'\b\w+\b', quote_normalized))
    source_words = set(re.findall(r'\b\w+\b', source_normalized))

    if not quote_words:
        return False

    overlap = len(quote_words & source_words) / len(quote_words)
    return overlap >= min_overlap


__all__ = [
    "RichCitation",
    "format_rich_context",
    "generate_reference_section",
    "validate_answer_citations",
    "extract_inline_quotes",
    "verify_quote_in_source",
]
