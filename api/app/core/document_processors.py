"""Advanced document processors for structured content extraction.

This module provides specialized processors for:
- Tables: Markdown, HTML, and delimited tables
- Lists: Bullet points, numbered lists
- Code blocks: With language detection
- Headers: Document structure extraction
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StructuredContentType(str, Enum):
    """Types of structured content in documents."""
    TABLE = "table"
    LIST = "list"
    CODE_BLOCK = "code_block"
    HEADER = "header"
    PARAGRAPH = "paragraph"


@dataclass
class StructuredContent:
    """Extracted structured content with metadata."""
    content_type: StructuredContentType
    raw_text: str
    processed_text: str  # Human-readable version
    start_pos: int
    end_pos: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TableCell:
    """A single cell in a table."""
    content: str
    row: int
    col: int
    is_header: bool = False


@dataclass
class ExtractedTable:
    """A fully extracted table with structure."""
    headers: list[str]
    rows: list[list[str]]
    caption: str = ""

    def to_text(self) -> str:
        """Convert to readable text format."""
        lines = []
        if self.caption:
            lines.append(f"Table: {self.caption}")
        if self.headers:
            lines.append("Columns: " + " | ".join(self.headers))
        for row in self.rows:
            lines.append(" | ".join(row))
        return "\n".join(lines)

    def to_qa_pairs(self) -> list[tuple[str, str]]:
        """Convert to Q&A pairs for better retrieval."""
        pairs = []
        if not self.headers or not self.rows:
            return pairs

        for row in self.rows:
            for i, (header, value) in enumerate(zip(self.headers, row, strict=False)):
                if value.strip():
                    q = f"What is the {header} of {row[0]}?" if i > 0 else f"What is {header}?"
                    a = value
                    pairs.append((q, a))
        return pairs


class TableExtractor:
    """Extracts tables from various formats."""

    # Markdown table pattern
    MD_TABLE_PATTERN = re.compile(
        r'(\|[^\n]+\|\n)'  # Header row
        r'(\|[-:\s|]+\|\n)'  # Separator row
        r'((?:\|[^\n]+\|\n?)+)',  # Data rows
        re.MULTILINE
    )

    # Simple delimited table (CSV-like)
    DELIMITED_PATTERN = re.compile(
        r'^([^\n,\t]+[,\t][^\n]+)(\n[^\n,\t]+[,\t][^\n]+)+$',
        re.MULTILINE
    )

    def extract_markdown_tables(self, text: str) -> list[ExtractedTable]:
        """Extract markdown-formatted tables."""
        tables = []

        for match in self.MD_TABLE_PATTERN.finditer(text):
            header_row = match.group(1)
            data_rows = match.group(3)

            # Parse headers
            headers = [
                cell.strip()
                for cell in header_row.strip().split('|')
                if cell.strip()
            ]

            # Parse data rows
            rows = []
            for line in data_rows.strip().split('\n'):
                if line.strip():
                    cells = [
                        cell.strip()
                        for cell in line.split('|')
                        if cell.strip()
                    ]
                    if cells:
                        rows.append(cells)

            if headers and rows:
                tables.append(ExtractedTable(headers=headers, rows=rows))

        return tables

    def extract_html_tables(self, text: str) -> list[ExtractedTable]:
        """Extract HTML tables."""
        tables = []

        # Simple HTML table extraction
        table_pattern = re.compile(r'<table[^>]*>(.*?)</table>', re.DOTALL | re.IGNORECASE)
        row_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
        cell_pattern = re.compile(r'<t[dh][^>]*>(.*?)</t[dh]>', re.DOTALL | re.IGNORECASE)

        for table_match in table_pattern.finditer(text):
            table_content = table_match.group(1)

            headers = []
            rows = []

            for i, row_match in enumerate(row_pattern.finditer(table_content)):
                row_content = row_match.group(1)
                cells = [
                    re.sub(r'<[^>]+>', '', cell.group(1)).strip()
                    for cell in cell_pattern.finditer(row_content)
                ]

                # First row with th or first row is headers
                if i == 0 and '<th' in row_content.lower() or i == 0 and not headers:
                    headers = cells
                else:
                    rows.append(cells)

            if headers or rows:
                tables.append(ExtractedTable(headers=headers, rows=rows))

        return tables

    def extract_all_tables(self, text: str) -> list[ExtractedTable]:
        """Extract tables from all supported formats."""
        tables = []
        tables.extend(self.extract_markdown_tables(text))
        tables.extend(self.extract_html_tables(text))
        return tables


class ListExtractor:
    """Extracts lists from documents."""

    # Bullet list patterns
    BULLET_PATTERN = re.compile(
        r'^[\s]*[-*•]\s+(.+)$',
        re.MULTILINE
    )

    # Numbered list patterns
    NUMBERED_PATTERN = re.compile(
        r'^[\s]*\d+[.)]\s+(.+)$',
        re.MULTILINE
    )

    def extract_bullet_lists(self, text: str) -> list[list[str]]:
        """Extract bullet point lists."""
        lists = []
        current_list = []

        for line in text.split('\n'):
            match = self.BULLET_PATTERN.match(line)
            if match:
                current_list.append(match.group(1).strip())
            else:
                if current_list and len(current_list) >= 2:
                    lists.append(current_list)
                current_list = []

        if current_list and len(current_list) >= 2:
            lists.append(current_list)

        return lists

    def extract_numbered_lists(self, text: str) -> list[list[str]]:
        """Extract numbered lists."""
        lists = []
        current_list = []

        for line in text.split('\n'):
            match = self.NUMBERED_PATTERN.match(line)
            if match:
                current_list.append(match.group(1).strip())
            else:
                if current_list and len(current_list) >= 2:
                    lists.append(current_list)
                current_list = []

        if current_list and len(current_list) >= 2:
            lists.append(current_list)

        return lists


class CodeBlockExtractor:
    """Extracts code blocks with language detection."""

    # Fenced code block pattern
    FENCED_PATTERN = re.compile(
        r'```(\w*)\n(.*?)```',
        re.DOTALL
    )

    # Indented code block (4 spaces or 1 tab)
    INDENTED_PATTERN = re.compile(
        r'^(?:[ ]{4}|\t)(.+)$',
        re.MULTILINE
    )

    def extract_fenced_blocks(self, text: str) -> list[tuple[str, str]]:
        """Extract fenced code blocks with language.

        Returns:
            List of (language, code) tuples
        """
        blocks = []

        for match in self.FENCED_PATTERN.finditer(text):
            language = match.group(1) or "text"
            code = match.group(2)
            blocks.append((language, code.strip()))

        return blocks


class HeaderExtractor:
    """Extracts document headers for structure analysis."""

    # Markdown header patterns
    MD_HEADER_PATTERN = re.compile(
        r'^(#{1,6})\s+(.+)$',
        re.MULTILINE
    )

    def extract_headers(self, text: str) -> list[tuple[int, str, int]]:
        """Extract headers with their levels.

        Returns:
            List of (level, text, position) tuples
        """
        headers = []

        for match in self.MD_HEADER_PATTERN.finditer(text):
            level = len(match.group(1))
            header_text = match.group(2).strip()
            position = match.start()
            headers.append((level, header_text, position))

        return headers

    def build_toc(self, text: str) -> list[dict]:
        """Build a table of contents from headers."""
        headers = self.extract_headers(text)
        toc = []

        for level, header_text, pos in headers:
            toc.append({
                "level": level,
                "text": header_text,
                "position": pos,
            })

        return toc


class DocumentProcessor:
    """Main document processor that coordinates all extractors."""

    def __init__(self) -> None:
        self.table_extractor = TableExtractor()
        self.list_extractor = ListExtractor()
        self.code_extractor = CodeBlockExtractor()
        self.header_extractor = HeaderExtractor()

    def process(self, text: str) -> dict[str, Any]:
        """Process document and extract all structured content."""
        return {
            "tables": [
                {
                    "headers": t.headers,
                    "rows": t.rows,
                    "text_representation": t.to_text(),
                }
                for t in self.table_extractor.extract_all_tables(text)
            ],
            "bullet_lists": self.list_extractor.extract_bullet_lists(text),
            "numbered_lists": self.list_extractor.extract_numbered_lists(text),
            "code_blocks": [
                {"language": lang, "code": code}
                for lang, code in self.code_extractor.extract_fenced_blocks(text)
            ],
            "headers": self.header_extractor.extract_headers(text),
            "toc": self.header_extractor.build_toc(text),
        }

    def enhance_text_for_retrieval(self, text: str) -> str:
        """Enhance document text for better retrieval.

        Converts structured content to more searchable text.
        """
        enhanced_parts = [text]

        # Add table Q&A pairs
        for table in self.table_extractor.extract_all_tables(text):
            for q, a in table.to_qa_pairs():
                enhanced_parts.append(f"Q: {q} A: {a}")

        # Add flattened lists
        for lst in self.list_extractor.extract_bullet_lists(text):
            enhanced_parts.append("List items: " + ", ".join(lst))

        return "\n\n".join(enhanced_parts)


__all__ = [
    "StructuredContentType",
    "StructuredContent",
    "ExtractedTable",
    "TableExtractor",
    "ListExtractor",
    "CodeBlockExtractor",
    "HeaderExtractor",
    "DocumentProcessor",
]
