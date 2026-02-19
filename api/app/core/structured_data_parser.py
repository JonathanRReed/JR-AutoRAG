"""Structured Data Parser: Convert structured data to retrievable text.

Parses and converts structured data formats to text suitable for RAG:
- JSON/JSONL: Flattened key-value pairs with path context
- CSV/TSV: Row-based text with header context
- YAML: Nested structure flattening
- XML: Tag-aware text extraction

Each record becomes a retrievable chunk with schema awareness.
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StructuredDataFormat(str, Enum):
    """Supported structured data formats."""
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    TSV = "tsv"
    YAML = "yaml"
    XML = "xml"


@dataclass
class StructuredRecord:
    """A single record from structured data."""
    text: str  # Text representation for retrieval
    raw_data: dict[str, Any]  # Original structured data
    record_index: int
    format: StructuredDataFormat
    schema_path: str = ""  # JSON path or XML xpath style
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredDataConfig:
    """Configuration for structured data parsing."""
    # JSON settings
    flatten_json: bool = True
    max_json_depth: int = 5
    include_null_values: bool = False

    # CSV settings
    csv_has_header: bool = True
    csv_delimiter: str = ","
    max_columns_per_chunk: int = 10

    # Text formatting
    key_value_separator: str = ": "
    path_separator: str = " > "
    record_separator: str = "\n---\n"

    # Chunking
    max_record_text_length: int = 1000
    group_related_records: bool = True


class StructuredDataParser:
    """Parse structured data formats for RAG retrieval.

    Converts structured data to natural language text while preserving
    schema information for better retrieval matching.

    Example JSON transformation:
    ```
    {"user": {"name": "John", "age": 30}}
    ```
    Becomes:
    ```
    Record 1:
    user > name: John
    user > age: 30
    ```
    """

    def __init__(self, config: StructuredDataConfig | None = None) -> None:
        """Initialize parser with configuration."""
        self.config = config or StructuredDataConfig()

    def detect_format(self, content: str, filename: str = "") -> StructuredDataFormat:
        """Auto-detect the structured data format.

        Uses filename extension and content inspection.
        """
        # Check filename extension
        ext_map = {
            ".json": StructuredDataFormat.JSON,
            ".jsonl": StructuredDataFormat.JSONL,
            ".csv": StructuredDataFormat.CSV,
            ".tsv": StructuredDataFormat.TSV,
            ".yaml": StructuredDataFormat.YAML,
            ".yml": StructuredDataFormat.YAML,
            ".xml": StructuredDataFormat.XML,
        }

        for ext, fmt in ext_map.items():
            if filename.lower().endswith(ext):
                return fmt

        # Content inspection
        content_stripped = content.strip()

        # Check for JSON
        if content_stripped.startswith('{') or content_stripped.startswith('['):
            if '\n{' in content_stripped:
                return StructuredDataFormat.JSONL
            return StructuredDataFormat.JSON

        # Check for XML
        if content_stripped.startswith('<?xml') or content_stripped.startswith('<'):
            return StructuredDataFormat.XML

        # Check for CSV (has commas in most lines)
        lines = content_stripped.split('\n')[:5]
        comma_lines = sum(1 for line in lines if ',' in line)
        if comma_lines >= len(lines) * 0.6:
            return StructuredDataFormat.CSV

        # Check for TSV
        tab_lines = sum(1 for line in lines if '\t' in line)
        if tab_lines >= len(lines) * 0.6:
            return StructuredDataFormat.TSV

        # Default to JSON
        return StructuredDataFormat.JSON

    def _flatten_json(
        self,
        data: Any,
        prefix: str = "",
        depth: int = 0,
    ) -> list[tuple[str, Any]]:
        """Flatten nested JSON to key-value pairs with paths."""
        if depth >= self.config.max_json_depth:
            return [(prefix, str(data))]

        pairs = []
        sep = self.config.path_separator

        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}{sep}{key}" if prefix else key
                pairs.extend(self._flatten_json(value, new_prefix, depth + 1))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_prefix = f"{prefix}[{i}]"
                pairs.extend(self._flatten_json(item, new_prefix, depth + 1))
        else:
            if data is None and not self.config.include_null_values:
                return []
            pairs.append((prefix, data))

        return pairs

    def _format_record(
        self,
        flat_pairs: list[tuple[str, Any]],
        record_index: int,
    ) -> str:
        """Format flattened key-value pairs as text."""
        lines = [f"Record {record_index + 1}:"]

        for path, value in flat_pairs:
            value_str = str(value) if value is not None else "(empty)"
            lines.append(f"{path}{self.config.key_value_separator}{value_str}")

        text = "\n".join(lines)

        if len(text) > self.config.max_record_text_length:
            text = text[:self.config.max_record_text_length - 3] + "..."

        return text

    def parse_json(self, content: str) -> list[StructuredRecord]:
        """Parse JSON content to structured records."""
        records = []

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return records

        # Handle array of objects
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    flat_pairs = self._flatten_json(item)
                    text = self._format_record(flat_pairs, i)
                    records.append(StructuredRecord(
                        text=text,
                        raw_data=item,
                        record_index=i,
                        format=StructuredDataFormat.JSON,
                    ))
        elif isinstance(data, dict):
            # Single object
            flat_pairs = self._flatten_json(data)
            text = self._format_record(flat_pairs, 0)
            records.append(StructuredRecord(
                text=text,
                raw_data=data,
                record_index=0,
                format=StructuredDataFormat.JSON,
            ))

        return records

    def parse_jsonl(self, content: str) -> list[StructuredRecord]:
        """Parse JSON Lines content."""
        records = []

        for i, line in enumerate(content.strip().split('\n')):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    flat_pairs = self._flatten_json(data)
                    text = self._format_record(flat_pairs, i)
                    records.append(StructuredRecord(
                        text=text,
                        raw_data=data,
                        record_index=i,
                        format=StructuredDataFormat.JSONL,
                    ))
            except json.JSONDecodeError:
                continue

        return records

    def parse_csv(self, content: str, delimiter: str | None = None) -> list[StructuredRecord]:
        """Parse CSV/TSV content."""
        records = []
        delimiter = delimiter or self.config.csv_delimiter

        reader = csv.reader(io.StringIO(content), delimiter=delimiter)
        rows = list(reader)

        if not rows:
            return records

        # Get headers
        headers = rows[0] if self.config.csv_has_header else [f"Column {i+1}" for i in range(len(rows[0]))]
        data_rows = rows[1:] if self.config.csv_has_header else rows

        for i, row in enumerate(data_rows):
            # Build key-value pairs
            pairs = []
            for j, (header, value) in enumerate(zip(headers, row, strict=False)):
                if j < self.config.max_columns_per_chunk:
                    pairs.append((header, value))

            text = self._format_record(pairs, i)

            # Create raw data dict
            raw_data = dict(zip(headers, row, strict=False))

            records.append(StructuredRecord(
                text=text,
                raw_data=raw_data,
                record_index=i,
                format=StructuredDataFormat.CSV if delimiter == "," else StructuredDataFormat.TSV,
            ))

        return records

    def parse_yaml(self, content: str) -> list[StructuredRecord]:
        """Parse YAML content.

        Falls back to treating as text if yaml not available.
        """
        try:
            import yaml
            data = yaml.safe_load(content)

            if isinstance(data, list):
                records = []
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        flat_pairs = self._flatten_json(item)
                        text = self._format_record(flat_pairs, i)
                        records.append(StructuredRecord(
                            text=text,
                            raw_data=item,
                            record_index=i,
                            format=StructuredDataFormat.YAML,
                        ))
                return records
            elif isinstance(data, dict):
                flat_pairs = self._flatten_json(data)
                text = self._format_record(flat_pairs, 0)
                return [StructuredRecord(
                    text=text,
                    raw_data=data,
                    record_index=0,
                    format=StructuredDataFormat.YAML,
                )]
        except ImportError:
            pass
        except Exception:
            pass

        return []

    def parse_xml(self, content: str) -> list[StructuredRecord]:
        """Parse XML content.

        Extracts text with tag path context.
        """
        records = []

        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(content)

            def extract_text(element: Any, path: str = "") -> list[tuple[str, str]]:
                current_path = f"{path} > {element.tag}" if path else element.tag
                pairs = []

                # Add element text
                if element.text and element.text.strip():
                    pairs.append((current_path, element.text.strip()))

                # Add attributes
                for attr, value in element.attrib.items():
                    pairs.append((f"{current_path}@{attr}", value))

                # Recurse to children
                for child in element:
                    pairs.extend(extract_text(child, current_path))

                return pairs

            all_pairs = extract_text(root)

            # Group into chunks
            chunk_size = self.config.max_columns_per_chunk
            for i in range(0, len(all_pairs), chunk_size):
                chunk_pairs = all_pairs[i:i + chunk_size]
                text = self._format_record(chunk_pairs, i // chunk_size)

                records.append(StructuredRecord(
                    text=text,
                    raw_data=dict(chunk_pairs),
                    record_index=i // chunk_size,
                    format=StructuredDataFormat.XML,
                    schema_path=chunk_pairs[0][0] if chunk_pairs else "",
                ))
        except Exception:
            pass

        return records

    def parse(
        self,
        content: str,
        format: StructuredDataFormat | None = None,
        filename: str = "",
    ) -> list[StructuredRecord]:
        """Parse structured content to records.

        Args:
            content: Raw content string
            format: Optional format override
            filename: Filename for format detection

        Returns:
            List of StructuredRecords ready for indexing
        """
        if format is None:
            format = self.detect_format(content, filename)

        parsers = {
            StructuredDataFormat.JSON: self.parse_json,
            StructuredDataFormat.JSONL: self.parse_jsonl,
            StructuredDataFormat.CSV: lambda c: self.parse_csv(c, ","),
            StructuredDataFormat.TSV: lambda c: self.parse_csv(c, "\t"),
            StructuredDataFormat.YAML: self.parse_yaml,
            StructuredDataFormat.XML: self.parse_xml,
        }

        parser = parsers.get(format, self.parse_json)
        return parser(content)

    def to_chunks(
        self,
        records: list[StructuredRecord],
        add_schema_context: bool = True,
    ) -> list[str]:
        """Convert records to chunk texts for indexing.

        Optionally adds schema context to improve retrieval.
        """
        chunks = []

        for record in records:
            text = record.text

            if add_schema_context and record.raw_data:
                # Add schema hint
                keys = list(record.raw_data.keys())[:5]
                schema_hint = f"Fields: {', '.join(keys)}"
                text = f"{schema_hint}\n\n{text}"

            chunks.append(text)

        return chunks


# Singleton for easy access
_parser: StructuredDataParser | None = None


def get_structured_data_parser(config: StructuredDataConfig | None = None) -> StructuredDataParser:
    """Get or create the structured data parser."""
    global _parser
    if _parser is None or config is not None:
        _parser = StructuredDataParser(config)
    return _parser


__all__ = [
    "StructuredDataFormat",
    "StructuredRecord",
    "StructuredDataConfig",
    "StructuredDataParser",
    "get_structured_data_parser",
]
