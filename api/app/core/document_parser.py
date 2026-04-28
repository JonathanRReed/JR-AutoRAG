"""Document parser provider layer with optional Docling support."""

from __future__ import annotations

import json
import mimetypes
import re
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Protocol

ParsedBlockType = Literal["text", "heading", "table", "image", "ocr", "metadata"]


@dataclass
class ParsedBlock:
    type: ParsedBlockType
    text: str = ""
    page: int | None = None
    heading_level: int | None = None
    confidence: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class ParsedPage:
    number: int
    text: str = ""
    blocks: list[ParsedBlock] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class ChunkProvenance:
    parser: str
    page_numbers: list[int] = field(default_factory=list)
    block_types: list[str] = field(default_factory=list)
    heading_path: list[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class ParserResult:
    provider: str
    engine: str
    text: str
    pages: list[ParsedPage] = field(default_factory=list)
    blocks: list[ParsedBlock] = field(default_factory=list)
    confidence: float = 1.0
    used_ocr: bool = False
    warnings: list[str] = field(default_factory=list)
    raw_metadata: dict[str, object] = field(default_factory=dict)

    def preview_payload(self, max_blocks: int = 80) -> dict[str, object]:
        pages = [
            {
                "number": page.number,
                "text": page.text[:1200],
                "confidence": page.confidence,
                "metadata": page.metadata,
                "blocks": [asdict(block) for block in page.blocks[:20]],
            }
            for page in self.pages[:20]
        ]
        return {
            "parser_provider": self.provider,
            "parser_engine": self.engine,
            "confidence": self.confidence,
            "used_ocr": self.used_ocr,
            "warnings": self.warnings,
            "page_count": len(self.pages),
            "block_count": len(self.blocks),
            "blocks": [asdict(block) for block in self.blocks[:max_blocks]],
            "pages": pages,
            "metadata": self.raw_metadata,
        }


class DocumentParserProvider(Protocol):
    name: str

    def available(self) -> bool:
        ...

    def parse(self, content: bytes, metadata: dict[str, str] | None = None) -> ParserResult:
        ...


def _infer_suffix(metadata: dict[str, str] | None) -> str:
    filename = (metadata or {}).get("filename") or (metadata or {}).get("original_filename")
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix:
            return suffix
    content_type = (metadata or {}).get("content_type")
    if content_type:
        return (mimetypes.guess_extension(content_type) or "").lower()
    return ".bin"


def _blocks_from_markdown(markdown: str) -> list[ParsedBlock]:
    blocks: list[ParsedBlock] = []
    current_page = 1
    heading_path: list[str] = []
    for raw in re.split(r"\n{2,}", markdown.replace("\r", "")):
        text = raw.strip()
        if not text:
            continue
        if "\f" in text or "[PAGE_BREAK]" in text:
            current_page += 1
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", text)
        if heading_match:
            level = len(heading_match.group(1))
            title = heading_match.group(2).strip()
            heading_path = heading_path[: max(level - 1, 0)] + [title]
            blocks.append(
                ParsedBlock(
                    type="heading",
                    text=title,
                    page=current_page,
                    heading_level=level,
                    metadata={"heading_path": list(heading_path)},
                )
            )
            continue
        block_type: ParsedBlockType = "table" if "|" in text and re.search(r"\n\s*\|?[-: ]+\|", text) else "text"
        blocks.append(
            ParsedBlock(
                type=block_type,
                text=text,
                page=current_page,
                metadata={"heading_path": list(heading_path)},
            )
        )
    return blocks


def _pages_from_blocks(blocks: list[ParsedBlock], fallback_text: str) -> list[ParsedPage]:
    by_page: dict[int, list[ParsedBlock]] = {}
    for block in blocks:
        by_page.setdefault(block.page or 1, []).append(block)
    if not by_page:
        by_page[1] = [ParsedBlock(type="text", text=fallback_text[:1200], page=1)]
    pages: list[ParsedPage] = []
    for page_number in sorted(by_page):
        page_blocks = by_page[page_number]
        text = "\n\n".join(block.text for block in page_blocks if block.text)
        confidence = sum(block.confidence for block in page_blocks) / max(len(page_blocks), 1)
        pages.append(ParsedPage(number=page_number, text=text, blocks=page_blocks, confidence=confidence))
    return pages


class DoclingDocumentParser:
    name = "docling"

    def available(self) -> bool:
        try:
            from docling.document_converter import DocumentConverter  # noqa: F401
        except Exception:
            return False
        return True

    def parse(self, content: bytes, metadata: dict[str, str] | None = None) -> ParserResult:
        try:
            from docling.document_converter import DocumentConverter
        except Exception as exc:
            raise RuntimeError(f"Docling is not installed: {exc}") from exc

        suffix = _infer_suffix(metadata)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / f"input{suffix}"
            path.write_bytes(content)
            converter = DocumentConverter()
            result = converter.convert(path)
            document = result.document
            text = document.export_to_markdown()
            raw_metadata: dict[str, object] = {}
            try:
                raw_metadata = document.export_to_dict()
            except Exception:
                raw_metadata = {}

        blocks = _blocks_from_markdown(text)
        pages = _pages_from_blocks(blocks, text)
        status = str(getattr(result, "status", "") or "")
        errors = getattr(result, "errors", []) or []
        warnings = [str(error) for error in errors][:8]
        confidence = 0.95 if text.strip() else 0.0
        if status and "success" not in status.lower():
            confidence = min(confidence, 0.5)
        return ParserResult(
            provider=self.name,
            engine="docling.DocumentConverter",
            text=text,
            pages=pages,
            blocks=blocks,
            confidence=confidence,
            used_ocr=any(block.type == "ocr" for block in blocks),
            warnings=warnings,
            raw_metadata={"status": status, "docling": raw_metadata},
        )


class NativeDocumentParser:
    name = "native"

    def available(self) -> bool:
        return True

    def parse(self, content: bytes, metadata: dict[str, str] | None = None) -> ParserResult:
        text = content.decode("utf-8", errors="ignore")
        confidence = 1.0 if text.strip() else 0.0
        blocks = _blocks_from_markdown(text)
        pages = _pages_from_blocks(blocks, text)
        return ParserResult(
            provider=self.name,
            engine="utf8-markdown-native",
            text=text,
            pages=pages,
            blocks=blocks,
            confidence=confidence,
        )


class DocumentParserRouter:
    """Prefer Docling when available, otherwise leave native extraction in control."""

    def __init__(self, prefer_docling: bool = True) -> None:
        self._docling = DoclingDocumentParser()
        self._native = NativeDocumentParser()
        self._prefer_docling = prefer_docling

    def parse(self, content: bytes, metadata: dict[str, str] | None = None) -> ParserResult:
        suffix = _infer_suffix(metadata)
        docling_candidate = suffix in {".pdf", ".docx", ".doc", ".pptx", ".html", ".htm", ".md", ".markdown", ".png", ".jpg", ".jpeg", ".tiff"}
        if self._prefer_docling and docling_candidate and self._docling.available():
            try:
                return self._docling.parse(content, metadata)
            except Exception as exc:
                native = self._native.parse(content, metadata)
                native.warnings.append(f"Docling fallback: {exc}")
                return native
        return self._native.parse(content, metadata)


def parser_result_from_text(
    text: str,
    provider: str = "native",
    engine: str = "stored-text",
    confidence: float = 0.75,
    used_ocr: bool = False,
    raw_metadata: dict[str, object] | None = None,
) -> ParserResult:
    blocks = _blocks_from_markdown(text)
    pages = _pages_from_blocks(blocks, text)
    return ParserResult(
        provider=provider,
        engine=engine,
        text=text,
        pages=pages,
        blocks=blocks,
        confidence=confidence,
        used_ocr=used_ocr,
        raw_metadata=raw_metadata or {},
    )


def parser_result_to_metadata(
    result: ParserResult,
    extraction_metadata: dict[str, str] | None = None,
) -> dict[str, str]:
    if extraction_metadata:
        merged = dict(result.raw_metadata)
        merged.update(
            {
                "extraction_method": extraction_metadata.get("extraction_method", ""),
                "extraction_engine": extraction_metadata.get("extraction_engine", ""),
                "ocr_policy": extraction_metadata.get("ocr_policy", ""),
                "ocr_used": extraction_metadata.get("ocr_used", "False"),
                "ocr_attempted": extraction_metadata.get("ocr_attempted", ""),
            }
        )
        result.raw_metadata = merged
    payload = result.preview_payload()
    block_types = sorted({block.type for block in result.blocks})
    headings = [block.text for block in result.blocks if block.type == "heading"][:20]
    return {
        "parser_provider": result.provider,
        "parser_engine": result.engine,
        "parser_confidence": f"{result.confidence:.3f}",
        "parser_page_count": str(len(result.pages)),
        "parser_block_count": str(len(result.blocks)),
        "parser_block_types": ",".join(block_types),
        "parser_headings_json": json.dumps(headings),
        "parser_used_ocr": str(result.used_ocr),
        "parser_preview_json": json.dumps(payload),
    }


def build_preview_from_document_metadata(metadata: dict[str, str], text: str) -> dict[str, object]:
    raw = metadata.get("parser_preview_json", "")
    if raw:
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass
    return parser_result_from_text(
        text=text,
        provider=metadata.get("parser_provider", "native"),
        engine=metadata.get("parser_engine", "stored-text"),
        confidence=float(metadata.get("parser_confidence", "0.75") or 0.75),
        used_ocr=metadata.get("parser_used_ocr", "").lower() == "true",
    ).preview_payload()
