"""Document ingestion pipeline with incremental indexing support.

Implements Workstream A1: Incremental ingestion and indexing.
- Content hashes per document for change detection
- Only re-chunk and re-embed changed documents
- Append to existing index without full rebuild when possible
"""

from __future__ import annotations

import hashlib
import io
import mimetypes
import os
import subprocess
import tempfile
import shutil
import zipfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:  # pragma: no cover - optional dependency
    from pypdf import PdfReader  # type: ignore
    print("PDF library (pypdf) loaded successfully")
except ImportError:  # pragma: no cover
    PdfReader = None  # type: ignore
    print("WARNING: PDF library (pypdf) failed to load")

try:  # pragma: no cover
    import docx  # type: ignore
    print("DOCX library loaded successfully")
except ImportError:  # pragma: no cover
    docx = None  # type: ignore
    print("WARNING: docx library failed to load")

try:  # pragma: no cover
    from pdf2image import convert_from_bytes  # type: ignore
except ImportError:  # pragma: no cover
    convert_from_bytes = None  # type: ignore

try:  # pragma: no cover
    import pytesseract  # type: ignore
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore

from .documents import DocumentStore
from .retrieval import RetrievalEngine


@dataclass
class IngestResult:
    """Result of document ingestion."""
    document_id: str
    title: str
    chunk_count: int
    was_modified: bool = True  # False if skipped due to unchanged content
    content_hash: str = ""  # SHA-256 hash of content for change tracking


class IngestPipeline:
    """Handles text/file ingestion with incremental indexing support.
    
    Key features (A1 requirement):
    - Content hashing for change detection
    - Skip re-processing of unchanged documents  
    - Contextualize chunks with document metadata
    """

    def __init__(self, store: DocumentStore, retrieval: RetrievalEngine) -> None:
        self._store = store
        self._retrieval = retrieval

    def ingest_text(self, title: str, text: str, metadata: dict[str, str] | None = None) -> IngestResult:
        """Ingest text with content hash tracking for change detection."""
        meta = self._prepare_metadata(metadata)
        meta.setdefault("processing_status", "processing")
        
        # Compute content hash for change detection (A1)
        content_hash = self._compute_content_hash(text)
        meta["content_hash"] = content_hash
        
        chunks = self._chunk(text)
        # Contextualize chunks with document header (D2)
        contextualized = self._contextualize_chunks(chunks, title, meta)
        combined = "\n\n".join(contextualized)
        
        doc = self._store.add(title=title, text=combined, metadata=meta)
        
        # Run heavy indexing in background thread to keep API responsive
        def do_build():
            try:
                self._retrieval.build()
                # Increment corpus version if retrieval engine supports it
                if hasattr(self._retrieval, 'increment_corpus_version'):
                    self._retrieval.increment_corpus_version()
                doc.metadata["processing_status"] = "ready"
                doc.metadata["processed_at"] = datetime.now(timezone.utc).isoformat()
                self._store.upsert(doc)
            except Exception as exc:
                doc.metadata["processing_status"] = "error"
                doc.metadata["processing_error"] = str(exc)
                self._store.upsert(doc)
        
        # Use a thread pool to avoid blocking the event loop
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(do_build)
        executor.shutdown(wait=False)  # Don't block, let it run in background
        
        return IngestResult(
            document_id=doc.id, 
            title=doc.title, 
            chunk_count=len(chunks),
            was_modified=True,
            content_hash=content_hash,
        )

    def ingest_file(self, title: str, content: bytes, metadata: dict[str, str] | None = None) -> IngestResult:
        meta = {**(metadata or {})}
        meta.setdefault("filename", title)
        meta.setdefault("original_filename", meta["filename"])
        meta.setdefault("content_type", mimetypes.guess_type(meta["filename"])[0] or "text/plain")
        meta["filesize"] = str(len(content))
        text = self._extract_text(content, meta)
        return self.ingest_text(title=title, text=text, metadata=meta)

    def ingest_incremental(
        self, 
        title: str, 
        text: str, 
        metadata: dict[str, str] | None = None,
    ) -> IngestResult:
        """Incremental ingest: skip if document content unchanged (A1).
        
        Args:
            title: Document title (used as identifier)
            text: Document text content
            metadata: Optional metadata dict
            
        Returns:
            IngestResult with was_modified=False if skipped
        """
        content_hash = self._compute_content_hash(text)
        
        # Check if document exists with same hash
        existing = self._store.get_by_title(title)
        if existing:
            existing_hash = existing.metadata.get("content_hash", "")
            if existing_hash == content_hash:
                # Document unchanged - skip re-processing
                return IngestResult(
                    document_id=existing.id,
                    title=existing.title,
                    chunk_count=0,
                    was_modified=False,
                    content_hash=content_hash,
                )
            # Document changed - delete old and re-ingest
            self._store.delete(existing.id)
        
        # Proceed with full ingest
        return self.ingest_text(title=title, text=text, metadata=metadata)

    def _compute_content_hash(self, text: str) -> str:
        """Compute SHA-256 hash of document content for change detection."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _contextualize_chunks(
        self,
        chunks: list[str],
        doc_title: str,
        metadata: dict[str, str] | None,
    ) -> list[str]:
        """Add header context to each chunk for better interpretability (D2).
        
        Makes isolated snippets self-contained with document info.
        """
        header_parts = [f"[Document: {doc_title}]"]
        if metadata:
            if "date" in metadata:
                header_parts.append(f"[Date: {metadata['date']}]")
            elif "uploaded_at" in metadata:
                # Use upload date if no explicit date
                upload_date = metadata["uploaded_at"][:10]  # YYYY-MM-DD
                header_parts.append(f"[Date: {upload_date}]")
            if "author" in metadata:
                header_parts.append(f"[Author: {metadata['author']}]")
        
        header = " ".join(header_parts)
        return [f"{header}\n{chunk}" for chunk in chunks]

    def _prepare_metadata(self, metadata: dict[str, str] | None) -> dict[str, str]:
        meta = {**(metadata or {})}
        meta.setdefault("uploaded_at", datetime.now(timezone.utc).isoformat())
        return meta

    def _infer_extension(self, metadata: dict[str, str] | None) -> str:
        if metadata:
            filename = metadata.get("filename")
            if filename:
                return Path(filename).suffix.lower()
            content_type = metadata.get("content_type")
            if content_type:
                return (mimetypes.guess_extension(content_type) or "").lower()
        return ""

    def _detect_magic_extension(self, content: bytes) -> str:
        if content.startswith(b"%PDF"):
            return ".pdf"
        if content[:4] == b"PK\x03\x04":
            try:
                with zipfile.ZipFile(io.BytesIO(content)) as archive:
                    if "word/document.xml" in archive.namelist():
                        return ".docx"
            except Exception:
                return ""
        if content[:8] == b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1":
            return ".doc"
        return ""

    def _extract_text(self, content: bytes, metadata: dict[str, str] | None = None) -> str:
        ext = self._infer_extension(metadata)
        if not ext:
            ext = self._detect_magic_extension(content)
        if ext in {".md", ".markdown"}:
            return self._extract_markdown(content)
        if ext == ".pdf":
            text = self._extract_pdf_text(content)
            if text.strip():
                return text
            text = self._extract_pdf_text_pdftotext(content)
            if text.strip():
                return text
            ocr_text = self._ocr_pdf(content)
            if ocr_text.strip():
                return ocr_text
            return (
                "(PDF extraction failed: no text found. "
                "If this is a scanned PDF, install poppler + tesseract for OCR. "
                "If it is text-based, try re-exporting the PDF.)"
            )
        if ext in {".doc", ".docx"}:
            if ext == ".docx":
                text = self._extract_docx_text(content)
                if text.strip():
                    return text
            if ext == ".doc":
                print("Tip: .doc files are legacy. Convert to .docx for better extraction.")

        # Fallback to plain text if not binary
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError:
            return f"(Binary content: {ext or 'unknown'})"

    def _extract_markdown(self, content: bytes) -> str:
        text = content.decode("utf-8", errors="ignore")
        # lightweight removal of common markdown tokens
        replacements = ["#", "*", "`", ">", "- ", "* "]
        for token in replacements:
            text = text.replace(token, "")
        return text

    def _extract_pdf_text(self, content: bytes) -> str:
        if not PdfReader:
            return ""
        try:
            reader = PdfReader(io.BytesIO(content), strict=False)  # type: ignore[name-defined]
            if getattr(reader, "is_encrypted", False):
                try:
                    reader.decrypt("")  # type: ignore[attr-defined]
                except Exception:
                    return ""
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        except Exception as exc:
            print(f"Error extracting PDF text: {exc}")
            return ""

    def _extract_pdf_text_pdftotext(self, content: bytes) -> str:
        """Fallback PDF extraction using pdftotext if installed."""
        try:
            pdftotext_bin = shutil.which("pdftotext")
            if not pdftotext_bin:
                return ""
            with tempfile.TemporaryDirectory() as tmpdir:
                pdf_path = Path(tmpdir) / "input.pdf"
                txt_path = Path(tmpdir) / "output.txt"
                pdf_path.write_bytes(content)
                result = subprocess.run(
                    [pdftotext_bin, "-enc", "UTF-8", str(pdf_path), str(txt_path)],
                    check=False,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if result.returncode != 0:
                    return ""
                if not txt_path.exists():
                    return ""
                return txt_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return ""

    def _extract_docx_text(self, content: bytes) -> str:
        """Extract text from a .docx using python-docx or a zip/XML fallback."""
        if docx:
            try:
                doc = docx.Document(io.BytesIO(content))
                text = "\n".join([para.text for para in doc.paragraphs])
                if text.strip():
                    return text
                table_text = []
                for table in doc.tables:
                    for row in table.rows:
                        table_text.append(" | ".join([cell.text for cell in row.cells]))
                return "\n".join(table_text)
            except Exception as exc:
                print(f"Error extracting Word text: {exc}")

        # Fallback: parse the document.xml from the docx zip
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as archive:
                xml_data = archive.read("word/document.xml")
            root = ET.fromstring(xml_data)
            namespace = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
            lines: list[str] = []
            buffer: list[str] = []
            for elem in root.iter():
                if elem.tag == f"{namespace}t" and elem.text:
                    buffer.append(elem.text)
                elif elem.tag == f"{namespace}p":
                    if buffer:
                        lines.append("".join(buffer).strip())
                        buffer = []
            if buffer:
                lines.append("".join(buffer).strip())
            return "\n".join([line for line in lines if line])
        except Exception as exc:
            print(f"Error parsing DOCX fallback: {exc}")
            return ""

    def _ocr_pdf(self, content: bytes) -> str:
        if not convert_from_bytes or not pytesseract:
            return ""
        poppler_bin = shutil.which("pdftoppm") or shutil.which("pdftocairo")
        poppler_path = str(Path(poppler_bin).parent) if poppler_bin else None
        tesseract_bin = shutil.which("tesseract")
        if tesseract_bin:
            pytesseract.pytesseract.tesseract_cmd = tesseract_bin  # type: ignore[attr-defined]
        try:
            images = convert_from_bytes(content, poppler_path=poppler_path)  # type: ignore[name-defined]
        except Exception as exc:
            print(f"Error converting PDF to images for OCR: {exc}")
            return ""
        text_chunks: list[str] = []
        for image in images:
            try:
                text = pytesseract.image_to_string(image)  # type: ignore[attr-defined]
                if text:
                    text_chunks.append(text)
            except Exception as exc:
                print(f"Error OCRing image page: {exc}")
            finally:
                image.close()
        return "\n".join(text_chunks)

    def _chunk(self, text: str, target: int = 800) -> list[str]:
        clean = text.replace("\r", "")
        paragraphs = [p.strip() for p in clean.split("\n\n") if p.strip()]
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for para in paragraphs:
            if current_len + len(para) > target and current:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            current.append(para)
            current_len += len(para)
        if current:
            chunks.append("\n".join(current))
        return chunks or [text.strip()]
