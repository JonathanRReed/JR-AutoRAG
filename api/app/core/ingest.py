"""Document ingestion pipeline."""

from __future__ import annotations

import io
import mimetypes
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

try:  # pragma: no cover - optional dependency
    from pypdf import PdfReader  # type: ignore
except ImportError:  # pragma: no cover
    PdfReader = None  # type: ignore

try:  # pragma: no cover
    import docx2txt  # type: ignore
except ImportError:  # pragma: no cover
    docx2txt = None  # type: ignore

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
    document_id: str
    title: str
    chunk_count: int


class IngestPipeline:
    """Handles text/file ingestion and triggers index rebuilds."""

    def __init__(self, store: DocumentStore, retrieval: RetrievalEngine) -> None:
        self._store = store
        self._retrieval = retrieval

    def ingest_text(self, title: str, text: str, metadata: Optional[Dict[str, str]] = None) -> IngestResult:
        meta = self._prepare_metadata(metadata)
        chunks = self._chunk(text)
        combined = "\n\n".join(chunks)
        doc = self._store.add(title=title, text=combined, metadata=meta)
        self._retrieval.build()
        return IngestResult(document_id=doc.id, title=doc.title, chunk_count=len(chunks))

    def ingest_file(self, title: str, content: bytes, metadata: Optional[Dict[str, str]] = None) -> IngestResult:
        meta = {**(metadata or {})}
        meta.setdefault("filename", title)
        meta.setdefault("original_filename", meta["filename"])
        meta.setdefault("content_type", mimetypes.guess_type(meta["filename"])[0] or "text/plain")
        meta["filesize"] = str(len(content))
        text = self._extract_text(content, meta)
        return self.ingest_text(title=title, text=text, metadata=meta)

    def _prepare_metadata(self, metadata: Optional[Dict[str, str]]) -> Dict[str, str]:
        meta = {**(metadata or {})}
        meta.setdefault("uploaded_at", datetime.now(timezone.utc).isoformat())
        return meta

    def _infer_extension(self, metadata: Optional[Dict[str, str]]) -> str:
        if metadata:
            filename = metadata.get("filename")
            if filename:
                return Path(filename).suffix.lower()
            content_type = metadata.get("content_type")
            if content_type:
                return (mimetypes.guess_extension(content_type) or "").lower()
        return ""

    def _extract_text(self, content: bytes, metadata: Optional[Dict[str, str]] = None) -> str:
        ext = self._infer_extension(metadata)
        if ext in {".md", ".markdown"}:
            return self._extract_markdown(content)
        if ext == ".pdf":
            text = self._extract_pdf_text(content)
            if text.strip():
                return text
            ocr_text = self._ocr_pdf(content)
            if ocr_text.strip():
                return ocr_text
        if ext in {".doc", ".docx"} and docx2txt:
            try:
                suffix = ext or ".docx"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(content)
                    tmp.flush()
                    tmp_path = tmp.name
                try:
                    return docx2txt.process(tmp_path)  # type: ignore[arg-type]
                finally:
                    os.unlink(tmp_path)
            except Exception:
                pass
        return content.decode("utf-8", errors="ignore")

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
            reader = PdfReader(io.BytesIO(content))  # type: ignore[name-defined]
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages)
        except Exception as exc:
            print(f"Error extracting PDF text: {exc}")
            return ""

    def _ocr_pdf(self, content: bytes) -> str:
        if not convert_from_bytes or not pytesseract:
            return ""
        try:
            images = convert_from_bytes(content)  # type: ignore[name-defined]
        except Exception as exc:
            print(f"Error converting PDF to images for OCR: {exc}")
            return ""
        text_chunks: List[str] = []
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

    def _chunk(self, text: str, target: int = 800) -> List[str]:
        clean = text.replace("\r", "")
        paragraphs = [p.strip() for p in clean.split("\n\n") if p.strip()]
        chunks: List[str] = []
        current: List[str] = []
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
