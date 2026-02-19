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
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

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

from ..schemas.config import AppConfig
from .audit import AuditAction, AuditEntry, get_audit_log
from .documents import DocumentStore
from .langextract_enricher import LangExtractEnricher
from .retrieval import RetrievalEngine

_INDEX_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="autorag-index")
_INDEX_LOCK = Lock()
_LANGEXTRACT_SYNTHETIC_BEGIN = "[[LANGEXTRACT_SYNTHETIC_FACTS_BEGIN]]"
_LANGEXTRACT_SYNTHETIC_END = "[[LANGEXTRACT_SYNTHETIC_FACTS_END]]"


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

    def __init__(
        self,
        store: DocumentStore,
        retrieval: RetrievalEngine,
        config_getter: Callable[[], AppConfig] | None = None,
        data_dir: Path | None = None,
    ) -> None:
        self._store = store
        self._retrieval = retrieval
        self._config_getter = config_getter
        self._langextract = LangExtractEnricher(data_dir=data_dir)

    def ingest_text(
        self,
        title: str,
        text: str,
        metadata: dict[str, str] | None = None,
        sync: bool = False,
        langextract_profile_override: str | None = None,
        langextract_prompt_override: str | None = None,
    ) -> IngestResult:
        """Ingest text with content hash tracking for change detection."""
        meta = self._prepare_metadata(metadata)
        meta.setdefault("processing_status", "processing")

        langextract_result = self._run_langextract(
            text=text,
            profile_override=langextract_profile_override,
            prompt_override=langextract_prompt_override,
        )
        augmented_text = self._append_langextract_sections(text, langextract_result.get("synthetic_sections", []))
        self._apply_langextract_metadata(meta, langextract_result)

        # Compute content hash for change detection (A1)
        content_hash = self._compute_content_hash(augmented_text)
        meta["content_hash"] = content_hash

        chunks = self._chunk(augmented_text)
        # Contextualize chunks with document header (D2)
        contextualized = self._contextualize_chunks(chunks, title, meta)
        combined = "\n\n".join(contextualized)

        doc = self._store.add(title=title, text=combined, metadata=meta)

        self._persist_langextract_artifact(doc.id, title, langextract_result, doc)
        self._log_langextract_audit(doc.id, title, langextract_result)

        # Run indexing in a shared background worker to keep API responsive
        def do_build() -> None:
            try:
                with _INDEX_LOCK:
                    if hasattr(self._retrieval, "index_documents"):
                        self._retrieval.index_documents([doc])
                    else:
                        self._retrieval.build()
                        if hasattr(self._retrieval, "increment_corpus_version"):
                            self._retrieval.increment_corpus_version()
                doc.metadata["processing_status"] = "ready"
                doc.metadata["processed_at"] = datetime.now(UTC).isoformat()
                self._store.upsert(doc)
            except Exception as exc:
                doc.metadata["processing_status"] = "error"
                doc.metadata["processing_error"] = str(exc)
                self._store.upsert(doc)

        if sync:
            do_build()
        else:
            _INDEX_EXECUTOR.submit(do_build)

        return IngestResult(
            document_id=doc.id,
            title=doc.title,
            chunk_count=len(chunks),
            was_modified=True,
            content_hash=content_hash,
        )

    def ingest_file(
        self,
        title: str,
        content: bytes,
        metadata: dict[str, str] | None = None,
        sync: bool = False,
        langextract_profile_override: str | None = None,
        langextract_prompt_override: str | None = None,
    ) -> IngestResult:
        meta = {**(metadata or {})}
        meta.setdefault("filename", title)
        meta.setdefault("original_filename", meta["filename"])
        meta.setdefault("content_type", mimetypes.guess_type(meta["filename"])[0] or "text/plain")
        meta["filesize"] = str(len(content))
        text = self._extract_text(content, meta)
        return self.ingest_text(
            title=title,
            text=text,
            metadata=meta,
            sync=sync,
            langextract_profile_override=langextract_profile_override,
            langextract_prompt_override=langextract_prompt_override,
        )

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

        # Proceed with full ingest (duplicate titles handled in store)
        return self.ingest_text(title=title, text=text, metadata=metadata)

    def _run_langextract(
        self,
        text: str,
        profile_override: str | None,
        prompt_override: str | None,
    ) -> dict[str, object]:
        cfg = self._read_config()
        if cfg is None:
            return {
                "status": "disabled",
                "profile": "",
                "model_source": "",
                "entities_count": 0,
                "relations_count": 0,
                "claims_count": 0,
                "warnings_count": 0,
                "synthetic_sections": [],
                "error": None,
                "raw": None,
            }

        override_bundle = {
            "langextract_profile_override": profile_override,
            "langextract_prompt_override": prompt_override,
        }
        if not self._langextract.is_enabled(cfg, per_doc_override=override_bundle):
            return {
                "status": "disabled",
                "profile": "",
                "model_source": getattr(cfg.retrieval, "langextract_model_source", "gatherer"),
                "entities_count": 0,
                "relations_count": 0,
                "claims_count": 0,
                "warnings_count": 0,
                "synthetic_sections": [],
                "error": None,
                "raw": None,
            }

        profile = (profile_override or "").strip() or getattr(
            cfg.retrieval,
            "langextract_profile_default",
            "generic_entities_v1",
        )
        model_source = getattr(cfg.retrieval, "langextract_model_source", "gatherer")
        timeout = int(getattr(cfg.retrieval, "langextract_timeout_sec", 20))
        max_chars = int(getattr(cfg.retrieval, "langextract_max_chars", 12000))
        max_facts = int(getattr(cfg.retrieval, "langextract_max_synthetic_facts", 200))

        return self._langextract.extract(
            text=text,
            provider=cfg.provider,
            profile=profile,
            prompt_override=prompt_override,
            timeout=timeout,
            model_source=model_source,
            max_chars=max_chars,
            max_synthetic_facts=max_facts,
        )

    def _append_langextract_sections(self, text: str, sections: list[str]) -> str:
        if not sections:
            return text
        body = "\n\n".join(section for section in sections if section.strip())
        if not body:
            return text
        clean_text = text.rstrip()
        return (
            f"{clean_text}\n\n{_LANGEXTRACT_SYNTHETIC_BEGIN}\n"
            f"{body}\n{_LANGEXTRACT_SYNTHETIC_END}\n"
        )

    def _apply_langextract_metadata(
        self,
        metadata: dict[str, str],
        result: dict[str, object],
    ) -> None:
        metadata["langextract_status"] = str(result.get("status", "disabled"))
        metadata["langextract_profile"] = str(result.get("profile", ""))
        metadata["langextract_model_source"] = str(result.get("model_source", ""))
        metadata["langextract_entities_count"] = str(result.get("entities_count", 0))
        metadata["langextract_relations_count"] = str(result.get("relations_count", 0))
        metadata["langextract_claims_count"] = str(result.get("claims_count", 0))
        metadata["langextract_warnings_count"] = str(result.get("warnings_count", 0))
        error = str(result.get("error", "") or "").strip()
        if error:
            metadata["langextract_error"] = error
        else:
            metadata.pop("langextract_error", None)

    def _persist_langextract_artifact(
        self,
        doc_id: str,
        title: str,
        result: dict[str, object],
        doc,
    ) -> None:
        status = str(result.get("status", "disabled"))
        if status == "disabled":
            return
        payload = {
            "document_id": doc_id,
            "title": title,
            "created_at": datetime.now(UTC).isoformat(),
            "status": status,
            "profile": result.get("profile"),
            "model_source": result.get("model_source"),
            "model_id": result.get("model_id"),
            "provider": result.get("provider"),
            "entities_count": result.get("entities_count", 0),
            "relations_count": result.get("relations_count", 0),
            "claims_count": result.get("claims_count", 0),
            "warnings_count": result.get("warnings_count", 0),
            "error": result.get("error"),
            "synthetic_sections": result.get("synthetic_sections", []),
            "raw": result.get("raw"),
        }
        try:
            artifact = self._langextract.persist_artifact(doc_id, payload)
            doc.metadata["langextract_artifact_path"] = str(artifact)
            self._store.upsert(doc)
        except Exception as exc:
            doc.metadata["langextract_error"] = str(exc)
            self._store.upsert(doc)

    def _log_langextract_audit(self, doc_id: str, title: str, result: dict[str, object]) -> None:
        status = str(result.get("status", "disabled"))
        details = {
            "document_id": doc_id,
            "title": title,
            "langextract_status": status,
            "langextract_profile": str(result.get("profile", "")),
            "langextract_model_source": str(result.get("model_source", "")),
            "langextract_entities_count": int(result.get("entities_count", 0) or 0),
            "langextract_relations_count": int(result.get("relations_count", 0) or 0),
            "langextract_claims_count": int(result.get("claims_count", 0) or 0),
            "langextract_warnings_count": int(result.get("warnings_count", 0) or 0),
            "langextract_enabled": status != "disabled",
        }
        get_audit_log().log(
            AuditEntry(
                timestamp=datetime.utcnow(),
                action=AuditAction.INGEST,
                details=details,
                success=True,
            )
        )

    def _read_config(self) -> AppConfig | None:
        if self._config_getter is None:
            return None
        try:
            return self._config_getter()
        except Exception:
            return None

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
        meta.setdefault("uploaded_at", datetime.now(UTC).isoformat())
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
                    capture_output=True,
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
                elif elem.tag == f"{namespace}p" and buffer:
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
