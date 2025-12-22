"""Document storage for ingestion and retrieval."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from threading import RLock


@dataclass
class Document:
    id: str
    title: str
    text: str
    metadata: dict[str, str]


class DocumentStore:
    """SQLite-backed document registry with title index."""

    def __init__(self, path: Path | None = None, legacy_json_path: Path | None = None) -> None:
        self._path = Path(path or Path.cwd() / "data" / "documents.db")
        self._legacy_json_path = Path(legacy_json_path) if legacy_json_path else None
        self._lock = RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()
        self._last_refresh = self._get_last_mutation()
        if self._is_empty():
            self._maybe_import_legacy()
        self._docs: dict[str, Document] = {}
        self._title_index: dict[str, str] = {}
        self._load_all()
        self._last_refresh = self._get_last_mutation()

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                title_normalized TEXT NOT NULL UNIQUE,
                text TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_title_norm ON documents(title_normalized)"
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('last_mutation', ?)",
            (time.time(),),
        )
        self._conn.commit()

    def _normalize_title(self, title: str) -> str:
        return title.strip().casefold()

    def _is_empty(self) -> bool:
        cursor = self._conn.execute("SELECT COUNT(*) AS count FROM documents")
        row = cursor.fetchone()
        return row is None or row["count"] == 0

    def _get_last_mutation(self) -> float:
        cursor = self._conn.execute("SELECT value FROM meta WHERE key = 'last_mutation'")
        row = cursor.fetchone()
        if row is None:
            return 0.0
        return float(row["value"])

    def _touch_mutation(self) -> None:
        value = time.time()
        self._conn.execute(
            """
            INSERT INTO meta (key, value)
            VALUES ('last_mutation', ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """,
            (value,),
        )
        self._conn.commit()
        self._last_refresh = value

    def _refresh_if_needed(self) -> None:
        with self._lock:
            current = self._get_last_mutation()
            if current > self._last_refresh:
                self._load_all()
                self._last_refresh = current

    def _maybe_import_legacy(self) -> None:
        if not self._legacy_json_path or not self._legacy_json_path.exists():
            return
        try:
            raw = json.loads(self._legacy_json_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(raw, list):
            return
        for item in raw:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            text = str(item.get("text", ""))
            metadata = item.get("metadata") or {}
            if not title:
                continue
            doc_id = str(item.get("id") or uuid.uuid4())
            normalized = self._normalize_title(title)
            metadata_json = json.dumps(metadata)
            updated_at = time.time()
            self._conn.execute(
                """
                INSERT OR REPLACE INTO documents
                (id, title, title_normalized, text, metadata_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (doc_id, title, normalized, text, metadata_json, updated_at),
            )
        self._conn.commit()
        self._touch_mutation()

    def _load_all(self) -> None:
        self._docs = {}
        self._title_index = {}
        cursor = self._conn.execute(
            "SELECT id, title, text, metadata_json, title_normalized FROM documents"
        )
        for row in cursor.fetchall():
            metadata = {}
            try:
                metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
            except Exception:
                metadata = {}
            doc = Document(
                id=row["id"],
                title=row["title"],
                text=row["text"],
                metadata=metadata,
            )
            self._docs[doc.id] = doc
            if row["title_normalized"]:
                self._title_index[row["title_normalized"]] = doc.id

    def _write_doc(self, doc: Document) -> None:
        normalized = self._normalize_title(doc.title)
        metadata_json = json.dumps(doc.metadata)
        updated_at = time.time()
        self._conn.execute(
            """
            INSERT INTO documents (id, title, title_normalized, text, metadata_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                title_normalized=excluded.title_normalized,
                text=excluded.text,
                metadata_json=excluded.metadata_json,
                updated_at=excluded.updated_at
            """,
            (doc.id, doc.title, normalized, doc.text, metadata_json, updated_at),
        )
        self._conn.commit()
        self._touch_mutation()

    def add(
        self,
        title: str,
        text: str,
        metadata: dict[str, str] | None = None,
        on_duplicate: str = "replace",
    ) -> Document:
        normalized = self._normalize_title(title)
        if not normalized:
            raise ValueError("Document title cannot be empty")
        with self._lock:
            self._refresh_if_needed()
            existing_id = self._title_index.get(normalized)
            if existing_id:
                if on_duplicate == "reject":
                    raise ValueError(f"Document title already exists: {title}")
                doc_id = existing_id
            else:
                doc_id = str(uuid.uuid4())
            meta = dict(metadata or {})
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            meta.setdefault("created_at", now)
            meta["updated_at"] = now
            doc = Document(id=doc_id, title=title, text=text, metadata=meta)
            try:
                self._write_doc(doc)
            except sqlite3.IntegrityError:
                if on_duplicate == "reject":
                    raise
                self._refresh_if_needed()
                existing_id = self._title_index.get(normalized)
                if not existing_id:
                    raise
                doc.id = existing_id
                self._write_doc(doc)
            self._docs[doc.id] = doc
            self._title_index[normalized] = doc.id
            return doc

    def list(self) -> list[Document]:
        self._refresh_if_needed()
        with self._lock:
            return list(self._docs.values())

    def get(self, doc_id: str) -> Document | None:
        self._refresh_if_needed()
        with self._lock:
            return self._docs.get(doc_id)

    def get_by_title(self, title: str) -> Document | None:
        normalized = self._normalize_title(title)
        if not normalized:
            return None
        self._refresh_if_needed()
        with self._lock:
            doc_id = self._title_index.get(normalized)
            if not doc_id:
                return None
            return self._docs.get(doc_id)

    def upsert(self, doc: Document) -> None:
        normalized = self._normalize_title(doc.title)
        if not normalized:
            raise ValueError("Document title cannot be empty")
        with self._lock:
            self._refresh_if_needed()
            existing_id = self._title_index.get(normalized)
            if existing_id and existing_id != doc.id:
                raise ValueError(f"Document title already exists: {doc.title}")
            meta = dict(doc.metadata or {})
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            meta.setdefault("created_at", now)
            meta["updated_at"] = now
            doc.metadata = meta
            try:
                self._write_doc(doc)
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"Document title already exists: {doc.title}") from exc
            self._docs[doc.id] = doc
            self._title_index[normalized] = doc.id

    def delete(self, doc_id: str) -> None:
        with self._lock:
            self._refresh_if_needed()
            doc = self._docs.get(doc_id)
            if not doc:
                return
            normalized = self._normalize_title(doc.title)
            self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            self._conn.commit()
            self._touch_mutation()
            del self._docs[doc_id]
            if normalized in self._title_index:
                del self._title_index[normalized]

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM documents")
            self._conn.commit()
            self._touch_mutation()
            self._docs = {}
            self._title_index = {}
