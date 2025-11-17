"""Document storage for ingestion and retrieval."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from threading import RLock
from typing import Dict, List, Optional
import json
import uuid


@dataclass
class Document:
    id: str
    title: str
    text: str
    metadata: Dict[str, str]


class DocumentStore:
    """Filesystem-backed document registry."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path or Path.cwd() / "data" / "documents.json")
        self._lock = RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._docs: Dict[str, Document] = {}
        if self._path.exists():
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._docs = {item["id"]: Document(**item) for item in raw}

    def _persist(self) -> None:
        payload = [asdict(doc) for doc in self._docs.values()]
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def add(self, title: str, text: str, metadata: Optional[Dict[str, str]] = None) -> Document:
        with self._lock:
            doc = Document(id=str(uuid.uuid4()), title=title, text=text, metadata=metadata or {})
            self._docs[doc.id] = doc
            self._persist()
            return doc

    def list(self) -> List[Document]:
        with self._lock:
            return list(self._docs.values())

    def get(self, doc_id: str) -> Optional[Document]:
        with self._lock:
            return self._docs.get(doc_id)

    def upsert(self, doc: Document) -> None:
        with self._lock:
            self._docs[doc.id] = doc
            self._persist()

    def delete(self, doc_id: str) -> None:
        with self._lock:
            if doc_id in self._docs:
                del self._docs[doc_id]
                self._persist()
