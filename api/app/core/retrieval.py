"""Hybrid retrieval engine (skeleton implementation)."""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .documents import Document, DocumentStore


@dataclass
class RetrievalResult:
    document: Document
    score: float


class RetrievalEngine:
    """Very small retrieval engine backed by TF-IDF."""

    def __init__(self, documents: DocumentStore) -> None:
        self._docs = documents
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = None
        # Map row_idx -> (doc_id, chunk_text)
        self._chunk_map: list[tuple[str, str]] = []

    def _chunk_text(self, text: str, chunk_size: int = 800) -> list[str]:
        # Simple paragraph-based chunking
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks = []
        current = []
        current_len = 0
        for p in paragraphs:
            if current_len + len(p) > chunk_size and current:
                chunks.append("\n".join(current))
                current = []
                current_len = 0
            current.append(p)
            current_len += len(p)
        if current:
            chunks.append("\n".join(current))
        return chunks if chunks else [text]

    def build(self) -> None:
        corpus = []
        self._chunk_map = []

        for doc in self._docs.list():
            chunks = self._chunk_text(doc.text)
            for chunk in chunks:
                corpus.append(chunk)
                self._chunk_map.append((doc.id, chunk))

        if corpus:
            self._matrix = self._vectorizer.fit_transform(corpus)
        else:
            self._matrix = None

    def query(self, text: str, top_k: int = 5) -> list[RetrievalResult]:
        if self._matrix is None:
            self.build()
        if self._matrix is None or not text.strip():
            return []

        query_vec = self._vectorizer.transform([text])
        scores = cosine_similarity(query_vec, self._matrix).flatten()

        # Get top indices
        top_indices = scores.argsort()[::-1][:top_k]

        results: list[RetrievalResult] = []
        id_to_doc = {doc.id: doc for doc in self._docs.list()}

        for idx in top_indices:
            score = float(scores[idx])
            if score > 0 and idx < len(self._chunk_map):
                doc_id, chunk_text = self._chunk_map[idx]
                doc = id_to_doc.get(doc_id)
                if doc:
                    # Create a transient document representing this chunk
                    # but keeping the original metadata
                    chunk_doc = Document(
                        id=f"{doc.id}-{idx}", # Unique ID for the chunk
                        title=doc.title,
                        text=chunk_text,
                        metadata=doc.metadata
                    )
                    results.append(RetrievalResult(document=chunk_doc, score=score))

        return results
