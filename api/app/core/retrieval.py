"""Hybrid retrieval engine (skeleton implementation)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .documents import DocumentStore, Document


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
        self._doc_ids: List[str] = []

    def build(self) -> None:
        corpus = [doc.text for doc in self._docs.list()]
        self._doc_ids = [doc.id for doc in self._docs.list()]
        if corpus:
            self._matrix = self._vectorizer.fit_transform(corpus)
        else:
            self._matrix = None

    def query(self, text: str, top_k: int = 5) -> List[RetrievalResult]:
        if self._matrix is None:
            self.build()
        if self._matrix is None or not text.strip():
            return []
        query_vec = self._vectorizer.transform([text])
        scores = cosine_similarity(query_vec, self._matrix).flatten()
        idx_scores: List[Tuple[int, float]] = sorted(
            [(idx, float(score)) for idx, score in enumerate(scores)],
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_k]
        id_to_doc = {doc.id: doc for doc in self._docs.list()}
        results: List[RetrievalResult] = []
        for idx, score in idx_scores:
            if idx < len(self._doc_ids):
                doc_id = self._doc_ids[idx]
                doc = id_to_doc.get(doc_id)
                if doc:
                    results.append(RetrievalResult(document=doc, score=score))
        return results
