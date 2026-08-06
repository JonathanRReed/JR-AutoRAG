"""Tests for CorpusHealthChecker against a realistic retrieval engine shape.

The retrieval engine stores chunks as ``list[tuple[str, Chunk]]`` and exposes a
``get_readiness_snapshot()`` contract. These tests pin the corrected behavior so
the health dashboard no longer pokes nonexistent ``_faiss``/``_documents``/``.text``
attributes.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.core.corpus_health import CorpusHealthChecker


@dataclass
class _Chunk:
    text: str


class _FakeDocs:
    def __init__(self, docs: list[object]) -> None:
        self._docs = docs

    def list(self) -> list[object]:
        return list(self._docs)


class _FakeEngine:
    """Minimal stand-in matching HybridRetrievalEngine's real attributes."""

    def __init__(self, chunks, docs, dense_ready=True) -> None:
        self._chunks = chunks
        self._docs = _FakeDocs(docs)
        self._dense_ready = dense_ready

    def get_readiness_snapshot(self) -> dict:
        return {
            "document_count": len(self._docs.list()),
            "chunk_count": len(self._chunks),
            "dense_ready": self._dense_ready,
            "sparse_ready": True,
            "index_ready": len(self._docs.list()) == 0 or len(self._chunks) > 0,
            "model_status": {"embedding_model": {"status": "ready"}},
        }


def test_stats_with_ready_engine():
    chunks = [
        ("doc-1", _Chunk("one two three four five")),
        ("doc-1", _Chunk("six seven eight nine ten")),
    ]
    engine = _FakeEngine(chunks, [object(), object()], dense_ready=True)
    checker = CorpusHealthChecker(retrieval=engine)
    stats = checker.get_stats()

    assert stats.document_count == 2
    assert stats.chunk_count == 2
    assert stats.index_status == "ready"
    assert stats.embedding_status == "ready"
    assert stats.avg_chunk_size > 0


def test_stats_with_empty_corpus():
    engine = _FakeEngine([], [], dense_ready=False)
    checker = CorpusHealthChecker(retrieval=engine)
    stats = checker.get_stats()

    assert stats.document_count == 0
    assert stats.chunk_count == 0
    assert stats.index_status == "missing"


def test_report_overall_status_is_valid():
    chunks = [("doc-1", _Chunk("alpha beta gamma delta epsilon"))]
    engine = _FakeEngine(chunks, [object()], dense_ready=True)
    checker = CorpusHealthChecker(retrieval=engine)
    report = checker.generate_report()

    assert report.overall_status in {"healthy", "warning", "critical"}
    assert len(report.checks) > 0


def test_stats_fallback_without_snapshot():
    """When the engine lacks get_readiness_snapshot, fall back to private attrs."""

    class _BareEngine:
        def __init__(self) -> None:
            self._chunks = [("d", _Chunk("one two three four"))]
            self._docs = _FakeDocs([object()])
            self._embeddings = [0.1, 0.2]

        def get_model_status(self) -> dict:
            return {"embedding_model": {"status": "ready"}}

    checker = CorpusHealthChecker(retrieval=_BareEngine())
    stats = checker.get_stats()

    assert stats.chunk_count == 1
    assert stats.document_count == 1
    assert stats.index_status == "ready"
    assert stats.embedding_status == "ready"
