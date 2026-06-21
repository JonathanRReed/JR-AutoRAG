"""Security regression tests for full-query cache scoping."""

from __future__ import annotations

from app.core.orchestrator import Orchestrator
from app.core.persistence import DiskQueryCache, QueryCacheConfig


def test_disk_query_cache_varies_by_scope_key(tmp_path):
    """Same normalized query must not hit across different request scopes."""
    cache = DiskQueryCache(QueryCacheConfig(db_path=tmp_path / "query_cache.db"))
    try:
        cache.set(
            " What is the launch code? ",
            {"answer": "secret", "chunks": [{"id": "secret-doc:1"}]},
            corpus_version="v1",
            retrieval_mode=1,
            preset_id="balanced",
            model_ids={"generator": "test-model"},
            scope_key="secret-doc-scope",
        )

        assert cache.get(
            "what IS the launch code?",
            corpus_version="v1",
            retrieval_mode=1,
            preset_id="balanced",
            model_ids={"generator": "test-model"},
            scope_key="public-doc-scope",
        ) is None
        assert cache.get(
            "what IS the launch code?",
            corpus_version="v1",
            retrieval_mode=1,
            preset_id="balanced",
            model_ids={"generator": "test-model"},
            scope_key="secret-doc-scope",
        )["answer"] == "secret"
    finally:
        cache.close()


def test_orchestrator_cache_scope_includes_request_context():
    """Document filters and private chat history must change full-query cache scope."""
    orchestrator = Orchestrator.__new__(Orchestrator)

    base_scope = orchestrator._query_cache_scope(document_ids=["doc-a"], cache_scope="user-a")
    same_scope = orchestrator._query_cache_scope(document_ids=["doc-a"], cache_scope="user-a")
    other_doc_scope = orchestrator._query_cache_scope(document_ids=["doc-b"], cache_scope="user-a")
    history_scope = orchestrator._query_cache_scope(
        document_ids=["doc-a"],
        history=[{"role": "user", "content": "Private project is Orchid"}],
        cache_scope="user-a",
    )
    other_history_scope = orchestrator._query_cache_scope(
        document_ids=["doc-a"],
        history=[{"role": "user", "content": "Private project is Juniper"}],
        cache_scope="user-a",
    )

    assert base_scope == same_scope
    assert base_scope != other_doc_scope
    assert base_scope != history_scope
    assert history_scope != other_history_scope
    assert orchestrator._query_cache_scope() is None


def test_document_mutation_invalidates_in_memory_and_disk_query_caches(tmp_path, monkeypatch):
    """Document delete handlers must drop cached answers that may contain deleted text."""
    from app.core.cache import get_cache_manager
    from app.routers import documents

    disk_cache = DiskQueryCache(QueryCacheConfig(db_path=tmp_path / "query_cache.db"))
    monkeypatch.setattr(documents, "get_disk_query_cache", lambda: disk_cache)

    memory_cache = get_cache_manager().queries
    memory_cache.invalidate_all()
    try:
        memory_cache.set(
            "What is the launch code?",
            {"answer": "SECRET-ALPHA-12345", "chunks": [{"snippet": "SECRET-ALPHA-12345"}]},
            config_hash="cfg",
            corpus_version="v1",
            retrieval_mode=1,
        )
        disk_cache.set(
            "What is the launch code?",
            {"answer": "SECRET-ALPHA-12345", "chunks": [{"snippet": "SECRET-ALPHA-12345"}]},
            corpus_version="v1",
            retrieval_mode=1,
            preset_id="balanced",
            model_ids={"generator": "test-model"},
            scope_key="scope",
        )

        assert memory_cache.get(
            "What is the launch code?",
            config_hash="cfg",
            corpus_version="v1",
            retrieval_mode=1,
        ) is not None
        assert disk_cache.get(
            "What is the launch code?",
            corpus_version="v1",
            retrieval_mode=1,
            preset_id="balanced",
            model_ids={"generator": "test-model"},
            scope_key="scope",
        ) is not None

        documents._invalidate_query_caches_after_document_mutation()

        assert memory_cache.get(
            "What is the launch code?",
            config_hash="cfg",
            corpus_version="v1",
            retrieval_mode=1,
        ) is None
        assert disk_cache.get(
            "What is the launch code?",
            corpus_version="v1",
            retrieval_mode=1,
            preset_id="balanced",
            model_ids={"generator": "test-model"},
            scope_key="scope",
        ) is None
    finally:
        memory_cache.invalidate_all()
        disk_cache.close()
