"""Security regression tests for full-query cache scoping."""

from __future__ import annotations

from app.core.orchestrator import Orchestrator
from app.core.persistence import DiskQueryCache, QueryCacheConfig
from app.core.query_mode import QueryMode


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
    grounded_scope = orchestrator._query_cache_scope(
        document_ids=["doc-a"],
        cache_scope="user-a",
        query_mode=QueryMode.GROUNDED,
    )
    open_domain_scope = orchestrator._query_cache_scope(
        document_ids=["doc-a"],
        cache_scope="user-a",
        query_mode=QueryMode.OPEN_DOMAIN,
    )

    assert history_scope != other_history_scope
    assert grounded_scope != open_domain_scope
    assert orchestrator._query_cache_scope() is None
