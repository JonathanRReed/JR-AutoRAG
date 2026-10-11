"""Behavioral regressions for scoped ranking and evidence diversity.

Core retrieval uses real SQLite documents and BM25. Numerical dense/reranker
fixtures exercise the ranking boundary without model downloads or API calls.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from app.core.cache import CacheManager
from app.core.chunking import ChunkingStrategy
from app.core.documents import DocumentStore
from app.core.hybrid_retrieval import HybridConfig, HybridRetrievalEngine


@pytest.fixture
def make_engine(tmp_path):
    def create(documents, **overrides):
        store = DocumentStore(tmp_path / f"documents-{len(list(tmp_path.iterdir()))}.db")
        docs = [store.add(title=title, text=text, metadata=metadata) for title, text, metadata in documents]
        config = HybridConfig(
            embedding_model="", use_reranking=False,
            chunking_strategy=ChunkingStrategy.FIXED, recency_weight=0.0,
            **overrides,
        )
        engine = HybridRetrievalEngine(store, config, persist_path=None)
        engine._cache_manager = CacheManager()
        engine.build()
        return engine, docs
    return create


@pytest.mark.asyncio
async def test_scope_is_applied_before_sparse_and_literal_candidate_budget(make_engine):
    documents = [(f"Distractor {i}", "Orion launch date launch date is October 4.", {}) for i in range(25)]
    documents.append(("Selected source", "Orion launch date is October 11.", {}))
    engine, docs = make_engine(documents, rerank_top_k=5)

    results = await engine.query("Orion launch date", top_k=1, document_ids=[docs[-1].id])

    assert [result.chunk_text for result in results] == [docs[-1].text]
    assert all(result.chunk_id.startswith(docs[-1].id + "-") for result in results)


@pytest.mark.asyncio
async def test_scope_is_applied_before_dense_candidate_budget(make_engine):
    engine, docs = make_engine([(f"Document {i}", f"Passage number {i}.", {}) for i in range(8)], rerank_top_k=2)

    class NumericalEmbedder:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0] for _ in texts])

    engine._embedder = NumericalEmbedder()
    engine._embeddings = np.array([[1.0, 0.0]] * 7 + [[0.5, 0.5]])
    results = await engine.query("stellar departure", top_k=1, document_ids=[docs[-1].id])

    assert [result.chunk_text for result in results] == [docs[-1].text]


@pytest.mark.asyncio
async def test_out_of_scope_text_never_reaches_reranker(make_engine):
    engine, docs = make_engine([
        ("Outside scope", "Orion launch date launch date is October 4.", {}),
        ("Selected source", "Orion launch date is October 11.", {}),
    ])

    class RecordingReranker:
        def __init__(self):
            self.passages = []

        def predict(self, pairs):
            self.passages.extend(passage for _, passage in pairs)
            return np.ones(len(pairs))

    reranker = RecordingReranker()
    engine._reranker = reranker
    engine._config.use_reranking = True
    results = await engine.query("Orion launch date", top_k=1, document_ids=[docs[1].id])

    assert results
    assert reranker.passages == [docs[1].text]


@pytest.mark.asyncio
async def test_unknown_scope_does_not_fall_back_to_other_documents(make_engine):
    engine, _ = make_engine([("Orion", "Orion launch is October 11.", {})])
    assert await engine.query("Orion launch", document_ids=["missing"]) == []


def test_sparse_metadata_boosts_are_ranked_before_top_k(make_engine):
    documents = [
        ("General notes", "orion " * 20, {}),
        ("Orion", "orion launch schedule", {}),
    ] + [(f"Filler {i}", f"unrelated material {i}", {}) for i in range(8)]
    engine, docs = make_engine(documents)

    ranked = engine._sparse_search("orion", top_k=1, title_boost=20.0)

    assert engine._chunks[ranked[0][0]][0] == docs[1].id


@pytest.mark.parametrize("bm25_available", [True, False])
def test_title_only_match_is_available_to_sparse_ranking(make_engine, bm25_available):
    engine, docs = make_engine([
        ("Zephyr", "The permitted departure date is October 11.", {}),
        ("Other", "Some unrelated content.", {}),
    ])
    if not bm25_available:
        engine._bm25 = None

    ranked = engine._sparse_search("zephyr", top_k=1, title_boost=1.0)

    assert ranked and engine._chunks[ranked[0][0]][0] == docs[0].id


@pytest.mark.asyncio
async def test_empty_incremental_replacement_refreshes_title_ordinals(make_engine):
    engine, docs = make_engine([
        ("Alpha", "Content one.", {}), ("Beta", "Content two.", {}),
    ])
    previous_version = engine.get_corpus_version()
    docs[0].text = ""
    engine._docs.upsert(docs[0])
    engine.index_documents([docs[0]])

    results = await engine.query("beta")

    assert [result.chunk_text for result in results] == [docs[1].text]
    assert engine.get_corpus_version() != previous_version
    assert engine._doc_chunk_indices == {docs[1].id: [0]}


@pytest.mark.asyncio
async def test_failed_rebuild_cannot_publish_mixed_passage_and_token_state(make_engine, monkeypatch):
    engine, docs = make_engine([
        ("Schedule", "Orion launch date is October 4.", {}),
        ("Notes", "Unrelated pottery conventions.", {}),
    ])
    docs[0].text = "Lunar pottery focuses on glaze techniques."
    engine._docs.upsert(docs[0])
    from app.core.chunking import FixedChunker

    class FailingChunker:
        calls = 0

        def chunk(self, text):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("synthetic indexing failure")
            return FixedChunker().chunk(text)

    with monkeypatch.context() as scoped:
        scoped.setattr("app.core.hybrid_retrieval.get_chunker", lambda **kwargs: FailingChunker())
        with pytest.raises(RuntimeError, match="synthetic indexing failure"):
            engine.build()
    assert engine.get_chunk_records() == []
    assert engine._tokenized_corpus == []
    assert engine._bm25 is None
    assert await engine.query("orion launch") == []


@pytest.mark.asyncio
async def test_source_edit_before_background_indexing_cannot_return_stale_span(make_engine):
    from app.core.documents import Document

    engine, docs = make_engine([("Schedule", "Orion launch date is October 4.", {})])
    old_version = engine.get_corpus_version()
    engine._docs.upsert(Document(docs[0].id, docs[0].title, "New source.", dict(docs[0].metadata)))

    assert engine.get_corpus_version() != old_version
    assert await engine.query("orion launch") == []
    assert engine.get_last_cache_info()["source_changed"] == "retrieval_aborted"


@pytest.mark.asyncio
async def test_source_edit_after_final_document_snapshot_is_detected(make_engine, monkeypatch):
    from app.core.documents import Document

    engine, docs = make_engine([("Schedule", "Orion launch date is October 4.", {})])
    original = engine._docs.list
    changed = False

    def edit_after_snapshot():
        nonlocal changed
        snapshot = original()
        if not changed:
            changed = True
            engine._docs.upsert(Document(docs[0].id, docs[0].title, "New source.", dict(docs[0].metadata)))
        return snapshot

    monkeypatch.setattr(engine._docs, "list", edit_after_snapshot)
    assert await engine.query("orion launch") == []
    assert changed


def test_full_rebuild_empty_corpus_resets_all_ordinal_data(make_engine):
    engine, docs = make_engine([("Orion", "Orion launch date is October 4.", {})])
    previous_version = engine.get_corpus_version()
    engine._docs.delete(docs[0].id)
    engine.build()
    assert engine.get_corpus_version() != previous_version
    assert engine._chunks == engine._tokenized_corpus == []
    assert engine._bm25 is engine._embeddings is None
    assert engine._doc_chunk_indices == engine._doc_title_tokens == {}


def test_in_progress_mutation_cannot_reuse_stable_answer_cache_key(make_engine):
    engine, _ = make_engine([("Orion", "Orion launch date is October 4.", {})])
    stable_version = engine.get_corpus_version()
    with engine._mutating_index():
        assert engine.get_corpus_version().startswith("updating:")
        assert engine.get_corpus_version() != stable_version
    assert engine.get_corpus_version() != stable_version


@pytest.mark.asyncio
async def test_embedding_cache_separates_model_identity_and_dimensions(make_engine):
    engine, docs = make_engine([("Orion", "Orion launch date is October 11.", {})])

    class NumericalEmbedder:
        def __init__(self, dimension):
            self.dimension = dimension
            self.calls = 0

        def encode(self, texts, **kwargs):
            self.calls += 1
            return np.ones((len(texts), self.dimension))

    engine._config.embedding_model = "synthetic-model-a"
    engine._embedder = NumericalEmbedder(2)
    engine.build()
    await engine.query("orion launch")
    engine._config.embedding_model = "synthetic-model-b"
    engine._embedder = NumericalEmbedder(3)
    engine.build()
    results = await engine.query("orion launch")
    assert results and results[0].chunk_text == docs[0].text
    assert engine._embedder.calls == 2  # New corpus plus a new query encoding.


def test_literal_search_does_not_treat_substrings_as_evidence(make_engine):
    engine, _ = make_engine([("Maps", "Cartography describes maps.", {})])
    assert engine._literal_search("art", top_k=5) == ([], 0)


@pytest.mark.asyncio
async def test_no_evidence_for_substring_only_query(make_engine):
    engine, _ = make_engine([("Maps", "Cartography describes maps.", {})])
    assert await engine.query("art") == []


def test_reranker_scores_remain_associated_with_valid_candidates(make_engine):
    engine, _ = make_engine([("Orion", "Launch occurs October 11.", {})])

    class NumericalReranker:
        def predict(self, pairs):
            return np.array([0.9] * len(pairs))

    engine._reranker = NumericalReranker()
    assert engine._rerank("launch", [(999, 0.99), (0, 0.5)], top_k=1) == [(0, 0.9)]


@pytest.mark.asyncio
async def test_scope_is_rechecked_when_corpus_rebuilds_before_search(make_engine):
    engine, docs = make_engine([
        ("Outside scope", "Orion launch notice outside the selected corpus.", {}),
        ("Selected source", "Orion launch date is October 11.", {}),
    ])

    class NumericalEmbedder:
        def encode(self, texts, **kwargs):
            return np.array([[1.0, 0.0] for _ in texts])

    engine._embedder = NumericalEmbedder()
    engine._embeddings = np.array([[1.0, 0.0], [1.0, 0.0]])
    rebuilt = False

    def rebuild_at_search_boundary(message, progress):
        nonlocal rebuilt
        if message.startswith("Semantic search") and not rebuilt:
            rebuilt = True
            engine._docs.delete(docs[1].id)
            engine.build()

    results = await engine.query("Orion launch", document_ids=[docs[1].id], on_progress=rebuild_at_search_boundary)
    assert rebuilt and results == []


@pytest.mark.asyncio
async def test_scope_is_rechecked_if_chunk_identity_changes_before_reranking(make_engine):
    engine, docs = make_engine([
        ("Selected source", "Orion launch date is October 11.", {}),
        ("Outside scope", "Orion launch date outside the selected corpus.", {}),
    ])

    class RecordingReranker:
        def __init__(self):
            self.passages = []

        def predict(self, pairs):
            self.passages.extend(passage for _, passage in pairs)
            return np.ones(len(pairs))

    reranker = RecordingReranker()
    engine._reranker = reranker
    engine._config.use_reranking = True

    def rebuild_at_rerank_boundary(message, progress):
        if message.startswith("Reranking candidates"):
            engine._docs.delete(docs[0].id)
            engine.build()

    results = await engine.query("Orion launch", document_ids=[docs[0].id], on_progress=rebuild_at_rerank_boundary)
    assert results == []
    assert reranker.passages == []


@pytest.mark.asyncio
async def test_rebuild_within_same_document_retries_with_exact_new_evidence(make_engine):
    engine, docs = make_engine([("Schedule", "Orion launch date is October 4.", {})], chunk_size=60)
    latest_fact = "Orion launch has been postponed to October 11."
    latest = "Lunar pottery conventions focus on glaze techniques.\n\n" + latest_fact
    rebuilt = False

    def rebuild_after_discovery(message, progress):
        nonlocal rebuilt
        if message.startswith("Fusing results") and not rebuilt:
            rebuilt = True
            docs[0].text = latest
            engine._docs.upsert(docs[0])
            engine.build()

    results = await engine.query("Orion launch", top_k=1, on_progress=rebuild_after_discovery)

    assert rebuilt
    assert [result.chunk_text for result in results] == [latest_fact]
    assert all(latest[result.start_char:result.end_char] == result.chunk_text for result in results)


@pytest.mark.asyncio
async def test_repeated_index_mutation_aborts_instead_of_returning_misranked_evidence(make_engine):
    engine, docs = make_engine([("Orion", "Orion launch date is October 4.", {})])
    rebuilds = 0

    def rebuild_after_discovery(message, progress):
        nonlocal rebuilds
        if message.startswith("Fusing results"):
            rebuilds += 1
            docs[0].text = f"Orion launch has revision {rebuilds}."
            engine._docs.upsert(docs[0])
            engine.build()

    assert await engine.query("Orion launch", on_progress=rebuild_after_discovery) == []
    assert rebuilds == 2
    assert engine.get_last_cache_info()["index_changed"] == "retrieval_aborted"


@pytest.mark.asyncio
async def test_clearing_index_mid_query_retries_from_source(make_engine):
    engine, docs = make_engine([("Orion", "Orion launch date is October 11.", {})])
    cleared = False

    def clear_after_discovery(message, progress):
        nonlocal cleared
        if message.startswith("Fusing results") and not cleared:
            cleared = True
            engine.clear_cache(include_disk=False)

    results = await engine.query("Orion launch", on_progress=clear_after_discovery)
    assert cleared and [result.chunk_text for result in results] == [docs[0].text]


@pytest.mark.asyncio
async def test_loading_cached_index_mid_query_cannot_repoint_scores(make_engine, tmp_path):
    engine, docs = make_engine([
        ("Orion", "Orion launch date is October 11.", {}),
        ("Pottery", "Lunar pottery conventions focus on glaze techniques.", {}),
    ])
    engine._persist_path = str(tmp_path / "indexes")
    engine._embeddings = np.ones((2, 2), dtype=np.float32)
    assert engine.save_index()
    engine.index_documents([docs[0]])  # Orion moves to ordinal 1; old cache has ordinal 0.
    loaded = False

    def load_after_discovery(message, progress):
        nonlocal loaded
        if message.startswith("Fusing results") and not loaded:
            loaded = engine.load_index()

    results = await engine.query("Orion launch", top_k=1, on_progress=load_after_discovery)
    assert loaded and [result.chunk_text for result in results] == [docs[0].text]


@pytest.mark.asyncio
async def test_diversity_can_select_evidence_beyond_initial_top_k(make_engine):
    duplicate = "Orion launch date is October 11."
    alternative = "Orion launch crew training begins October 5."
    engine, docs = make_engine([
        ("Launch summary", duplicate, {}),
        ("Launch copy", duplicate, {}),
        ("Training schedule", alternative, {}),
    ], diversity=0.6)

    results = await engine.query("Orion launch", top_k=2)

    assert len(results) == 2
    assert {result.chunk_text for result in results} == {duplicate, alternative}
    assert any(result.chunk_id.startswith(docs[2].id + "-") for result in results)


def test_mmr_selection_is_invariant_to_positive_score_scale(make_engine):
    engine, _ = make_engine([
        ("A", "Orion launch date is October 11.", {}),
        ("B", "Orion launch date is October 11.", {}),
        ("C", "Crew training uses a flight simulator.", {}),
    ])
    candidates = [(0, 0.90), (1, 0.89), (2, 0.80)]
    selected = engine._apply_diversity_rerank(candidates, 2, 0.6)
    scaled = engine._apply_diversity_rerank([(idx, score * 1000) for idx, score in candidates], 2, 0.6)

    assert [idx for idx, _ in selected] == [0, 2]
    assert [idx for idx, _ in scaled] == [idx for idx, _ in selected]


@pytest.mark.asyncio
async def test_recency_can_promote_a_candidate_before_final_top_k(make_engine):
    engine, docs = make_engine([
        ("Archive", "Orion launch date is October 4.", {"updated_at": "2000-01-01T00:00:00+00:00"}),
        ("Latest", "Orion launch date is October 11.", {"updated_at": datetime.now(UTC).isoformat()}),
    ])
    # SQLite records import/update times; inject controlled publication dates
    # into the scoring cache to isolate the final-selection ordering.
    engine._doc_timestamps[docs[0].id] = datetime(2000, 1, 1, tzinfo=UTC)
    engine._doc_timestamps[docs[1].id] = datetime.now(UTC)

    results = await engine.query("Orion launch date", top_k=1, routing_params={"recency_weight": 5.0})

    assert results[0].chunk_id.startswith(docs[1].id + "-")


@pytest.mark.asyncio
async def test_top_k_larger_than_rerank_pool_is_honored(make_engine):
    engine, _ = make_engine([(f"Document {i}", f"Orion launch notice {i}.", {}) for i in range(7)], rerank_top_k=2)
    assert len(await engine.query("Orion launch", top_k=5)) == 5


@pytest.mark.asyncio
@pytest.mark.parametrize("top_k", [0, -1])
async def test_nonpositive_top_k_returns_no_evidence(make_engine, top_k):
    engine, _ = make_engine([(f"Orion {i}", f"Orion launch notice {i}.", {}) for i in range(3)])
    assert await engine.query("Orion launch", top_k=top_k) == []
