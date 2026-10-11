"""Literal-search prefilter regression without models or network calls."""
import re
from types import SimpleNamespace

import pytest

from app.core import hybrid_retrieval as hybrid
from app.core.chunking import ChunkingStrategy
from app.core.documents import DocumentStore


def make_engine(tmp_path, texts):
    store = DocumentStore(tmp_path / "documents.db")
    for i, text in enumerate(texts):
        store.add(title=f"Source {i}", text=text)
    engine = hybrid.HybridRetrievalEngine(
        store, hybrid.HybridConfig(
            embedding_model="", use_reranking=False,
            chunking_strategy=ChunkingStrategy.FIXED, recency_weight=0.0,
        ), persist_path=None,
    )
    engine.build()
    return engine


def test_absent_phrase_skips_regex_scan_but_keeps_token_matches(tmp_path, monkeypatch):
    texts = ["robotics reference", "launch orion schedule", "Orion launch", "orion launcher"]
    engine = make_engine(tmp_path, texts)
    scanned = []
    original_compile = re.compile

    def compile_spy(pattern):
        compiled = original_compile(pattern)

        def findall(text):
            scanned.append(text)
            return compiled.findall(text)

        return SimpleNamespace(findall=findall)

    monkeypatch.setattr(hybrid, "re", SimpleNamespace(compile=compile_spy, findall=re.findall))
    ranked, total = engine._literal_search("orion launch", 10)

    assert ranked == [(2, 8.0), (1, 2.0), (3, 1.0)]
    assert total == 3
    assert scanned == ["orion launch", "orion launcher"]


@pytest.mark.parametrize("query", ["cat", "CAT", " cat ", "c++", "a.b", "東京", ""])
def test_phrase_prefilter_preserves_boundaries_counts_and_scope(tmp_path, query):
    texts = ["cat cat", "concatenate", "cat_cat", "écat", "C++ and a.b", "東京 東京", "other"]
    engine = make_engine(tmp_path, texts)
    allowed = {0, 2, 3, 4, 5, 6}
    cleaned = query.strip().lower()
    expected = []
    if cleaned:
        pattern = re.compile(r"(?<!\w)" + re.escape(cleaned) + r"(?!\w)")
        tokens = [t for t in engine._tokenize(cleaned) if len(t) > 2]
        for idx, text in enumerate(texts):
            if idx not in allowed:
                continue
            count = len(pattern.findall(text.lower()))
            score = (5.0 + count if count else 0.0) + sum(
                1.0 for token in tokens if token in set(engine._tokenize(text))
            )
            if score:
                expected.append((idx, score))
        expected.sort(key=lambda item: item[1], reverse=True)

    assert engine._literal_search(query, 10, allowed_indices=allowed) == (expected, len(expected))
