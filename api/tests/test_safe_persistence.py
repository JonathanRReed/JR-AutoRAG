"""Safe persistence regression tests for issue #135.

Pickle is used only to construct adversarial fixtures; loaders must never execute it.
"""

from __future__ import annotations

import ast
import copy
import inspect
import json
import pickle
from unittest.mock import patch

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

from app.core import persistence as module
from app.core.chunking import Chunk
from app.core.hierarchy import DocumentTree, HierarchyNode
from app.core.persistence import (
    DiskQueryCache,
    IndexMetadata,
    IndexPersistence,
    QueryCacheConfig,
)


def execute_legacy_payload():
    raise AssertionError("Legacy pickle executed")


class LegacyPayload:
    def __reduce__(self):
        return execute_legacy_payload, ()


@pytest.fixture
def no_pickle():
    with (
        patch("pickle.load", side_effect=AssertionError("pickle.load called")),
        patch("pickle.loads", side_effect=AssertionError("pickle.loads called")),
        patch("pickle.dump", side_effect=AssertionError("pickle.dump called")),
        patch("pickle.dumps", side_effect=AssertionError("pickle.dumps called")),
    ):
        yield


@pytest.fixture
def metadata():
    return IndexMetadata("v1", "cfg", 2, 1.0, "test-model")


@pytest.fixture
def chunks():
    return [
        ("doc-a", Chunk("alpha", 0, 0, 5, {"heading": "A"})),
        ("doc-b", Chunk("beta", 0, 0, 4)),
    ]


@pytest.fixture
def query_cache(tmp_path):
    cache = DiskQueryCache(QueryCacheConfig(tmp_path / "query.db"))
    try:
        yield cache
    finally:
        cache.close()


def overwrite_query(cache, payload):
    cache.set("question", {"answer": "seed"}, corpus_version="v1")
    cache._get_conn().execute("UPDATE query_cache SET result = ?", (payload,))
    cache._get_conn().commit()


def test_query_roundtrip_json_and_hits(query_cache, no_pickle):
    value = {"answer": "ok", "chunks": [{"id": "a", "score": 0.25}], "optional": None}
    query_cache.set("question", value, corpus_version="v1")
    assert query_cache.get("question", corpus_version="v1") == value
    assert query_cache.get_last_event().hit
    stored = query_cache._get_conn().execute("SELECT result FROM query_cache").fetchone()[0]
    assert json.loads(stored) == value


@pytest.mark.parametrize(
    "payload",
    [
        b'{"',
        b"[]",
        b"null",
        b'{"score": NaN}',
        b'{"nested": [Infinity]}',
        b'{"answer": "a", "answer": "b"}',
        b"\xff\xfe",
    ],
)
def test_query_malformed_is_miss_without_counting_hit(query_cache, payload):
    overwrite_query(query_cache, payload)
    with patch("pickle.loads", side_effect=AssertionError("unpickled")):
        assert query_cache.get("question", corpus_version="v1") is None
    assert not query_cache.get_last_event().hit
    assert query_cache.get_last_event().reason == "invalid_data"
    assert query_cache.stats()["total_hits"] == 0


def test_query_legacy_is_miss(query_cache):
    payload = pickle.dumps(LegacyPayload())
    overwrite_query(query_cache, payload)
    with patch("pickle.loads", side_effect=AssertionError("unpickled")) as loader:
        assert query_cache.get("question", corpus_version="v1") is None
        loader.assert_not_called()


def test_query_refuses_invalid_writes(query_cache):
    with pytest.raises(ValueError):
        query_cache.set("question", {"score": float("nan")})
    with pytest.raises(ValueError):
        query_cache.set("question", {"object": object()})
    assert query_cache.stats()["total_entries"] == 0


def test_dense_roundtrip_restores_chunk_objects(tmp_path, metadata, chunks, no_pickle):
    store = IndexPersistence(tmp_path)
    vectors = np.array([[0.25, 0.75], [-0.5, 0.5]], dtype=np.float32)
    store.save_dense_index("dense", vectors, chunks, metadata)
    loaded_vectors, loaded_chunks, loaded_metadata = store.load_dense_index("dense")
    np.testing.assert_array_equal(loaded_vectors, vectors)
    assert loaded_chunks == chunks
    assert loaded_metadata == metadata
    assert (tmp_path / "dense_chunks.json").exists()
    assert not (tmp_path / "dense_chunks.pkl").exists()


@pytest.mark.parametrize(
    "vectors",
    [
        np.array([1.0, 2.0]),
        np.empty((2, 0)),
        np.ones((3, 2)),
        np.array([[float("nan"), 0], [0, 1]]),
        np.array([[float("inf"), 0], [0, 1]]),
        np.array([["a", "b"], ["c", "d"]]),
    ],
)
def test_dense_malformed_array_is_miss(tmp_path, metadata, chunks, vectors):
    store = IndexPersistence(tmp_path)
    store.save_dense_index("dense", np.ones((2, 2)), chunks, metadata)
    np.save(tmp_path / "dense_embeddings.npy", vectors, allow_pickle=False)
    assert store.load_dense_index("dense") == (None, None, None)


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data.update(embedding_dimensions=3),
        lambda data: data["chunks"].pop(),
        lambda data: data["chunks"][0]["chunk"].update(index=True),
        lambda data: data["chunks"][0]["chunk"].update(end_char=-1),
        lambda data: data["chunks"][0]["chunk"].update(metadata={"score": float("nan")}),
        lambda data: data.update(format_version=999),
    ],
)
def test_dense_malformed_chunks_is_miss(tmp_path, metadata, chunks, change):
    store = IndexPersistence(tmp_path)
    store.save_dense_index("dense", np.ones((2, 2)), chunks, metadata)
    path = tmp_path / "dense_chunks.json"
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))
    assert store.load_dense_index("dense") == (None, None, None)


def test_dense_rejects_legacy_and_numpy_objects(tmp_path, metadata, chunks):
    store = IndexPersistence(tmp_path)
    np.save(tmp_path / "dense_embeddings.npy", np.ones((2, 2)))
    (tmp_path / "dense_metadata.json").write_text(json.dumps(metadata.to_dict()))
    payload = pickle.dumps(LegacyPayload())
    (tmp_path / "dense_chunks.pkl").write_bytes(payload)
    with patch("pickle.load", side_effect=AssertionError("unpickled")):
        assert store.load_dense_index("dense") == (None, None, None)
    store.save_dense_index("dense", np.ones((2, 2)), chunks, metadata)
    (tmp_path / "dense_chunks.json").write_bytes(payload)
    assert store.load_dense_index("dense") == (None, None, None)
    store.save_dense_index("dense", np.ones((2, 2)), chunks, metadata)
    np.save(tmp_path / "dense_embeddings.npy", np.array([LegacyPayload()], dtype=object))
    with patch("pickle.load", side_effect=AssertionError("unpickled")):
        assert store.load_dense_index("dense") == (None, None, None)


@pytest.fixture
def graph():
    return {
        "entities": {
            "alpha": {"name": "Alpha", "type": "concept", "description": "", "mentions": ["0"]},
            "beta": {"name": "Beta", "type": "other", "description": "", "mentions": ["1"]},
        },
        "relationships": [
            {"source": "Alpha", "target": "Beta", "relation": "related_to", "weight": 0.5, "chunk_ids": ["0"]}
        ],
        "communities": [{"id": 0, "entities": ["alpha", "beta"], "summary": "summary"}],
        "chunk_documents": {"0": "doc-a", "1": "doc-b"},
    }


@pytest.fixture
def trees():
    root = HierarchyNode("root", 0, "Root", "", "Root summary", children=["leaf"])
    leaf = HierarchyNode("leaf", 1, "Leaf", "", "Leaf summary", parent_id="root", chunk_ids=["0"])
    return {"doc-a": DocumentTree("root", {"root": root, "leaf": leaf}, "doc-a").to_dict()}


@pytest.mark.parametrize("kind", ["graph", "trees"])
def test_structured_index_roundtrip(tmp_path, metadata, graph, trees, kind, no_pickle):
    store = IndexPersistence(tmp_path)
    value = graph if kind == "graph" else trees
    path = getattr(store, f"save_{kind}")("index", value, metadata)
    assert path.suffix == ".json"
    loaded, loaded_metadata = getattr(store, f"load_{kind}")("index")
    assert loaded == value
    assert loaded_metadata == metadata


@pytest.mark.parametrize("kind", ["graph", "trees"])
@pytest.mark.parametrize("payload", [b"[]", b"null", b'{"', b'{"value": NaN}', b"\xff\xfe"])
def test_structured_index_malformed_is_miss(tmp_path, metadata, graph, trees, kind, payload):
    store = IndexPersistence(tmp_path)
    value = graph if kind == "graph" else trees
    path = getattr(store, f"save_{kind}")("index", value, metadata)
    path.write_bytes(payload)
    assert getattr(store, f"load_{kind}")("index") == (None, None)


@pytest.mark.parametrize("kind", ["graph", "trees"])
def test_structured_index_legacy_is_miss(tmp_path, metadata, graph, trees, kind):
    store = IndexPersistence(tmp_path)
    payload = pickle.dumps(LegacyPayload())
    (tmp_path / f"index_{kind}.pkl").write_bytes(payload)
    (tmp_path / f"index_{kind}_metadata.json").write_text(json.dumps(metadata.to_dict()))
    with patch("pickle.load", side_effect=AssertionError("unpickled")):
        assert getattr(store, f"load_{kind}")("index") == (None, None)
    path = getattr(store, f"save_{kind}")("index", graph if kind == "graph" else trees, metadata)
    path.write_bytes(payload)
    assert getattr(store, f"load_{kind}")("index") == (None, None)


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data["entities"]["alpha"].update(type="invalid"),
        lambda data: data["relationships"][0].update(weight=float("inf")),
        lambda data: data["relationships"][0].update(source=[]),
        lambda data: data["entities"]["alpha"].update(mentions="0"),
        lambda data: data["communities"][0].update(id=True),
        lambda data: data.update(chunk_documents=[]),
    ],
)
def test_graph_shape_validation(tmp_path, metadata, graph, change):
    store = IndexPersistence(tmp_path)
    path = store.save_graph("index", graph, metadata)
    bad = copy.deepcopy(graph)
    change(bad)
    path.write_text(json.dumps(bad))
    assert store.load_graph("index") == (None, None)


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data["doc-a"].update(root_id="missing"),
        lambda data: data["doc-a"]["nodes"]["leaf"].update(level=True),
        lambda data: data["doc-a"]["nodes"]["leaf"].update(parent_id="leaf", children=["leaf"]),
        lambda data: data["doc-a"]["nodes"]["root"].update(children=["missing"]),
        lambda data: data["doc-a"]["nodes"]["leaf"].update(chunk_ids=0),
        lambda data: data["doc-a"].update(document_id="another-doc"),
    ],
)
def test_tree_shape_and_cycle_validation(tmp_path, metadata, trees, change):
    store = IndexPersistence(tmp_path)
    path = store.save_trees("index", trees, metadata)
    bad = copy.deepcopy(trees)
    change(bad)
    path.write_text(json.dumps(bad))
    assert store.load_trees("index") == (None, None)


@pytest.mark.parametrize(
    "params",
    [
        [],
        {"k1": "1.5", "b": 0.75, "epsilon": 0.25},
        {"k1": True, "b": 0.75, "epsilon": 0.25},
        {"k1": float("nan"), "b": 0.75, "epsilon": 0.25},
        {"k1": 1.5, "b": 2.0, "epsilon": 0.25},
        {"k1": 1.5, "b": 0.75, "epsilon": -1.0},
    ],
)
def test_sparse_invalid_params_is_miss(tmp_path, metadata, params):
    store = IndexPersistence(tmp_path)
    corpus = [["alpha"], ["beta"]]
    store.save_sparse_index("index", BM25Okapi(corpus), corpus, metadata)
    (tmp_path / "index_bm25.json").write_text(json.dumps(params))
    assert store.load_sparse_index("index") == (None, None, None)


@pytest.mark.parametrize("payload", [b"[]", b'[[]]', b'["alpha"]', b"\xff\xfe", b"{"])
def test_sparse_malformed_corpus_is_miss(tmp_path, metadata, payload):
    store = IndexPersistence(tmp_path)
    corpus = [["alpha"], ["beta"]]
    store.save_sparse_index("index", BM25Okapi(corpus), corpus, metadata)
    (tmp_path / "index_tokenized.json").write_bytes(payload)
    assert store.load_sparse_index("index") == (None, None, None)


def test_sparse_rejects_pickle_in_json_paths(tmp_path, metadata):
    store = IndexPersistence(tmp_path)
    corpus = [["alpha"], ["beta"]]
    store.save_sparse_index("index", BM25Okapi(corpus), corpus, metadata)
    (tmp_path / "index_tokenized.json").write_bytes(pickle.dumps(LegacyPayload()))
    with patch("pickle.load", side_effect=AssertionError("unpickled")):
        assert store.load_sparse_index("index") == (None, None, None)


@pytest.mark.parametrize("value", [[], {"chunk_count": True}, {"created_at": float("nan")}])
def test_metadata_invalid_is_not_valid(tmp_path, metadata, value):
    store = IndexPersistence(tmp_path)
    data = metadata.to_dict()
    if isinstance(value, dict):
        data.update(value)
    else:
        data = value
    (tmp_path / "index_metadata.json").write_text(json.dumps(data))
    assert not store.is_valid("index", "v1", "cfg")


def test_runtime_persistence_has_no_pickle_import_or_calls():
    tree = ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name != "pickle" for alias in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module != "pickle"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert not (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "pickle"
            )


@pytest.mark.parametrize(
    "value",
    [
        {"answer": []},
        {"chunks": {}},
        {"chunks": [{"id": []}]},
        {"chunks": [{"score": True}]},
        {"sources": [0]},
        {"metrics": []},
        {"steps": [0]},
        {"steps": [{"duration_ms": "1"}]},
        {"trace_id": False},
        {"confidence": []},
        {"needs_clarification": "yes"},
    ],
)
def test_query_known_fields_have_valid_shapes(query_cache, value):
    overwrite_query(query_cache, json.dumps(value).encode("utf-8"))
    assert query_cache.get("question", corpus_version="v1") is None
    assert query_cache.get_last_event().reason == "invalid_data"


@pytest.mark.parametrize("kind", ["dense", "sparse", "graph", "trees"])
def test_each_index_rejects_malformed_metadata(tmp_path, metadata, chunks, graph, trees, kind):
    store = IndexPersistence(tmp_path)
    if kind == "dense":
        store.save_dense_index("index", np.ones((2, 2)), chunks, metadata)
        path = tmp_path / "index_metadata.json"
    elif kind == "sparse":
        corpus = [["alpha"], ["beta"]]
        store.save_sparse_index("index", BM25Okapi(corpus), corpus, metadata)
        path = tmp_path / "index_sparse_metadata.json"
    else:
        getattr(store, f"save_{kind}")("index", graph if kind == "graph" else trees, metadata)
        path = tmp_path / f"index_{kind}_metadata.json"
    path.write_text('{"corpus_version": [], "config_hash": "cfg", "chunk_count": 2, "created_at": 1.0}')
    loaded = getattr(store, f"load_{kind}_index" if kind in ("dense", "sparse") else f"load_{kind}")("index")
    assert loaded == ((None, None, None) if kind in ("dense", "sparse") else (None, None))
