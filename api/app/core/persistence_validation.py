"""Validation for disposable persistence artifacts.

Only data-only JSON and numeric NumPy arrays are accepted. Invalid artifacts
are rebuildable; callers must never attempt to migrate them by unpickling.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np


INVALID_DATA_ERRORS = (
    OSError,
    UnicodeError,
    ValueError,
    TypeError,
    KeyError,
    OverflowError,
    RecursionError,
    EOFError,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def validate_json(value: Any, depth: int = 0) -> None:
    require(depth <= 100, "JSON nesting exceeds the persistence limit")
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        require(finite_number(value), "JSON numbers must be finite")
    elif isinstance(value, list):
        for item in value:
            validate_json(item, depth + 1)
    elif isinstance(value, dict):
        for key, item in value.items():
            require(isinstance(key, str), "JSON object keys must be strings")
            validate_json(item, depth + 1)
    else:
        raise ValueError("Unsupported persistence value")


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON constant: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON object key")
        result[key] = value
    return result


def decode_json(raw: str | bytes) -> Any:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    require(isinstance(raw, str), "Expected JSON text")
    data = json.loads(
        raw, parse_constant=_reject_constant, object_pairs_hook=_unique_object
    )
    validate_json(data)
    return data


def encode_json(value: Any) -> str:
    validate_json(value)
    return json.dumps(value, allow_nan=False)


def validate_vector(value: Any) -> list[float]:
    require(isinstance(value, list) and bool(value), "Expected a nonempty vector")
    require(all(finite_number(item) for item in value), "Invalid vector values")
    return [float(item) for item in value]


def validate_metadata(data: Any) -> None:
    require(isinstance(data, dict), "Expected index metadata object")
    required = {"corpus_version", "config_hash", "chunk_count", "created_at"}
    require(required <= data.keys(), "Missing index metadata fields")
    require(data.keys() <= required | {"model_name"}, "Unknown index metadata fields")
    require(isinstance(data["corpus_version"], str), "Invalid corpus version")
    require(isinstance(data["config_hash"], str), "Invalid config hash")
    require(nonnegative_int(data["chunk_count"]), "Invalid chunk count")
    require(
        finite_number(data["created_at"]) and data["created_at"] >= 0,
        "Invalid creation timestamp",
    )
    require(isinstance(data.get("model_name", ""), str), "Invalid model name")


def validate_dense_array(embeddings: Any, chunk_count: int, dimensions: int) -> None:
    require(isinstance(embeddings, np.ndarray), "Expected a NumPy array")
    require(embeddings.ndim == 2, "Expected a two-dimensional embedding matrix")
    require(type(dimensions) is int and dimensions > 0, "Invalid embedding dimensions")
    require(
        embeddings.shape == (chunk_count, dimensions),
        "Embedding rows or dimensions do not match chunks",
    )
    require(embeddings.dtype.kind in "fiu", "Expected real numeric embeddings")
    require(bool(np.isfinite(embeddings).all()), "Non-finite embeddings")


def validate_chunk_records(data: Any) -> list[dict[str, Any]]:
    require(isinstance(data, dict), "Expected dense chunk object")
    require(type(data.get("format_version")) is int and data["format_version"] == 1, "Unknown dense format")
    require(
        type(data.get("embedding_dimensions")) is int and data["embedding_dimensions"] > 0,
        "Invalid embedding dimensions",
    )
    records = data.get("chunks")
    require(isinstance(records, list), "Expected dense chunk list")
    for record in records:
        require(isinstance(record, dict), "Expected chunk record object")
        require(isinstance(record.get("doc_id"), str), "Invalid document ID")
        chunk = record.get("chunk")
        require(isinstance(chunk, dict), "Expected chunk object")
        require(
            {"text", "index", "start_char", "end_char"} <= chunk.keys()
            and chunk.keys() <= {"text", "index", "start_char", "end_char", "metadata"},
            "Invalid chunk fields",
        )
        require(isinstance(chunk["text"], str), "Invalid chunk text")
        for field in ("index", "start_char", "end_char"):
            require(nonnegative_int(chunk[field]), f"Invalid chunk {field}")
        require(chunk["end_char"] >= chunk["start_char"], "Invalid chunk span")
        metadata = chunk.get("metadata")
        require(
            metadata is None
            or (
                isinstance(metadata, dict)
                and all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items())
            ),
            "Invalid chunk metadata",
        )
    return records


def validate_sparse(params: Any, corpus: Any, chunk_count: int) -> None:
    require(isinstance(params, dict), "Expected BM25 parameter object")
    for key, default in (("k1", 1.5), ("b", 0.75), ("epsilon", 0.25)):
        require(finite_number(params.get(key, default)), f"Invalid BM25 {key}")
    require(params.get("k1", 1.5) > 0, "BM25 k1 must be positive")
    require(0 <= params.get("b", 0.75) <= 1, "BM25 b must be in [0, 1]")
    require(params.get("epsilon", 0.25) >= 0, "BM25 epsilon must be nonnegative")
    require(isinstance(corpus, list) and bool(corpus), "Expected nonempty tokenized corpus")
    require(len(corpus) == chunk_count, "Sparse corpus row count mismatch")
    require(
        all(isinstance(doc, list) and all(isinstance(tok, str) for tok in doc) for doc in corpus),
        "Invalid tokenized corpus",
    )
    require(any(corpus), "Tokenized corpus contains no tokens")


def string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def validate_graph(data: Any) -> None:
    require(isinstance(data, dict), "Expected graph object")
    entities = data.get("entities")
    relationships = data.get("relationships")
    communities = data.get("communities")
    documents = data.get("chunk_documents")
    require(isinstance(entities, dict), "Invalid graph entities")
    require(isinstance(relationships, list), "Invalid graph relationships")
    require(isinstance(communities, list), "Invalid graph communities")
    require(
        isinstance(documents, dict)
        and all(isinstance(k, str) and isinstance(v, str) for k, v in documents.items()),
        "Invalid graph chunk documents",
    )
    types = {"person", "organization", "location", "concept", "event", "product", "technology", "other"}
    for name, entity in entities.items():
        require(isinstance(name, str) and isinstance(entity, dict), "Invalid graph entity")
        require(isinstance(entity.get("name"), str), "Invalid entity name")
        require(entity.get("type") in types, "Invalid entity type")
        require(isinstance(entity.get("description", ""), str), "Invalid entity description")
        require(string_list(entity.get("mentions", [])), "Invalid entity mentions")
    for relationship in relationships:
        require(isinstance(relationship, dict), "Invalid relationship object")
        for key in ("source", "target", "relation"):
            require(isinstance(relationship.get(key), str), f"Invalid relationship {key}")
        require(finite_number(relationship.get("weight", 1.0)), "Invalid relationship weight")
        require(string_list(relationship.get("chunk_ids", [])), "Invalid relationship chunk IDs")
    for community in communities:
        require(isinstance(community, dict), "Invalid community object")
        require(nonnegative_int(community.get("id")), "Invalid community ID")
        require(string_list(community.get("entities")), "Invalid community entities")
        require(isinstance(community.get("summary", ""), str), "Invalid community summary")


def validate_trees(data: Any) -> None:
    require(isinstance(data, dict), "Expected document trees object")
    for doc_id, tree in data.items():
        require(isinstance(doc_id, str) and isinstance(tree, dict), "Invalid document tree")
        require(tree.get("document_id") == doc_id, "Tree document ID mismatch")
        root_id, nodes = tree.get("root_id"), tree.get("nodes")
        require(isinstance(root_id, str) and isinstance(nodes, dict), "Invalid tree root or nodes")
        require(root_id in nodes, "Missing root node")
        for node_id, node in nodes.items():
            require(isinstance(node_id, str) and isinstance(node, dict), "Invalid tree node")
            require(node.get("id") == node_id, "Tree node ID mismatch")
            require(nonnegative_int(node.get("level")), "Invalid tree level")
            for key in ("title", "summary"):
                require(isinstance(node.get(key), str), f"Invalid node {key}")
            require(isinstance(node.get("text", ""), str), "Invalid node text")
            parent = node.get("parent_id")
            require(parent is None or isinstance(parent, str), "Invalid parent ID")
            require(string_list(node.get("children", [])), "Invalid node children")
            require(string_list(node.get("chunk_ids", [])), "Invalid node chunk IDs")
            children = node.get("children", [])
            require(len(children) == len(set(children)), "Duplicate child ID")
            if node_id == root_id:
                require(parent is None, "Root cannot have a parent")
            else:
                require(parent in nodes, "Missing parent node")
            for child in children:
                require(child in nodes, "Missing child node")
                require(isinstance(nodes[child], dict), "Invalid child node")
                require(nodes[child].get("parent_id") == node_id, "Inconsistent child parent")
        # Traverse iteratively: a malformed cycle must not hang tree consumers.
        visited: set[str] = set()
        pending = [root_id]
        while pending:
            current = pending.pop()
            require(current not in visited, "Tree cycle or repeated node")
            visited.add(current)
            pending.extend(nodes[current].get("children", []))
        require(visited == set(nodes), "Disconnected tree nodes")
