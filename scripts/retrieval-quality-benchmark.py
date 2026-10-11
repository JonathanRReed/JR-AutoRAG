"""Offline synthetic regression benchmark, not a state-of-the-art leaderboard.

Run with the repository's locked core/dev Python environment. --api-root lets
the same driver and fixture compare an immutable baseline checkout. No model
downloads, generation, API requests, credentials, or user documents are used.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import io
import json
import math
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def evidence_metrics(retrieved: list[str], groups: list[list[str]], top_k: int) -> dict:
    """Score distinct evidence groups so copied passages cannot inflate recall."""
    if not groups:
        return {"recall_at_k": None, "mrr": None, "ndcg_at_k": None,
                "no_evidence_accuracy": float(not retrieved)}
    seen = set()
    first_rank = None
    dcg = 0.0
    for rank, source_id in enumerate(retrieved[:top_k], 1):
        for group_index, group in enumerate(groups):
            if source_id in group and group_index not in seen:
                seen.add(group_index)
                first_rank = first_rank or rank
                dcg += 1 / math.log2(rank + 1)
                break
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(top_k, len(groups)) + 1))
    return {"recall_at_k": len(seen) / len(groups),
            "mrr": 1 / first_rank if first_rank else 0.0,
            "ndcg_at_k": dcg / ideal if ideal else 0.0,
            "no_evidence_accuracy": None}


async def evaluate_case(case: dict, directory: Path) -> dict:
    from app.core.cache import CacheManager
    from app.core.chunking import ChunkingStrategy
    from app.core.documents import DocumentStore
    from app.core.hybrid_retrieval import HybridConfig, HybridRetrievalEngine

    store = DocumentStore(directory / "documents.db")
    actual_ids = {}
    sources = {}
    distractors = case.get("distractors", {})
    documents = [
        {"id": f"distractor-{i}", "title": f"Distractor {i}", "text": distractors["text"]}
        for i in range(distractors.get("count", 0))
    ] + case["documents"]
    for record in documents:
        doc = store.add(title=record["title"], text=record["text"], metadata={"fixture_id": record["id"]})
        actual_ids[record["id"]] = doc.id
        sources[doc.id] = record["text"]
    settings = {"embedding_model": "", "use_reranking": False,
                "chunking_strategy": ChunkingStrategy.FIXED, "recency_weight": 0.0}
    settings.update(case.get("config", {}))
    if settings.get("embedding_model") or any(settings.get(flag) for flag in ("use_reranking", "use_colbert", "graph")):
        raise ValueError("The offline benchmark rejects model-enabled fixture controls")
    engine = HybridRetrievalEngine(store, HybridConfig(**settings), persist_path=None)
    engine._cache_manager = CacheManager()
    engine.build()
    scope = [actual_ids.get(source_id, source_id) for source_id in case["scope"]] if "scope" in case else None
    start = time.perf_counter()
    results = await engine.query(case["query"], top_k=case["top_k"], document_ids=scope,
                                 routing_params=case.get("routing"))
    elapsed = (time.perf_counter() - start) * 1000
    retrieved = [result.document.metadata["fixture_id"] for result in results]
    valid_spans = sum(
        sources[result.chunk_id.rsplit("-", 1)[0]][result.start_char:result.end_char] == result.chunk_text
        for result in results
    )
    context = "\n".join(result.chunk_text for result in results).casefold()
    points = case["answer_points"]
    covered = sum(point.casefold() in context for point in points)
    return {"id": case["id"], "top_k": case["top_k"], "retrieved_ids": retrieved,
            **evidence_metrics(retrieved, case["expected_evidence_groups"], case["top_k"]),
            "evidence_answer_point_coverage": covered / len(points) if points else None,
            "source_span_fidelity": valid_spans / len(results) if results else None,
            "valid_source_spans": valid_spans, "returned_chunks": len(results),
            "duration_ms": round(elapsed, 3)}


def evaluate_chunk_spans() -> dict:
    from app.core.chunking import get_chunker

    samples = [
        "Alpha paragraph.\n\nBeta paragraph.\n\nGamma paragraph.",
        "  First sentence.\nSecond sentence.  Third sentence.  ",
        "Repeated paragraph.\n\nRepeated paragraph.\n\nLast paragraph.",
        "Header\r\n\r\nA paragraph with café and 東京.\r\nAnother line.",
        "abcdefghij" * 10, "\n\t  ", "",
    ]
    rows = []
    for strategy in ("fixed", "semantic", "recursive", "late"):
        for index, text in enumerate(samples):
            try:
                chunks = get_chunker(strategy, target_size=24, overlap=0).chunk(text)
                valid = all(0 <= chunk.start_char <= chunk.end_char <= len(text)
                            and text[chunk.start_char:chunk.end_char] == chunk.text for chunk in chunks)
                rows.append({"strategy": strategy, "sample": index, "valid": valid})
            except Exception as exc:  # noqa: BLE001 - record a baseline chunker failure, then continue.
                rows.append({"strategy": strategy, "sample": index, "valid": False,
                             "error": f"{type(exc).__name__}: {exc}"})
    return {"valid_cases": sum(row["valid"] for row in rows), "total_cases": len(rows), "cases": rows}


def sparse_performance(sizes=(10000, 50000), repetitions=5) -> list[dict]:
    """Isolated CPU sparse-stage cost, excluding indexing and all ML inference."""
    from app.core.chunking import Chunk
    from app.core.documents import DocumentStore
    from app.core.hybrid_retrieval import HybridConfig, HybridRetrievalEngine
    from rank_bm25 import BM25Okapi

    rows = []
    with tempfile.TemporaryDirectory(prefix="jr-sparse-benchmark-") as directory:
        engine = HybridRetrievalEngine(DocumentStore(Path(directory) / "documents.db"),
                                        HybridConfig(embedding_model="", use_reranking=False), persist_path=None)
        for size in sizes:
            engine._chunks = []
            engine._doc_title_tokens = {}
            for index in range(size):
                text = ("orion launch schedule " if index % 1000 == 0 else "botanical moss shade ") + "reference material " * 24
                engine._chunks.append((f"doc-{index}", Chunk(text=text, index=0, start_char=0, end_char=len(text))))
                engine._doc_title_tokens[f"doc-{index}"] = {"orion"} if index % 1000 == 0 else {"reference"}
            engine._tokenized_corpus = [engine._tokenize(chunk.text) for _, chunk in engine._chunks]
            engine._bm25 = BM25Okapi(engine._tokenized_corpus)
            for scoped in (False, True):
                # The baseline did not expose pre-budget scope, so measure the
                # same global primitive on both revisions and scope separately
                # only where the implementation supports the optional mask.
                if scoped and "allowed_indices" not in engine._sparse_search.__code__.co_varnames:
                    continue
                kwargs = {"title_boost": 0.6, "heading_boost": 0.4, "proximity_weight": 0.5}
                if scoped:
                    kwargs["allowed_indices"] = set(range(0, size, 1000))
                engine._sparse_search("orion launch schedule", 20, **kwargs)
                durations = []
                for _ in range(repetitions):
                    start = time.perf_counter()
                    result = engine._sparse_search("orion launch schedule", 20, **kwargs)
                    durations.append((time.perf_counter() - start) * 1000)
                rows.append({"corpus_chunks": size, "tokens_per_chunk": 51, "scoped": scoped,
                             "repetitions": repetitions, "returned_candidates": len(result),
                             "median_ms": round(statistics.median(durations), 3),
                             "max_ms": round(max(durations), 3)})
    return rows


def source_identity(api_root: Path) -> dict:
    hashes = {name: hashlib.sha256((api_root / "app" / "core" / name).read_bytes()).hexdigest()
              for name in ("chunking.py", "hybrid_retrieval.py", "bq_hybrid_retrieval.py",
                           "documents.py", "orchestrator.py")}
    commit = subprocess.run(["git", "-C", str(api_root.parent), "rev-parse", "HEAD"],
                            text=True, capture_output=True, check=False).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(api_root.parent), "diff", "--quiet"],
                           check=False).returncode != 0
    return {"checkout_commit": commit, "tracked_worktree_dirty": dirty, "source_sha256": hashes,
            "benchmark_driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


async def run(fixture: Path, api_root: Path, performance: bool) -> dict:
    cases = json.loads(fixture.read_text())["cases"]
    results = []
    with tempfile.TemporaryDirectory(prefix="jr-quality-benchmark-") as directory:
        for case in cases:
            case_dir = Path(directory) / case["id"]
            case_dir.mkdir()
            results.append(await evaluate_case(case, case_dir))
    means = {}
    for key in ("recall_at_k", "mrr", "ndcg_at_k", "evidence_answer_point_coverage", "no_evidence_accuracy"):
        values = [row[key] for row in results if row[key] is not None]
        means[key] = round(statistics.mean(values), 6) if values else None
    chunk_count = sum(row["returned_chunks"] for row in results)
    valid_spans = sum(row["valid_source_spans"] for row in results)
    means["source_span_fidelity"] = valid_spans / chunk_count if chunk_count else None
    return {"schema_version": 1, "benchmark": "synthetic_retrieval_regressions_v1",
            "limitations": "Purpose-built regression cases; no external benchmark or neural/generative quality claim.",
            "model_calls": 0, "fixture_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
            **source_identity(api_root), "case_count": len(results), "aggregate": means,
            "cases": results, "chunk_source_spans": evaluate_chunk_spans(),
            "sparse_performance": sparse_performance() if performance else []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-root", type=Path, default=ROOT / "api")
    parser.add_argument("--fixture", type=Path, default=ROOT / "api" / "tests" / "fixtures" / "retrieval_quality.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--performance", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.api_root.resolve()))
    with contextlib.redirect_stdout(io.StringIO()):
        report = asyncio.run(run(args.fixture, args.api_root, args.performance))
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
