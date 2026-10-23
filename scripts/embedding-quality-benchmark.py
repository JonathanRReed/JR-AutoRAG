"""Local real-model diagnostic. Uses only the supplied public/synthetic fixture.

Run one model per process for honest load/RSS measurements. Downloads public
weights if not cached; never sends fixture text to a hosted embedding service.
This small set cannot establish SOTA or semantic faithfulness of generated text.
"""
import argparse
import hashlib
import json
import math
import os
import resource
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))


def save_checkpoint(path, identity, vectors, elapsed):
    import numpy as np
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, identity=identity, vectors=vectors, elapsed=elapsed)
    temporary.replace(path)


def load_checkpoint(path, identity, document_count):
    import numpy as np
    with np.load(path, allow_pickle=False) as state:
        if str(state["identity"].item()) != identity:
            raise ValueError("Checkpoint identity differs from model, runtime, or fixture")
        vectors = state["vectors"]
        elapsed = float(state["elapsed"].item())
    if (vectors.ndim != 2 or not 0 < len(vectors) <= document_count
            or not np.isfinite(vectors).all()
            or (np.linalg.norm(vectors, axis=1) <= 0).any()
            or not math.isfinite(elapsed) or elapsed < 0):
        raise ValueError("Checkpoint vectors or timing are invalid")
    return vectors, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--fixture", type=Path, default=ROOT / "api/tests/fixtures/embedding_quality.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Optional local restart state outside the repository")
    args = parser.parse_args()
    import numpy as np
    from app.embedding_profiles import (
        EMBEDDING_PROFILES,
        embedding_fingerprint,
        encode_query,
        load_profiled_embedder,
    )
    from sentence_transformers import SentenceTransformer

    os.environ["AUTORAG_EMBEDDING_DEVICE"] = "cpu"
    fixture = json.loads(args.fixture.read_text())
    started = time.perf_counter()
    model = (load_profiled_embedder(SentenceTransformer, args.model)
             if args.model in EMBEDDING_PROFILES else SentenceTransformer(args.model, device="cpu", trust_remote_code=False))
    load_seconds = time.perf_counter() - started
    documents = fixture["documents"]
    fixture_hash = hashlib.sha256(args.fixture.read_bytes()).hexdigest()
    runtime_versions = {"sentence_transformers": __import__("sentence_transformers").__version__,
                        "transformers": __import__("transformers").__version__}
    identity = json.dumps([embedding_fingerprint(args.model), fixture_hash, runtime_versions], sort_keys=True)
    batches = []
    prior_seconds = 0.0
    prior_count = 0
    if args.checkpoint and args.checkpoint.exists():
        prior_vectors, prior_seconds = load_checkpoint(args.checkpoint, identity, len(documents))
        batches.append(prior_vectors)
        prior_count = len(prior_vectors)
        print(f"resumed={prior_count}/{len(documents)}", flush=True)
    started = time.perf_counter()
    for offset in range(prior_count, len(documents), 32):
        batches.append(model.encode([doc["text"] for doc in documents[offset:offset + 32]],
                                   convert_to_numpy=True, normalize_embeddings=True, batch_size=4))
        if args.checkpoint:
            save_checkpoint(args.checkpoint, identity, np.vstack(batches), prior_seconds + time.perf_counter() - started)
        print(f"indexed={min(offset + 32, len(documents))}/{len(documents)}", flush=True)
    vectors = np.vstack(batches)
    index_seconds = prior_seconds + time.perf_counter() - started
    rows = []
    for query in fixture["queries"]:
        started = time.perf_counter()
        vector = encode_query(model, [query["text"]], convert_to_numpy=True, normalize_embeddings=True)[0]
        order = np.argsort(-(vectors @ vector), kind="stable")
        ranked = [documents[i]["id"] for i in order]
        relevant = set(query["relevant"])
        rank = next((i for i, item in enumerate(ranked, 1) if item in relevant), None)
        dcg = sum(1 / math.log2(i + 1) for i, item in enumerate(ranked[:5], 1) if item in relevant)
        ideal = sum(1 / math.log2(i + 1) for i in range(1, min(5, len(relevant)) + 1))
        rows.append({"id":query["id"], "top5":ranked[:5], "recall_at_5":len(set(ranked[:5]) & relevant)/len(relevant),
                     "mrr":1/rank if rank else 0, "ndcg_at_5":dcg/ideal, "latency_ms":(time.perf_counter()-started)*1000})
    report = {"model":args.model, "fingerprint":embedding_fingerprint(args.model),
              "fixture_sha256":hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
              "kind":fixture.get("kind", "small_synthetic_real_model_diagnostic_not_sota"), "device":"cpu",
              "document_count":len(documents), "query_count":len(rows),
              "source":fixture.get("source"), "source_archive_sha256":fixture.get("archive_sha256"),
              "runtime_versions": runtime_versions, "resumed_document_count": prior_count,
              "memory_scope": "current process; earlier interrupted processes are not included",
              "load_seconds":load_seconds, "index_seconds":index_seconds,
              "peak_rss_platform_units":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              "summary":{key:sum(row[key] for row in rows)/len(rows) for key in ("recall_at_5","mrr","ndcg_at_5","latency_ms")}, "queries":rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({"model":args.model, **report["summary"]}), flush=True)


if __name__ == "__main__":
    main()
