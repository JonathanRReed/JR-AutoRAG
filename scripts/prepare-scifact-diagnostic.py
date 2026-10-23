"""Prepare a fixed public-data diagnostic outside the repository.

Source: BEIR SciFact, https://huggingface.co/datasets/BeIR/scifact (CC BY-SA 4.0).
This gold-inclusive 512-document / 50-query subset is easier than the complete
BEIR task. Scores MUST NOT be presented as full BEIR or held-out SOTA scores.
No model outputs influence sampling. The output contains public source text;
keep it outside the code checkout rather than committing dataset copies.
"""
import argparse
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SEED = "jr-autorag-2026-10-21"


def order(identifier):
    return hashlib.sha256((SEED + identifier).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with urllib.request.urlopen(URL, timeout=60) as response:
        archive = response.read(32 * 1024 * 1024 + 1)
    if len(archive) > 32 * 1024 * 1024:
        raise ValueError("Unexpected archive size")
    with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
        # Read only known data members; never extract paths or execute code.
        for name in ("scifact/corpus.jsonl", "scifact/queries.jsonl", "scifact/qrels/test.tsv"):
            if zipped.getinfo(name).file_size > 64 * 1024 * 1024:
                raise ValueError("Unexpected uncompressed member size")
        corpus = {row["_id"]: row for row in map(json.loads, zipped.read("scifact/corpus.jsonl").decode().splitlines())}
        queries = {row["_id"]: row for row in map(json.loads, zipped.read("scifact/queries.jsonl").decode().splitlines())}
        qrels = {}
        for row in csv.DictReader(io.StringIO(zipped.read("scifact/qrels/test.tsv").decode()), delimiter="\t"):
            if int(row["score"]) > 0:
                qrels.setdefault(row["query-id"], []).append(row["corpus-id"])
    selected_queries = sorted(qrels, key=order)[:50]
    gold = {doc for query in selected_queries for doc in qrels[query]}
    pool = gold | set(sorted(set(corpus) - gold, key=order)[:512 - len(gold)])
    result = {
        "kind": "scifact_gold_inclusive_diagnostic_subset_not_full_beir",
        "description": __doc__, "source": URL, "license": "CC-BY-SA-4.0",
        "archive_sha256": hashlib.sha256(archive).hexdigest(), "sampling_seed": SEED,
        "documents": [{"id": doc, "text": corpus[doc]["title"] + "\n\n" + corpus[doc]["text"]} for doc in sorted(pool)],
        "queries": [{"id": query, "text": queries[query]["text"], "relevant": qrels[query]} for query in selected_queries],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({"documents": len(pool), "queries": len(selected_queries), "archive_sha256": result["archive_sha256"]}))


if __name__ == "__main__":
    main()
