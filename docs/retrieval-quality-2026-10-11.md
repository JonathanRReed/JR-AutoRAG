# Retrieval quality and evidence integrity, 11 October 2026

## Result and scope

This update fixes demonstrated candidate-selection and evidence-integrity bugs
in the existing local-first architecture. It adds no dependencies, providers,
credentials, model downloads, or document transmission. It does not establish
state-of-the-art performance on an external benchmark.

Baseline: `3253a80b8fcc4b8856aa383530f4ef5c64fdeac3`. The same driver, fixtures,
locked Python environment and cloud CPU were used for both revisions. Raw
reports record the fixture and source-file SHA-256 hashes:

- [Baseline report](benchmarks/retrieval-baseline-2026-10-11.json)
- [Improved report](benchmarks/retrieval-improved-2026-10-11.json)
- [Authored fixture](../api/tests/fixtures/retrieval_quality.json)
- [Benchmark driver](../scripts/retrieval-quality-benchmark.py)

The improved report was captured before committing the patch, so its checkout
commit is the baseline; its source-file hashes identify the changed code.

## Measured comparison

The 16 purpose-built cases contain 13 answerable and three no-evidence queries:
keywords, identifiers, numbers, phrases, selected-document scope with 25
distractors, copied evidence, title-only matches, a metadata boost, requested
budgets, multiple sources, and abstention. Dense models, neural reranking and
generation are disabled. Unsafe model-enabled fixture controls are rejected.

| Metric | Baseline | Improved |
| --- | ---: | ---: |
| Mean distinct-evidence recall@k, answerable cases | 0.4835 | 0.9780 |
| Mean reciprocal rank, answerable cases | 0.6154 | 1.0000 |
| Mean duplicate-aware nDCG@k, answerable cases | 0.5215 | 1.0000 |
| Answer-point coverage in retrieved evidence | 0.4835 | 0.9780 |
| No-evidence accuracy, three negative cases | 2/3 | 3/3 |
| Exact source-span fidelity, returned chunks | 0% | 100% |
| Direct chunker round trips, four strategies × seven inputs | 5/28 | 28/28 |

Equivalent copied passages share one evidence group, so copies cannot inflate
recall or nDCG. The ideal nDCG ranking is limited to requested k. One case has
seven relevant notices and k=5, limiting its recall/answer-point coverage to
5/7 even with ideal ranking. No-answer queries are excluded from answerable
averages and scored separately.

Answer-point coverage means a required string is present in retrieved context.
It is **not generated-answer accuracy, claim entailment, or faithfulness**.
Five additional tests exercise the real no-provider orchestrator, gatherer,
compression, citations and SQLite answer cache. The first four fail on the baseline
and all five pass with the changes: a scoped launch fact, substring-only abstention,
an update during generation, changed retrieval configuration, and an update during
cache lookup. This verifies
the deterministic context-summary path, not a neural generator.

### CPU cost

Five warmed repetitions of the isolated sparse stage, synthetic 51-token
chunks, rare three-term query, title/heading/proximity boosts enabled:

| Corpus | Baseline global median | Improved global median | Improved selected-scope median |
| --- | ---: | ---: | ---: |
| 10,000 chunks | 4.092 ms | 6.464 ms | 2.635 ms |
| 50,000 chunks | 25.183 ms | 31.500 ms | 13.552 ms |

Scope selects every thousandth chunk. The baseline primitive has no pre-budget
scope parameter, so the scoped column has no direct baseline equivalent.
Correctly ranking all boost-eligible candidates costs more than boosting only
an already truncated top-k. Reusing indexed tokens and expanding only eligible
metadata/proximity matches avoids re-tokenizing every passage. These are
single-machine sparse-stage timings, not end-to-end p95/SLA, concurrency,
indexing, or model-inference measurements.

## Runtime changes

1. **Scope before candidate budgets.** Float32 dense, BM25/fallback and literal
   retrieval select candidates within the requested document scope. A final
   scope check remains. Out-of-scope passages are excluded before reranking.
   Binary retrieval retains the store's pre-budget scope and checks scope/source
   identity before reranking and result construction.
2. **Rank boosts before cutoff.** Title, heading and proximity scores participate
   in sparse candidate ranking. Title-only matches can become candidates.
   Literal matching uses token/phrase boundaries: `art` no longer cites
   `cartography` merely because it contains that substring.
3. **Retain alternatives.** Reranking and recency keep the candidate pool until
   final selection; diversity widens the pool to at least four times requested
   k. MMR normalizes relevance inside its objective while retaining original
   returned scores. Nonpositive k returns no evidence. Valid reranker pairs
   retain their original candidate identities after an invalid ordinal is skipped.
4. **Exact indexed-source spans.** Fixed, semantic, recursive and `late` splitters
   preserve original text coordinates, including repeated text, internal
   whitespace and CRLF. Separator-free hard splits are bounded. Nonpositive chunk sizes are rejected. Useful overlap
   remains; contained tails, including those created by whitespace trimming,
   are omitted. Empty/whitespace text yields an empty span at zero.
5. **Coherent mutation boundaries.** Serialized, nested writer generations cover
   build, incremental replacement, configuration, cache clearing and loading.
   Float32 discovery/reranking/finalization detect generation changes, retry once
   and then fail closed. Binary mutations include binary-store rebuilds and
   fail closed while updating. Failed writers clear partial ordinal-dependent
   state rather than exposing mixed passages and tokens. Empty updates refresh
   metadata ordinals and invalidate answer-cache versions.
6. **Source and cache identity.** Pending source edits fail closed through source
   hashes and exact slice checks. Persisted metadata describes immutable indexed
   source snapshots; saving refuses changed live sources, and loading validates
   source spans. The source-span version and content fingerprint reject old
   caches, triggering a one-time rebuild without deleting source documents.
   Durable document revisions and in-progress generation keys invalidate answer
   caches; configuration identity also enters disk keys. An answer generated
   across a corpus update is not cached under the new version. Shared float/binary
   query-vector caches include embedding model, backend and deployment identity.

Scope and source checks are integrity/privacy defenses, not a complete access
control or prompt-injection proof. A query already generating during an update
can finish with its earlier evidence; it is prevented from polluting the newer
answer cache. Continuously changing indexes may temporarily return no evidence.

## Verification

The original core/dev API suite passed **495 tests**. The updated suite passed
**595 tests**, including deterministic SQLite/BM25 tests, numerical dense and
reranker boundaries, in-process binary-store cases, faulted writers, actual
background rebuilds and no-provider answers. An independent reviewer supplied
nine additional edge-case reproductions; the fixed candidate passes them.

Core/dev dependencies came from the existing frozen uv lockfile. Cloud SOCKS
proxy environment variables cause five unrelated HTTPX provider tests to fail
without optional `socksio`; the offline/stubbed suite was run with those variables
unset rather than adding a dependency. ML extras and live-provider/model quality
were not tested. `bun run verify` passed API lint/tests, evidence and handoff
regressions, architecture checks, TypeScript, 12 frontend tests and the web build
(824.96 KB JavaScript, 97.55 KB CSS). The local release gate also passed with
`--skip-container-smoke`: doctor/install smoke, container manifest, secret scan,
high-level Bun audit/SBOM receipts, and aggregate verification. Docker is absent
from this cloud worker, so actual image build/runtime smoke must still pass in CI.
The audit required the normal HTTPS proxy; only SOCKS `ALL_PROXY` was removed
for the successful aggregate gate.

Reproduce from the repository root:

```bash
uv sync --frozen --project api --all-groups
uv run --project api python scripts/retrieval-quality-benchmark.py --performance
uv run --project api pytest api/tests/core/test_retrieval_candidate_selection.py \
  api/tests/core/test_chunking_source_spans.py api/tests/core/test_chunking_index_migration.py \
  api/tests/core/test_scoped_answer_quality.py api/tests/core/test_binary_evidence_coherency.py \
  api/tests/core/test_retrieval_quality_benchmark.py -q
bun run verify
```

Compare a detached baseline worktree with the same driver and fixture:

```bash
git worktree add --detach ../jr-autorag-baseline 3253a80b8fcc4b8856aa383530f4ef5c64fdeac3
uv run --project api python scripts/retrieval-quality-benchmark.py \
  --api-root ../jr-autorag-baseline/api --performance --output baseline.json
```

## What advanced labels currently mean

- **Late chunking:** window splitting only; the engine calls `.encode(chunk_texts)`
  independently. True late chunking encodes full-document tokens before pooling
  per-chunk vectors. The [paper](https://arxiv.org/abs/2409.04701) and
  [official Jina implementation](https://github.com/jina-ai/late-chunking) describe
  that distinction.
- **Matryoshka:** `matryoshka_dim` is a field, not a wired truncated-search/full-vector
  rescore path. Compatible model training is a prerequisite.
- **ColBERT-style:** current MaxSim uses generic MiniLM token embeddings by default,
  not a trained ColBERT checkpoint. Neither accuracy nor equivalence was measured.
- **Contextual enrichment:** the default title/section/summary/window enrichment is
  heuristic. Anthropic's [official contextual retrieval report](https://www.anthropic.com/engineering/contextual-retrieval)
  evaluates a different model-generated procedure. Ingestion enriches/recombines
  text, so exact offsets here refer to stored indexed text, not original PDF/DOCX
  bytes or page coordinates.
- **Graph/hierarchy:** the float32 query builds an `expanded_indices` set but does
  not consume it for final candidate selection. The separate orchestrator graph
  path and hierarchy context are present; their quality is not established here.
- **Citations/grounding:** ID verification checks citation membership, not factual
  entailment; missing citations can satisfy that check vacuously. The firewall,
  reflection and several evaluation metrics are heuristic. Injection/poison
  detection is defense in depth, not an immunity guarantee.
- **Query planning:** current feature-based fusion weights and query expansion need
  representative ablations; no universal superiority claim is justified.
- **Incremental ingestion:** content hashes and incremental indexing exist; this
  change fixes empty replacement/cache coherency. It does not add scalable posting
  lists, transactional ingest jobs, or original-file provenance mapping.

## Prioritized future roadmap

1. **Representative evaluation first.** Add public/local text-and-table datasets,
   duplicate/synergy evidence groups, negatives, numeric/identifier queries and
   claim-level answer grading. Keep held-out sets distinct from these authored
   regressions. [MIGRASCOPE](https://arxiv.org/abs/2602.21553) motivates measuring
   evidence complementarity; [GaRAGe](https://arxiv.org/abs/2506.07671) motivates
   fine-grained grounding rather than citation-ID scores.
2. **Local model ablations.** Benchmark present embeddings/cross-encoders against
   compatible newer local candidates with recall, entailment, peak RAM and cold/
   warm p95. Official [Sentence Transformers retrieve/rerank guidance](https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
   supports staged candidate pools, while [cross-encoder outputs](https://sbert.net/docs/package_reference/cross_encoder/model.html)
   can be logits rather than calibrated probabilities. [Qwen3-Embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)
   requires appropriate query instructions; [Qwen3-Reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)
   uses generative yes/no scoring and needs an explicit adapter, not a blind model
   name swap. Do not change defaults without workload/hardware evidence.
3. **Learn where retrieval helps.** Test chunk size/overlap, hybrid weights,
   reranking, bounded query expansion and contextual enrichment independently.
   [Text-and-table retrieval benchmarking](https://arxiv.org/abs/2604.01733) and
   [BM25 at scale](https://arxiv.org/abs/2607.26497) report workload-dependent
   tradeoffs; they do not establish a winner for this app.
4. **Grounding and provenance.** Map stored/enriched chunks back to original files,
   verify each factual claim against cited evidence, test contradiction and absent
   support, and include injection/poisoned-document negatives with independent
   graders. Require explicit local privacy policy for any future external calls.
5. **Scale only after profiling.** Measure ingestion/rebuild time, per-query source
   hashing, concurrent mutation behavior, sparse posting lists and memory. Add true
   late pooling, trained late interaction or Matryoshka search behind compatible
   adapters and measured gates. Change vector storage only when scale warrants it.
