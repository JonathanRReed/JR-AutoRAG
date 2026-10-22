# JR-AutoRAG modernization: evidence, experiments, and launch gates

This is an engineering plan, not a claim of state-of-the-art performance. Research reviewed 2026-10-21. Preserve local-first operation and reproducible, independently owned deployments. No hosted model calls or user-corpus uploads are necessary for the local product.

## What the current code actually does

- Hybrid retrieval combines Sentence Transformers embeddings, BM25, literal matching, fusion, and optional cross-encoder reranking. Embeddings are not a new feature.
- ML is an optional installation extra. Without it, semantic retrieval is unavailable; keyword retrieval must remain usable and the degraded state visible.
- Baseline embedder: BGE-base-en-v1.5. Baseline reranker: MS-MARCO MiniLM L6. Some role-model recommendations still list Llama 3.1/3.2 and Qwen 2.5.
- The previous encoder path treated queries and documents identically. New instruction-sensitive models need task-specific formatting, not just model-name replacement.
- Persistence already fingerprints model name and chunking. Modern profiles additionally need immutable revisions and encoding contracts so equally sized but incompatible spaces never mix.
- The previous evidence contract accepted a citation marker alone as support. Numerical and polarity differences were also erased by lexical matching. Fix these before adding more reasoning features.
- The existing offline retrieval-quality benchmark explicitly disables ML. It is a regression benchmark, not evidence that a new embedding model is better.

## Embedding candidates

1. Microsoft Harrier OSS v1 0.6B: initial balanced default candidate (superseded by the measured decision below), MIT, 1024 dimensions, 32K context. Publisher-reported multilingual MTEB v2 69.0. Validate on this product's corpora; do not transfer benchmark superiority automatically. https://huggingface.co/microsoft/harrier-oss-v1-0.6b
2. Google EmbeddingGemma 2: lightweight text configuration, Apache-2.0, 270M text / 740M complete multimodal model, 768 dimensions, 8K context. Disable unused vision/audio encoders. FP16 is unsupported; FP32/BF16 only. This integration does not itself implement image/audio/video retrieval. https://ai.google.dev/gemma/docs/embeddinggemma and https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers
3. Qwen3 Embedding 4B: higher-memory local option, Apache-2.0, 2560 dimensions, conservative 32K upstream limit. CPU FP32 weights alone approach 16GB; not a reasonable universal default. https://huggingface.co/Qwen/Qwen3-Embedding-4B

Pin exact revisions and prompts. Preserve old explicitly configured models. For saved partial configurations, preserve the legacy default. Compare actual retrieval, latency and peak memory before promoting a model as universally better.

## Research-driven priorities

### P0: measurable quality and trustworthy evidence

- Build a reproducible dense/hybrid comparison harness alongside the model-free regression suite. Separate recall, ranking quality, citation correctness, answer correctness, latency and memory. Include exact identifiers, quantities, negation, multilingual paraphrases, distractors, multi-hop and unanswerable queries.
- Use paired comparisons on fixed queries, not anecdotal demos. Avoid training/tuning on the evaluation split. Record model revision, prompts, index settings and corpus hashes.
- Eliminate marker-only evidence acceptance. Numeric/polarity screening is only a conservative lexical filter, not a semantic entailment proof. Future entailment models require separate calibration and refusal evaluation.
- Keep keyword/literal retrieval. A 2026 text/table study found hybrid plus reranking effective and BM25 particularly strong for numerical financial documents; HyDE was not universally helpful. These are that study's results, not reproduced JR-AutoRAG results. https://arxiv.org/abs/2604.01733
- Use TREC RAG's distinction between retrieval and grounded generation to avoid a single opaque quality score. Public benchmark tooling can supplement, not replace, product-specific tests. https://trec-rag.github.io/

### P1: bounded evidence navigation rather than blind repeated search

A 2026 enterprise study exposes search, find, open, and summarize to the model. It reports benefits from agentic retrieval and within-document navigation, but model/backend/corpus differences limit transfer. Implement only with per-query budgets, cancellation, repeated-query detection, scoped document access and preserved original source spans. https://arxiv.org/abs/2605.05538

Candidate experiment: retrieve a passage, open neighboring source spans, search within that document, and stop when the evidence target is met. Compare against the current iterative planner. Never interpret retrieved instructions as tool authority.

### P1: structure-aware ingestion

Preserve table row/column headers, units, page provenance and section hierarchy. Test scanned documents separately from text PDFs. Make parsing diagnostics actionable. Contextual enrichment must remain derived metadata linked to original text; it must never replace source evidence.

### P2: ambitious but opt-in experiments

- Compiled wiki/graph navigation: LLM-Wiki reports multi-hop gains using linked compiled pages and an error book. Test on multi-document tasks, track compilation cost and invalidation, and never let generated summaries masquerade as originals. Not a launch default without ablation. https://arxiv.org/abs/2605.25480
- Multimodal evidence retrieval using EmbeddingGemma 2: requires actual page-image/audio/video ingestion, licensing/resource evaluation, source coordinates and modality-aware UI. Text-only support is not multimodal support.
- Learned routing and adaptive compute: train only on a separate labeled corpus, retain deterministic fallback and strict budgets. More model calls are not automatically higher quality.
- Late-interaction retrieval / genuine late chunking: verify that implementation really pools contextual token embeddings. Do not rename ordinary chunk embeddings as these techniques.

## Bun and product engineering

Use Bun for the web/server parts it supports well: streaming responses without buffering, propagated cancellation, bounded upload handling, deterministic dependency installation, production builds, and repeatable browser-flow tests. Keep Python where the ML ecosystem is materially stronger. A rewrite for tool fashion is not a performance optimization; benchmark the bottleneck first.

## Acceptance gates

- Core and frontend regression tests, type checks, lint and production build pass against final code.
- Actual model inference returns finite nonzero normalized vectors of the declared dimensions; retrieval ordering verified on a fixed public/synthetic fixture. Unit mocks alone do not qualify runtime support.
- Switching models cannot query old vectors; existing configuration and source documents preserved. Failed initialization produces actionable degradation instead of fabricated quality claims.
- Fabricated references, changed numbers, contrary polarity, missing evidence and evidence from another scope are rejected by regression tests.
- Quality claims include exact evaluated dataset/configuration, baseline and limitations. No broad SOTA or launch-ready claim from a small smoke test.
- Hardware-specific and GPU checks are recorded separately. Publishing a branch, merging, deploying, paid APIs and third-party uploads remain subject to their actual authorization.

## First measured diagnostic (not a release claim)

On the versioned 12-query synthetic text fixture, Harrier 0.6B and EmbeddingGemma 2 both ranked the expected document first on all queries. The existing BGE-base configuration had MRR 0.9444 and nDCG@5 0.9583; all three had recall@5 1.0. Approximate CPU query latency in these runs was Harrier 424 ms, Gemma 169 ms, and BGE 84 ms. Results are tiny-set diagnostics, not a statistically established superiority claim. Corpus/fixture hashes and per-query rankings are recorded in `docs/benchmarks/embedding-*-2026-10-21.json`. Qwen4B has not been inference-tested in this memory-limited environment.

Actual Gemma2 loading exposed a required image-processor dependency even with vision/audio encoders disabled. The ML extra therefore includes Sentence Transformers' image dependencies. Modern encoder profiles use runtime device detection; `AUTORAG_EMBEDDING_DEVICE=cpu` forces CPU. FP32 remains the compatibility precision, and batches default to four to limit peak activation memory.

Frontend lifecycle tests on unchanged main also exceeded the 5-second unit-test default and cascaded after aborted React `act` work. Running in isolation with a bounded 15-second per-test budget passed 101 tests. Only that full-DOM lifecycle test file receives the longer budget; application timeout assertions remain unchanged. No automatic retries or skipped tests were added.

## Validation and remaining release gates

The backend regression suite passed 793 tests after the generation-control fix and benchmark checkpoint tests. Ollama chat, streaming chat, and completion now share the same explicit generation-control allowlist. A failing regression demonstrated completion previously discarded those controls.

An initial frontend run passed 105 tests after fixing toast timeout cleanup, disabling animation choreography in transport lifecycle tests, and releasing detached test DOM graphs. A subsequent full run failed from an intermittent connection lifecycle timeout and cascading aborted React act work. An instrumented isolated run of all 83 lifecycle tests passed, but that does not establish resolution. Frontend stability remains an open gate. Type checks and production builds also passed; these are not substitutes for rendered browser testing.

Doctor, installation smoke, evidence-bundle, and client-handoff script tests passed. The installation smoke reused the prepared environment; it does not prove a clean install on every supported platform. The container manifest check passed, but Docker is not installed here, so an actual container build/run remains unverified. Chromium launch is blocked by this execution environment, including its supported escalation attempt; no rendered-browser screenshots or visual QA pass are claimed. Qwen4B, GPU/MPS behavior, and full end-to-end generation with a real local generator remain unverified.

The larger retrieval diagnostic uses 50 deterministically sampled SciFact queries and a 512-document gold-inclusive corpus. It is intentionally labeled a diagnostic subset, not full BEIR. The original BGE profile retains its own native input-length behavior while new profiles use their published context windows, so this compares product configurations rather than isolating architecture alone. Indexing time and memory matter as well as ranking quality. No paid inference service or private user documents are used.

The real-model harness supports an optional local checkpoint outside the repository. Checkpoints are atomically replaced after each batch and validated against model-profile identity, fixture hash, runtime versions, vector validity, and document count before resuming. Indexing time accumulates completed work across resumes; peak RSS and model-load time describe only the final process. Run offline with cached weights when network access is unavailable. Interrupted runs are not reported as completed comparisons.

## Measured default decision

The 50-query / 512-document SciFact diagnostic finished with identical source/fixture hashes across all three evaluated models:

| Configuration | Recall@5 | MRR | nDCG@5 | Mean query ms | Index seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy BGE-base | 0.896 | 0.8110 | 0.8198 | 96 | 245 |
| Harrier 0.6B | 0.936 | 0.8846 | 0.8892 | 467 | 2178 |
| EmbeddingGemma 2 text | 0.936 | 0.9040 | 0.9067 | 231 | 568 |

EmbeddingGemma 2 is the new-install default: on this diagnostic it matches Harrier recall, has slightly higher ranking scores, and materially lower CPU costs. This is a pragmatic product selection, not statistically established universal superiority or full BEIR performance. Harrier remains an alternative; Qwen4B remains an unverified high-memory option. Existing saved configurations remain unchanged.

Gemma's completed run deliberately resumed at 448 documents to verify restart behavior. Its indexing time accumulates completed batches, not discarded work or downtime, and final-process peak RSS is not comparable to a full uninterrupted run's peak. CPU timings were not measured on a dedicated unloaded benchmark host. The retained Harrier result's fingerprint predates its display-label change from default to alternative; revision, prompts, dtype, and encoding behavior are unchanged.

A later full frontend run, with no concurrent model inference, passed all 105 tests in 106 seconds. The earlier timeout remains documented rather than erased by that pass; rendered browser QA remains blocked in this environment.

## Follow-up recovery audit (October 22 UTC)

A new regression reproduced an incremental-index alignment failure: when an
existing collection had lost its dense matrix after an embedding failure, adding
one document encoded only the new chunks but retained all old chunk ordinals.
The resulting vector rows could resolve to the wrong source. Incremental recovery
now re-embeds the retained and new chunks together when the dense matrix is absent
or its row count differs, and rejects incorrect output batch lengths. Regression
coverage checks both missing and empty matrices and preserves vector/source order.
The full verification script passed after this change: 795 backend tests, 114
frontend tests (519 assertions), lint, type checks, evidence/handoff checks, and
the production build. The rerun used HF_HUB_OFFLINE=1 and
TRANSFORMERS_OFFLINE=1 after an earlier run stopped at a blocked model-network
request. This validates the offline regression suite, not fresh model downloads
or real-generator end-to-end behavior.
