# JR-AutoRAG Upgrade Plan — 2026

> Supersedes `JR-AutoRAG-Upgrade-Doc-Local-First.md`.
> Grounded in a 2025–2026 state-of-the-art RAG survey and a full read-only
> architecture audit of the current repository.

## 1. Where the project actually is

A codebase audit (123 Python files in `api/app/core` + `api/app/routers`, plus the
Bun/React frontend) shows JR-AutoRAG already has a **surprisingly complete skeleton**:

- Hybrid dense + BM25 + RRF + cross-encoder rerank (`hybrid_retrieval.py`).
- Binary-quantized retrieval variant (`bq_hybrid_retrieval.py`, `binary_vector_store.py`).
- Pluggable OCR router with `off/auto/vision/dedicated/hybrid` policies (`ocr.py`).
- Contextual enrichment, hierarchy/RAPTOR, GraphRAG, memory, HyDE, FLARE, Self-RAG.
- Citation formatting/verification, hallucination firewall, evidence contracts.
- RAGAS-style eval, golden sets, eval gates, A/B testing, corpus health.
- Auth, rate limiting, prompt guard, PII redaction, secrets vault, audit, ACL.
- Tracing, telemetry, cost tracking, trace export/replay, install report.

**But** the audit also found real gaps that matter more than new features:

1. **Dead code.** Many SOTA modules are written and exported but never imported by
   the orchestrator or routers (`advanced_retrieval`, `auto_weights`,
   `retrieval_cascade`, `multi_granularity`, `circuit_breaker`, `loop_budgets`,
   `stage_budgets`, `execution_control`, `batch_processor`, `multimodal`,
   `trace_replay`, `span_citations`, `model_roles`, `feature_flags`). They add
   maintenance and audit surface without delivering value.
2. **`corpus_health.py` is broken.** It pokes `_faiss`, `_documents`, and `.text`
   on chunks that the retrieval engine does not expose in that shape. The health
   dashboard reports wrong status.
3. **Chunking is not wired.** `chunking.py` defines `FIXED/SEMANTIC/RECURSIVE`
   chunkers and a `get_chunker()` factory, but `IngestPipeline._chunk()` uses its
   own inline splitter and ignores `retrieval.chunking_strategy` from config.
4. **Frontend route ordering.** `src/index.ts` registers `/*` before `/api/*`,
   which can shadow API proxy routes depending on Bun's matcher.
5. **`flare.py` `stop()` is a no-op**, called by `orchestrator.stop()`.
6. **Package identity** is still the template name `bun-react-template`.
7. **Pluggable vector store is only half-real.** `VectorStore` ABC exists but the
   primary runtime path uses ad-hoc numpy arrays; the BQ store is the only real
   alternate backend.

The 2025–2026 SOTA baseline (see research summary below) is a **multi-stage
retrieval/orchestration pipeline**: rich ingestion → hybrid retrieval → learned
routing → reranking → grounded generation → continuous evaluation → memory &
observability. JR-AutoRAG has the bones for all of it; the work is
**productization and calibration, not a rewrite**.

## 2. Principles for this upgrade

1. **Local-first stays a hard constraint.** Every subsystem keeps a local-only
   path; cloud is always opt-in via `DeploymentProfile`.
2. **Wire before adding.** Prefer connecting existing dead modules to the runtime
   over writing new ones. Delete what is genuinely unused and off-roadmap.
3. **No new heavy dependencies unless they earn their place.** Qdrant/ColPali/etc.
   are documented as optional next steps, not forced into this pass.
4. **Every change is verifiable** with `ruff`, `pytest`, `bun test`,
   `bun run typecheck`, and `bun run doctor`.
5. **Config-driven, not code-driven.** Behavior switches live in
   `RetrievalDefaults` / `AppConfig`, not in hidden constants.

## 3. SOTA findings that shaped this plan

| Theme | 2025–2026 consensus | JR-AutoRAG status |
| --- | --- | --- |
| Hybrid retrieval | Dense + sparse + RRF + rerank is the default; per-query weights beat global. | Has it; weights are global. |
| Reranking | Cross-encoder over top 20–50; ColBERT/ColPali late-interaction for quality & visual docs. | Cross-encoder yes; ColBERT stubbed; ColPali absent. |
| Contextual retrieval | Anthropic: −49% failed retrievals; −67% with rerank. Cache by content hash. | Implemented but optional, not default. |
| Late chunking | Long-context embedder → pool token reps per chunk window. | Not implemented. |
| Parent-child / small-to-big | Retrieve small children, expand to parents for context. | `hierarchy.py` has blocks, not wired. |
| Binary quantization + Matryoshka | BQ needs oversampling + rescore; Matryoshka truncation for speed. | BQ done without rescore; Matryoshka absent. |
| Query processing | Rewrite, multi-query, HyDE, routing, agentic loops, FLARE, answerability/abstention. | All present; some not on the runtime path. |
| Generation/grounding | Inline citations, claim decomposition, NLI entailment verify, structured output, prompt guarding, canary tokens. | Most present; NLI verifier & canary tokens absent. |
| Evaluation | RAGAS, LLM-as-judge, golden sets, eval gates, nDCG/MRR/Recall@k, drift detection. | Present; not all wired into CI/handoff. |
| Memory/graph | Conversation + episodic memory; GraphRAG global/local search; LightRAG dual-level. | Present; graph not a first-class retrieval mode. |
| Observability | Trace replay, per-query cost, token budgets, drift dashboards. | Present; budgets not enforced by orchestrator. |
| Local models | bge-m3 (dense+sparse+ColBERT), mxbai/nomic Matryoshka, bge-reranker-v2-m3, ColQwen for visual. | Presets exist; bge-m3 not first-class. |
| Security | Layered prompt-injection defense, poisoned-chunk scanning, Presidio PII, knowledge-extraction refusal. | Regex guards only; chunk-side scanning weak. |

Full source list is in the research artifact that produced this plan.

## 4. Phased work

### Phase 0 — Correctness & tech debt (this pass, done)

Low-risk, high-value fixes that make the existing system honest:

- **Fix `corpus_health.py`** to use `get_readiness_snapshot()` / `get_model_status()`
  and read `_chunks` as `list[tuple[str, Chunk]]` instead of poking nonexistent
  `_faiss` / `_documents` / `.text`.
- **Fix `src/index.ts` route ordering** so `/api/*` and `/__api/*` are matched
  before the `/*` catch-all.
- **Wire `chunking.py` into `IngestPipeline._chunk()`** so
  `retrieval.chunking_strategy` (`fixed`/`semantic`/`recursive`) actually selects
  the chunker, with the inline splitter kept as the `fixed` fallback.
- **Implement `flare.py` `stop()`** to cancel active generation.
- **Rename package** from `bun-react-template` to `jr-autorag`.

### Phase 1 — Wire existing dead modules (this pass, done where safe)

- **Contextual enrichment on by default** for new corpora, with a heuristic
  fallback when no local LLM is available, and caching keyed by content hash
  (already present in `EnrichedChunk`).
- **Circuit breakers / stage budgets** documented as the next wiring target; only
  wired in this pass where it is provably safe and tested.

### Phase 2 — Retrieval quality calibration (next pass)

- Per-query `dense_weight` / `sparse_weight` via the learned router.
- BQ oversampling + full-vector rescore.
- `matryoshka_dim` on `HybridConfig` (index truncated, rescore full).
- bge-m3 as a first-class preset emitting dense + sparse + ColBERT vectors.
- MMR diversity rerank (`diversity` field already a placeholder).

### Phase 3 — Modern retrieval backends (optional, next pass)

- `VectorStoreProvider` interface with Qdrant Edge and LanceDB as local backends;
  keep the pure-Python BQ store as the zero-dependency fallback.
- ColBERT/ColPali late-interaction reranker for visually complex/scanned PDFs.
- Late chunking behind a feature flag (requires a long-context embedder).

### Phase 4 — Agentic & generation hardening (optional, next pass)

- `AgenticRAGPlanner` with `max_steps` + `token_budget`, exposed in `TraceLog`.
- FLARE wired into the streaming response path with logprob/entropy uncertainty.
- NLI entailment verifier in the hallucination firewall.
- Canary tokens + poisoned-chunk scanning in the prompt guard.

### Phase 5 — Eval & observability productization (optional, next pass)

- RAGAS package integration (optional) alongside the heuristic evaluator.
- Eval gates wired into `evidence:bundle` and `handoff:gate`.
- A/B test UI in `QualityCockpit`; drift detection dashboard.
- Trace replay endpoint that re-runs a query with the same config/corpus version.

## 5. Local-first model recommendations (reference)

| Role | Local-first | Optional hybrid/cloud |
| --- | --- | --- |
| Embedding | `BAAI/bge-m3` (dense+sparse+ColBERT, 8192 ctx) or `BAAI/bge-base-en-v1.5` | `openai/text-embedding-3-large` |
| Reranker | `BAAI/bge-reranker-v2-m3` (GPU) / `mixedbread-ai/mxbai-rerank-xsmall-v1` (CPU) | Cohere Rerank |
| LLM | Ollama/LM Studio with Qwen2.5, Llama, Mistral GGUF | OpenAI/Anthropic via explicit profile |
| VLM/OCR | Qwen2.5-VL, GOT-OCR2.0, moondream, minicpm-v | — |
| Vector store | Qdrant Edge / LanceDB / Chroma (pure-Python BQ fallback) | — |
| Graph | NetworkX in-memory / SQLite triples | Neo4j (optional) |
| PII | Microsoft Presidio (regex fallback) | — |

## 6. Verification gates

Every phase must keep these green before it lands:

```bash
bun install
bun run api:sync
bun run typecheck          # tsc --noEmit
bun test                   # frontend unit tests
cd api && uv run ruff check app tests
cd api && uv run pytest -q
bun run doctor
```

For client handoff readiness:

```bash
bun run evidence:bundle
bun run handoff:gate -- evidence/install/<timestamp>-install-evidence
```

## 7. What this pass delivered

See the commit history and `CHANGELOG.md` for the exact set. Phase 0 + the safe
subset of Phase 1 are the scope of this upgrade: correctness fixes, wiring, and
dependency/lockfile refresh — no behavior-changing new dependencies.
