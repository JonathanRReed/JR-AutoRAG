# JR-AutoRAG Upgrade Plan — 2026

> Supersedes the original `JR-AutoRAG-Upgrade-Doc-Local-First.md` (deleted).
> Historical upgrade record, corrected against the runtime in October 2026.
> The [measured retrieval audit](docs/retrieval-quality-2026-10-11.md) is the
> current evidence and prioritized roadmap. Feature names are not evidence of
> state-of-the-art quality or security guarantees.

## 0. Completed Work (Prior Upgrade Record)

### Phase 1: Dead Code Removal (11,532 lines deleted)

AST-based import-graph audit identified 33 dead Python modules with zero runtime
importers (no app code, no tests, no `__init__` re-exports). All were stale
replacements superseded by live modules:

- `milvus_store` → `binary_vector_store`
- `metrics` → `telemetry`
- `router` → `learned_router` / `query_mode`
- `evaluator` / `eval_harness` → `ragas_eval` / `golden_eval`
- `vector_store` → `documents` / `binary_vector_store`
- `trace_export_v2` / `tracing` → `trace_export` / `telemetry`
- Plus 27 more modules

Frontend cleanup removed 8 dead components: `ArtifactViewer` (unreachable),
`CacheEventBadge`, `GroundingBadge`, `PipelineTimeline`, `EnterpriseControls`,
`APITester`, and unused `ui/field` + `ui/input-group` primitives.

### Phase 2: SOTA Retrieval Upgrades (historical goals, 2025-2026)

**Late Chunking** (arXiv 2409.04701):

- New `ChunkingStrategy.LATE` enum value + `LateChunker` class
- Produces non-overlapping source windows, now with exact source offsets
- Wired into `ingest.py` `_chunk()` routing
- Retrieval still embeds each window independently. Full-document token
  encoding followed by contextual pooling from the paper is **not wired**.

**Per-Query Hybrid Weights** (`AutoHybridWeights`):

- Analyzes query features (length, question words, quotes, numbers, NL indicators)
- Computes per-query dense/sparse fusion weights instead of global defaults
- Wired into `HybridRetrievalEngine.query()` when no explicit overrides provided
- These are heuristics, not learned routing or a measured universal improvement

**MMR Diversity Rerank** (upgraded):

- Uses embedding cosine similarity when available
- Falls back to token Jaccard for zero-dependency operation
- October fixes retain a larger candidate pool and normalize relevance within
  the MMR objective so diversity is not overwhelmed by score scale

**Matryoshka Embedding Support**:

- `HybridConfig.matryoshka_dim` is a configuration field only
- Truncated-vector search and full-dimensional rescoring are **not wired**

**Contextual Enrichment as Default** (Anthropic Contextual Retrieval):

- `ContextualEnricher.enrich_chunks_sync()` adds document title, section header,
  heuristic chunk summary, and context window to every chunk at ingest time
- Falls back to simple header prepend on failure
- Wired as default in `IngestPipeline.ingest_text()`
- This heuristic enrichment is not equivalent to the evaluated model-generated
  contextual retrieval described by Anthropic. Offsets refer to the resulting
  stored/indexed text, not necessarily the original uploaded file.

### Phase 3: Eval Gates CI Integration

- New `/evaluation/gates/{set_name}` endpoint exposes `GatedEvaluator`
- Supports `strict` and `lenient` threshold presets
- Returns pass/fail status for citation coverage, recall@k, faithfulness, latency p95

### Phase 4: Security Hardening (OWASP LLM01/02)

**Canary Token Manager** (OWASP LLM01 — Prompt Injection):

- `CanaryTokenManager` injects unique canary tokens into system prompts
- Verifies token presence in LLM output — missing canary indicates hijack
- Singleton accessor: `get_canary_manager()`

**Poisoned Chunk Scanner** (OWASP LLM02 — Knowledge Base Poisoning):

- `PoisonedChunkScanner` detects anomalous chunks via:
  - Embedded instruction patterns ("ignore previous", "system:", etc.)
  - Excessive repetition (adversarial padding)
  - Unusual character distribution (encoded payloads)
  - Short instruction payloads
- Wired into ingest pipeline — suspicious chunks flagged in document metadata
- Singleton accessor: `get_poison_scanner()`

### Phase 5: UI/UX P0 Fixes

**Color Contrast (WCAG AA)**:

- Darkened light-mode primary from `#ff5d73` to `#d92647` (~4.8:1 on white)
- Dark mode keeps brand pink (passes on dark background)
- Separated `--destructive` from `--primary` (were same color)
- Lightened dark-mode `--muted-foreground` for WCAG compliance

**Tab Navigation (ARIA Tablist)**:

- Roving tabindex (active tab = 0, inactive = -1)
- Arrow-key navigation (left/right cycles tabs)
- `id` + `role="tabpanel"` + `tabIndex` on all 6 panels
- `aria-controls` now points to real panel ids

**Accessibility**:

- Source items and saved session rows are now keyboard-focusable
- aria-labels on all icon-only buttons (sidebar toggles, close buttons)
- `aria-live="polite"` on toast viewport and streaming answer
- Hover-only action buttons now show on keyboard focus
- Removed triple focus rings (consolidated to single outline)
- Scoped global `*` transition to interactive elements only

**Functional Fixes**:

- Wired `cacheStats`/`onClearCache`/`isClearingCache` to `MetricsDashboard`
- Fixed script loading: `defer` instead of `async`, moved to body end
- API URL visible on `lg+` screens (was hidden below `2xl`)
- Added meta description

### Historical Verification

The prior session recorded the following checks. They are historical evidence,
not a current release receipt. Current commands, counts, and benchmark limits
are in the [October report](docs/retrieval-quality-2026-10-11.md).

- `uv run ruff check app tests` — clean
- `uv run pytest -q` — 363 tests pass (334 original + 29 new)
- `bun run typecheck` — clean
- `bun test` — 12 tests pass
- `bun run build` — 840 KB JS + 102 KB CSS

---

## 1. Where the project actually is (post-upgrade)

After this upgrade session, the codebase has:

- Hybrid dense + BM25 + RRF + cross-encoder rerank with **per-query weights**
- Binary-quantized retrieval variant (`bq_hybrid_retrieval.py`, `binary_vector_store.py`)
- A `late` window-splitting strategy; contextual token pooling remains future work
- Heuristic contextual enrichment as default; no measured Anthropic-equivalent claim
- **MMR diversity** with embedding cosine similarity
- Pluggable OCR router with `off/auto/vision/dedicated/hybrid` policies
- Hierarchy/RAPTOR, GraphRAG, memory, HyDE, FLARE, Self-RAG
- Citation verification, hallucination firewall, evidence contracts
- Citation ID checks do not prove that a source entails each answer claim;
  current firewall/quality metrics are heuristic and need independent evaluation
- RAGAS-style eval, golden sets, **eval gates endpoint for CI**
- **Canary tokens** + **poisoned chunk scanner** (OWASP LLM01/02)
- Auth, rate limiting, prompt guard, PII redaction, secrets vault, audit, ACL
- Tracing, telemetry, trace export/replay, install report
- **WCAG AA compliant** color contrast
- **ARIA-compliant** tab navigation with keyboard support
- **Accessible** interactive elements (focusable, labeled, live regions)

## 2. Remaining work (future phases)

### Phase B — Measured Retrieval and Model Selection

- Evaluate fixed/semantic/recursive chunking, dense/sparse fusion, reranking,
  redundancy and no-answer queries on representative public/local fixtures
- Compare local embedding/reranking models against present presets, including
  query instruction formatting, peak memory, cold/warm latency and answer support
- Implement true contextual late pooling only behind a tested compatible encoder
- Wire Matryoshka dimensions only for compatible models, with recall/latency gates
- Generic MiniLM token MaxSim is not a trained ColBERT model; evaluate a real
  late-interaction implementation before claiming comparable quality
- Add a new vector backend only if measured corpus scale requires one

### Phase C — Agentic & Advanced Security

- Promote `smart_planner.py` to `AgenticRAGPlanner` with `max_steps`/`token_budget`
- Add ReAct-style prompt formatting with trace visualization
- Add Presidio-backed PII redaction (optional backend)
- Add embedding-based anomaly detection for prompt injection defense
- Add metamorphic drift detection to eval harness
- Measure claim-level citation entailment and unsupported-claim abstention with
  human/independent grading; retain prompt-injection and poisoned-corpus negatives
- Bound query expansion/planning by evidence gain, latency and local model cost

### Phase D — Frontend Polish

- Enable bundle splitting (`splitting: true` in `build.ts`)
- Add `AbortController` to `fetchJson` and cancel streams on unmount
- Stabilize handlers with `useCallback` and memoize heavy chat subcomponents
- Split `AdvancedRAGSettings` into grouped accordions with sticky actions
- Add ESLint + `eslint-plugin-jsx-a11y` for regression prevention
