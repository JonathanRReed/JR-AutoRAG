# Iterative retrieval score contract

Retrieval scores describe one query and one ranking stage. They are not
confidence probabilities and cannot be compared across retrieval calls.

`HybridRetrievalEngine.query` still returns a single channel's raw score when
only one channel has results, weighted RRF when several channels contribute,
and reranker outputs when reranking runs. Recency may adjust these values.
MMR selects evidence using its own normalized objective and preserves the
returned diagnostic scores. This change preserves those values and the
engine's channel fusion, reranking, and selection behavior.

## Aggregation

Gatherer records `EvidenceChunk.retrieval_rank`, the one-based position in
each engine response after all ranking and MMR stages. It leaves `score`
unchanged. Each primary, slot-fill, clarification, decomposition, or contract
retry call has its own rank positions. Concatenating their evidence does not
turn them into a single ranked list.

The orchestrator deduplicates by chunk ID, retaining the observation with the
lowest returned rank. Equal ranks keep the first observation. It sorts by
that best rank; ties between IDs follow the IDs' first appearance. Repeating
a query does not add votes. Empty snippets are excluded. Unranked RAPTOR,
Graph, web, and multi-resolution supplements follow ranked evidence in arrival
order. Legacy manually constructed evidence without rank metadata follows
that same fallback. Multi-resolution and evidence-contract retries use the
same aggregation rule.

This is an ordinal selection policy, not relevance calibration. Rank one from
a short targeted list has the same priority as rank one from a broad list.
A rank-one hit from a weak query can tie a hit from a useful query. The policy
deliberately discards score magnitude across queries and does not introduce
rank weights, consensus bonuses, or per-query min-max confidence values.

Extractive compression ranks sentences by query-term relevance. Stable ties
preserve aggregate evidence order. It no longer adds a raw chunk-score bonus
that could undo the aggregation policy. Public chunk and source `score` fields
remain diagnostics from the retained observation.

Supplemental corrective context remains available when the context budget
permits. It has no reserved budget: simple truncation can fill the budget with
ranked evidence first, and extractive compression can omit a supplement when
its sentences have equal or lower relevance. A regression records both
inclusion with enough space and omission under a tight budget. This policy
does not guarantee that RAPTOR, Graph, or web corrections reach generation.

## Stopping

`compute_marginal_gain` returns the number of novel nonempty chunks divided by
the number of nonempty returned chunks. A chunk is redundant if its ID or
full case/whitespace-normalized text has already appeared, including earlier
in the same batch. IDs prevent snippet trimming from making a retrieved chunk
look new. Empty batches return zero. Duplicate returns remain in the
denominator, so repeated evidence lowers novelty. This is exact evidence
novelty, not semantic novelty or answer quality.

Nine repeated chunks and one new chunk now produce gain `0.1`, regardless of
the new chunk's diagnostic score. The comparison remains strictly
`gain < stop_threshold`; equality does not stop. The planner's threshold
values are unchanged, but now refer to this novelty fraction. Both query
coverage and plan coverage must still be sufficient. The correct-verdict
early stop keeps the same coverage requirements.

The heuristic retrieval verdict uses its existing text overlap, relevance,
and noise terms, with the same thresholds. Its raw retrieval-score bonus is
removed because it could independently trigger the correct-verdict early
stop. No scores or thresholds were clamped or retuned to claim calibration.
An empty iteration keeps the prior evidence explicitly rather than relying
on the `[:-len(iteration_chunks)]` slice. The trace's iteration count records
completed iterations rather than the loop counter plus one.

Traces identify `novel_nonempty_fraction` and
`best_returned_rank_stable_ties` as the stopping and aggregation contracts.

## Verification and limits

Synthetic regressions exercise the actual engine, Gatherer, and no-provider
orchestrator paths. Numerical channel/reranker fixtures isolate score units;
the sparse fallback check uses real documents and token matching. Tests cover
positive scaling and additive shifts that preserve order, mixed single- and
multi-channel iterations, reranked and non-reranked results, MMR order,
duplicate and empty evidence, independent ranked-list boundaries, supplemental
context, contract retries, and query/plan coverage gating.

The initial regression run on the previous implementation produced 20
failures and 7 passes. The finished implementation passes the complete API
suite: 705 tests, including 35 focused score-contract tests. Restoring the old
orchestrator separately reproduces eight failures across mixed-channel,
reranked, multi-resolution, and contract-retry cases. API Ruff checks
and Python compilation also pass. No models, paid providers, user documents,
or large corpus benchmarks were used.

Verification commands, after installing the locked dependencies:

```bash
cd api
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 \
  .venv/bin/python -m pytest tests/ -q --tb=short --timeout=60
.venv/bin/ruff check app tests
.venv/bin/ruff check . --select F821,F811,E9
.venv/bin/python -m compileall -q app
.venv/bin/python -c "from app.main import app; print('API imports OK')"
cd ..
bun run typecheck
bun test
bun run build
git diff --check
```

All commands pass locally. The frontend run has 12 passing tests. Container
builds, optional ML extras, and real model quality evaluations were not run.

This does not calibrate retrieval or answer confidence. The optional
abstention rule still checks average raw chunk score against its existing
threshold; quality/conflict consumers also retain their existing score
semantics. Those consumers require a separate policy review. The literal
channel still includes generic token overlap; the offline experiment that
removed it is not included. Synthetic score-contract checks establish
behavioral invariance where returned ranking is unchanged, not end-to-end
retrieval quality or generated-answer faithfulness.
