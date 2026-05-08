# JR AutoRAG Research-Backed Architecture Matrix

Last verified: 2026-05-08.

This file maps current RAG research directions to JR AutoRAG implementation evidence. It is intended for client handoff, technical due diligence, and deciding what must be hardened next before a paid local install.

## Architecture Position

JR AutoRAG should be treated as a local-first, evidence-first AutoRAG system, not a single-pass chatbot. The current implementation already aligns with the main modern RAG directions identified by recent surveys: hybrid retrieval, context filtering, grounded generation, robustness checks, structured reasoning, and retrieval-aware evaluation.

The 2025 comprehensive RAG survey frames the field around retriever-centric, generator-centric, hybrid, and robustness-oriented systems, and highlights open challenges in adaptive retrieval, structured multi-hop reasoning, privacy-preserving retrieval, and retrieval-aware evaluation. JR AutoRAG should keep these four lanes explicit in product surfaces and evidence bundles.

Source: https://arxiv.org/abs/2506.00054

## Technique Matrix

| Capability | Research basis | Product reason | Current implementation evidence | Readiness |
| --- | --- | --- | --- | --- |
| Hybrid dense plus sparse retrieval | Modern RAG surveys identify retrieval optimization and context filtering as core quality levers. | Enterprise corpora contain exact terms, aliases, abbreviations, and semantic matches. Hybrid retrieval protects both precision and recall. | `api/app/core/hybrid_retrieval.py`, `api/app/services.py`, `api/app/schemas/config.py` | Implemented |
| Cross-encoder reranking | Retrieval-aware systems need stronger context ordering before generation. | Reranking reduces irrelevant chunks before answer synthesis and improves evidence quality. | `api/app/core/hybrid_retrieval.py`, `api/app/core/presets.py`, `src/components/features/AdvancedRAGSettings.tsx` | Implemented |
| GraphRAG for global sensemaking | GraphRAG targets corpus-level questions by building an entity graph and community summaries over private corpora. | Client installs often ask global questions like themes, risk clusters, org relationships, and project summaries. | `api/app/core/graph_rag.py`, `api/app/core/orchestrator.py`, `api/app/routers/artifact_routes.py` | Implemented, needs larger corpus stress tests |
| RAPTOR-style hierarchical retrieval | RAPTOR recursively embeds, clusters, and summarizes chunks so retrieval can use multiple abstraction levels. | Long client documents need section and summary retrieval, not only flat chunks. | `api/app/core/hierarchy.py`, `api/app/core/orchestrator.py`, `src/components/features/EnterpriseStatusPanel.tsx` | Implemented, needs client-corpus benchmarks |
| HyDE query expansion | HyDE generates a hypothetical answer document and uses its embedding to improve zero-shot dense retrieval. | Local installs often lack labeled relevance data, so zero-shot retrieval improvements matter. | `api/app/core/hyde.py`, `api/app/core/orchestrator.py`, `api/app/schemas/config.py` | Implemented |
| Active and iterative retrieval | FLARE shows retrieval should happen across generation when long-form answers need more evidence. | Enterprise answers often require staged evidence gathering instead of one fixed retrieval call. | `api/app/core/orchestrator.py`, `api/app/core/gatherer.py`, `api/app/core/learned_router.py` | Implemented |
| Self-reflection and corrective retrieval | Self-RAG argues retrieval should be used when useful and critiqued for relevance and support. | The product should say when evidence is weak, ask for more context, or retry instead of hallucinating. | `api/app/core/orchestrator.py`, `api/app/core/evaluator.py`, `api/app/core/abstention.py` | Implemented |
| Evidence contracts and citation verification | RAGAS and recent surveys emphasize faithfulness, context relevance, and generation quality as separate evaluation dimensions. | Client reports need answer claims tied to source chunks and auditable failure modes. | `api/app/core/evidence_contract.py`, `api/app/core/citation_verifier.py`, `api/app/core/golden_eval.py`, `src/components/features/GroundingBadge.tsx` | Implemented |
| Golden evaluation receipts | RAGAS provides reference-free RAG metrics, while CRAG and mmRAG show the need for benchmarked, component-level evaluation. | A B2B install needs a repeatable quality receipt, not only a demo answer. | `api/app/core/golden_eval.py`, `api/app/routers/evaluation.py`, `src/components/features/QualityCockpit.tsx` | Implemented |
| Modular evaluation across text, tables, and knowledge graphs | mmRAG argues for granular evaluation of retrieval and routing across text, tables, and graph-shaped evidence. | Enterprise data is mixed-format. The product must expose parser quality and retrieval artifacts by modality. | `api/app/core/document_parser.py`, `api/app/core/langextract_enricher.py`, `api/app/core/install_report.py`, `src/components/features/QualityCockpit.tsx` | Partially implemented |
| Local-first enterprise security filtering | Secure Multifaceted-RAG motivates local open-source generation and selective external calls to reduce proprietary data exposure. | B2B installs must default to client-owned storage and local inference unless an engagement explicitly permits external calls. | `api/app/core/local_first.py`, `api/app/core/security_posture.py`, `api/app/core/security_middleware.py`, `Public/SECURITY.md` | Implemented |
| Handoff evidence bundle | Enterprise delivery needs reproducible install evidence, policy proof, hashes, and live readiness results. | Installers need something concrete to give a client after setup. | `scripts/evidence-bundle.sh`, `api/app/core/install_report.py`, `README.md` | Implemented |

## Source Notes

- RAG survey, 2025: Taxonomy for retriever-centric, generator-centric, hybrid, and robustness-oriented systems. Also notes open challenges in adaptive retrieval, structured reasoning, privacy-preserving retrieval, and evaluation. https://arxiv.org/abs/2506.00054
- Secure Multifaceted-RAG, 2025: Enterprise RAG framing with local generation and selective external generation to reduce leakage risk. https://arxiv.org/abs/2504.13425
- mmRAG, 2025: Modular evaluation across text, tables, and knowledge graphs, with granular retrieval and routing assessment. https://arxiv.org/abs/2505.11180
- GraphRAG, revised 2025: Graph indexing and community summaries for global questions over private corpora. https://arxiv.org/abs/2404.16130
- RAPTOR, 2024: Recursive clustering and summaries for tree-organized retrieval over long documents. https://arxiv.org/abs/2401.18059
- Self-RAG, 2023: Retrieval and critique through self-reflection. https://arxiv.org/abs/2310.11511
- FLARE, 2023: Active retrieval during generation for long-form knowledge-intensive answers. https://arxiv.org/abs/2305.06983
- HyDE, 2022: Hypothetical document embeddings for zero-shot dense retrieval without labels. https://arxiv.org/abs/2212.10496
- RAGAS, revised 2025: Reference-free metrics for retrieval quality, faithfulness, and answer quality. https://arxiv.org/abs/2309.15217
- CRAG, 2024: Comprehensive benchmark showing trustworthy QA remains difficult even with advanced systems. https://arxiv.org/abs/2406.04744

## Remaining Research Gaps

1. Add benchmark packs for mixed PDFs, tables, spreadsheets, and graph-like evidence so the quality cockpit is not text-only.
2. Add robustness tests for adversarial retrieval, poisoned documents, prompt injection inside documents, and membership inference risk.
3. Add calibrated refusal thresholds per corpus type, with saved receipts showing when the system abstained.
4. Add client-safe external-call policy receipts when hybrid or cloud-accelerated modes are used.
5. Add larger-corpus GraphRAG and RAPTOR timing receipts so sales demos do not imply unverified scale.
