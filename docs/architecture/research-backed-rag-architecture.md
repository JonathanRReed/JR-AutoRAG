# JR AutoRAG Research-Backed Architecture Matrix

Last verified: 2026-05-12.

This file maps current RAG research directions to JR AutoRAG implementation evidence. It is intended for client handoff, technical due diligence, and deciding what must be hardened next before a paid local install.

## Architecture Position

JR AutoRAG should be treated as a local-first, evidence-first AutoRAG system, not a single-pass chatbot. The current implementation already aligns with the main modern RAG directions identified by recent surveys: hybrid retrieval, context filtering, grounded generation, robustness checks, structured reasoning, and retrieval-aware evaluation.

The 2025 comprehensive RAG survey frames the field around retriever-centric, generator-centric, hybrid, and robustness-oriented systems, and highlights open challenges in adaptive retrieval, structured multi-hop reasoning, privacy-preserving retrieval, and retrieval-aware evaluation. Agentic RAG and secure RAG literature sharpens that product bar: modern systems need adaptive retrieval interfaces, governance around autonomous tool use, security benchmarks for poisoning and extraction attacks, and evidence receipts that distinguish retrieval, generation, routing, and policy failures. JR AutoRAG should keep these lanes explicit in product surfaces and evidence bundles.

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
| Agentic hierarchical retrieval interfaces | A-RAG argues that frontier models should participate in retrieval decisions through tools such as keyword search, semantic search, and chunk reads, rather than only receiving one static top-k context. Agentic RAG surveys also identify planning, reflection, tool use, and governance as core design patterns. | Client corpora require multi-step search, narrow reads, retries, and transparent control over how much evidence is pulled. | `api/app/core/gatherer.py`, `api/app/core/learned_router.py`, `api/app/core/orchestrator.py`, `src/components/features/ChatInterface.tsx`, `src/components/features/QualityCockpit.tsx` | Implemented, needs trajectory-level eval receipts |
| Self-reflection and corrective retrieval | Self-RAG argues retrieval should be used when useful and critiqued for relevance and support. | The product should say when evidence is weak, ask for more context, or retry instead of hallucinating. | `api/app/core/orchestrator.py`, `api/app/core/evaluator.py`, `api/app/core/abstention.py` | Implemented |
| Evidence contracts and citation verification | RAGAS and recent surveys emphasize faithfulness, context relevance, and generation quality as separate evaluation dimensions. | Client reports need answer claims tied to source chunks and auditable failure modes. | `api/app/core/evidence_contract.py`, `api/app/core/citation_verifier.py`, `api/app/core/golden_eval.py`, `src/components/features/GroundingBadge.tsx` | Implemented |
| Fine-grained evaluation receipts | RAGAS provides reference-free RAG metrics, RAGChecker argues for diagnostic retriever and generator metrics, and CRAG and mmRAG show the need for benchmarked, component-level evaluation. | A B2B install needs repeatable quality receipts, not only a demo answer. The receipt should identify whether failures came from retrieval, generation, routing, parser, or policy gates. | `api/app/core/golden_eval.py`, `api/app/core/eval_gates.py`, `api/app/routers/evaluation.py`, `src/components/features/QualityCockpit.tsx` | Implemented |
| Agentic capability benchmarks | RAGCap-Bench argues that agentic RAG should evaluate intermediate capabilities such as planning, retrieval decisions, and multi-hop reasoning instead of only final answers. | Client handoff should prove the system can expose the retrieval path, not merely produce a plausible response. | `api/app/core/eval_gates.py`, `api/app/core/golden_eval.py`, `scripts/client-handoff-gate.sh`, `src/components/features/QualityCockpit.tsx` | Implemented for client-readiness receipts, needs larger trajectory corpora |
| Modular evaluation across text, tables, and knowledge graphs | mmRAG argues for granular evaluation of retrieval and routing across text, tables, and graph-shaped evidence. | Enterprise data is mixed-format. The product must expose parser quality and retrieval artifacts by modality. | `api/app/core/document_parser.py`, `api/app/core/langextract_enricher.py`, `api/app/core/install_report.py`, `src/components/features/QualityCockpit.tsx` | Partially implemented |
| Memory-efficient retrieval modes | 2025 and 2026 vector compression work shows that embedding storage and refinement costs matter for real deployments, but also that binary quantization can have dataset-specific accuracy cliffs. | Local B2B installs need a cost and memory lever for larger corpora, but the product must expose binary retrieval as a measured mode with fallback, not as a universal quality claim. | `api/app/core/bq_retrieval.py`, `api/app/core/binary_quantization.py`, `api/app/core/binary_vector_store.py`, `api/tests/test_bq_retrieval.py`, `src/components/features/AdvancedRAGSettings.tsx` | Implemented with fallback, needs corpus-specific receipts |
| Local-first enterprise security filtering | Secure Multifaceted-RAG motivates local open-source generation and selective external calls to reduce proprietary data exposure. | B2B installs must default to client-owned storage and local inference unless an engagement explicitly permits external calls. | `api/app/core/local_first.py`, `api/app/core/security_posture.py`, `api/app/core/security_middleware.py`, `Public/SECURITY.md` | Implemented |
| RAG threat and leakage controls | 2026 secure RAG surveys and extraction benchmarks highlight poisoning, adversarial retrieval, membership inference, and knowledge-extraction attacks as first-class RAG risks. | A sellable local install must treat retrieved documents, external pages, model outputs, and tool arguments as untrusted, then prove controls with receipts. | `api/app/core/security_middleware.py`, `api/app/core/document_acl.py`, `api/app/core/tools.py`, `api/app/core/rate_limiter.py`, `api/tests/test_tool_safety.py`, `api/tests/test_rate_limiter.py`, `scripts/secret-scan.sh` | Implemented, needs adversarial corpus benchmark pack |
| Handoff-gated robustness benchmark | Secure RAG and knowledge-extraction work show that client RAG systems need specific receipts for poisoning, leakage, and extraction resistance. | The installer should not be able to mark a bundle client-ready unless the golden run covers prompt injection, poisoned documents, knowledge extraction, and abstention. | `api/app/core/eval_gates.py`, `api/app/core/install_report.py`, `scripts/client-handoff-gate.sh`, `api/app/core/onboarding.py` | Implemented |
| Handoff evidence bundle | Enterprise delivery needs reproducible install evidence, policy proof, hashes, and live readiness results. | Installers need something concrete to give a client after setup. | `scripts/evidence-bundle.sh`, `api/app/core/install_report.py`, `README.md` | Implemented |

## Source Notes

- RAG survey, 2025: Taxonomy for retriever-centric, generator-centric, hybrid, and robustness-oriented systems. Also notes open challenges in adaptive retrieval, structured reasoning, privacy-preserving retrieval, and evaluation. https://arxiv.org/abs/2506.00054
- Agentic RAG survey, revised 2026: Agentic design patterns for planning, tool use, reflection, collaboration, governance, and enterprise document processing. https://arxiv.org/abs/2501.09136
- A-RAG, 2026: Hierarchical retrieval interfaces where the model can choose keyword search, semantic search, and chunk reads across granularities. https://arxiv.org/abs/2602.03442
- RAGCap-Bench, 2025: Capability-level benchmark for agentic RAG workflows and intermediate reasoning tasks. https://arxiv.org/abs/2510.13910
- LinearRAG, revised 2025: Relation-free hierarchical graph retrieval that reduces noisy relation extraction for large corpora. https://arxiv.org/abs/2510.10114
- Towards Agentic RAG with Deep Reasoning, 2025: Survey of reasoning-retrieval systems and iterative search/reasoning loops. https://arxiv.org/abs/2507.09477
- Chain-of-Retrieval Augmented Generation, 2025: Iterative retrieval and reasoning approach for knowledge-intensive tasks. https://arxiv.org/abs/2501.14342
- Secure RAG survey, 2026: Pipeline-level taxonomy of poisoning, adversarial, membership-inference, and leakage defenses plus benchmark consolidation. https://arxiv.org/abs/2603.21654
- RAG knowledge-extraction benchmark, 2026: Systematic attack and defense benchmark for recovering sensitive knowledge-base content from RAG systems. https://arxiv.org/abs/2602.09319
- FaTRQ, 2026: Tiered residual quantization for ANNS refinement in RAG and large embedding search systems. https://arxiv.org/abs/2601.09985
- Secure Multifaceted-RAG, 2025: Enterprise RAG framing with local generation and selective external generation to reduce leakage risk. https://arxiv.org/abs/2504.13425
- mmRAG, 2025: Modular evaluation across text, tables, and knowledge graphs, with granular retrieval and routing assessment. https://arxiv.org/abs/2505.11180
- RAG embedding storage optimization, 2025: Evaluates float16, int8, binary, float8, and dimensionality reduction trade-offs on MTEB for RAG storage. https://arxiv.org/abs/2505.00105
- Efficient vector retrieval quantization analysis, 2025: Shows quantization choices are dataset-dependent and binary retrieval can fail on some tasks. https://arxiv.org/abs/2511.13057
- GraphRAG, revised 2025: Graph indexing and community summaries for global questions over private corpora. https://arxiv.org/abs/2404.16130
- RAPTOR, 2024: Recursive clustering and summaries for tree-organized retrieval over long documents. https://arxiv.org/abs/2401.18059
- Self-RAG, 2023: Retrieval and critique through self-reflection. https://arxiv.org/abs/2310.11511
- FLARE, 2023: Active retrieval during generation for long-form knowledge-intensive answers. https://arxiv.org/abs/2305.06983
- HyDE, 2022: Hypothetical document embeddings for zero-shot dense retrieval without labels. https://arxiv.org/abs/2212.10496
- RAGAS, revised 2025: Reference-free metrics for retrieval quality, faithfulness, and answer quality. https://arxiv.org/abs/2309.15217
- RAGChecker, 2024: Fine-grained diagnostic metrics for retrieval and generation modules. https://arxiv.org/abs/2408.08067
- CRAG, 2024: Comprehensive benchmark showing trustworthy QA remains difficult even with advanced systems. https://arxiv.org/abs/2406.04744

## Remaining Research Gaps

1. Add benchmark packs for mixed PDFs, tables, spreadsheets, and graph-like evidence so the quality cockpit is not text-only.
2. Expand robustness tests beyond the built-in client-readiness pack with larger adversarial retrieval, poisoned-document, prompt-injection, and membership-inference corpora.
3. Add calibrated refusal thresholds per corpus type, with saved receipts showing when the system abstained.
4. Add client-safe external-call policy receipts when hybrid or cloud-accelerated modes are used.
5. Add larger-corpus GraphRAG and RAPTOR timing receipts so sales demos do not imply unverified scale.
6. Add trajectory-level evaluation for agentic retrieval decisions, including retries, chunk reads, and tool failures.
7. Add per-corpus binary retrieval quality receipts before recommending compressed retrieval for a client install.
