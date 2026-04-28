export type ProviderConfig = {
    name: string;
    base_url: string;
    planner_model?: string;
    gatherer_model?: string;
    generator_model?: string;
    api_key?: string;
};

export type ApiClientSettings = {
    baseUrl: string;
    apiKey?: string;
};

export type ProviderProfile = {
    name: string;
    provider: ProviderConfig;
};

export type DeploymentProfile = "local_only" | "hybrid" | "cloud_accelerated";
export type BackendMode = "local" | "hybrid" | "cloud";
export type CapabilityClass = "low" | "medium" | "high";
export type SubsystemType =
    | "document_parser"
    | "ocr"
    | "embedding"
    | "reranker"
    | "vector_store"
    | "sparse_index"
    | "graph_store"
    | "llm"
    | "memory"
    | "eval"
    | "telemetry";

export type BackendCapabilities = {
    mode: BackendMode;
    requires_network: boolean;
    supports_batching: boolean;
    supports_streaming: boolean;
    supports_multimodal: boolean;
    estimated_latency_class: CapabilityClass;
    estimated_memory_class: CapabilityClass;
};

export type SubsystemBackendConfig = {
    subsystem: SubsystemType;
    backend_id: string;
    label: string;
    enabled: boolean;
    capabilities: BackendCapabilities;
    settings: Record<string, string | number | boolean>;
};

export type FallbackConfig = {
    enabled: boolean;
    order: string[];
    on_failure: "error" | "fallback";
};

export type OCRPolicy = "off" | "auto" | "vision_model" | "dedicated_ocr" | "hybrid";

export type OCRSettings = {
    policy: OCRPolicy;
    extractable_text_threshold: number;
    min_characters: number;
    allow_cloud_fallback: boolean;
    preferred_backends: string[];
    dual_merge_strategy: "highest_confidence" | "prefer_text_parser";
};

export type IngestSettings = {
    ocr: OCRSettings;
    parsing_stack: string[];
    attach_processing_trace: boolean;
};

export type RetrievalDefaults = {
    hybrid: boolean;
    dense_k: number;
    sparse_k: number;
    dense_weight?: number;
    sparse_weight?: number;
    rerank_pool: number;
    top_n: number;
    compression: boolean;
    target_tokens: number;
    raptor: boolean;
    graph: boolean;
    coverage_target: number;
    max_context_tokens: number;
    // New hybrid retrieval options
    chunking_strategy: "fixed" | "semantic" | "recursive";
    embedding_model: string;
    reranker_model: string;
    use_reranking: boolean;
    use_colbert?: boolean;
    colbert_model?: string;
    colbert_top_k?: number;
    chunk_size: number;
    chunk_overlap: number;
    planner_mode: "simple" | "smart";
    flare_generation?: boolean;
    enforce_evidence_contract?: boolean;
    multi_resolution?: boolean;
    recency_weight?: number;
    recency_half_life_days?: number;
    title_boost?: number;
    heading_boost?: number;
    proximity_weight?: number;
    diversity?: number;
    use_hyde?: boolean;
    abstain_when_unverified?: boolean;
    self_rag_critic?: boolean;
    // v2 Binary Quantization settings
    retrieval_mode?: "float32" | "binary";
    bq_enabled?: boolean;
    bq_normalize?: boolean;
    bq_rule?: string;
    bq_two_stage?: boolean;
    bq_stage1_candidates?: number;
    bq_fallback_enabled?: boolean;
    bq_fallback_threshold?: number;
    milvus_host?: string;
    milvus_port?: number;
    milvus_collection?: string;
    milvus_index_type?: "BIN_FLAT" | "BIN_IVF_FLAT";
    milvus_metric?: "HAMMING" | string;
    milvus_nlist?: number;
    milvus_nprobe?: number;
    langextract_enabled?: boolean;
    langextract_profile_default?: string;
    langextract_model_source?: "planner" | "gatherer" | "generator";
    langextract_timeout_sec?: number;
    langextract_max_chars?: number;
    langextract_max_synthetic_facts?: number;
};

// 3.0: Stage budget configuration
export type StageBudgets = {
    planner_timeout_ms: number;
    gatherer_timeout_ms: number;
    rerank_timeout_ms: number;
    compression_timeout_ms: number;
    generation_timeout_ms: number;
    verification_timeout_ms: number;
    total_timeout_ms: number;
    retrieval_token_budget: number;
    rerank_pool_budget: number;
    compression_token_budget: number;
    answer_token_budget: number;
};

// 3.0: Query mode types
export type QueryMode = "grounded" | "open_domain";

export type AppConfig = {
    profile: string;
    deployment_profile: DeploymentProfile;
    provider?: ProviderConfig;
    provider_profiles: ProviderProfile[];
    retrieval: RetrievalDefaults;
    ingest: IngestSettings;
    backends: Record<string, SubsystemBackendConfig>;
    fallbacks: Record<string, FallbackConfig>;
    // 3.0: Query mode and stage budgets
    query_mode?: QueryMode;
    stage_budgets?: StageBudgets;
};

export type ModelStatus = {
    embedding: "installed" | "missing" | "unknown" | "error";
    reranker: "installed" | "missing" | "unknown" | "error";
    embedding_message?: string;
    reranker_message?: string;
    deployment_profile?: DeploymentProfile;
    local_only_ready?: boolean;
};

export type DocumentOut = {
    id: string;
    title: string;
    text: string;
    metadata: Record<string, string>;
    created_at?: string;
    chunk_count?: number;
    processing_status?: string;
    processing_error?: string;
};

export type ParsedBlock = {
    type: string;
    text: string;
    page?: number | null;
    heading_level?: number | null;
    confidence: number;
    metadata: Record<string, unknown>;
};

export type ParsedPage = {
    number: number;
    text: string;
    confidence: number;
    metadata: Record<string, unknown>;
    blocks: ParsedBlock[];
};

export type DocumentPreview = {
    document_id: string;
    title: string;
    parser_provider: string;
    parser_engine: string;
    confidence: number;
    used_ocr: boolean;
    page_count: number;
    block_count: number;
    warnings: string[];
    blocks: ParsedBlock[];
    pages: ParsedPage[];
};

export type EvalMetricResult = {
    name: string;
    value: number;
    provider: string;
    direction: string;
    details: Record<string, unknown>;
};

export type ExperimentConfig = {
    name: string;
    description?: string;
    parser?: string[];
    chunker?: string[];
    embedding?: string[];
    dense_weight?: number[];
    sparse_weight?: number[];
    reranker?: boolean[];
    graph?: boolean[];
    raptor?: boolean[];
    ocr_policy?: string[];
    questions?: string[];
};

export type ExperimentRun = {
    id: string;
    config: ExperimentConfig;
    status: string;
    created_at: string;
    completed_at?: string | null;
    metrics: EvalMetricResult[];
    winning_preset?: string | null;
    config_snapshot: Record<string, unknown>;
    traces: string[];
    promoted_at?: string | null;
};

export type EvalRunSummary = {
    run_id: string;
    golden_set_name: string;
    timestamp: string;
    retrieval_metrics: {
        recall_at_k: number;
        mrr: number;
        ndcg: number;
        citation_coverage: number;
    };
    answer_metrics: {
        faithfulness: number;
        completeness: number;
        refusal_accuracy: number;
        coherence: number;
    };
    duration_ms: number;
};

export type QualityRecommendation = {
    id: string;
    title: string;
    priority: "high" | "medium" | "low" | string;
    detail: string;
    action: string;
};

export type QualityRecommendations = {
    deployment_profile: DeploymentProfile;
    document_count: number;
    parser_counts: Record<string, number>;
    low_confidence_documents: number;
    processing_errors: number;
    active_features: Record<string, boolean>;
    recommendations: QualityRecommendation[];
};

export type IngestResponse = {
    document_id: string;
    title: string;
    chunk_count: number;
};

export type ChunkOut = {
    id: string;
    title: string;
    snippet: string;
    score: number;
};

export type PipelineStep = {
    name: string;
    duration_ms: number;
    details: Record<string, unknown>;
    status: string;
    started_at?: string;
    completed_at?: string;
};

// 3.0: Grounding info for answers
export type GroundingInfo = {
    grounded: boolean;
    docs_used: number;
    citations_kept: number;
    chunks_dropped: number;
    mode?: QueryMode;
    no_evidence_response?: {
        found_evidence: boolean;
        message: string;
        suggested_actions: Array<{
            label: string;
            description: string;
            action_type: string;
        }>;
    };
};

// 3.0: Cache event for tracing
export type CacheEvent = {
    hit: boolean;
    key: string;
    reason?: string; // "not_found" | "expired" | "version_mismatch"
    corpus_version: string;
    retrieval_mode: number;
    preset_id: string;
};

export type QueryResponse = {
    answer: string;
    chunks: ChunkOut[];
    sources?: Array<{
        id: string;
        title: string;
        snippet_preview?: string;
        citation_number?: number;
        score?: number;
    }>;
    trace_id: string;
    metrics: Record<string, number | string>;
    steps: PipelineStep[];
    confidence?: {
        overall: number;
        factors?: {
            retrieval?: number;
            generation?: number;
            citation?: number;
        };
        hallucination_pass?: boolean;
        evidence_contract_pass?: boolean;
    };
    trace_bundle_available?: boolean;
    needs_clarification?: boolean;
    // 3.0: New grounding and cache fields
    grounding?: GroundingInfo;
    from_cache?: boolean;
    cache_event?: CacheEvent;
};

export type TraceOut = {
    id: string;
    prompt: string;
    answer: string;
    metrics: Record<string, number | string>;
    steps: PipelineStep[];
};

export type CacheStats = {
    embeddings?: { hits: number; misses: number; size: number };
    queries?: { hits: number; misses: number; size: number };
};

export type ProviderKind = "ollama" | "ollama_cloud" | "lmstudio" | "openai" | "openrouter";

export type LocalProviderInfo = {
    kind: ProviderKind;
    name: string;
    base_url: string;
    models: string[];
    running: string[];
    version?: string;
    status?: string;
    error_message?: string;
};

export type RoleSelection = {
    planner: string;
    gatherer: string;
    generator: string;
};

export type OpenRouterStatus = {
    available: boolean;
    api_key_configured: boolean;
    default_model: string;
    error_message?: string;
};

export type OpenRouterModel = {
    id: string;
    name: string;
    context_length?: number;
    pricing?: Record<string, unknown>;
};

export type RAGFuzzStatus = {
    status: string;
    ragfuzz_enabled: boolean;
    corpus_size: number;
    providers_available: string[];
    version: string;
};

export type ChatSession = {
    id: string;
    title: string;
    history: { role: string; content: string }[];
    queryResult: QueryResponse | null;
    createdAt: string;
};

// Preset system types
export type PresetLevel = "turbo" | "fast" | "balanced" | "thorough" | "ultra_accurate";

export type PresetInfo = {
    level: PresetLevel;
    name: string;
    description: string;
    icon: string;
    features: string[];
};

export const PRESET_DEFINITIONS: PresetInfo[] = [
    {
        level: "turbo",
        name: "Turbo",
        description: "Fastest responses",
        icon: "⚡",
        features: ["Basic retrieval", "No reranking"],
    },
    {
        level: "fast",
        name: "Fast",
        description: "Quick & good",
        icon: "🚀",
        features: ["Reranking"],
    },
    {
        level: "balanced",
        name: "Balanced",
        description: "Speed & accuracy",
        icon: "⚖️",
        features: ["Reranking", "Multi-resolution"],
    },
    {
        level: "thorough",
        name: "Thorough",
        description: "Deep research",
        icon: "🔍",
        features: ["FLARE", "RAPTOR", "Iterative"],
    },
    {
        level: "ultra_accurate",
        name: "Ultra Accurate",
        description: "Maximum accuracy",
        icon: "🎯",
        features: ["All SOTA features", "Evidence contract", "GraphRAG"],
    },
];
