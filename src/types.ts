export type ProviderConfig = {
    name: string;
    base_url: string;
    planner_model?: string;
    gatherer_model?: string;
    generator_model?: string;
    api_key?: string;
};

export type ProviderProfile = {
    name: string;
    provider: ProviderConfig;
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
    provider?: ProviderConfig;
    provider_profiles: ProviderProfile[];
    retrieval: RetrievalDefaults;
    // 3.0: Query mode and stage budgets
    query_mode?: QueryMode;
    stage_budgets?: StageBudgets;
};

export type ModelStatus = {
    embedding: "installed" | "missing" | "unknown" | "error";
    reranker: "installed" | "missing" | "unknown" | "error";
    embedding_message?: string;
    reranker_message?: string;
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

export type ProviderKind = "ollama" | "lmstudio" | "openai";

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
