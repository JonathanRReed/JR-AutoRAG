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
    chunk_size: number;
    chunk_overlap: number;
    planner_mode: "simple" | "smart";
};

export type AppConfig = {
    profile: string;
    provider?: ProviderConfig;
    provider_profiles: ProviderProfile[];
    retrieval: RetrievalDefaults;
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
