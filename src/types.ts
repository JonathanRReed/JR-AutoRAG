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
    raptor: string;
    graph: boolean;
    coverage_target: number;
    max_context_tokens: number;
};

export type AppConfig = {
    profile: string;
    provider?: ProviderConfig;
    provider_profiles: ProviderProfile[];
    retrieval: RetrievalDefaults;
};

export type DocumentOut = {
    id: string;
    title: string;
    text: string;
    metadata: Record<string, string>;
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
};

export type QueryResponse = {
    answer: string;
    chunks: ChunkOut[];
    trace_id: string;
    metrics: Record<string, number>;
    steps: PipelineStep[];
};

export type TraceOut = {
    id: string;
    prompt: string;
    answer: string;
    metrics: Record<string, number>;
    steps: PipelineStep[];
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
