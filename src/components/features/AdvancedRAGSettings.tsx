import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Boxes, ChevronDown, GitMerge, Layers3, Package, Sliders, Info, ShieldCheck, Loader2, Binary, Zap } from "lucide-react";
import type { FallbackConfig, RetrievalDefaults, SubsystemBackendConfig } from "@/types";

interface AdvancedRAGSettingsProps {
    retrieval: RetrievalDefaults | undefined;
    backends: Record<string, SubsystemBackendConfig> | undefined;
    fallbacks: Record<string, FallbackConfig> | undefined;
    updateBackend: (subsystem: string, patch: Partial<SubsystemBackendConfig>) => void;
    updateRetrieval: (field: keyof RetrievalDefaults, value: string | number | boolean) => void;
    onSave: () => void;
    isSaving: boolean;
    modelStatus: {
        embedding: "installed" | "missing" | "unknown" | "error";
        reranker: "installed" | "missing" | "unknown" | "error";
        embedding_message?: string;
        reranker_message?: string;
    };
    isCheckingModels: boolean;
    modelActionMessage: string;
    onRefreshModelStatus: () => void;
    onDownloadEmbedding: () => void;
    onDownloadReranker: () => void;
    onDeleteEmbedding: () => void;
    onDeleteReranker: () => void;
    isDownloadingEmbedding: boolean;
    isDownloadingReranker: boolean;
}

function InlineHint({ label, detail }: { label: string; detail: string }) {
    return (
        <span
            className="inline-flex items-center gap-2 rounded-full bg-muted px-2.5 py-1 text-[11px] font-medium text-muted-foreground"
            title={detail}
        >
            <Info className="h-3.5 w-3.5 text-muted-foreground" />
            <span className="truncate">{label}</span>
        </span>
    );
}

export function AdvancedRAGSettings({
    retrieval,
    backends,
    fallbacks,
    updateBackend,
    updateRetrieval,
    onSave,
    isSaving,
    modelStatus,
    isCheckingModels,
    modelActionMessage,
    onRefreshModelStatus,
    onDownloadEmbedding,
    onDownloadReranker,
    onDeleteEmbedding,
    onDeleteReranker,
    isDownloadingEmbedding,
    isDownloadingReranker,
}: AdvancedRAGSettingsProps) {
    const [isExpanded, setIsExpanded] = useState(true);
    const chunkSizeOptions = [100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 800, 1000];
    const overlapOptions = [0, 50, 100, 150, 200];
    const targetTokenOptions = [800, 1000, 1200, 1500, 1800, 2000, 2400, 3000, 3600];
    const maxContextOptions = [2048, 3072, 4096, 6144, 8192, 12288];
    const coverageOptions = [0.5, 0.6, 0.7, 0.8, 0.9];
    const topNOptions = [3, 5, 8, 10, 12, 15, 20];
    const denseKOptions = [5, 10, 15, 20, 30, 40, 50];
    const sparseKOptions = [10, 20, 30, 40, 60, 80, 100];
    const rerankPoolOptions = [10, 20, 30, 40, 50, 60, 80, 100];
    const langExtractProfiles = ["generic_entities_v1", "compliance_risk_v1", "contract_terms_v1"];
    const langExtractTimeouts = [10, 15, 20, 30, 45, 60];
    const langExtractMaxChars = [4000, 8000, 12000, 16000, 24000];
    const langExtractMaxFacts = [50, 100, 150, 200, 300, 500];
    const embeddingModels = [
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-large-en-v1.5",
        "BAAI/bge-small-en-v1.5",
        "intfloat/e5-base-v2",
        "intfloat/e5-large-v2",
        "intfloat/e5-small-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
        "thenlper/gte-base",
        "thenlper/gte-large",
        "BAAI/bge-m3",
    ];
    const rerankerModels = [
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "cross-encoder/ms-marco-MiniLM-L-12-v2",
        "cross-encoder/ms-marco-TinyBERT-L-2-v2",
        "cross-encoder/ms-marco-electra-base",
        "BAAI/bge-reranker-base",
        "BAAI/bge-reranker-large",
        "mixedbread-ai/mxbai-rerank-base-v1",
    ];
    const retrievalBackends = [
        ["embedding", "Embedding Lane"],
        ["reranker", "Reranker Lane"],
        ["vector_store", "Vector Store"],
        ["sparse_index", "Sparse Index"],
        ["graph_store", "Graph Store"],
    ] as const;
    return (
        <Card className="overflow-hidden">
            <CardHeader className="bg-muted/20">
                <div className="flex items-center justify-between gap-4">
                    <CardTitle className="flex items-center gap-2 text-foreground">
                        <Sliders className="h-5 w-5" />
                        Advanced Engineering
                    </CardTitle>
                    <button
                        type="button"
                        onClick={() => setIsExpanded((prev) => !prev)}
                        className="inline-flex items-center gap-2 rounded-md border border-border/60 bg-card/60 px-3 py-1 text-xs font-medium text-foreground transition-colors hover:bg-muted/40"
                        aria-expanded={isExpanded}
                    >
                        {isExpanded ? "Collapse" : "Expand"}
                        <ChevronDown className={`h-4 w-4 transition-transform ${isExpanded ? "rotate-180" : ""}`} />
                    </button>
                </div>
                <CardDescription className="text-muted-foreground">
                    Fine-tune the retrieval engine, chunking strategies, and reranking parameters.
                </CardDescription>
                <div className="mt-3 flex flex-wrap gap-2">
                    <InlineHint label="Presets" detail="Apply tuned defaults for speed, balance, or depth." />
                    <InlineHint label="Chunking" detail="Control segment size/overlap to balance recall vs speed." />
                    <InlineHint label="Hybrid + rerank" detail="Blend dense/sparse retrieval; rerank for precision." />
                </div>
                <div className="mt-3 flex flex-wrap gap-2 text-xs text-muted-foreground">
                    <span className="inline-flex items-center gap-2 rounded-lg bg-muted px-2.5 py-1">
                        <ShieldCheck className="h-3.5 w-3.5 text-primary" />
                        {modelStatus.embedding === "installed" ? "Embedding model ready" : "Embedding model not installed"}
                    </span>
                    <span className="inline-flex items-center gap-2 rounded-lg bg-muted px-2.5 py-1">
                        <ShieldCheck className="h-3.5 w-3.5 text-primary" />
                        {modelStatus.reranker === "installed" ? "Reranker ready" : "Reranker not installed"}
                    </span>
                    {isCheckingModels && (
                        <span className="inline-flex items-center gap-2 rounded-lg bg-muted px-2.5 py-1">
                            <Loader2 className="h-3.5 w-3.5 animate-spin text-muted-foreground" />
                            Checking models...
                        </span>
                    )}
                </div>
            </CardHeader>
            {isExpanded && (
                <CardContent className="pt-8">
                    <div className="grid gap-6">
                        <section className="space-y-4 rounded-lg border border-border/60 bg-muted/10 p-4">
                            <div className="flex items-center gap-2">
                                <div className="h-4 w-1 rounded-full bg-border" />
                                <div>
                                    <h4 className="text-sm font-semibold uppercase tracking-wider text-foreground">Backend Routing</h4>
                                    <p className="text-xs text-muted-foreground">Control local-first retrieval backends and fallback chains.</p>
                                </div>
                            </div>
                            <div className="grid gap-4 sm:grid-cols-2">
                                {retrievalBackends.map(([key, label]) => {
                                    const backend = backends?.[key];
                                    const fallback = fallbacks?.[key];
                                    return (
                                        <div key={key} className="space-y-2">
                                            <Label htmlFor={`routing-${key}`} className="text-xs">{label}</Label>
                                            <Input
                                                id={`routing-${key}`}
                                                value={backend?.backend_id ?? ""}
                                                onChange={(e) => updateBackend(key, { backend_id: e.target.value, label: backend?.label ?? label })}
                                                className="font-mono text-xs"
                                            />
                                            <p className="text-[10px] uppercase tracking-tight text-muted-foreground">
                                                {backend?.capabilities?.mode ?? "local"} / fallback {fallback?.enabled ? "on" : "off"} / {fallback?.order?.[0] ?? "none"}
                                            </p>
                                        </div>
                                    );
                                })}
                            </div>
                        </section>

                        {/* Chunking Strategy */}
                        <section className="space-y-4 rounded-lg border border-border/60 bg-muted/10 p-4">
                            <div className="flex items-center gap-2">
                                <div className="h-4 w-1 rounded-full bg-border" />
                                <div>
                                    <h4 className="text-sm font-semibold uppercase tracking-wider text-foreground">Chunking Configuration</h4>
                                    <p className="text-xs text-muted-foreground">Control how documents are segmented for retrieval.</p>
                                </div>
                            </div>
                            <div className="grid gap-4 sm:grid-cols-3">
                                <div className="space-y-2">
                                    <Label htmlFor="chunkingStrategy" className="text-xs">Strategy</Label>
                                    <Select
                                        value={retrieval?.chunking_strategy ?? "fixed"}
                                        onValueChange={(value) => updateRetrieval("chunking_strategy", value)}
                                    >
                                        <SelectTrigger id="chunkingStrategy" className="h-9">
                                            <SelectValue placeholder="Select strategy" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="fixed">Fixed Char Length</SelectItem>
                                            <SelectItem value="semantic">Semantic (LLM-based)</SelectItem>
                                            <SelectItem value="recursive">Recursive Character</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="chunkSize" className="text-xs">Chunk Size</Label>
                                    <Select
                                        value={String(retrieval?.chunk_size ?? 400)}
                                        onValueChange={(value) => updateRetrieval("chunk_size", Number(value))}
                                    >
                                        <SelectTrigger id="chunkSize" className="h-9">
                                            <SelectValue placeholder="Select size" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {chunkSizeOptions.map(option => (
                                                <SelectItem key={option} value={String(option)}>
                                                    {option}
                                                </SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="chunkOverlap" className="text-xs">Overlap</Label>
                                    <Select
                                        value={String(retrieval?.chunk_overlap ?? 50)}
                                        onValueChange={(value) => updateRetrieval("chunk_overlap", Number(value))}
                                    >
                                        <SelectTrigger id="chunkOverlap" className="h-9">
                                            <SelectValue placeholder="Select overlap" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {overlapOptions.map(option => (
                                                <SelectItem key={option} value={String(option)}>
                                                    {option}
                                                </SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                </div>
                            </div>
                        </section>

                        {/* Embedding & Reranking */}
                        <section className="space-y-4 rounded-lg border border-border/60 bg-muted/10 p-4">
                            <div className="flex items-center gap-2">
                                <div className="h-4 w-1 rounded-full bg-border" />
                                <div>
                                    <h4 className="text-sm font-semibold uppercase tracking-wider text-foreground">Models & Reranking</h4>
                                    <p className="text-xs text-muted-foreground">Select embedding and reranking models for quality control.</p>
                                </div>
                            </div>
                            <div className="grid gap-4 sm:grid-cols-2">
                                <div className="space-y-2">
                                    <Label htmlFor="embeddingModel" className="text-xs">Dense Embedding Model</Label>
                                    <Select
                                        value={retrieval?.embedding_model ?? "BAAI/bge-base-en-v1.5"}
                                        onValueChange={(value) => updateRetrieval("embedding_model", value)}
                                    >
                                        <SelectTrigger id="embeddingModel" className="h-9 font-mono text-xs">
                                            <SelectValue placeholder="Select model" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {embeddingModels.map(model => (
                                                <SelectItem key={model} value={model} className="font-mono text-xs">
                                                    {model}
                                                </SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                    <p className="text-[10px] text-muted-foreground uppercase tracking-tight">Hugging Face identifier</p>
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="rerankerModel" className="text-xs">Cross-Encoder Reranker</Label>
                                    <Select
                                        value={retrieval?.reranker_model ?? "cross-encoder/ms-marco-MiniLM-L-6-v2"}
                                        onValueChange={(value) => updateRetrieval("reranker_model", value)}
                                    >
                                        <SelectTrigger id="rerankerModel" className="h-9 font-mono text-xs">
                                            <SelectValue placeholder="Select reranker" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {rerankerModels.map(model => (
                                                <SelectItem key={model} value={model} className="font-mono text-xs">
                                                    {model}
                                                </SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                    <p className="text-[10px] text-muted-foreground uppercase tracking-tight">Used for high-precision ranking</p>
                                </div>
                            </div>
                            <div className="grid gap-4 sm:grid-cols-2">
                                <div className="space-y-2">
                                    <Label htmlFor="useReranking" className="text-xs">Reranking</Label>
                                    <Select
                                        value={(retrieval?.use_reranking ?? true) ? "on" : "off"}
                                        onValueChange={(value) => updateRetrieval("use_reranking", value === "on")}
                                    >
                                        <SelectTrigger id="useReranking" className="h-9">
                                            <SelectValue placeholder="Select mode" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="on">Enabled (higher precision)</SelectItem>
                                            <SelectItem value="off">Disabled (faster)</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>
                                <div className="space-y-2">
                                    <Label className="text-xs">Model Assets</Label>
                                    <div className="flex flex-wrap items-center gap-2">
                                        <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ${modelStatus.embedding === "installed"
                                            ? "bg-primary/10 text-primary"
                                            : modelStatus.embedding === "missing"
                                                ? "bg-muted text-muted-foreground"
                                                : modelStatus.embedding === "error"
                                                    ? "bg-destructive/20 text-destructive"
                                                    : "bg-muted/60 text-muted-foreground"
                                            }`}>
                                            Embedding: {modelStatus.embedding}
                                        </span>
                                        <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ${modelStatus.reranker === "installed"
                                            ? "bg-primary/10 text-primary"
                                            : modelStatus.reranker === "missing"
                                                ? "bg-muted text-muted-foreground"
                                                : modelStatus.reranker === "error"
                                                    ? "bg-destructive/20 text-destructive"
                                                    : "bg-muted/60 text-muted-foreground"
                                            }`}>
                                            Reranker: {modelStatus.reranker}
                                        </span>
                                        <button
                                            type="button"
                                            onClick={onRefreshModelStatus}
                                            className="text-[10px] font-semibold text-muted-foreground hover:text-foreground"
                                            disabled={isCheckingModels}
                                        >
                                            {isCheckingModels ? "Checking..." : "Refresh"}
                                        </button>
                                    </div>
                                    {(modelStatus.embedding_message || modelStatus.reranker_message) && (
                                        <p className="text-[10px] text-muted-foreground">
                                            {modelStatus.embedding_message || modelStatus.reranker_message}
                                        </p>
                                    )}
                                    {modelActionMessage && (
                                        <p className="text-[10px] text-muted-foreground">{modelActionMessage}</p>
                                    )}
                                    <div className="flex flex-wrap gap-2">
                                        <Button
                                            type="button"
                                            variant="outline"
                                            size="sm"
                                            onClick={onDownloadEmbedding}
                                            disabled={isDownloadingEmbedding}
                                            className="h-8 text-xs"
                                        >
                                            {isDownloadingEmbedding ? "Downloading Embedding..." : "Download Embedding"}
                                        </Button>
                                        <Button
                                            type="button"
                                            variant="outline"
                                            size="sm"
                                            onClick={onDownloadReranker}
                                            disabled={isDownloadingReranker}
                                            className="h-8 text-xs"
                                        >
                                            {isDownloadingReranker ? "Downloading Reranker..." : "Download Reranker"}
                                        </Button>
                                        <Button
                                            type="button"
                                            variant="outline"
                                            size="sm"
                                            onClick={onDeleteEmbedding}
                                            className="h-8 text-xs"
                                        >
                                            Remove Embedding
                                        </Button>
                                        <Button
                                            type="button"
                                            variant="outline"
                                            size="sm"
                                            onClick={onDeleteReranker}
                                            className="h-8 text-xs"
                                        >
                                            Remove Reranker
                                        </Button>
                                    </div>
                                </div>
                            </div>
                        </section>

                        {/* Retrieval Parameters */}
                        <section className="space-y-4 rounded-lg border border-border/60 bg-muted/10 p-4">
                            <div className="flex items-center gap-2">
                                <div className="h-4 w-1 rounded-full bg-border" />
                                <div>
                                    <h4 className="text-sm font-semibold uppercase tracking-wider text-foreground">Operational Parameters</h4>
                                    <p className="text-xs text-muted-foreground">Tune recall, coverage, and rerank pool sizes.</p>
                                </div>
                            </div>
                            <div className="grid gap-4 grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
                                <div className="space-y-2">
                                    <Label htmlFor="targetTokens" className="text-xs text-muted-foreground">Target Tokens</Label>
                                    <Select
                                        value={String(retrieval?.target_tokens ?? 1600)}
                                        onValueChange={(value) => updateRetrieval("target_tokens", Number(value))}
                                    >
                                        <SelectTrigger id="targetTokens" className="h-9">
                                            <SelectValue placeholder="Select target" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {targetTokenOptions.map(option => (
                                                <SelectItem key={option} value={String(option)}>
                                                    {option}
                                                </SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="maxContextTokens" className="text-xs text-muted-foreground">Max Context</Label>
                                    <Select
                                        value={String(retrieval?.max_context_tokens ?? 4096)}
                                        onValueChange={(value) => updateRetrieval("max_context_tokens", Number(value))}
                                    >
                                        <SelectTrigger id="maxContextTokens" className="h-9">
                                            <SelectValue placeholder="Select max" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {maxContextOptions.map(option => (
                                                <SelectItem key={option} value={String(option)}>
                                                    {option}
                                                </SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="coverageTarget" className="text-xs text-muted-foreground">Coverage</Label>
                                    <Select
                                        value={String(retrieval?.coverage_target ?? 0.7)}
                                        onValueChange={(value) => updateRetrieval("coverage_target", Number(value))}
                                    >
                                        <SelectTrigger id="coverageTarget" className="h-9">
                                            <SelectValue placeholder="Select coverage" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {coverageOptions.map(option => (
                                                <SelectItem key={option} value={String(option)}>
                                                    {option}
                                                </SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="topN" className="text-xs text-muted-foreground">Top N</Label>
                                    <Select
                                        value={String(retrieval?.top_n ?? 5)}
                                        onValueChange={(value) => updateRetrieval("top_n", Number(value))}
                                    >
                                        <SelectTrigger id="topN" className="h-9">
                                            <SelectValue placeholder="Select top N" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {topNOptions.map(option => (
                                                <SelectItem key={option} value={String(option)}>
                                                    {option}
                                                </SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="denseK" className="text-xs text-muted-foreground">Dense K</Label>
                                    <Select
                                        value={String(retrieval?.dense_k ?? 5)}
                                        onValueChange={(value) => updateRetrieval("dense_k", Number(value))}
                                    >
                                        <SelectTrigger id="denseK" className="h-9">
                                            <SelectValue placeholder="Select dense K" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {denseKOptions.map(option => (
                                                <SelectItem key={option} value={String(option)}>
                                                    {option}
                                                </SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="sparseK" className="text-xs text-muted-foreground">Sparse K</Label>
                                    <Select
                                        value={String(retrieval?.sparse_k ?? 10)}
                                        onValueChange={(value) => updateRetrieval("sparse_k", Number(value))}
                                    >
                                        <SelectTrigger id="sparseK" className="h-9">
                                            <SelectValue placeholder="Select sparse K" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {sparseKOptions.map(option => (
                                                <SelectItem key={option} value={String(option)}>
                                                    {option}
                                                </SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="rerankPool" className="text-xs text-muted-foreground">Rerank Pool</Label>
                                    <Select
                                        value={String(retrieval?.rerank_pool ?? 20)}
                                        onValueChange={(value) => updateRetrieval("rerank_pool", Number(value))}
                                    >
                                        <SelectTrigger id="rerankPool" className="h-9">
                                            <SelectValue placeholder="Select rerank pool" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {rerankPoolOptions.map(option => (
                                                <SelectItem key={option} value={String(option)}>
                                                    {option}
                                                </SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                </div>
                            </div>
                        </section>

                        {/* Feature Toggles */}
                        <section className="space-y-4 rounded-lg border border-border/60 bg-muted/10 p-4">
                            <div className="flex items-center gap-2">
                                <div className="h-4 w-1 rounded-full bg-border" />
                                <div>
                                    <h4 className="text-sm font-semibold uppercase tracking-wider text-foreground">Pipeline Features</h4>
                                    <p className="text-xs text-muted-foreground">Enable optional stages for higher fidelity.</p>
                                </div>
                            </div>
                            <div className="grid gap-4 sm:grid-cols-[minmax(0,240px)_minmax(0,1fr)] sm:items-center">
                                <div className="space-y-2">
                                    <Label htmlFor="plannerMode" className="text-xs">Planner Mode</Label>
                                    <Select
                                        value={retrieval?.planner_mode ?? "smart"}
                                        onValueChange={(value) => updateRetrieval("planner_mode", value)}
                                    >
                                        <SelectTrigger id="plannerMode" className="h-9">
                                            <SelectValue placeholder="Select mode" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="smart">Smart (LLM-assisted)</SelectItem>
                                            <SelectItem value="simple">Simple (fast)</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>
                                <p className="text-xs text-muted-foreground">
                                    Smart planner decomposes complex queries; simple planner is faster and uses defaults.
                                </p>
                            </div>
                            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                                {[
                                    { id: "hybrid", label: "Hybrid Search", icon: Layers3 },
                                    { id: "compression", label: "Compression", icon: Package },
                                    { id: "raptor", label: "RAPTOR", icon: Boxes },
                                    { id: "graph", label: "Knowledge Graph", icon: GitMerge },
                                ].map((feature) => (
                                    <div key={feature.id} className="space-y-2 rounded-lg border border-border/60 bg-card p-3">
                                        <div className="flex items-center gap-2">
                                            <feature.icon className="h-4 w-4 text-foreground" />
                                            <span className="text-xs font-semibold">{feature.label}</span>
                                        </div>
                                        <Select
                                            value={retrieval?.[feature.id as keyof RetrievalDefaults] ? "on" : "off"}
                                            onValueChange={(value) => updateRetrieval(feature.id as keyof RetrievalDefaults, value === "on")}
                                        >
                                            <SelectTrigger
                                                className="h-8 text-xs"
                                                aria-label={`${feature.label} setting`}
                                            >
                                                <SelectValue placeholder="Select mode" />
                                            </SelectTrigger>
                                            <SelectContent>
                                                <SelectItem value="on">Enabled</SelectItem>
                                                <SelectItem value="off">Disabled</SelectItem>
                                            </SelectContent>
                                        </Select>
                                    </div>
                                ))}
                            </div>
                        </section>

                        {/* Binary Quantization (v2) */}
                        <section className="space-y-4 rounded-lg border border-border/60 bg-muted/10 p-4">
                            <div className="flex items-center gap-2">
                                <div className="h-4 w-1 rounded-full bg-primary" />
                                <div>
                                    <h4 className="text-sm font-semibold uppercase tracking-wider text-foreground flex items-center gap-2">
                                        <Binary className="h-4 w-4" />
                                        Binary Quantization (v2)
                                    </h4>
                                    <p className="text-xs text-muted-foreground">Memory-efficient retrieval with ~32x storage savings using binary vectors.</p>
                                </div>
                            </div>
                            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                                <div className="space-y-2">
                                    <Label htmlFor="retrievalMode" className="text-xs">Retrieval Mode</Label>
                                    <Select
                                        value={retrieval?.retrieval_mode ?? "float32"}
                                        onValueChange={(value) => updateRetrieval("retrieval_mode", value)}
                                    >
                                        <SelectTrigger id="retrievalMode" className="h-9">
                                            <SelectValue placeholder="Select mode" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="float32">Float32 (Standard)</SelectItem>
                                            <SelectItem value="binary">Binary (32x smaller)</SelectItem>
                                        </SelectContent>
                                    </Select>
                                    <p className="text-[10px] text-muted-foreground">Binary mode uses Hamming distance for fast search</p>
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="bqEnabled" className="text-xs">Binary Quantization</Label>
                                    <Select
                                        value={(retrieval?.bq_enabled ?? false) ? "on" : "off"}
                                        onValueChange={(value) => updateRetrieval("bq_enabled", value === "on")}
                                    >
                                        <SelectTrigger id="bqEnabled" className="h-9">
                                            <SelectValue placeholder="Select" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="on">Enabled</SelectItem>
                                            <SelectItem value="off">Disabled</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="bqNormalize" className="text-xs">Normalize Before Quantization</Label>
                                    <Select
                                        value={(retrieval?.bq_normalize ?? false) ? "on" : "off"}
                                        onValueChange={(value) => updateRetrieval("bq_normalize", value === "on")}
                                    >
                                        <SelectTrigger id="bqNormalize" className="h-9">
                                            <SelectValue placeholder="Select" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="on">L2 Normalize</SelectItem>
                                            <SelectItem value="off">No Normalization</SelectItem>
                                        </SelectContent>
                                    </Select>
                                    <p className="text-[10px] text-muted-foreground">L2-normalize vectors before sign thresholding</p>
                                </div>
                            </div>
                            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                                <div className="space-y-2">
                                    <Label htmlFor="bqTwoStage" className="text-xs">Two-Stage Retrieval</Label>
                                    <Select
                                        value={(retrieval?.bq_two_stage ?? false) ? "on" : "off"}
                                        onValueChange={(value) => updateRetrieval("bq_two_stage", value === "on")}
                                    >
                                        <SelectTrigger id="bqTwoStage" className="h-9">
                                            <SelectValue placeholder="Select" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="on">Enabled (binary + rerank)</SelectItem>
                                            <SelectItem value="off">Disabled</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="bqStage1Candidates" className="text-xs">Stage 1 Candidates</Label>
                                    <Select
                                        value={String(retrieval?.bq_stage1_candidates ?? 50)}
                                        onValueChange={(value) => updateRetrieval("bq_stage1_candidates", Number(value))}
                                    >
                                        <SelectTrigger id="bqStage1Candidates" className="h-9">
                                            <SelectValue placeholder="Select" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {[20, 50, 100, 150, 200].map(n => (
                                                <SelectItem key={n} value={String(n)}>{n}</SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="bqFallback" className="text-xs">Fallback to Float32</Label>
                                    <Select
                                        value={(retrieval?.bq_fallback_enabled ?? true) ? "on" : "off"}
                                        onValueChange={(value) => updateRetrieval("bq_fallback_enabled", value === "on")}
                                    >
                                        <SelectTrigger id="bqFallback" className="h-9">
                                            <SelectValue placeholder="Select" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="on">Auto-fallback on low confidence</SelectItem>
                                            <SelectItem value="off">No fallback</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="bqFallbackThreshold" className="text-xs">Fallback Threshold</Label>
                                    <Select
                                        value={String(retrieval?.bq_fallback_threshold ?? 500)}
                                        onValueChange={(value) => updateRetrieval("bq_fallback_threshold", Number(value))}
                                    >
                                        <SelectTrigger id="bqFallbackThreshold" className="h-9">
                                            <SelectValue placeholder="Select" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {[200, 300, 400, 500, 600, 700, 800].map(n => (
                                                <SelectItem key={n} value={String(n)}>{n}</SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                    <p className="text-[10px] text-muted-foreground">Hamming distance threshold</p>
                                </div>
                            </div>
                            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                                <div className="space-y-2">
                                    <Label htmlFor="bqRule" className="text-xs">Quantization Rule</Label>
                                    <Select
                                        value={retrieval?.bq_rule ?? "sign_threshold_0"}
                                        onValueChange={(value) => updateRetrieval("bq_rule", value)}
                                    >
                                        <SelectTrigger id="bqRule" className="h-9">
                                            <SelectValue placeholder="Select rule" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="sign_threshold_0">Sign threshold (&gt;= 0)</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="milvusIndexType" className="text-xs">Milvus Index Type</Label>
                                    <Select
                                        value={retrieval?.milvus_index_type ?? "BIN_FLAT"}
                                        onValueChange={(value) => updateRetrieval("milvus_index_type", value)}
                                    >
                                        <SelectTrigger id="milvusIndexType" className="h-9">
                                            <SelectValue placeholder="Select index" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="BIN_FLAT">BIN_FLAT</SelectItem>
                                            <SelectItem value="BIN_IVF_FLAT">BIN_IVF_FLAT</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="milvusMetric" className="text-xs">Milvus Metric</Label>
                                    <Select
                                        value={retrieval?.milvus_metric ?? "HAMMING"}
                                        onValueChange={(value) => updateRetrieval("milvus_metric", value)}
                                    >
                                        <SelectTrigger id="milvusMetric" className="h-9">
                                            <SelectValue placeholder="Select metric" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="HAMMING">HAMMING</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>
                            </div>
                            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                                <div className="space-y-2">
                                    <Label htmlFor="milvusHost" className="text-xs">Milvus Host</Label>
                                    <Input
                                        id="milvusHost"
                                        value={retrieval?.milvus_host ?? "localhost"}
                                        onChange={(e) => updateRetrieval("milvus_host", e.target.value)}
                                        placeholder="localhost"
                                        className="h-9"
                                    />
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="milvusPort" className="text-xs">Milvus Port</Label>
                                    <Input
                                        id="milvusPort"
                                        type="number"
                                        value={String(retrieval?.milvus_port ?? 19530)}
                                        onChange={(e) => updateRetrieval("milvus_port", Number(e.target.value))}
                                        placeholder="19530"
                                        className="h-9"
                                    />
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="milvusCollection" className="text-xs">Collection Name</Label>
                                    <Input
                                        id="milvusCollection"
                                        value={retrieval?.milvus_collection ?? "jr_autorag_chunks_bq"}
                                        onChange={(e) => updateRetrieval("milvus_collection", e.target.value)}
                                        placeholder="jr_autorag_chunks_bq"
                                        className="h-9"
                                    />
                                </div>
                            </div>
                            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-2">
                                <div className="space-y-2">
                                    <Label htmlFor="milvusNlist" className="text-xs">Milvus nlist</Label>
                                    <Input
                                        id="milvusNlist"
                                        type="number"
                                        value={String(retrieval?.milvus_nlist ?? 128)}
                                        onChange={(e) => updateRetrieval("milvus_nlist", Number(e.target.value))}
                                        placeholder="128"
                                        className="h-9"
                                    />
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="milvusNprobe" className="text-xs">Milvus nprobe</Label>
                                    <Input
                                        id="milvusNprobe"
                                        type="number"
                                        value={String(retrieval?.milvus_nprobe ?? 16)}
                                        onChange={(e) => updateRetrieval("milvus_nprobe", Number(e.target.value))}
                                        placeholder="16"
                                        className="h-9"
                                    />
                                </div>
                            </div>
                        </section>

                        {/* LangExtract Enrichment */}
                        <section className="space-y-4 rounded-lg border border-border/60 bg-muted/10 p-4">
                            <div className="flex items-center gap-2">
                                <div className="h-4 w-1 rounded-full bg-primary" />
                                <div>
                                    <h4 className="text-sm font-semibold uppercase tracking-wider text-foreground">
                                        LangExtract Enrichment
                                    </h4>
                                    <p className="text-xs text-muted-foreground">
                                        Optional ingestion-time structured fact extraction (fail-open).
                                    </p>
                                </div>
                            </div>
                            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                                <div className="space-y-2">
                                    <Label htmlFor="langextractEnabled" className="text-xs">Enabled</Label>
                                    <Select
                                        value={(retrieval?.langextract_enabled ?? false) ? "on" : "off"}
                                        onValueChange={(value) => updateRetrieval("langextract_enabled", value === "on")}
                                    >
                                        <SelectTrigger id="langextractEnabled" className="h-9">
                                            <SelectValue placeholder="Select mode" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="off">Disabled</SelectItem>
                                            <SelectItem value="on">Enabled</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="langextractProfileDefault" className="text-xs">Default Profile</Label>
                                    <Select
                                        value={retrieval?.langextract_profile_default ?? "generic_entities_v1"}
                                        onValueChange={(value) => updateRetrieval("langextract_profile_default", value)}
                                    >
                                        <SelectTrigger id="langextractProfileDefault" className="h-9">
                                            <SelectValue placeholder="Select profile" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {langExtractProfiles.map(profile => (
                                                <SelectItem key={profile} value={profile}>{profile}</SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="langextractModelSource" className="text-xs">Model Source</Label>
                                    <Select
                                        value={retrieval?.langextract_model_source ?? "gatherer"}
                                        onValueChange={(value) => updateRetrieval("langextract_model_source", value)}
                                    >
                                        <SelectTrigger id="langextractModelSource" className="h-9">
                                            <SelectValue placeholder="Select role" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="planner">planner</SelectItem>
                                            <SelectItem value="gatherer">gatherer</SelectItem>
                                            <SelectItem value="generator">generator</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="langextractTimeout" className="text-xs">Timeout (sec)</Label>
                                    <Select
                                        value={String(retrieval?.langextract_timeout_sec ?? 20)}
                                        onValueChange={(value) => updateRetrieval("langextract_timeout_sec", Number(value))}
                                    >
                                        <SelectTrigger id="langextractTimeout" className="h-9">
                                            <SelectValue placeholder="Select timeout" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {langExtractTimeouts.map(option => (
                                                <SelectItem key={option} value={String(option)}>{option}</SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="langextractMaxChars" className="text-xs">Max Source Chars</Label>
                                    <Select
                                        value={String(retrieval?.langextract_max_chars ?? 12000)}
                                        onValueChange={(value) => updateRetrieval("langextract_max_chars", Number(value))}
                                    >
                                        <SelectTrigger id="langextractMaxChars" className="h-9">
                                            <SelectValue placeholder="Select max chars" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {langExtractMaxChars.map(option => (
                                                <SelectItem key={option} value={String(option)}>{option}</SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                </div>
                                <div className="space-y-2">
                                    <Label htmlFor="langextractMaxFacts" className="text-xs">Max Synthetic Facts</Label>
                                    <Select
                                        value={String(retrieval?.langextract_max_synthetic_facts ?? 200)}
                                        onValueChange={(value) => updateRetrieval("langextract_max_synthetic_facts", Number(value))}
                                    >
                                        <SelectTrigger id="langextractMaxFacts" className="h-9">
                                            <SelectValue placeholder="Select cap" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            {langExtractMaxFacts.map(option => (
                                                <SelectItem key={option} value={String(option)}>{option}</SelectItem>
                                            ))}
                                        </SelectContent>
                                    </Select>
                                </div>
                            </div>
                        </section>

                        <section className="pt-2">
                            <div className="flex justify-end">
                                <Button
                                    onClick={onSave}
                                    disabled={isSaving}
                                    className="px-8"
                                >
                                    {isSaving ? "Saving..." : "Apply Advanced Settings"}
                                </Button>
                            </div>
                        </section>
                    </div>
                </CardContent>
            )}
        </Card>
    );
}
