import { useMemo, useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { Boxes, ChevronDown, GitMerge, Layers3, Package, Sliders, Info, ShieldCheck, Loader2 } from "lucide-react";
import type { RetrievalDefaults } from "@/types";

interface AdvancedRAGSettingsProps {
    retrieval: RetrievalDefaults | undefined;
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
            <Info className="h-3.5 w-3.5 text-secondary-foreground" />
            <span className="truncate">{label}</span>
        </span>
    );
}

export function AdvancedRAGSettings({
    retrieval,
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
    const [preset, setPreset] = useState("balanced");
    const chunkSizeOptions = [100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 800, 1000];
    const overlapOptions = [0, 50, 100, 150, 200];
    const targetTokenOptions = [800, 1000, 1200, 1500, 1800, 2000, 2400, 3000, 3600];
    const maxContextOptions = [2048, 3072, 4096, 6144, 8192, 12288];
    const coverageOptions = [0.5, 0.6, 0.7, 0.8, 0.9];
    const topNOptions = [3, 5, 8, 10, 12, 15, 20];
    const denseKOptions = [5, 10, 15, 20, 30, 40, 50];
    const sparseKOptions = [10, 20, 30, 40, 60, 80, 100];
    const rerankPoolOptions = [10, 20, 30, 40, 50, 60, 80, 100];
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
    const presets: Record<string, Partial<RetrievalDefaults>> = useMemo(() => ({
        balanced: {
            chunking_strategy: "fixed",
            chunk_size: 400,
            chunk_overlap: 50,
            target_tokens: 1600,
            max_context_tokens: 4096,
            coverage_target: 0.7,
            top_n: 5,
            dense_k: 10,
            sparse_k: 20,
            rerank_pool: 20,
            hybrid: true,
            compression: false,
            raptor: false,
            graph: false,
            use_reranking: true,
            embedding_model: "BAAI/bge-base-en-v1.5",
            reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2",
        },
        fast: {
            chunking_strategy: "fixed",
            chunk_size: 600,
            chunk_overlap: 50,
            target_tokens: 1200,
            max_context_tokens: 3072,
            coverage_target: 0.6,
            top_n: 3,
            dense_k: 5,
            sparse_k: 10,
            rerank_pool: 10,
            hybrid: false,
            compression: false,
            raptor: false,
            graph: false,
            use_reranking: false,
            embedding_model: "BAAI/bge-small-en-v1.5",
            reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2",
        },
        deep: {
            chunking_strategy: "semantic",
            chunk_size: 450,
            chunk_overlap: 100,
            target_tokens: 2400,
            max_context_tokens: 8192,
            coverage_target: 0.8,
            top_n: 8,
            dense_k: 20,
            sparse_k: 40,
            rerank_pool: 40,
            hybrid: true,
            compression: true,
            raptor: true,
            graph: false,
            use_reranking: true,
            embedding_model: "BAAI/bge-large-en-v1.5",
            reranker_model: "BAAI/bge-reranker-large",
        },
        recall: {
            chunking_strategy: "recursive",
            chunk_size: 400,
            chunk_overlap: 100,
            target_tokens: 2000,
            max_context_tokens: 6144,
            coverage_target: 0.9,
            top_n: 12,
            dense_k: 30,
            sparse_k: 60,
            rerank_pool: 60,
            hybrid: true,
            compression: false,
            raptor: true,
            graph: true,
            use_reranking: true,
            embedding_model: "intfloat/e5-large-v2",
            reranker_model: "cross-encoder/ms-marco-MiniLM-L-12-v2",
        },
        precise: {
            chunking_strategy: "fixed",
            chunk_size: 300,
            chunk_overlap: 50,
            target_tokens: 1500,
            max_context_tokens: 4096,
            coverage_target: 0.7,
            top_n: 3,
            dense_k: 10,
            sparse_k: 20,
            rerank_pool: 30,
            hybrid: true,
            compression: true,
            raptor: false,
            graph: false,
            use_reranking: true,
            embedding_model: "BAAI/bge-base-en-v1.5",
            reranker_model: "mixedbread-ai/mxbai-rerank-base-v1",
        },
    }), []);
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
                            <Loader2 className="h-3.5 w-3.5 animate-spin text-secondary-foreground" />
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
                                    <h4 className="text-sm font-semibold uppercase tracking-wider text-foreground">Presets</h4>
                                    <p className="text-xs text-muted-foreground">Apply tuned defaults for common workflows.</p>
                                </div>
                            </div>
                            <div className="grid gap-4 sm:grid-cols-[minmax(0,240px)_minmax(0,1fr)] sm:items-center">
                                <div className="space-y-2">
                                    <Label htmlFor="presetSelect" className="text-xs">Quick Start</Label>
                                    <Select
                                        value={preset}
                                        onValueChange={(value) => {
                                            setPreset(value);
                                            const selected = presets[value];
                                            if (!selected) return;
                                            (Object.entries(selected) as Array<[keyof RetrievalDefaults, RetrievalDefaults[keyof RetrievalDefaults]]>)
                                                .forEach(([key, val]) => updateRetrieval(key, val));
                                        }}
                                    >
                                        <SelectTrigger id="presetSelect" className="h-9">
                                            <SelectValue placeholder="Select a preset" />
                                        </SelectTrigger>
                                        <SelectContent>
                                            <SelectItem value="balanced">Balanced (recommended)</SelectItem>
                                            <SelectItem value="fast">Fast answers</SelectItem>
                                            <SelectItem value="deep">Deep analysis</SelectItem>
                                            <SelectItem value="recall">High recall</SelectItem>
                                            <SelectItem value="precise">Precision mode</SelectItem>
                                        </SelectContent>
                                    </Select>
                                </div>
                                <p className="text-xs text-muted-foreground">
                                    Presets apply a tuned mix of chunking, retrieval, and reranking for common use cases.
                                </p>
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
                                        <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ${
                                            modelStatus.embedding === "installed"
                                                ? "bg-primary/10 text-primary"
                                                : modelStatus.embedding === "missing"
                                                    ? "bg-muted text-muted-foreground"
                                                    : modelStatus.embedding === "error"
                                                        ? "bg-destructive/20 text-destructive"
                                                        : "bg-muted/60 text-muted-foreground"
                                        }`}>
                                            Embedding: {modelStatus.embedding}
                                        </span>
                                        <span className={`rounded-full px-2.5 py-1 text-[10px] font-semibold ${
                                            modelStatus.reranker === "installed"
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
                                            <SelectTrigger className="h-8 text-xs">
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
