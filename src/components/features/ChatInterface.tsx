import { useMemo, useState } from "react";
import {
    Activity,
    AlertTriangle,
    Box,
    CheckCircle2,
    Database,
    FileSearch,
    Loader2,
    MessageSquare,
    Search,
    Sparkles,
    Target,
    Gauge,
    FileText,
    Layers,
    ArrowRightLeft,
    HelpCircle,
    Info,
    UploadCloud,
    ShieldCheck,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import type { DocumentOut, PipelineStep, ProviderConfig, QueryResponse } from "@/types";

interface ChatInterfaceProps {
    question: string;
    setQuestion: (value: string) => void;
    isQuerying: boolean;
    handleAsk: () => void;
    queryResult: QueryResponse | null;
    documents: DocumentOut[];
    selectedDocumentIds: string[];
    setSelectedDocumentIds: React.Dispatch<React.SetStateAction<string[]>>;
    providerConfig?: ProviderConfig;
    activeStage?: string | null;
}

function StepIcon({ name, status }: { name: string; status: string }) {
    const isComplete = status === "completed";
    const isSkipped = status === "skipped";
    const icons: Record<string, typeof Target> = {
        cache: Database,
        planning: Target,
        gatherer: FileSearch,
        retrieval: Search,
        generation: Sparkles,
        compression: Box,
        reflection: Activity,
    };
    const Icon = icons[name] ?? Target;
    return (
        <span className={`flex h-8 w-8 items-center justify-center rounded-lg text-lg transition-all ${isComplete
                ? "bg-primary/10"
                : isSkipped
                    ? "bg-muted"
                    : "bg-secondary/30"
            }`}>
            <Icon className="h-4 w-4 text-foreground" />
        </span>
    );
}

function StepDetails({ step }: { step: PipelineStep }) {
    const details = step.details;

    if (step.name === "planning") {
        const queries = (details.queries as string[] | undefined) ?? [];
        const queryType = details.query_type as string | undefined;
        const plannerMode = details.planner_mode as string | undefined;
        const expandedTerms = (details.expanded_terms as string[] | undefined) ?? [];
        return (
            <div className="mt-2 space-y-2 text-sm">
                {(queryType || plannerMode) && (
                    <span className="inline-flex items-center rounded-full bg-secondary/30 px-2.5 py-0.5 text-xs font-medium text-secondary-foreground">
                        {queryType ?? "planning"}{plannerMode ? ` • ${plannerMode}` : ""}
                    </span>
                )}
                {expandedTerms.length > 0 && (
                    <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                        {expandedTerms.slice(0, 5).map(term => (
                            <span key={term} className="rounded-full bg-muted px-2 py-0.5">
                                {term}
                            </span>
                        ))}
                    </div>
                )}
                {queries.length > 0 && (
                    <div className="space-y-1">
                        <p className="text-xs font-medium text-muted-foreground">Sub-queries:</p>
                        <ul className="ml-4 space-y-1 text-muted-foreground">
                            {queries.map((q, i) => (
                                <li key={i} className="flex items-start gap-2">
                                    <span className="text-muted-foreground">•</span>
                                    <span>{q}</span>
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
            </div>
        );
    }

    if (step.name === "retrieval") {
        const totalChunks = details.total_chunks as number | undefined;
        const uniqueSources = details.unique_sources as number | undefined;
        const documentFilter = details.document_filter as string[] | undefined;
        const embeddingHits = details.embedding_cache_hits as number | undefined;
        const embeddingMisses = details.embedding_cache_misses as number | undefined;
        const denseEnabled = details.dense_enabled as boolean | undefined;
        const rerankerEnabled = details.reranker_enabled as boolean | undefined;
        return (
            <div className="mt-2 flex flex-wrap gap-3 text-sm">
                <span className="inline-flex items-center gap-1.5 rounded-lg bg-muted px-2.5 py-1 text-xs">
                    <FileText className="h-3.5 w-3.5" />
                    {totalChunks ?? 0} chunks
                </span>
                <span className="inline-flex items-center gap-1.5 rounded-lg bg-muted px-2.5 py-1 text-xs">
                    <Layers className="h-3.5 w-3.5" />
                    {uniqueSources ?? 0} sources
                </span>
                {documentFilter && (
                    <span className="inline-flex items-center gap-1.5 rounded-lg bg-muted px-2.5 py-1 text-xs">
                        <Target className="h-3.5 w-3.5" />
                        {documentFilter.length} docs filtered
                    </span>
                )}
                {(typeof embeddingHits === "number" || typeof embeddingMisses === "number") && (
                    <span className="inline-flex items-center gap-1.5 rounded-lg bg-muted px-2.5 py-1 text-xs">
                        <Database className="h-3.5 w-3.5" />
                        {embeddingHits ?? 0} hit / {embeddingMisses ?? 0} miss
                    </span>
                )}
                {typeof denseEnabled === "boolean" && (
                    <span className="inline-flex items-center gap-1.5 rounded-lg bg-muted px-2.5 py-1 text-xs">
                        Dense: {denseEnabled ? "on" : "off"}
                    </span>
                )}
                {typeof rerankerEnabled === "boolean" && (
                    <span className="inline-flex items-center gap-1.5 rounded-lg bg-muted px-2.5 py-1 text-xs">
                        Rerank: {rerankerEnabled ? "on" : "off"}
                    </span>
                )}
            </div>
        );
    }

    if (step.name === "generation") {
        const provider = details.provider as string | undefined;
        const model = details.model as string | undefined;
        const fallback = details.fallback as boolean | undefined;
        return (
            <div className="mt-2 space-y-1 text-sm">
                <div className="flex flex-wrap gap-2">
                    {provider && (
                        <span className="inline-flex items-center rounded-lg bg-muted px-2.5 py-1 text-xs">
                            <Gauge className="mr-1 h-3.5 w-3.5" />
                            {provider}
                        </span>
                    )}
                    {model && (
                        <span className="inline-flex items-center rounded-lg bg-muted px-2.5 py-1 text-xs font-mono">
                            {model}
                        </span>
                    )}
                </div>
                {fallback && (
                    <p className="text-xs text-destructive">
                        <AlertTriangle className="mr-1 inline h-3.5 w-3.5" />
                        Using fallback (no LLM configured)
                    </p>
                )}
            </div>
        );
    }

    if (step.name === "compression") {
        const enabled = details.enabled as boolean | undefined;
        const chunksUsed = details.chunks_used as number | undefined;
        const chunksTotal = details.chunks_total as number | undefined;
        const estimatedTokens = details.estimated_tokens as number | undefined;
        return (
            <div className="mt-2 flex flex-wrap gap-3 text-sm">
                <span className="inline-flex items-center gap-1.5 rounded-lg bg-muted px-2.5 py-1 text-xs">
                    {enabled ? <CheckCircle2 className="h-3.5 w-3.5" /> : <ArrowRightLeft className="h-3.5 w-3.5" />}
                    {enabled ? "compression" : "skipped"}
                </span>
                <span className="inline-flex items-center gap-1.5 rounded-lg bg-muted px-2.5 py-1 text-xs">
                    <FileText className="h-3.5 w-3.5" />
                    {chunksUsed ?? 0}/{chunksTotal ?? 0}
                </span>
                {typeof estimatedTokens === "number" && (
                    <span className="inline-flex items-center gap-1.5 rounded-lg bg-muted px-2.5 py-1 text-xs">
                        <FileSearch className="h-3.5 w-3.5" />
                        ~{estimatedTokens} tokens
                    </span>
                )}
            </div>
        );
    }

    if (step.name === "reflection") {
        const quality = details.quality as string | undefined;
        const confidence = details.confidence as number | undefined;
        const shouldRetry = details.should_retry as boolean | undefined;
        return (
            <div className="mt-2 flex flex-wrap gap-3 text-sm">
                {quality && (
                    <span className="inline-flex items-center gap-1.5 rounded-lg bg-muted px-2.5 py-1 text-xs">
                        <Activity className="h-3.5 w-3.5" />
                        {quality}
                    </span>
                )}
                {typeof confidence === "number" && (
                    <span className="inline-flex items-center gap-1.5 rounded-lg bg-muted px-2.5 py-1 text-xs">
                        <Gauge className="h-3.5 w-3.5" />
                        {(confidence * 100).toFixed(0)}%
                    </span>
                )}
                {typeof shouldRetry === "boolean" && (
                    <span className="inline-flex items-center gap-1.5 rounded-lg bg-muted px-2.5 py-1 text-xs">
                        {shouldRetry ? "Retry" : "No retry"}
                    </span>
                )}
            </div>
        );
    }

    if (step.name === "gatherer") {
        const subQueries = (details.sub_queries as Array<{ query: string; duration_ms?: number; literal_hits?: number }> | undefined) ?? [];
        const embeddingHits = details.embedding_cache_hits as number | undefined;
        const embeddingMisses = details.embedding_cache_misses as number | undefined;
        const literalHits = details.literal_hits as number | undefined;
        return (
            <div className="mt-2 space-y-2 text-sm">
                <div className="flex flex-wrap gap-2">
                    <span className="inline-flex items-center rounded-lg bg-muted px-2.5 py-1 text-xs">
                        {subQueries.length} sub-queries
                    </span>
                    {(typeof embeddingHits === "number" || typeof embeddingMisses === "number") && (
                        <span className="inline-flex items-center rounded-lg bg-muted px-2.5 py-1 text-xs">
                            Cache: {embeddingHits ?? 0} hit / {embeddingMisses ?? 0} miss
                        </span>
                    )}
                    {typeof literalHits === "number" && (
                        <span className="inline-flex items-center rounded-lg bg-muted px-2.5 py-1 text-xs">
                            Literal hits: {literalHits}
                        </span>
                    )}
                </div>
                {subQueries.length > 0 && (
                    <div className="space-y-1 text-xs text-muted-foreground">
                        {subQueries.slice(0, 3).map((item, idx) => (
                            <div key={idx} className="flex items-center justify-between gap-2">
                                <span className="truncate">{item.query}</span>
                                {typeof item.duration_ms === "number" && (
                                    <span className="shrink-0">{item.duration_ms.toFixed(0)}ms</span>
                                )}
                            </div>
                        ))}
                    </div>
                )}
            </div>
        );
    }

    if (step.name === "cache") {
        const queryCache = details.query_cache as string | undefined;
        return (
            <div className="mt-2 flex flex-wrap gap-3 text-sm">
                <span className="inline-flex items-center gap-1.5 rounded-lg bg-muted px-2.5 py-1 text-xs">
                    <Database className="h-3.5 w-3.5" />
                    Query cache: {queryCache ?? "unknown"}
                </span>
            </div>
        );
    }

    return null;
}

function PipelinePanel({ steps, metrics }: { steps: PipelineStep[]; metrics: Record<string, number | string> }) {
    const [expanded, setExpanded] = useState(true);
    const totalMs = typeof metrics.duration_ms === "number" ? metrics.duration_ms : 0;
    const formatStepMs = (ms: number) => (ms < 1 ? "<1ms" : `${ms.toFixed(0)}ms`);
    const formatTime = (value?: string) => {
        if (!value) return "";
        const date = new Date(value);
        if (Number.isNaN(date.getTime())) return "";
        return date.toLocaleTimeString();
    };

    return (
        <div className="rounded-lg border border-border/60 bg-card p-4">
            <button
                type="button"
                className="flex w-full flex-wrap items-center justify-between gap-3 text-left"
                onClick={() => setExpanded(!expanded)}
            >
                <div className="flex min-w-0 items-center gap-3">
                    <div className="min-w-0">
                        <span className="font-semibold text-foreground">Pipeline Details</span>
                        <span className="ml-2 text-sm text-muted-foreground whitespace-nowrap">
                            {totalMs.toFixed(0)}ms total
                        </span>
                    </div>
                </div>
                <span className="text-xs font-medium text-muted-foreground whitespace-nowrap">
                    {expanded ? "Hide ▲" : "Show ▼"}
                </span>
            </button>

            {expanded && (
                <div className="mt-4 space-y-3">
                    {steps.map((step, idx) => (
                        <div
                            key={idx}
                            className="rounded-md border border-border/60 bg-card p-4 transition-colors"
                        >
                            <div className="flex flex-wrap items-center justify-between gap-2">
                                <div className="flex min-w-0 flex-1 items-center gap-3">
                                    <StepIcon name={step.name} status={step.status} />
                                    <span className="min-w-0 truncate font-medium capitalize text-foreground">{step.name}</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className={`text-xs font-medium ${step.status === "completed"
                                            ? "text-primary"
                                            : step.status === "skipped"
                                                ? "text-muted-foreground"
                                                : "text-secondary-foreground"
                                        }`}>
                                        {step.status === "skipped" ? "skipped" : step.status}
                                    </span>
                                    <span className="text-xs text-muted-foreground whitespace-nowrap">
                                        {formatStepMs(step.duration_ms)}
                                    </span>
                                </div>
                            </div>
                            {(step.started_at || step.completed_at) && (
                                <div className="mt-2 text-[10px] uppercase tracking-wide text-muted-foreground">
                                    {step.started_at ? `Start ${formatTime(step.started_at)}` : ""}
                                    {step.started_at && step.completed_at ? " • " : ""}
                                    {step.completed_at ? `End ${formatTime(step.completed_at)}` : ""}
                                </div>
                            )}
                            <StepDetails step={step} />
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

type CitationSource = {
    id: string;
    title: string;
    snippet_preview?: string;
    citation_number?: number;
    score?: number;
};

function SkeletonBlock({ className }: { className?: string }) {
    return (
        <div className={`animate-pulse rounded-lg bg-muted/60 ${className ?? ""}`} />
    );
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

function SourcesList({ sources }: { sources: CitationSource[] }) {
    const [showAll, setShowAll] = useState(false);
    const displaySources = showAll ? sources : sources.slice(0, 3);

    return (
        <div className="space-y-3 min-w-0">
            <div className="flex items-center justify-between gap-3 min-w-0">
                <h4 className="text-sm font-semibold text-foreground truncate">
                    Sources ({sources.length})
                </h4>
                {sources.length > 3 && (
                    <button
                        type="button"
                        onClick={() => setShowAll(!showAll)}
                        className="text-xs font-medium text-secondary-foreground hover:text-foreground whitespace-nowrap"
                    >
                        {showAll ? "Show less" : `Show all ${sources.length}`}
                    </button>
                )}
            </div>
            <div className="grid gap-2 min-w-0">
                {displaySources.map((source, idx) => (
                    <div
                        key={`${source.id}-${idx}`}
                        className="group flex min-w-0 gap-3 rounded-md border border-border/60 bg-card p-3 transition-colors hover:bg-muted/20"
                    >
                        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-secondary/40 text-xs font-bold text-foreground">
                            {source.citation_number ?? idx + 1}
                        </span>
                        <div className="flex-1 min-w-0">
                            <div className="flex min-w-0 items-center gap-2">
                                <span className="min-w-0 truncate font-medium text-foreground">{source.title}</span>
                                {typeof source.score === "number" && (
                                    <span className="shrink-0 rounded-full bg-primary/10 px-2 py-0.5 text-xs font-medium text-primary">
                                        {(source.score * 100).toFixed(0)}%
                                    </span>
                                )}
                            </div>
                            <p className="mt-1 text-sm text-muted-foreground line-clamp-2">
                                {source.snippet_preview ?? "No snippet available."}
                            </p>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}

export function ChatInterface({
    question,
    setQuestion,
    isQuerying,
    handleAsk,
    queryResult,
    documents,
    selectedDocumentIds,
    setSelectedDocumentIds,
    providerConfig,
    activeStage,
}: ChatInterfaceProps) {
    const liveStages = useMemo(() => ([
        { name: "cache", label: "Cache", icon: Database },
        { name: "planning", label: "Planning", icon: Target },
        { name: "gatherer", label: "Gatherer", icon: FileSearch },
        { name: "retrieval", label: "Retrieval", icon: Search },
        { name: "compression", label: "Compression", icon: Box },
        { name: "generation", label: "Generation", icon: Sparkles },
        { name: "reflection", label: "Reflection", icon: Activity },
    ]), []);
    const completedStages = useMemo(() => new Set(
        (queryResult?.steps ?? []).map(step => step.name)
    ), [queryResult?.steps]);
    const displaySteps = useMemo(() => {
        const steps = queryResult?.steps ?? [];
        if (!steps.length) {
            return steps;
        }
        const hasGatherer = steps.some(step => step.name === "gatherer");
        const shouldShowGatherer = Boolean(providerConfig?.gatherer_model) || activeStage === "gatherer" || isQuerying;
        if (hasGatherer || !shouldShowGatherer) {
            return steps;
        }
        const insertIndex = (() => {
            const retrievalIndex = steps.findIndex(step => step.name === "retrieval");
            if (retrievalIndex !== -1) return retrievalIndex;
            const planningIndex = steps.findIndex(step => step.name === "planning");
            if (planningIndex !== -1) return planningIndex + 1;
            return 0;
        })();
        const status = activeStage === "gatherer" ? "running" : isQuerying ? "pending" : "skipped";
        const gathererStep: PipelineStep = {
            name: "gatherer",
            status,
            duration_ms: 0,
            details: {},
        };
        const next = [...steps];
        next.splice(insertIndex, 0, gathererStep);
        return next;
    }, [queryResult?.steps, providerConfig?.gatherer_model, activeStage, isQuerying]);
    const hasPipeline = Boolean(queryResult?.steps && queryResult.steps.length > 0);
    const hasAnswer = Boolean(queryResult?.answer);
    const sourcesForDisplay = queryResult
        ? (queryResult.sources && queryResult.sources.length > 0
            ? queryResult.sources
            : queryResult.chunks.map((chunk, idx) => ({
                id: chunk.id,
                title: chunk.title,
                snippet_preview: chunk.snippet,
                citation_number: idx + 1,
                score: chunk.score,
            })))
        : [];
    const hasSources = sourcesForDisplay.length > 0;
    const showPipelineSkeleton = isQuerying && !hasPipeline;
    const showAnswerSkeleton = isQuerying && !hasAnswer;
    const showSourcesSkeleton = isQuerying && !hasSources;
    const layoutHasPipeline = hasPipeline || isQuerying;
    const layoutHasSources = hasSources || isQuerying;
    const resultGridClass = layoutHasPipeline && layoutHasSources
        ? "xl:grid-cols-[minmax(240px,1fr)_minmax(0,2fr)_minmax(240px,1fr)]"
        : layoutHasPipeline || layoutHasSources
            ? "xl:grid-cols-[minmax(0,1fr)_minmax(0,2fr)]"
            : "lg:grid-cols-1";

    const allSelected = documents.length > 0 && selectedDocumentIds.length === documents.length;
    const providerReady = Boolean(providerConfig?.generator_model || providerConfig?.planner_model || providerConfig?.gatherer_model);
    const hasDocs = documents.length > 0;
    const missingDocs = !hasDocs;
    const missingProvider = !providerReady;

    const activeStageCopy: Record<string, string> = {
        cache: "Checking cache hits...",
        planning: "Planning queries...",
        gatherer: "Gathering documents...",
        retrieval: "Retrieving relevant chunks...",
        compression: "Compressing context...",
        generation: "Generating answer...",
        reflection: "Evaluating answer quality...",
    };

    const toggleAll = () => {
        if (selectedDocumentIds.length === 0) {
            setSelectedDocumentIds(documents.map(doc => doc.id));
        } else {
            setSelectedDocumentIds([]);
        }
    };

    const toggleDoc = (docId: string) => {
        setSelectedDocumentIds(prev => (
            prev.includes(docId)
                ? prev.filter(id => id !== docId)
                : [...prev, docId]
        ));
    };

    return (
        <Card className="overflow-hidden">
            <CardHeader className="bg-muted/20">
                <CardTitle className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                        <MessageSquare className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                        <span>Query Interface</span>
                    </div>
                </CardTitle>
                <CardDescription>
                    Query your documents through the full RAG pipeline with transparent step-by-step processing.
                </CardDescription>
                <div className="mt-3 flex flex-wrap gap-2">
                    <InlineHint
                        label={hasDocs ? "Step 1: Documents ready" : "Step 1: Add documents"}
                        detail="Upload PDFs, DOCX, Markdown, or paste text before asking."
                    />
                    <InlineHint
                        label={providerReady ? "Step 2: Provider set" : "Step 2: Pick models"}
                        detail="Choose planner/gatherer/generator models in Provider Settings."
                    />
                    <InlineHint
                        label="Step 3: Ask & observe"
                        detail="Run a question, then watch pipeline + sources update live."
                    />
                </div>
            </CardHeader>
            <CardContent className="space-y-6 pt-6">
                {/* Premium setup checklist */}
                {(missingDocs || missingProvider) && (
                    <div className="flex flex-col gap-3 rounded-lg border border-border/70 bg-muted/10 p-4">
                        <div className="flex items-center justify-between gap-3">
                            <div className="flex items-center gap-2">
                                <ShieldCheck className="h-4 w-4 text-primary" />
                                <p className="text-sm font-semibold text-foreground">Getting ready</p>
                            </div>
                            <span className="text-xs text-muted-foreground">1–2 minutes</span>
                        </div>
                        <div className="space-y-2 text-sm">
                            <div className="flex items-center gap-2">
                                <span className={`h-2 w-2 rounded-full ${missingDocs ? "bg-amber-500" : "bg-emerald-500"}`} />
                                <span className="text-foreground">{missingDocs ? "Add at least one document" : "Documents detected"}</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className={`h-2 w-2 rounded-full ${missingProvider ? "bg-amber-500" : "bg-emerald-500"}`} />
                                <span className="text-foreground">{missingProvider ? "Select models in Provider Settings" : "Models configured"}</span>
                            </div>
                        </div>
                        <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                            <span className="rounded-full bg-muted px-2 py-1">Planner • Gatherer • Generator</span>
                            <span className="rounded-full bg-muted px-2 py-1">Uses selected docs; falls back to all if none chosen</span>
                        </div>
                        {missingDocs && (
                            <div className="flex items-center gap-2 text-sm text-muted-foreground">
                                <UploadCloud className="h-4 w-4" />
                                <span>Upload PDFs/DOCX/Markdown/TXT to unlock querying.</span>
                            </div>
                        )}
                    </div>
                )}

                {/* Query Scope */}
                <div className="rounded-lg border border-border/60 bg-muted/10 p-4">
                    <div className="flex items-center justify-between gap-4">
                        <div>
                            <p className="text-sm font-semibold text-foreground">Query Scope</p>
                            <p className="text-xs text-muted-foreground">
                            {selectedDocumentIds.length === 0 || allSelected
                                ? "Searching all documents"
                                : `Searching ${selectedDocumentIds.length} selected document(s)`}
                            </p>
                        </div>
                        <button
                            type="button"
                            onClick={toggleAll}
                            className="text-xs font-medium text-secondary-foreground hover:text-foreground"
                            title="Switch between all documents and a selected subset."
                        >
                            {selectedDocumentIds.length === 0 ? "Select specific" : "Use all"}
                        </button>
                    </div>
                    {documents.length > 0 && selectedDocumentIds.length > 0 && (
                        <div className="mt-3 grid gap-2 sm:grid-cols-2">
                            {documents.map(doc => (
                                <label
                                    key={doc.id}
                                    className={`flex items-center gap-2 rounded-md border px-3 py-2 text-xs transition-colors ${selectedDocumentIds.includes(doc.id)
                                        ? "border-primary/30 bg-primary/10"
                                        : "border-border/60 bg-card"
                                    }`}
                                >
                                    <input
                                        type="checkbox"
                                        checked={selectedDocumentIds.includes(doc.id)}
                                        onChange={() => toggleDoc(doc.id)}
                                        className="h-3.5 w-3.5 rounded border-border text-primary focus:ring-primary"
                                    />
                                    <span className="truncate">{doc.title}</span>
                                </label>
                            ))}
                        </div>
                    )}
                </div>

                {/* Query Status */}
                <div className="rounded-lg border border-border/60 bg-card p-4">
                    <div className="flex flex-wrap items-center justify-between gap-4">
                        <div>
                            <p className="text-sm font-semibold text-foreground">Query Status</p>
                            <p className="text-xs text-muted-foreground">
                                {isQuerying ? "Running query..." : queryResult ? "Last query complete" : "Idle"}
                            </p>
                        </div>
                        <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
                            <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2.5 py-1">
                                <FileText className="h-3 w-3" />
                                {selectedDocumentIds.length === 0 || selectedDocumentIds.length === documents.length
                                    ? "All docs"
                                    : `${selectedDocumentIds.length} docs`}
                            </span>
                            {providerConfig?.planner_model && (
                                <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2.5 py-1">
                                    Planner: {providerConfig.planner_model}
                                </span>
                            )}
                            {providerConfig?.gatherer_model && (
                                <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2.5 py-1">
                                    Gatherer: {providerConfig.gatherer_model}
                                </span>
                            )}
                            {providerConfig?.generator_model && (
                                <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2.5 py-1">
                                    Generator: {providerConfig.generator_model}
                                </span>
                            )}
                            {queryResult?.metrics?.duration_ms && typeof queryResult.metrics.duration_ms === "number" && (
                                <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2.5 py-1">
                                    <Activity className="h-3 w-3" />
                                    {queryResult.metrics.duration_ms.toFixed(0)}ms
                                </span>
                            )}
                        </div>
                    </div>
                </div>

                {/* Onboarding nudges */}
                {!providerReady && (
                    <div className="flex items-start gap-3 rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
                        <HelpCircle className="mt-0.5 h-4 w-4" />
                        <div>
                            <p className="font-semibold">Select your models to run queries</p>
                            <p className="text-xs text-amber-800">
                                Open <strong>Provider Settings</strong> and choose planner / gatherer / generator models (or apply an auto-detected provider).
                            </p>
                        </div>
                    </div>
                )}

                {/* Live Observability */}
                <div className="rounded-lg border border-border/60 bg-muted/10 p-4">
                    <div className="flex items-center justify-between gap-4">
                        <div>
                            <p className="text-sm font-semibold text-foreground">Live Pipeline</p>
                            <p className="text-xs text-muted-foreground">
                                {isQuerying
                                    ? (activeStage ? activeStageCopy[activeStage] ?? "Processing pipeline..." : (completedStages.size === 0 ? "Awaiting pipeline response..." : "Step telemetry received."))
                                    : "Idle"}
                            </p>
                        </div>
                        {providerConfig && (
                            <div className="text-right text-xs text-muted-foreground">
                                <div>Planner: {providerConfig.planner_model || "default"}</div>
                                <div>Gatherer: {providerConfig.gatherer_model || "default"}</div>
                                <div>Generator: {providerConfig.generator_model || "default"}</div>
                            </div>
                        )}
                    </div>
                    <div className="mt-3 flex flex-wrap gap-2">
                        {liveStages.map(stage => {
                            const isComplete = completedStages.has(stage.name);
                            const isActive = isQuerying && activeStage === stage.name;
                            const StageIcon = stage.icon;
                            return (
                                <span
                                    key={stage.name}
                                    className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-medium ${isComplete
                                        ? "bg-primary text-primary-foreground"
                                        : isActive
                                            ? "bg-secondary text-secondary-foreground"
                                            : "bg-muted/60 text-muted-foreground"
                                    }`}
                                >
                                    {isComplete ? (
                                        <CheckCircle2 className="h-3.5 w-3.5" />
                                    ) : isActive ? (
                                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                    ) : (
                                        <StageIcon className="h-3.5 w-3.5" />
                                    )}
                                    {stage.label}
                                </span>
                            );
                        })}
                    </div>
                </div>

                {/* Query Input */}
                <div className="flex flex-col gap-3 sm:flex-row">
                    <Input
                        value={question}
                        onChange={e => setQuestion(e.target.value)}
                        placeholder="Type your question here..."
                        className="flex-1 text-base"
                        onKeyDown={e => e.key === "Enter" && !isQuerying && handleAsk()}
                    />
                    <Button
                        disabled={isQuerying}
                        onClick={handleAsk}
                        className=""
                    >
                        {isQuerying ? (
                            <>
                                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                Processing...
                            </>
                        ) : (
                            "Run Query"
                        )}
                    </Button>
                </div>

                {/* Loading State */}
                {isQuerying && (
                    <div className="flex items-center gap-3 rounded-lg border border-border/60 bg-muted/20 p-4">
                        <Loader2 className="h-5 w-5 animate-spin text-secondary-foreground" />
                        <div>
                            <p className="font-medium text-foreground">
                                Processing your query...
                            </p>
                            <p className="text-sm text-muted-foreground">
                                Planning → Retrieval → Generation
                            </p>
                        </div>
                    </div>
                )}

                {/* Results */}
                {(queryResult || isQuerying) && (
                    <div className="space-y-6">
                        <div className={`grid gap-6 min-w-0 lg:items-start ${resultGridClass}`}>
                            {hasPipeline && queryResult && (
                                <div className="space-y-6 min-w-0 xl:sticky xl:top-6 xl:self-start">
                                    <PipelinePanel steps={displaySteps} metrics={queryResult.metrics ?? {}} />
                                </div>
                            )}
                            {showPipelineSkeleton && (
                                <div className="space-y-4 min-w-0 xl:sticky xl:top-6 xl:self-start">
                                    <div className="rounded-lg border border-border/60 bg-card p-4">
                                        <div className="flex items-center justify-between">
                                            <SkeletonBlock className="h-5 w-40" />
                                            <SkeletonBlock className="h-4 w-16" />
                                        </div>
                                        <div className="mt-4 space-y-3">
                                            <SkeletonBlock className="h-24 w-full" />
                                            <SkeletonBlock className="h-24 w-full" />
                                        </div>
                                    </div>
                                </div>
                            )}

                            <div className="space-y-6 min-w-0">
                                {queryResult?.answer && (
                                    <div className="rounded-lg border border-border/60 bg-card p-6">
                                        <div className="mb-3 flex items-center gap-2">
                                            <Sparkles className="h-4 w-4" />
                                            <h3 className="font-semibold text-foreground">Answer</h3>
                                        </div>
                                        <p className="whitespace-pre-wrap text-base leading-relaxed text-foreground">
                                            {queryResult.answer}
                                        </p>
                                    </div>
                                )}
                                {showAnswerSkeleton && (
                                    <div className="rounded-lg border border-border/60 bg-card p-6">
                                        <SkeletonBlock className="h-5 w-24" />
                                        <div className="mt-4 space-y-3">
                                            <SkeletonBlock className="h-4 w-full" />
                                            <SkeletonBlock className="h-4 w-11/12" />
                                            <SkeletonBlock className="h-4 w-10/12" />
                                            <SkeletonBlock className="h-4 w-9/12" />
                                        </div>
                                    </div>
                                )}
                            </div>

                            {hasSources && queryResult && (
                                <div className="space-y-6 min-w-0 xl:sticky xl:top-6 xl:self-start">
                                    <div className="rounded-lg border border-border/60 p-4 overflow-hidden">
                                        <SourcesList
                                            sources={sourcesForDisplay}
                                        />
                                    </div>
                                </div>
                            )}
                            {showSourcesSkeleton && (
                                <div className="space-y-4 min-w-0 xl:sticky xl:top-6 xl:self-start">
                                    <div className="rounded-lg border border-border/60 bg-card p-4">
                                        <div className="flex items-center justify-between">
                                            <SkeletonBlock className="h-5 w-24" />
                                            <SkeletonBlock className="h-4 w-16" />
                                        </div>
                                        <div className="mt-4 space-y-3">
                                            <SkeletonBlock className="h-16 w-full" />
                                            <SkeletonBlock className="h-16 w-full" />
                                            <SkeletonBlock className="h-16 w-full" />
                                        </div>
                                    </div>
                                </div>
                            )}
                        </div>

                        {/* Quick Stats */}
                        {queryResult && (
                            <div className="flex flex-wrap gap-3">
                                <span className="inline-flex items-center gap-1.5 rounded-lg bg-muted px-3 py-1.5 text-xs font-medium">
                                    <FileText className="h-3.5 w-3.5" />
                                    {(typeof queryResult.metrics.chunks === "number" ? queryResult.metrics.chunks : queryResult.chunks.length)} chunks
                                </span>
                                <span className="inline-flex items-center gap-1.5 rounded-lg bg-muted px-3 py-1.5 text-xs font-medium">
                                    <Gauge className="h-3.5 w-3.5" />
                                    {(((typeof queryResult.metrics.coverage === "number" ? queryResult.metrics.coverage : 0) * 100).toFixed(0))}% coverage
                                </span>
                                <span className="inline-flex items-center gap-1.5 rounded-lg bg-muted px-3 py-1.5 text-xs font-medium">
                                    <FileSearch className="h-3.5 w-3.5" />
                                    {typeof queryResult.metrics.tokens === "number" ? queryResult.metrics.tokens : 0} tokens
                                </span>
                                <span className="inline-flex items-center gap-1.5 rounded-lg bg-muted px-3 py-1.5 text-xs font-medium">
                                    <Activity className="h-3.5 w-3.5" />
                                    {(typeof queryResult.metrics.duration_ms === "number" ? queryResult.metrics.duration_ms.toFixed(0) : "N/A")}ms
                                </span>
                            </div>
                        )}
                    </div>
                )}

                {/* Empty State */}
                {!queryResult && !isQuerying && (
                    <div className="rounded-lg border border-dashed border-border/70 p-12 text-center">
                        <Search className="mx-auto h-8 w-8 text-muted-foreground" />
                        <p className="mt-3 text-muted-foreground">
                            Enter a question above to search your documents
                        </p>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}
