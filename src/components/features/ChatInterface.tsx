
import React, { useMemo, useState, useEffect } from "react";
import {
    Activity,
    AlertTriangle,
    Box,
    Brain,
    CheckCircle2,
    Database,
    FileSearch,
    GitMerge,
    Loader2,
    MessageSquare,
    Search,
    Sparkles,
    Target,
    Gauge,
    FileText,
    Network,
    PanelLeftClose,
    PanelLeftOpen,
    PanelRightClose,
    PanelRightOpen,
    ArrowRightLeft,
    HelpCircle,
    Info,
    UploadCloud,
    ShieldCheck,
    PlusCircle,
    Trash2,
    XCircle,
    History,
    Copy,
    RefreshCw,
    ArrowRight
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { Progress as ProgressIndicator } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
import { TypingAnimation } from "@/components/ui/typing-animation";
import { useToast } from "@/components/ui/toast";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { PresetSelector } from "./PresetSelector";
import { ConfidenceIndicator } from "./ConfidenceIndicator";
import { QueryModeToggle } from "./QueryModeToggle";
import type { DocumentOut, PipelineStep, ProviderConfig, QueryResponse, ChatSession, PresetLevel, QueryMode } from "@/types";

interface ProgressData {
    stage: string;
    message: string;
    detail?: string;
    progress?: number;
    elapsed_ms?: number;
    estimated_remaining_ms?: number;
}

interface ChatInterfaceProps {
    question: string;
    setQuestion: (value: string) => void;
    isQuerying: boolean;
    handleAsk: (overrideQuestion?: string, isRegenerate?: boolean) => void;
    queryResult: QueryResponse | null;
    documents: DocumentOut[];
    selectedDocumentIds: string[];
    setSelectedDocumentIds: React.Dispatch<React.SetStateAction<string[]>>;
    providerConfig?: ProviderConfig;
    baseUrl?: string;
    apiKey?: string;
    activeStage?: string | null;
    progress?: ProgressData | null;
    history?: { role: string; content: string }[];
    onNewChat?: () => void;
    onCancel?: () => void;
    savedSessions?: ChatSession[];
    onLoadSession?: (session: ChatSession) => void;
    onDeleteSession?: (id: string) => void;
    onClearHistory?: () => void;
    scrollRef?: React.RefObject<HTMLDivElement | null>;
    currentSessionId?: string | null;
    preset?: PresetLevel;
    onPresetChange?: (preset: PresetLevel) => void;
    // 3.0: Query mode toggle
    queryMode?: QueryMode;
    onQueryModeChange?: (mode: QueryMode) => void;
}

const MessageContent = ({ content, onCitationClick }: { content: string; onCitationClick?: (id: string) => void }) => {
    // Component to render text with clickable citations
    const TextWithCitations = ({ text }: { text: string }) => {
        const parts = text.split(/(\[\d+\])/g);
        return (
            <>
                {parts.map((part, i) => {
                    const match = part.match(/^\[(\d+)\]$/);
                    if (match && match[1]) {
                        const id = match[1];
                        return (
                            <button
                                key={i}
                                onClick={() => onCitationClick?.(id)}
                                className="inline-flex items-center justify-center -translate-y-0.5 mx-0.5 h-4 min-w-[1rem] rounded-full bg-primary/20 px-1 text-[9px] font-bold text-primary hover:bg-primary hover:text-primary-foreground transition-all duration-200 shadow-sm border border-primary/10"
                                title={`Jump to source ${id}`}
                                aria-label={`Citation ${id} - jump to source`}
                            >
                                {id}
                            </button>
                        );
                    }
                    return <span key={i}>{part}</span>;
                })}
            </>
        );
    };

    // Helper to process children for citations
    const processNode = (node: any): any => {
        if (typeof node === 'string') {
            return <TextWithCitations text={node} />;
        }
        if (Array.isArray(node)) {
            return node.map((child, i) => <React.Fragment key={i}>{processNode(child)}</React.Fragment>);
        }
        if (React.isValidElement(node) && (node.props as any).children) {
            return React.cloneElement(node, {
                ...(node.props as any),
                children: processNode((node.props as any).children)
            });
        }
        return node;
    };

    return (
        <div className="prose prose-sm dark:prose-invert max-w-none text-foreground leading-relaxed transition-all duration-200">
            <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                    p: ({ children }) => <p className="mb-3 last:mb-0">{processNode(children)}</p>,
                    li: ({ children }) => <li>{processNode(children)}</li>,
                    blockquote: ({ children }) => <blockquote className="border-l-2 border-primary/30 pl-4 py-1 my-3 bg-muted/10 italic">{processNode(children)}</blockquote>,
                    a: ({ node: _node, children, ...props }) => (
                        <a {...props} className="text-primary hover:underline" target="_blank" rel="noopener noreferrer">
                            {children}
                        </a>
                    ),
                    code: ({ node, ...props }) => (
                        <code {...props} className="bg-muted px-1.5 py-0.5 rounded text-xs font-mono font-medium border border-border/40" />
                    ),
                    pre: ({ node, ...props }) => (
                        <pre {...props} className="bg-muted/50 p-3 rounded-lg border border-border/40 overflow-x-auto my-3 text-xs font-mono" />
                    ),
                    table: ({ children }) => (
                        <div className="overflow-x-auto my-4 border border-border/40 rounded-lg shadow-sm">
                            <table className="w-full text-left text-xs border-collapse">
                                {children}
                            </table>
                        </div>
                    ),
                    th: ({ children }) => <th className="p-2 bg-muted/50 font-semibold border-b border-border/40">{children}</th>,
                    td: ({ children }) => <td className="p-2 border-b border-border/40">{children}</td>,
                    ul: ({ children }) => <ul className="list-disc pl-5 mb-3 space-y-1">{children}</ul>,
                    ol: ({ children }) => <ol className="list-decimal pl-5 mb-3 space-y-1">{children}</ol>,
                    // Override text nodes to handle [n] citations
                    // This is slightly tricky in ReactMarkdown v9+ as it doesn't expose a 'text' component easily
                    // But we can usually rely on simple string children in many cases or use a custom plugin.
                    // For now, we'll keep it simple: if it's just a string, we process it.
                }}
            >
                {content}
            </ReactMarkdown>
        </div>
    );
};

const HistoryItem = ({
    role,
    content,
    onCitationClick,
    onRegenerate,
}: {
    role: string;
    content: string;
    onCitationClick?: (id: string) => void;
    onRegenerate?: () => void;
}) => {
    const { toast } = useToast();
    const isAssistant = role === "assistant";

    const handleCopy = async () => {
        try {
            await navigator.clipboard.writeText(content);
            toast({
                title: "Copied!",
                description: "Message content saved to clipboard.",
            });
        } catch (err) {
            console.error("Failed to copy", err);
        }
    };

    return (
        <div className={`flex w-full ${isAssistant ? "justify-start" : "justify-end"} group mb-6 animate-in fade-in slide-in-from-bottom-2 duration-300`}>
            <div className={`flex gap-3 max-w-[85%] ${isAssistant ? "flex-row" : "flex-row-reverse"}`}>
                <div className={`flex h-8 w-8 shrink-0 items-center justify-center rounded-lg shadow-sm border transition-colors ${isAssistant
                    ? "bg-primary/10 border-primary/20 text-primary"
                    : "bg-primary border-primary text-primary-foreground"
                    }`}>
                    {isAssistant ? <Sparkles className="h-4 w-4" /> : <Target className="h-4 w-4" />}
                </div>

                <div className="flex flex-col gap-1.5 min-w-0">
                    <div className={`relative px-4 py-3 rounded-2xl shadow-sm border transition-all duration-300 ${isAssistant
                        ? "bg-card border-border/40 rounded-tl-none hover:border-primary/30"
                        : "bg-muted/30 border-border/40 rounded-tr-none hover:bg-muted/50"
                        }`}>
                        <MessageContent content={content} onCitationClick={onCitationClick} />

                        {isAssistant && (
                            <div className="absolute top-2 right-2 flex items-center gap-1 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity">
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    className="h-6 w-6 text-muted-foreground hover:text-primary hover:bg-primary/10"
                                    onClick={handleCopy}
                                    title="Copy to clipboard"
                                    aria-label="Copy message to clipboard"
                                >
                                    <Copy className="h-3 w-3" />
                                </Button>
                                {onRegenerate && (
                                    <Button
                                        variant="ghost"
                                        size="icon"
                                        className="h-6 w-6 text-muted-foreground hover:text-primary hover:bg-primary/10"
                                        onClick={onRegenerate}
                                        title="Regenerate response"
                                        aria-label="Regenerate response"
                                    >
                                        <RefreshCw className="h-3 w-3" />
                                    </Button>
                                )}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};


function StepIcon({ name, status }: { name: string; status: string }) {
    const isComplete = ["completed", "passed", "repaired"].includes(status);
    const isSkipped = status === "skipped";
    const isFailed = status === "failed" || status === "error";
    const icons: Record<string, typeof Target> = {
        cache: Database,
        planning: Target,
        gating: HelpCircle,
        routing: GitMerge,
        graph_build: Network,
        graph_retrieval: GitMerge,
        gatherer: FileSearch,
        retrieval: Search,
        generation: Sparkles,
        compression: Box,
        conflict_detection: AlertTriangle,
        thinking: Brain,
        verification: ShieldCheck,
        verification_retry: ShieldCheck,
        evidence_contract: ShieldCheck,
        citation_verification: ShieldCheck,
        retrieval_retry: Search,
        compression_retry: Box,
        generation_retry: Sparkles,
        reflection: Activity,
    };
    const Icon = icons[name] ?? Target;
    return (
        <span className={`flex h-8 w-8 items-center justify-center rounded-lg text-lg transition-all ${isComplete
            ? "bg-primary/10"
            : isSkipped
                ? "bg-muted"
                : isFailed
                    ? "bg-destructive/10"
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
                    <span className="inline-flex items-center rounded-full bg-secondary/30 px-2.5 py-0.5 text-xs font-medium text-foreground">
                        {queryType ?? "planning"}{plannerMode ? ` / ${plannerMode}` : ""}
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
                                    <span className="text-muted-foreground">-</span>
                                    <span>{q}</span>
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
            </div>
        );
    }

    if (step.name === "gating") {
        const decision = details.decision as string | undefined;
        const confidence = details.confidence as number | undefined;
        const reasoning = details.reasoning as string | undefined;
        const clarification = details.clarification as string | undefined;
        return (
            <div className="mt-2 space-y-2 text-sm">
                {decision && (
                    <span className="inline-flex items-center rounded-full bg-muted px-2.5 py-0.5 text-xs font-medium text-muted-foreground">
                        Decision: {decision}
                    </span>
                )}
                {typeof confidence === "number" && (
                    <span className="inline-flex items-center rounded-full bg-muted px-2.5 py-0.5 text-xs font-medium text-muted-foreground">
                        Confidence: {(confidence * 100).toFixed(0)}%
                    </span>
                )}
                {(reasoning || clarification) && (
                    <p className="text-xs text-muted-foreground">
                        {clarification ? `Clarification: ${clarification}` : reasoning}
                    </p>
                )}
            </div>
        );
    }

    if (step.name === "routing") {
        const useRaptor = details.use_raptor as boolean | undefined;
        const useGraph = details.use_graph as boolean | undefined;
        const useColbert = details.use_colbert as boolean | undefined;
        const rerankEnabled = details.rerank_enabled as boolean | undefined;
        return (
            <div className="mt-2 flex flex-wrap gap-2 text-xs text-muted-foreground">
                {typeof useRaptor === "boolean" && (
                    <span className="inline-flex items-center rounded-full bg-muted px-2.5 py-0.5">
                        RAPTOR: {useRaptor ? "on" : "off"}
                    </span>
                )}
                {typeof useGraph === "boolean" && (
                    <span className="inline-flex items-center rounded-full bg-muted px-2.5 py-0.5">
                        GraphRAG: {useGraph ? "on" : "off"}
                    </span>
                )}
                {typeof useColbert === "boolean" && (
                    <span className="inline-flex items-center rounded-full bg-muted px-2.5 py-0.5">
                        ColBERT: {useColbert ? "on" : "off"}
                    </span>
                )}
                {typeof rerankEnabled === "boolean" && (
                    <span className="inline-flex items-center rounded-full bg-muted px-2.5 py-0.5">
                        Rerank: {rerankEnabled ? "on" : "off"}
                    </span>
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
                    <FileText className="h-3.5 w-3.5" />
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
                {typeof rerankerEnabled === "boolean" && (
                    <span className="inline-flex items-center gap-1.5 rounded-lg bg-muted px-2.5 py-1 text-xs">
                        Rerank: {rerankerEnabled ? "on" : "off"}
                    </span>
                )}
                {typeof details.raptor_chunks_added === "number" && (
                    <span className="inline-flex items-center gap-1.5 rounded-lg bg-primary/10 text-primary px-2.5 py-1 text-xs font-medium">
                        <Box className="h-3 w-3" />
                        RAPTOR (+{details.raptor_chunks_added})
                    </span>
                )}
                {typeof details.graph_chunks_added === "number" && (
                    <span className="inline-flex items-center gap-1.5 rounded-lg bg-secondary text-secondary-foreground px-2.5 py-1 text-xs font-medium">
                        <Network className="h-3 w-3" />
                        GraphRAG (+{details.graph_chunks_added})
                    </span>
                )}
            </div>
        );
    }

    if (step.name === "graph_build") {
        const ready = details.graph_ready as boolean | undefined;
        const entityCount = details.entity_count as number | undefined;
        const alreadyReady = details.already_ready as boolean | undefined;
        const reason = details.reason as string | undefined;
        return (
            <div className="mt-2 flex flex-wrap gap-2 text-xs text-muted-foreground">
                <span className="inline-flex items-center rounded-full bg-muted px-2.5 py-0.5">
                    {alreadyReady ? "Using cached graph" : ready ? "Graph ready" : "Graph build failed"}
                </span>
                {reason && (
                    <span className="inline-flex items-center rounded-full bg-muted px-2.5 py-0.5">
                        {reason}
                    </span>
                )}
                {typeof entityCount === "number" && (
                    <span className="inline-flex items-center rounded-full bg-muted px-2.5 py-0.5">
                        {entityCount} entities
                    </span>
                )}
            </div>
        );
    }

    if (step.name === "graph_retrieval") {
        const chunksAdded = details.chunks_added as number | undefined;
        const durationMs = details.duration_ms as number | undefined;
        return (
            <div className="mt-2 flex flex-wrap gap-2 text-xs text-muted-foreground">
                <span className="inline-flex items-center rounded-full bg-muted px-2.5 py-0.5">
                    {chunksAdded ?? 0} graph chunks
                </span>
                {typeof durationMs === "number" && durationMs > 0 && (
                    <span className="inline-flex items-center rounded-full bg-muted px-2.5 py-0.5">
                        {durationMs.toFixed(0)}ms
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

    if (step.name === "thinking") {
        const outline = details.outline as string | undefined;
        const reason = details.reason as string | undefined;
        return (
            <div className="mt-2 space-y-2 text-sm">
                {outline ? (
                    <div className="rounded-md border border-border/60 bg-muted/30 p-2 text-xs text-foreground whitespace-pre-wrap">
                        {outline}
                    </div>
                ) : (
                    <span className="inline-flex items-center rounded-full bg-muted px-2.5 py-0.5 text-xs text-muted-foreground">
                        Outline unavailable{reason ? ` (${reason})` : ""}
                    </span>
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

    if (step.name === "conflict_detection") {
        const hasConflicts = details.has_conflicts as boolean | undefined;
        const conflictCount = details.conflict_count as number | undefined;
        return (
            <div className="mt-2 flex flex-wrap gap-2 text-xs text-muted-foreground">
                <span className="inline-flex items-center rounded-full bg-muted px-2.5 py-0.5">
                    {hasConflicts ? "Conflicts detected" : "No conflicts"}
                </span>
                {typeof conflictCount === "number" && (
                    <span className="inline-flex items-center rounded-full bg-muted px-2.5 py-0.5">
                        {conflictCount} conflicts
                    </span>
                )}
            </div>
        );
    }

    if (step.name === "verification" || step.name === "verification_retry") {
        const passRate = details.pass_rate as number | undefined;
        const flagged = details.flagged_claims_count as number | undefined;
        return (
            <div className="mt-2 flex flex-wrap gap-2 text-xs text-muted-foreground">
                {typeof passRate === "number" && (
                    <span className="inline-flex items-center rounded-full bg-muted px-2.5 py-0.5">
                        Pass rate: {(passRate * 100).toFixed(0)}%
                    </span>
                )}
                {typeof flagged === "number" && (
                    <span className="inline-flex items-center rounded-full bg-muted px-2.5 py-0.5">
                        {flagged} flagged
                    </span>
                )}
            </div>
        );
    }

    if (step.name === "evidence_contract") {
        const coverageRatio = details.coverage_ratio as number | undefined;
        const passThreshold = details.pass_threshold as number | undefined;
        return (
            <div className="mt-2 flex flex-wrap gap-2 text-xs text-muted-foreground">
                {typeof coverageRatio === "number" && (
                    <span className="inline-flex items-center rounded-full bg-muted px-2.5 py-0.5">
                        Coverage: {(coverageRatio * 100).toFixed(0)}%
                    </span>
                )}
                {typeof passThreshold === "number" && (
                    <span className="inline-flex items-center rounded-full bg-muted px-2.5 py-0.5">
                        Target: {(passThreshold * 100).toFixed(0)}%
                    </span>
                )}
            </div>
        );
    }

    if (step.name === "citation_verification") {
        const passRate = details.citation_check_pass_rate as number | undefined;
        const invalid = details.invalid_citations as Array<{ id: string }> | undefined;
        return (
            <div className="mt-2 flex flex-wrap gap-2 text-xs text-muted-foreground">
                {typeof passRate === "number" && (
                    <span className="inline-flex items-center rounded-full bg-muted px-2.5 py-0.5">
                        Pass rate: {(passRate * 100).toFixed(0)}%
                    </span>
                )}
                {invalid && (
                    <span className="inline-flex items-center rounded-full bg-muted px-2.5 py-0.5">
                        {invalid.length} invalid
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
                    {expanded ? "Hide details" : "Show details"}
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
                                        || step.status === "passed"
                                        || step.status === "repaired"
                                        ? "text-primary"
                                        : step.status === "skipped"
                                            ? "text-muted-foreground"
                                            : step.status === "failed"
                                                ? "text-destructive"
                                                : "text-muted-foreground"
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
                                    {step.started_at && step.completed_at ? " - " : ""}
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

function SourceDetailPanel({ doc, onClose }: { doc: DocumentOut; onClose: () => void }) {
    return (
        <div className="absolute inset-0 z-50 flex flex-col bg-background animate-in slide-in-from-right duration-300">
            <div className="flex items-center justify-between p-4 border-b border-border/40 bg-muted/20">
                <h3 className="text-sm font-semibold truncate flex-1 mr-4">{doc.title}</h3>
                <Button variant="ghost" size="icon" onClick={onClose} aria-label="Close document preview" className="h-8 w-8 shrink-0">
                    <PanelRightClose className="h-4 w-4" />
                </Button>
            </div>
            <div className="flex-1 overflow-y-auto p-6 font-serif">
                <p className="text-sm leading-relaxed whitespace-pre-wrap text-foreground/90">
                    {doc.text}
                </p>
            </div>
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
        <Skeleton className={className} />
    );
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

function SourcesList({ sources, highlightedSourceId, onSourceClick }: { sources: CitationSource[]; highlightedSourceId?: string | null; onSourceClick?: (s: CitationSource) => void }) {
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
                        className="text-xs font-medium text-primary hover:text-foreground whitespace-nowrap"
                    >
                        {showAll ? "Show less" : `Show all ${sources.length}`}
                    </button>
                )}
            </div>
            <div className="grid gap-2 min-w-0">
                {displaySources.map((source, idx) => (
                    <div
                        key={`${source.id}-${idx}`}
                        id={`source-${source.citation_number ?? idx + 1}`}
                        onClick={() => onSourceClick?.(source)}
                        onKeyDown={(e) => {
                            if (e.key === "Enter" || e.key === " ") {
                                e.preventDefault();
                                onSourceClick?.(source);
                            }
                        }}
                        role="button"
                        tabIndex={0}
                        aria-label={`Source ${source.citation_number ?? idx + 1}: ${source.title}`}
                        className={`group flex min-w-0 gap-3 rounded-md border p-3 transition-all duration-500 cursor-pointer ${highlightedSourceId === String(source.citation_number ?? idx + 1)
                            ? "bg-primary/10 border-primary ring-1 ring-primary"
                            : "bg-card border-border/60 hover:bg-muted/20 focus-within:bg-muted/20"
                            }`}
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

function formatStageLabel(value?: string | null) {
    if (!value) {
        return "Ready";
    }
    return value
        .split("_")
        .filter(Boolean)
        .map(part => `${part.charAt(0).toUpperCase()}${part.slice(1)}`)
        .join(" ");
}

function WhyAnswerPanel({
    queryResult,
    sources,
    queryMode,
    isQuerying,
    activeStage,
    progress,
}: {
    queryResult: QueryResponse | null;
    sources: CitationSource[];
    queryMode: QueryMode;
    isQuerying: boolean;
    activeStage?: string | null;
    progress?: ProgressData | null;
}) {
    const steps = queryResult?.steps ?? [];
    const routingStep = steps.find(step => step.name === "routing");
    const retrievalStep = steps.find(step => step.name === "retrieval");
    const evidenceStep = steps.find(step => step.name === "evidence_contract");
    const citationStep = steps.find(step => step.name === "citation_verification");
    const confidence = queryResult?.confidence?.overall;
    const confidencePercent = typeof confidence === "number" ? Math.round(confidence * 100) : null;
    const grounding = queryResult?.grounding;
    const noEvidence = Boolean(grounding?.no_evidence_response || (!isQuerying && queryResult && sources.length === 0));

    const routingDetails = routingStep?.details ?? {};
    const retrievalDetails = retrievalStep?.details ?? {};
    const strategyBadges = [
        typeof retrievalDetails.dense_enabled === "boolean" ? `Dense ${retrievalDetails.dense_enabled ? "on" : "off"}` : null,
        typeof routingDetails.use_graph === "boolean" ? `GraphRAG ${routingDetails.use_graph ? "on" : "off"}` : null,
        typeof routingDetails.use_raptor === "boolean" ? `RAPTOR ${routingDetails.use_raptor ? "on" : "off"}` : null,
        typeof routingDetails.rerank_enabled === "boolean" ? `Rerank ${routingDetails.rerank_enabled ? "on" : "off"}` : null,
    ].filter(Boolean) as string[];

    const totalChunks = typeof retrievalDetails.total_chunks === "number" ? retrievalDetails.total_chunks : queryResult?.chunks?.length ?? 0;
    const uniqueSources = typeof retrievalDetails.unique_sources === "number" ? retrievalDetails.unique_sources : sources.length;
    const progressValue = typeof progress?.progress === "number" ? Math.round(progress.progress * 100) : isQuerying ? 35 : 100;

    return (
        <div className="flex flex-col gap-3 rounded-lg border border-border/60 bg-background p-4 shadow-sm">
            <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                    <p className="text-sm font-semibold text-foreground">Why this answer</p>
                    <p className="mt-1 text-xs text-muted-foreground">
                        Trace {queryResult?.trace_id || "pending"}, {queryMode === "grounded" ? "grounded mode" : "open domain mode"}
                    </p>
                </div>
                <Badge variant={noEvidence ? "destructive" : confidencePercent !== null && confidencePercent < 50 ? "secondary" : "outline"}>
                    {noEvidence ? "No evidence" : confidencePercent !== null ? `${confidencePercent}%` : isQuerying ? "Streaming" : "Ready"}
                </Badge>
            </div>

            {isQuerying && (
                <div className="flex flex-col gap-2">
                    <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
                        <span>{formatStageLabel(activeStage || progress?.stage)}</span>
                        <span>{progressValue}%</span>
                    </div>
                    <ProgressIndicator value={progressValue} />
                    {progress?.detail && (
                        <p className="text-xs text-muted-foreground">{progress.detail}</p>
                    )}
                </div>
            )}

            {!isQuerying && noEvidence && (
                <Alert variant="warning">
                    <AlertTriangle data-icon="inline-start" />
                    <AlertTitle>Evidence is not strong enough</AlertTitle>
                    <AlertDescription>
                        Grounded mode did not find a reliable source set. Add documents, broaden the scope, or switch modes for a general response.
                    </AlertDescription>
                </Alert>
            )}

            <Separator />

            <div className="grid grid-cols-3 gap-2 text-xs">
                <div className="rounded-md bg-muted/40 p-2">
                    <p className="text-muted-foreground">Chunks</p>
                    <p className="mt-1 font-mono text-sm text-foreground">{totalChunks}</p>
                </div>
                <div className="rounded-md bg-muted/40 p-2">
                    <p className="text-muted-foreground">Sources</p>
                    <p className="mt-1 font-mono text-sm text-foreground">{uniqueSources}</p>
                </div>
                <div className="rounded-md bg-muted/40 p-2">
                    <p className="text-muted-foreground">Checks</p>
                    <p className="mt-1 font-mono text-sm text-foreground">
                        {[evidenceStep, citationStep].filter(Boolean).length}/2
                    </p>
                </div>
            </div>

            <div className="flex flex-wrap gap-2">
                {(strategyBadges.length ? strategyBadges : ["Hybrid retrieval", "Citation verification"]).map(item => (
                    <Badge key={item} variant="muted">{item}</Badge>
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
    baseUrl,
    apiKey,
    activeStage,
    progress,
    history = [],
    onNewChat,
    onCancel,
    savedSessions = [],
    onLoadSession,
    onDeleteSession,
    onClearHistory,
    scrollRef,
    currentSessionId,
    preset = "balanced",
    onPresetChange,
    queryMode = "grounded",
    onQueryModeChange,
}: ChatInterfaceProps) {
    const [highlightedSourceId, setHighlightedSourceId] = useState<string | null>(null);
    const [selectedDoc, setSelectedDoc] = useState<DocumentOut | null>(null);
    const [showHistory, setShowHistory] = useState(() => (typeof window === "undefined" ? true : window.innerWidth >= 1024));
    const [showSources, setShowSources] = useState(() => (typeof window === "undefined" ? true : window.innerWidth >= 1280));

    // Persist sources panel when query results are available
    useEffect(() => {
        const wideEnoughForSources = typeof window === "undefined" || window.innerWidth >= 1280;
        if (queryResult && !showSources && wideEnoughForSources) {
            setShowSources(true);
        }
    }, [queryResult, showSources]);

    useEffect(() => {
        if (typeof window === "undefined") {
            return;
        }
        const media = window.matchMedia("(max-width: 1023px)");
        const syncPanels = () => {
            if (media.matches) {
                setShowHistory(false);
                setShowSources(false);
            }
        };
        syncPanels();
        media.addEventListener("change", syncPanels);
        return () => media.removeEventListener("change", syncPanels);
    }, []);

    // Keyboard shortcuts
    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if ((e.metaKey || e.ctrlKey) && e.key === "k") {
                e.preventDefault();
                onNewChat?.();
            }
        };
        window.addEventListener("keydown", handleKeyDown);
        return () => window.removeEventListener("keydown", handleKeyDown);
    }, [onNewChat]);

    const handleCitationClick = (id: string) => {
        setHighlightedSourceId(id);
        if (!showSources) setShowSources(true);

        const source = sourcesForDisplay.find(s => String(s.citation_number) === id);
        if (source) {
            const doc = documents.find(d => d.id === source.id || d.title === source.title);
            if (doc) {
                setSelectedDoc(doc);
            }
        }

        setTimeout(() => {
            const element = document.getElementById(`source-${id}`);
            if (element) {
                element.scrollIntoView({ behavior: "smooth", block: "center" });
            }
        }, 100);
    };

    const handleDeleteSaved = (e: React.MouseEvent, id: string) => {
        e.stopPropagation();
        onDeleteSession?.(id);
    };

    const liveStages = useMemo(() => ([
        { name: "cache", label: "Cache", icon: Database },
        { name: "planning", label: "Planning", icon: Target },
        { name: "gating", label: "Gating", icon: HelpCircle },
        { name: "routing", label: "Routing", icon: GitMerge },
        { name: "graph_build", label: "Graph Build", icon: Network },
        { name: "gatherer", label: "Gatherer", icon: FileSearch },
        { name: "graph_retrieval", label: "Graph Retrieval", icon: Network },
        { name: "retrieval", label: "Retrieval", icon: Search },
        { name: "compression", label: "Compression", icon: Box },
        { name: "conflict_detection", label: "Conflict Check", icon: AlertTriangle },
        { name: "thinking", label: "Thinking", icon: Brain },
        { name: "generation", label: "Generation", icon: Sparkles },
        { name: "verification", label: "Verification", icon: ShieldCheck },
        { name: "evidence_contract", label: "Evidence Contract", icon: ShieldCheck },
        { name: "citation_verification", label: "Citation Check", icon: ShieldCheck },
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
    const layoutHasSources = hasSources || isQuerying || hasAnswer; // Keep visible after query completes
    // Grid columns based on visibility
    // Default: 250px (History) | 1fr (Chat) | 300px (Sources)
    const gridTemplate = `
        ${showHistory ? "260px" : "0px"} 
        minmax(0, 1fr) 
        ${showSources && layoutHasSources ? "320px" : "0px"}`;

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
        <div className="flex flex-col h-full gap-4 p-6 overflow-hidden">
            {/* Main Content Area - Full height grid */}
            <div
                className="flex-1 min-h-0 grid transition-[grid-template-columns] duration-300 ease-in-out gap-0 overflow-hidden rounded-xl border border-border/40 bg-background shadow-sm"
                style={{ gridTemplateColumns: gridTemplate }}
            >

                {/* Left Column: History/Threads */}
                <div className={`flex flex-col border-r border-border/40 bg-muted/20 h-full overflow-hidden transition-all duration-300 ${!showHistory && "opacity-0 invisible w-0 border-none"}`}>
                    <div className="p-4 border-b border-border/40 flex items-center justify-between">
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-2">
                            <History className="h-3.5 w-3.5" /> History
                        </h3>
                        <div className="flex items-center gap-1">
                            {onClearHistory && savedSessions.length > 0 && (
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    onClick={onClearHistory}
                                    className="h-7 w-7 text-muted-foreground hover:text-destructive hover:bg-destructive/10"
                                    title="Clear All History"
                                >
                                    <Trash2 className="h-4 w-4" />
                                </Button>
                            )}
                            {onNewChat && (
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    onClick={onNewChat}
                                    className="h-7 w-7 text-muted-foreground hover:text-primary hover:bg-primary/10"
                                    title="New Chat"
                                >
                                    <PlusCircle className="h-4 w-4" />
                                </Button>
                            )}
                        </div>
                    </div>
                    <div className="flex-1 overflow-y-auto p-2 space-y-1">
                        <div className="mt-2 mb-2 px-2 text-[10px] uppercase font-bold text-muted-foreground/60 tracking-widest">
                            Archived
                        </div>

                        {savedSessions.length === 0 ? (
                            <div className="p-4 text-xs text-muted-foreground text-center italic opacity-60">
                                No archived chats
                            </div>
                        ) : (
                            savedSessions.map((session) => {
                                const isActive = session.id === currentSessionId;
                                return (
                                    <div
                                        key={session.id}
                                        className={`group flex items-center justify-between p-2.5 rounded-lg border transition-all cursor-pointer ${isActive
                                            ? "bg-primary/10 border-primary/30 shadow-sm"
                                            : "bg-transparent border-transparent hover:bg-muted/50 hover:border-border/40 focus-within:bg-muted/50 focus-within:border-border/40"
                                            }`}
                                        onClick={() => onLoadSession?.(session)}
                                        onKeyDown={(e) => {
                                            if (e.key === "Enter" || e.key === " ") {
                                                e.preventDefault();
                                                onLoadSession?.(session);
                                            }
                                        }}
                                        role="button"
                                        tabIndex={0}
                                        aria-label={`Load chat session: ${session.title || session.id}`}
                                    >
                                        <div className="flex items-center gap-3 min-w-0">
                                            <MessageSquare className={`h-4 w-4 shrink-0 ${isActive ? "text-primary" : "text-muted-foreground"}`} />
                                            <div className="min-w-0">
                                                <div className={`text-sm truncate ${isActive ? "font-semibold text-primary" : "text-foreground/80 font-medium"}`}>
                                                    {session.title}
                                                </div>
                                                <div className="text-[10px] text-muted-foreground mt-0.5">
                                                    {new Date(session.createdAt).toLocaleDateString()}
                                                </div>
                                            </div>
                                        </div>
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            aria-label={`Delete chat session: ${session.title || session.id}`}
                                            title={`Delete ${session.title || "chat session"}`}
                                            className={`h-7 w-7 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity hover:bg-destructive/10 hover:text-destructive ${isActive ? "opacity-100 text-primary/60" : ""}`}
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                onDeleteSession?.(session.id);
                                            }}
                                        >
                                            <Trash2 className="h-3.5 w-3.5" />
                                        </Button>
                                    </div>
                                );
                            })
                        )}
                    </div>
                </div>

                {/* Center Column: Chat & Input */}
                <div className="flex flex-col h-full relative min-w-0 overflow-hidden bg-background/50">
                    {/* Toggle header */}
                    <div className="absolute top-4 left-4 z-10 transition-opacity duration-300">
                        {!showHistory && (
                            <Button variant="ghost" size="icon" onClick={() => setShowHistory(true)} aria-label="Show chat history sidebar" className="h-8 w-8 text-muted-foreground hover:text-foreground">
                                <PanelLeftOpen className="h-4 w-4" />
                            </Button>
                        )}
                        {showHistory && (
                            <Button variant="ghost" size="icon" onClick={() => setShowHistory(false)} aria-label="Hide chat history sidebar" className="h-8 w-8 text-muted-foreground hover:text-foreground">
                                <PanelLeftClose className="h-4 w-4" />
                            </Button>
                        )}
                    </div>
                    {/* Scrollable Chat Area */}
                    <div className="flex-1 overflow-y-auto px-4 py-6 scroll-smooth">
                        <div className="max-w-3xl mx-auto flex flex-col gap-6">

                            {/* Internal Header Content / Setup Widgets */}
                            <div className="flex flex-col gap-4 mb-4">
                                {/* Premium setup checklist */}
                                {(missingDocs || missingProvider) && (
                                    <div className="flex flex-col gap-3 rounded-lg border border-border/70 bg-muted/10 p-4">
                                        <div className="flex items-center justify-between gap-3">
                                            <div className="flex items-center gap-2">
                                                <ShieldCheck className="h-4 w-4 text-primary" />
                                                <p className="text-sm font-semibold text-foreground">Getting ready</p>
                                            </div>
                                            <span className="text-xs text-muted-foreground">1-2 minutes</span>
                                        </div>
                                        <div className="space-y-2 text-sm">
                                            <div className="flex items-center gap-2">
                                                <span className={`h-2 w-2 rounded-full ${missingDocs ? "bg-secondary" : "bg-primary"}`} />
                                                <span className="text-foreground">{missingDocs ? "Add at least one document" : "Documents detected"}</span>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <span className={`h-2 w-2 rounded-full ${missingProvider ? "bg-secondary" : "bg-primary"}`} />
                                                <span className="text-foreground">{missingProvider ? "Select models in Provider Settings" : "Models configured"}</span>
                                            </div>
                                        </div>
                                        <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                                            <span className="rounded-full bg-muted px-2 py-1">Planner / Gatherer / Generator</span>
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

                                <div className="grid gap-3 rounded-lg border border-border/60 bg-muted/10 p-4 lg:grid-cols-[minmax(0,1fr)_minmax(280px,0.9fr)]">
                                    <div className="flex min-w-0 flex-col gap-3">
                                        <div>
                                            <p className="text-sm font-semibold text-foreground">Answer Controls</p>
                                            <p className="mt-1 text-xs text-muted-foreground">
                                                Tune scope, routing depth, and evidence posture before streaming.
                                            </p>
                                        </div>
                                        <div className="flex flex-wrap items-center gap-2">
                                            <Badge variant="outline">Trace visible</Badge>
                                            <Badge variant="outline">Citations required</Badge>
                                            <Badge variant="outline">Hybrid retrieval</Badge>
                                        </div>
                                    </div>
                                    <div className="flex flex-col gap-3">
                                        <QueryModeToggle
                                            mode={queryMode}
                                            onChange={onQueryModeChange ?? (() => undefined)}
                                            disabled={isQuerying || !onQueryModeChange}
                                        />
                                        <PresetSelector
                                            value={preset}
                                            onChange={onPresetChange ?? (() => undefined)}
                                            disabled={isQuerying || !onPresetChange}
                                            compact
                                        />
                                    </div>
                                </div>

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
                                            className="text-xs font-medium text-primary hover:text-foreground"
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

                                {/* Enterprise Controls Removed */}
                            </div>

                            {/* Welcome / Empty State */}
                            {!hasAnswer && history.length === 0 && !isQuerying && (
                                <div className="flex flex-col items-center justify-center min-h-[50vh] text-center space-y-8 animate-in fade-in zoom-in duration-500">

                                    {/* System Quick Stats - Moved to Top */}
                                    <div className="flex items-center gap-6 mb-4 w-full max-w-2xl justify-center">
                                        <div className="flex flex-col items-center">
                                            <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Documents</span>
                                            <span className="text-xl font-bold font-mono text-foreground/80">{selectedDocumentIds.length || documents.length}</span>
                                        </div>
                                        <div className="w-px h-8 bg-border/40" />
                                        <div className="flex flex-col items-center max-w-[200px]">
                                            <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Models</span>
                                            <div className="flex flex-col items-center text-xs font-semibold text-primary/80">
                                                <span className="truncate max-w-[180px]">{providerConfig?.planner_model || "Not set"}</span>
                                            </div>
                                        </div>
                                        <div className="w-px h-8 bg-border/40" />
                                        <div className="flex flex-col items-center">
                                            <span className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Provider</span>
                                            <span className="text-sm font-bold text-foreground/80 uppercase">
                                                {providerConfig?.base_url?.includes("11434") ? "Ollama" :
                                                    providerConfig?.base_url?.includes("1234") ? "LM Studio" :
                                                        "Custom"}
                                            </span>
                                        </div>
                                    </div>

                                    <div className="space-y-4">
                                        <div className="h-20 w-20 rounded-2xl bg-primary/10 flex items-center justify-center mx-auto mb-6 shadow-lg">
                                            <Sparkles className="h-10 w-10 text-primary animate-pulse" />
                                        </div>
                                        <h2 className="text-3xl font-bold tracking-tight bg-gradient-to-br from-foreground to-foreground/60 bg-clip-text text-transparent italic">
                                            How can I help you today?
                                        </h2>
                                        <p className="text-muted-foreground max-w-lg mx-auto text-lg leading-relaxed font-light">
                                            <TypingAnimation duration={12} startOnView={false}>
                                                Ask anything about your documents. I will search, synthesize, and cite sources for every factual claim.
                                            </TypingAnimation>
                                        </p>
                                    </div>

                                    <div className="grid sm:grid-cols-2 gap-4 max-w-2xl w-full px-4">
                                        {[
                                            {
                                                title: "Technical Summary",
                                                desc: "Summarize the key architectural components of this system.",
                                                icon: Box
                                            },
                                            {
                                                title: "Identify Risks",
                                                desc: "Search for potential single points of failure mentioned in documents.",
                                                icon: AlertTriangle
                                            },
                                            {
                                                title: "Data Analysis",
                                                desc: "Provide a breakdown of all dates and timelines found in the corpus.",
                                                icon: Activity
                                            },
                                            {
                                                title: "Onboarding Memo",
                                                desc: "Give me a brief informed memo in the style of a formal email.",
                                                icon: HelpCircle
                                            }
                                        ].map((suggestion, i) => (
                                            <button
                                                key={i}
                                                onClick={() => setQuestion(suggestion.desc)}
                                                className="group relative flex flex-col items-start gap-2 rounded-xl border border-border/40 bg-card/50 p-4 text-left transition-all hover:bg-muted/50 hover:border-border/80 hover:shadow-sm"
                                            >
                                                <div className="flex w-full items-center justify-between">
                                                    <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
                                                        <suggestion.icon className="h-4 w-4 text-muted-foreground group-hover:text-primary transition-colors" />
                                                        {suggestion.title}
                                                    </div>
                                                    <ArrowRight className="h-4 w-4 opacity-0 -translate-x-2 transition-all group-hover:opacity-100 group-hover:translate-x-0 text-muted-foreground" />
                                                </div>
                                                <p className="text-xs text-muted-foreground line-clamp-2 leading-relaxed">
                                                    {suggestion.desc}
                                                </p>
                                            </button>
                                        ))}
                                    </div>

                                </div>
                            )}

                            {/* Chat History */}
                            {history.map((turn, idx) => (
                                <HistoryItem
                                    key={idx}
                                    role={turn.role}
                                    content={turn.content}
                                    onCitationClick={handleCitationClick}
                                    onRegenerate={turn.role === "assistant" && idx === history.length - 1 && !isQuerying ? () => {
                                        // Find the last user message to use as the question
                                        const lastUserMessage = [...history].reverse().find(m => m.role === "user");
                                        if (lastUserMessage) {
                                            handleAsk(lastUserMessage.content, true);
                                        }
                                    } : undefined}
                                />
                            ))}

                            {/* Streaming Result bubble — aria-live for screen reader feedback */}
                            {isQuerying && queryResult?.answer && (
                                <div aria-live="polite" aria-atomic="false">
                                    <HistoryItem
                                        role="assistant"
                                        content={queryResult.answer}
                                    />
                                </div>
                            )}

                            {!isQuerying && queryResult && !hasSources && (
                                <Alert variant="warning">
                                    <AlertTriangle data-icon="inline-start" />
                                    <AlertTitle>No cited evidence returned</AlertTitle>
                                    <AlertDescription>
                                        The answer did not include source evidence. Try a narrower question, select specific documents, or seed the demo corpus.
                                    </AlertDescription>
                                </Alert>
                            )}

                            {!isQuerying && queryResult?.confidence?.overall !== undefined && queryResult.confidence.overall < 0.5 && hasSources && (
                                <Alert variant="warning">
                                    <ShieldCheck data-icon="inline-start" />
                                    <AlertTitle>Low confidence answer</AlertTitle>
                                    <AlertDescription>
                                        The system found evidence, but confidence is below the normal review threshold. Inspect the citations and trace before using the answer.
                                    </AlertDescription>
                                </Alert>
                            )}

                            {/* Progress Indicator - Enhanced with typing dots and detailed step info */}
                            {isQuerying && (
                                <div className="flex items-start gap-3 mt-4">
                                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 border border-primary/20 text-primary">
                                        <Sparkles className="h-4 w-4" />
                                    </div>
                                    <div className="flex flex-col gap-2 flex-1">
                                        <div className="flex items-center gap-2 px-4 py-3 rounded-2xl rounded-tl-none bg-card border border-border/40">
                                            {/* Typing dots animation */}
                                            <div className="flex gap-1">
                                                <span className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                                                <span className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                                                <span className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                                            </div>
                                            <span className="text-sm text-muted-foreground ml-2">
                                                {progress?.message || "Thinking..."}
                                            </span>
                                        </div>
                                        {/* Detailed step info panel */}
                                        {progress && (
                                            <div className="px-4 py-2 rounded-lg bg-muted/30 border border-border/20 text-xs space-y-1">
                                                <div className="flex items-center gap-2">
                                                    <span className="font-medium text-foreground/80">
                                                        Stage: {progress.stage?.replace(/_/g, ' ').replace(/\b\w/g, (c: string) => c.toUpperCase()) || 'Processing'}
                                                    </span>
                                                    {progress.elapsed_ms && (
                                                        <span className="text-muted-foreground">
                                                            ({(progress.elapsed_ms / 1000).toFixed(1)}s)
                                                        </span>
                                                    )}
                                                </div>
                                                {progress.detail && (
                                                    <p className="text-muted-foreground/80">{progress.detail}</p>
                                                )}
                                                {progress.progress !== undefined && (
                                                    <ProgressIndicator value={Math.round(progress.progress * 100)} className="mt-1" />
                                                )}
                                            </div>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* Scroll Anchor */}
                            <div ref={scrollRef} className="h-4 w-full" />
                        </div>
                    </div>

                    {/* Standard Input Area (Flex Item, Not Absolute) */}
                    <div className="flex-none p-4 w-full bg-background border-t border-border/40">
                        <div className="max-w-3xl mx-auto">
                            <div className="relative group bg-background rounded-xl shadow-sm border border-border/40 focus-within:border-primary/50 focus-within:ring-1 focus-within:ring-primary/20 transition-all">
                                <textarea
                                    aria-label="Ask a grounded question"
                                    value={question}
                                    onChange={e => setQuestion(e.target.value)}
                                    // Submit on Enter
                                    onKeyDown={e => {
                                        if (e.key === "Enter" && !e.shiftKey) {
                                            e.preventDefault();
                                            if (!isQuerying) handleAsk();
                                        }
                                    }}
                                    placeholder="Create a report based on the documents..."
                                    className="w-full bg-transparent border-0 focus:ring-0 resize-none py-4 pl-4 pr-24 min-h-[50px] max-h-[200px] text-base outline-none scrollbar-hide font-normal leading-relaxed placeholder:text-muted-foreground/50"
                                    style={{ height: "auto" }}
                                />
                                <div className="absolute bottom-2.5 right-3 flex items-center gap-2">
                                    {isQuerying && onCancel && (
                                        <Button
                                            size="sm"
                                            variant="ghost"
                                            onClick={onCancel}
                                            className="h-8 px-3 text-xs flex items-center gap-1.5 text-destructive hover:text-destructive hover:bg-destructive/10"
                                        >
                                            <XCircle className="h-3.5 w-3.5" data-icon="inline-start" />
                                            Cancel
                                        </Button>
                                    )}
                                    <Button
                                        aria-label={isQuerying ? "Query running" : "Ask grounded question"}
                                        size="icon"
                                        disabled={isQuerying || !question.trim()}
                                        onClick={() => handleAsk()}
                                        className={`h-8 w-8 transition-all ${question.trim() ? "bg-primary text-primary-foreground shadow-md hover:scale-105 active:scale-95" : "bg-muted text-muted-foreground"}`}
                                    >
                                        {isQuerying ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
                                    </Button>
                                </div>
                            </div>
                            {/* Quick Suggestion Chips - Show when no active query and input is empty */}
                            {!isQuerying && !question.trim() && history.length === 0 && documents.length > 0 && (
                                <div className="flex flex-wrap gap-2 mt-3">
                                    <span className="text-[10px] text-muted-foreground/70 w-full mb-1">Try asking:</span>
                                    {[
                                        "Summarize the main points",
                                        "What are the key findings?",
                                        "Compare the approaches",
                                        "List important dates"
                                    ].map((suggestion) => (
                                        <button
                                            key={suggestion}
                                            onClick={() => setQuestion(suggestion)}
                                            className="text-xs px-3 py-1.5 rounded-full bg-muted/50 text-muted-foreground hover:bg-primary/10 hover:text-primary border border-border/40 hover:border-primary/30 transition-all"
                                        >
                                            {suggestion}
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* Right Column: Sources/Pipeline */}
                <div className={`flex flex-col border-l border-border/40 bg-muted/10 h-full overflow-hidden transition-all duration-300 ${!showSources && "opacity-0 invisible w-0 border-none"}`}>
                    <div className="p-4 border-b border-border/40 flex items-center justify-between bg-muted/20">
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-2">
                            <FileText className="h-3 w-3" /> Sources
                        </h3>
                        <Button variant="ghost" size="icon" onClick={() => setShowSources(false)} aria-label="Hide sources panel" className="h-6 w-6">
                            <PanelRightClose className="h-3 w-3" />
                        </Button>
                    </div>

                    <div className="flex-1 overflow-y-auto p-4 space-y-4">
                        <WhyAnswerPanel
                            queryResult={queryResult}
                            sources={sourcesForDisplay}
                            queryMode={queryMode}
                            isQuerying={isQuerying}
                            activeStage={activeStage}
                            progress={progress}
                        />

                        {/* Pipeline Status Widget - Hide when complete */}
                        {isQuerying && (
                            <div className="rounded-lg border border-border/60 bg-background p-3 mb-4 shadow-sm animate-in slide-in-from-top duration-300">
                                <div className="flex items-center justify-between text-xs mb-3">
                                    <span className="font-bold text-foreground">Pipeline Execution</span>
                                    <span className={`px-2 py-0.5 rounded-full ${isQuerying ? "bg-primary/10 text-primary animate-pulse" : "bg-muted text-muted-foreground"}`}>
                                        {isQuerying ? (activeStage || "Running") : "Complete"}
                                    </span>
                                </div>
                                <div className="space-y-2">
                                    {(isQuerying ? liveStages : (queryResult?.steps || [])).map((stage: any) => {
                                        const stepName = isQuerying ? stage.name : stage.name;
                                        const stepLabel = isQuerying ? stage.label : (stage.name.charAt(0).toUpperCase() + stage.name.slice(1));

                                        const isActive = isQuerying && (activeStage === stepName || progress?.stage === stepName);
                                        const isDone = !isQuerying || completedStages.has(stepName);

                                        return (
                                            <div key={stepName} className="flex items-center justify-between text-[11px]">
                                                <div className={`flex items-center gap-2 ${isActive ? "text-primary font-medium" : isDone ? "text-foreground/80" : "text-muted-foreground"}`}>
                                                    {isActive ? (
                                                        <Loader2 className="h-3 w-3 animate-spin" />
                                                    ) : isDone ? (
                                                        <CheckCircle2 className="h-3 w-3 text-primary" />
                                                    ) : (
                                                        <div className="h-1.5 w-1.5 rounded-full bg-muted-foreground/30 ml-0.5 mr-1" />
                                                    )}
                                                    {stepLabel}
                                                </div>
                                                {isDone && !isActive && !isQuerying && (
                                                    <span className="text-[9px] font-mono opacity-60">
                                                        {queryResult?.steps?.find((s: any) => s.name === stepName)?.duration_ms?.toFixed(0)}ms
                                                    </span>
                                                )}
                                            </div>
                                        )
                                    })}
                                </div>
                            </div>
                        )}

                        {/* Sources List */}
                        {showSourcesSkeleton ? (
                            <div className="space-y-4">
                                <SkeletonBlock className="h-24 w-full" />
                                <SkeletonBlock className="h-24 w-full" />
                            </div>
                        ) : !hasSources ? (
                            <Empty className="min-h-[180px]">
                                <EmptyHeader>
                                    <EmptyMedia>
                                        <FileSearch data-icon="inline-start" />
                                    </EmptyMedia>
                                    <EmptyTitle>No sources yet</EmptyTitle>
                                    <EmptyDescription>
                                        Ask a question to reveal cited documents, relevance scores, and source previews.
                                    </EmptyDescription>
                                </EmptyHeader>
                            </Empty>
                        ) : (
                            <SourcesList
                                sources={sourcesForDisplay}
                                highlightedSourceId={highlightedSourceId}
                                onSourceClick={(s) => {
                                    const doc = documents.find(d => d.id === s.id || d.title === s.title);
                                    if (doc) setSelectedDoc(doc);
                                }}
                            />
                        )}

                        {/* Full Document Detail View */}
                        {selectedDoc && (
                            <SourceDetailPanel
                                doc={selectedDoc}
                                onClose={() => setSelectedDoc(null)}
                            />
                        )}

                        {/* Confidence Indicator - Show after query completes */}
                        {!isQuerying && queryResult?.confidence && (
                            <div className="mt-4">
                                <ConfidenceIndicator
                                    confidence={queryResult.confidence.overall}
                                    factors={queryResult.confidence.factors}
                                    hallucinationPass={queryResult.confidence.hallucination_pass}
                                    evidenceContractPass={queryResult.confidence.evidence_contract_pass}
                                />
                            </div>
                        )}
                    </div>
                </div>

                {/* Right Panel Toggle (Absolute if closed) */}
                {!showSources && (
                    <div className="absolute top-4 right-4 z-10 hidden xl:block">
                        <Button variant="ghost" size="icon" onClick={() => setShowSources(true)} aria-label="Show sources panel" className="h-8 w-8 text-muted-foreground hover:text-foreground bg-background/50 backdrop-blur border border-border/20">
                            <PanelRightOpen className="h-4 w-4" />
                        </Button>
                    </div>
                )}
            </div>
        </div >
    );
}
