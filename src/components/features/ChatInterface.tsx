import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import type { PipelineStep, QueryResponse } from "@/types";

interface ChatInterfaceProps {
    question: string;
    setQuestion: (value: string) => void;
    isQuerying: boolean;
    handleAsk: () => void;
    queryResult: QueryResponse | null;
}

function StepIcon({ name, status }: { name: string; status: string }) {
    const iconClass = status === "completed" ? "text-green-600" : "text-amber-500";
    const icons: Record<string, string> = {
        planning: "1",
        retrieval: "2",
        generation: "3",
    };
    return (
        <span className={`flex h-6 w-6 items-center justify-center rounded-full bg-muted text-xs font-bold ${iconClass}`}>
            {icons[name] ?? "?"}
        </span>
    );
}

function StepDetails({ step }: { step: PipelineStep }) {
    const details = step.details;

    if (step.name === "planning") {
        const queries = (details.queries as string[] | undefined) ?? [];
        const targetTokens = details.target_tokens as number | undefined;
        const coverageTarget = details.coverage_target as number | undefined;
        return (
            <div className="mt-2 space-y-1 text-xs text-muted-foreground">
                <p>Target tokens: {targetTokens ?? "N/A"} | Coverage target: {coverageTarget ? (coverageTarget * 100).toFixed(0) : "N/A"}%</p>
                {queries.length > 0 && (
                    <div>
                        <p className="font-medium">Search queries:</p>
                        <ul className="ml-4 list-disc">
                            {queries.map((q, i) => <li key={i}>{q}</li>)}
                        </ul>
                    </div>
                )}
            </div>
        );
    }

    if (step.name === "retrieval") {
        const subQueries = (details.sub_queries as Array<{ query: string; chunks_found: number; duration_ms: number }> | undefined) ?? [];
        const totalChunks = details.total_chunks as number | undefined;
        const uniqueSources = details.unique_sources as number | undefined;
        return (
            <div className="mt-2 space-y-1 text-xs text-muted-foreground">
                <p>Found {totalChunks ?? 0} chunks from {uniqueSources ?? 0} sources</p>
                {subQueries.length > 0 && (
                    <div>
                        <p className="font-medium">Sub-queries:</p>
                        <ul className="ml-4 space-y-1">
                            {subQueries.map((sq, i) => (
                                <li key={i} className="flex items-center gap-2">
                                    <span className="text-foreground">{sq.query}</span>
                                    <span className="rounded bg-muted px-1">{sq.chunks_found} chunks</span>
                                    <span className="text-muted-foreground">{sq.duration_ms.toFixed(1)}ms</span>
                                </li>
                            ))}
                        </ul>
                    </div>
                )}
            </div>
        );
    }

    if (step.name === "generation") {
        const provider = details.provider as string | undefined;
        const model = details.model as string | undefined;
        const contextTokens = details.context_tokens as number | undefined;
        const fallback = details.fallback as boolean | undefined;
        const error = details.error as string | undefined;
        return (
            <div className="mt-2 space-y-1 text-xs text-muted-foreground">
                <p>
                    Provider: {provider ?? "none"}
                    {model && ` | Model: ${model}`}
                </p>
                <p>Context tokens: {contextTokens ?? "N/A"}</p>
                {fallback && <p className="text-amber-600">Using fallback (no LLM configured)</p>}
                {error && <p className="text-red-600">Error: {error}</p>}
            </div>
        );
    }

    return null;
}

function PipelinePanel({ steps, metrics }: { steps: PipelineStep[]; metrics: Record<string, number> }) {
    const [expanded, setExpanded] = useState(true);

    return (
        <div className="rounded-lg border border-blue-200 bg-blue-50/50 p-3">
            <button
                type="button"
                className="flex w-full items-center justify-between text-left"
                onClick={() => setExpanded(!expanded)}
            >
                <span className="text-sm font-semibold text-blue-800">
                    Pipeline Details
                    <span className="ml-2 font-normal text-blue-600">
                        ({metrics.duration_ms?.toFixed(0) ?? 0}ms total)
                    </span>
                </span>
                <span className="text-blue-600">{expanded ? "Hide" : "Show"}</span>
            </button>

            {expanded && (
                <div className="mt-3 space-y-3">
                    {steps.map((step, idx) => (
                        <div key={idx} className="rounded border border-border bg-white p-3">
                            <div className="flex items-center gap-2">
                                <StepIcon name={step.name} status={step.status} />
                                <span className="font-medium capitalize text-foreground">{step.name}</span>
                                <span className="ml-auto text-xs text-muted-foreground">{step.duration_ms.toFixed(1)}ms</span>
                            </div>
                            <StepDetails step={step} />
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}

export function ChatInterface({
    question,
    setQuestion,
    isQuerying,
    handleAsk,
    queryResult,
}: ChatInterfaceProps) {
    const [showEvidence, setShowEvidence] = useState(false);

    return (
        <Card>
            <CardHeader>
                <CardTitle>Ask a Question</CardTitle>
                <CardDescription>Send a query through the full RAG stack. See exactly what the AI does at each step.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
                <div className="flex flex-col gap-2 sm:flex-row">
                    <Input
                        value={question}
                        onChange={e => setQuestion(e.target.value)}
                        placeholder="Type your question here..."
                        onKeyDown={e => e.key === "Enter" && !isQuerying && handleAsk()}
                    />
                    <Button disabled={isQuerying} onClick={handleAsk}>
                        {isQuerying ? "Querying..." : "Run Query"}
                    </Button>
                </div>

                {isQuerying && (
                    <div className="flex items-center gap-2 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-800">
                        <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-amber-600 border-t-transparent" />
                        Processing your query through the RAG pipeline...
                    </div>
                )}

                {queryResult && (
                    <div className="space-y-4">
                        {/* Pipeline Steps */}
                        {queryResult.steps && queryResult.steps.length > 0 && (
                            <PipelinePanel steps={queryResult.steps} metrics={queryResult.metrics} />
                        )}

                        {/* Answer */}
                        <div className="rounded-lg border border-border p-4">
                            <p className="text-sm font-semibold text-muted-foreground">Answer</p>
                            <p className="mt-1 whitespace-pre-wrap text-base text-foreground">{queryResult.answer}</p>
                        </div>

                        {/* Evidence (collapsible) */}
                        <div className="rounded-lg border border-border p-4">
                            <button
                                type="button"
                                className="flex w-full items-center justify-between text-left"
                                onClick={() => setShowEvidence(!showEvidence)}
                            >
                                <span className="text-sm font-semibold text-muted-foreground">
                                    Evidence ({queryResult.chunks.length} chunks)
                                </span>
                                <span className="text-xs text-blue-600">{showEvidence ? "Hide" : "Show"}</span>
                            </button>
                            {showEvidence && (
                                <ul className="mt-3 space-y-2 text-sm">
                                    {queryResult.chunks.map(chunk => (
                                        <li key={chunk.id} className="rounded border border-border/50 bg-muted/30 p-2">
                                            <div className="flex items-center gap-2">
                                                <span className="font-medium text-foreground">{chunk.title}</span>
                                                <span className="rounded bg-green-100 px-1.5 py-0.5 text-xs text-green-700">
                                                    {(chunk.score * 100).toFixed(0)}% match
                                                </span>
                                            </div>
                                            <p className="mt-1 text-muted-foreground">{chunk.snippet.slice(0, 200)}...</p>
                                        </li>
                                    ))}
                                </ul>
                            )}
                        </div>

                        {/* Quick Stats */}
                        <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
                            <span>Chunks: {queryResult.metrics.chunks ?? 0}</span>
                            <span>Coverage: {((queryResult.metrics.coverage ?? 0) * 100).toFixed(0)}%</span>
                            <span>Tokens: {queryResult.metrics.tokens ?? 0}</span>
                            <span>Duration: {queryResult.metrics.duration_ms?.toFixed(0) ?? "N/A"}ms</span>
                        </div>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}
