import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Activity, ClipboardList, Inbox, Timer, Info, Sparkles } from "lucide-react";
import type { TraceOut } from "@/types";

interface TraceLogProps {
    isEvaluating: boolean;
    handleEvaluation: () => void;
    evaluationSummary: string;
    traces: TraceOut[];
    formatNumber: (value?: number) => string;
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

function TraceCard({ trace, formatNumber }: { trace: TraceOut; formatNumber: (value?: number) => string }) {
    const [expanded, setExpanded] = useState(false);
    const coverage = typeof trace.metrics.coverage === "number" ? trace.metrics.coverage : 0;
    const tokens = typeof trace.metrics.tokens === "number" ? trace.metrics.tokens : 0;
    const duration = typeof trace.metrics.duration_ms === "number" ? trace.metrics.duration_ms : 0;

    const coverageColor = coverage > 0.7
        ? "text-primary"
        : coverage > 0.4
            ? "text-secondary-foreground"
            : "text-destructive";

    return (
        <div className="rounded-lg border border-border/60 bg-card p-4 transition-colors">
            <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0">
                    <p className="font-medium text-foreground truncate">{trace.prompt}</p>
                    <p className="mt-1 text-sm text-muted-foreground line-clamp-2">
                        {trace.answer.slice(0, 150)}...
                    </p>
                </div>
                <button
                    type="button"
                    onClick={() => setExpanded(!expanded)}
                    className="shrink-0 rounded-md border border-border/60 px-2 py-1 text-xs font-medium text-muted-foreground hover:bg-muted/40"
                >
                    {expanded ? "Less" : "More"}
                </button>
            </div>

            {/* Quick Metrics */}
            <div className="mt-3 flex flex-wrap gap-2">
                <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${coverageColor} bg-current/10`}>
                    <Activity className="mr-1 h-3 w-3" />
                    {((coverage ?? 0) * 100).toFixed(0)}%
                </span>
                <span className="inline-flex items-center rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                    <ClipboardList className="mr-1 h-3 w-3" />
                    {formatNumber(tokens)} tokens
                </span>
                {duration > 0 && (
                    <span className="inline-flex items-center rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                        <Timer className="mr-1 h-3 w-3" />
                        {duration.toFixed(0)}ms
                    </span>
                )}
            </div>

            {/* Expanded Details */}
            {expanded && (
                    <div className="mt-4 space-y-3 border-t border-border/60 pt-4">
                    <div>
                        <p className="text-xs font-medium text-muted-foreground uppercase">Full Answer</p>
                        <p className="mt-1 text-sm text-foreground whitespace-pre-wrap">{trace.answer}</p>
                    </div>

                    {trace.steps && trace.steps.length > 0 && (
                        <div>
                            <p className="text-xs font-medium text-muted-foreground uppercase">Pipeline Steps</p>
                            <div className="mt-2 flex flex-wrap gap-2">
                                {trace.steps.map((step, idx) => (
                                    <span
                                        key={idx}
                                        className={`inline-flex items-center rounded-lg px-2 py-1 text-xs ${step.status === "completed"
                                                ? "bg-primary/10 text-primary"
                                                : step.status === "skipped"
                                                    ? "bg-muted text-muted-foreground"
                                                    : "bg-secondary/30 text-secondary-foreground"
                                            }`}
                                    >
                                        {step.name}: {step.duration_ms < 1 ? "<1ms" : `${step.duration_ms.toFixed(0)}ms`}
                                    </span>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

export function TraceLog({
    isEvaluating,
    handleEvaluation,
    evaluationSummary,
    traces,
    formatNumber,
}: TraceLogProps) {
    return (
        <Card>
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                    <ClipboardList className="h-4 w-4" />
                    Evaluation & Traces
                </CardTitle>
                <CardDescription>
                    Run evaluations and inspect query traces for debugging and optimization.
                </CardDescription>
                <div className="mt-3 flex flex-wrap gap-2">
                    <InlineHint label="Step 1: Run evaluation" detail="Smoke test sample questions to validate models and retrieval." />
                    <InlineHint label="Step 2: Inspect traces" detail="Review coverage, tokens, and timings per query." />
                    <InlineHint label="Step 3: Tune settings" detail="Adjust provider and retrieval settings to improve outcomes." />
                </div>
            </CardHeader>
            <CardContent className="space-y-6">
                {/* Evaluation Section */}
                <div className="rounded-lg border border-border/60 bg-muted/10 p-4">
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                        <div>
                            <h4 className="font-medium text-foreground">Quick Evaluation</h4>
                            <p className="text-sm text-muted-foreground">
                                Run a smoke test with sample questions
                            </p>
                        </div>
                        <Button
                            disabled={isEvaluating}
                            onClick={handleEvaluation}
                            variant="secondary"
                        >
                            {isEvaluating ? (
                                <>
                                    <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent mr-2" />
                                    Running...
                                </>
                            ) : (
                                "Run Evaluation"
                            )}
                        </Button>
                    </div>

                    {evaluationSummary && (
                        <div className="mt-4 rounded-lg border border-border/60 bg-card/60 p-3">
                            <p className="text-sm font-medium text-foreground">{evaluationSummary}</p>
                        </div>
                    )}
                </div>

                {/* Traces Section */}
                <div className="space-y-3">
                    <div className="flex items-center justify-between">
                        <h4 className="font-medium text-foreground">
                            Recent Traces
                            <span className="ml-2 rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                                {traces.length}
                            </span>
                        </h4>
                    </div>

                    {traces.length === 0 ? (
                        <div className="rounded-lg border border-dashed border-border/70 p-8 text-center">
                            <Inbox className="mx-auto h-8 w-8 text-muted-foreground" />
                            <p className="mt-2 text-muted-foreground">
                                No traces yet. Run a query to capture one.
                            </p>
                            <div className="mt-4 flex flex-col items-center gap-2 text-xs text-muted-foreground">
                                <div className="flex items-center gap-2">
                                    <span className="h-2 w-2 rounded-full bg-amber-500" />
                                    <span>Ensure documents and models are configured.</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <span className="h-2 w-2 rounded-full bg-emerald-500" />
                                    <span>Run a query, then refresh to see traces.</span>
                                </div>
                            </div>
                            <div className="mt-4">
                                <Button variant="secondary" size="sm" onClick={handleEvaluation}>
                                    <Sparkles className="mr-2 h-4 w-4" />
                                    Run quick evaluation
                                </Button>
                            </div>
                        </div>
                    ) : (
                        <div className="space-y-3 max-h-96 overflow-y-auto pr-2">
                            {traces.slice().reverse().map(trace => (
                                <TraceCard
                                    key={trace.id}
                                    trace={trace}
                                    formatNumber={formatNumber}
                                />
                            ))}
                        </div>
                    )}
                </div>
            </CardContent>
        </Card>
    );
}
