import { useState, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
    Activity,
    ClipboardList,
    Inbox,
    Timer,
    Info,
    Sparkles,
    Search,
    ChevronDown,
    ChevronUp,
    CheckCircle2,
    Clock,
    Filter,
    FileText,
} from "lucide-react";
import { GaugeRing, DurationBar } from "./LiveMetricsChart";
import type { TraceOut, PipelineStep } from "@/types";

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

function MiniTimeline({ steps }: { steps: PipelineStep[] }) {
    if (!steps || steps.length === 0) return null;

    const totalDuration = steps.reduce((sum, s) => sum + (s.duration_ms || 0), 0);

    return (
        <div className="flex items-center gap-1">
            {steps.map((step, idx) => {
                const width = totalDuration > 0
                    ? Math.max((step.duration_ms / totalDuration) * 100, 8)
                    : 100 / steps.length;
                const isComplete = step.status === "completed";
                const isSkipped = step.status === "skipped";

                return (
                    <div
                        key={idx}
                        className={`h-2 rounded-full transition-all ${isComplete ? "bg-primary" : isSkipped ? "bg-muted" : "bg-secondary"
                            }`}
                        style={{ width: `${width}%`, minWidth: 8 }}
                        title={`${step.name}: ${step.duration_ms?.toFixed(0) || 0}ms`}
                    />
                );
            })}
        </div>
    );
}

function TraceCard({ trace, formatNumber }: { trace: TraceOut; formatNumber: (value?: number) => string }) {
    const [expanded, setExpanded] = useState(false);
    const coverage = typeof trace.metrics.coverage === "number" ? trace.metrics.coverage : 0;
    const tokens = typeof trace.metrics.tokens === "number" ? trace.metrics.tokens : 0;
    const duration = typeof trace.metrics.duration_ms === "number" ? trace.metrics.duration_ms : 0;

    const coverageColor = coverage > 0.7
        ? "text-green-500"
        : coverage > 0.4
            ? "text-yellow-500"
            : "text-red-500";

    const qualityLabel = coverage > 0.7 ? "High" : coverage > 0.4 ? "Medium" : "Low";

    return (
        <div className="glass-card rounded-lg p-4 transition-all hover:shadow-lg">
            {/* Header row */}
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
                    className="shrink-0 flex items-center gap-1 rounded-md border border-border/60 px-2 py-1 text-xs font-medium text-muted-foreground hover:bg-muted/40 transition-colors"
                >
                    {expanded ? (
                        <>
                            Less <ChevronUp className="h-3 w-3" />
                        </>
                    ) : (
                        <>
                            More <ChevronDown className="h-3 w-3" />
                        </>
                    )}
                </button>
            </div>

            {/* Mini timeline */}
            {trace.steps && trace.steps.length > 0 && (
                <div className="mt-3">
                    <MiniTimeline steps={trace.steps} />
                </div>
            )}

            {/* Quick Metrics */}
            <div className="mt-3 flex flex-wrap items-center gap-3">
                <div className="flex items-center gap-2">
                    <GaugeRing value={coverage * 100} size={32} strokeWidth={3} showValue={false} />
                    <div className="text-xs">
                        <span className={`font-semibold ${coverageColor}`}>
                            {(coverage * 100).toFixed(0)}%
                        </span>
                        <span className="text-muted-foreground ml-1">coverage</span>
                    </div>
                </div>
                <span className="inline-flex items-center gap-1.5 rounded-full bg-muted px-2.5 py-1 text-xs text-muted-foreground">
                    <FileText className="h-3 w-3" />
                    {formatNumber(tokens)} tokens
                </span>
                {duration > 0 && (
                    <span className="inline-flex items-center gap-1.5 rounded-full bg-muted px-2.5 py-1 text-xs text-muted-foreground">
                        <Clock className="h-3 w-3" />
                        {duration.toFixed(0)}ms
                    </span>
                )}
                <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${coverage > 0.7 ? "bg-green-500/10 text-green-500" :
                    coverage > 0.4 ? "bg-yellow-500/10 text-yellow-500" :
                        "bg-red-500/10 text-red-500"
                    }`}>
                    {qualityLabel} quality
                </span>
            </div>

            {/* Expanded Details */}
            {expanded && (
                <div className="mt-4 space-y-4 border-t border-border/60 pt-4 fade-in">
                    <div>
                        <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">Full Answer</p>
                        <p className="mt-2 text-sm text-foreground whitespace-pre-wrap leading-relaxed">
                            {trace.answer}
                        </p>
                    </div>

                    {trace.steps && trace.steps.length > 0 && (
                        <div>
                            <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                                Pipeline Steps ({trace.steps.length})
                            </p>
                            <div className="mt-2 space-y-2">
                                {trace.steps.map((step, idx) => (
                                    <div
                                        key={idx}
                                        className={`step-card ${step.status === "error" ? "failed" : step.status} rounded-md border border-border/60 bg-card/50 p-3`}
                                    >
                                        <div className="flex items-center justify-between gap-2">
                                            <div className="flex items-center gap-2">
                                                {["completed", "passed", "repaired"].includes(step.status) ? (
                                                    <CheckCircle2 className="h-4 w-4 text-primary" />
                                                ) : step.status === "failed" ? (
                                                    <Activity className="h-4 w-4 text-destructive" />
                                                ) : (
                                                    <Activity className="h-4 w-4 text-muted-foreground" />
                                                )}
                                                <span className="font-medium capitalize text-foreground text-sm">
                                                    {step.name}
                                                </span>
                                            </div>
                                            <span className="text-xs text-muted-foreground">
                                                {step.duration_ms < 1 ? "<1ms" : `${step.duration_ms.toFixed(0)}ms`}
                                            </span>
                                        </div>
                                        {/* Step details - why this route */}
                                        {step.details && Object.keys(step.details).length > 0 && (
                                            <div className="mt-2 pt-2 border-t border-border/40">
                                                <div className="flex flex-wrap gap-2">
                                                    {Object.entries(step.details).slice(0, 4).map(([key, value]) => (
                                                        <span key={key} className="inline-flex items-center gap-1 text-[10px] text-muted-foreground bg-muted/50 px-1.5 py-0.5 rounded">
                                                            <span className="font-medium">{key}:</span>
                                                            <span>{typeof value === 'number' ? value.toFixed(2) : String(value).slice(0, 20)}</span>
                                                        </span>
                                                    ))}
                                                </div>
                                            </div>
                                        )}
                                    </div>
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
    const [searchQuery, setSearchQuery] = useState("");
    const [filterStatus, setFilterStatus] = useState<"all" | "high" | "medium" | "low">("all");

    const filteredTraces = useMemo(() => {
        let result = [...traces].reverse(); // newest first

        // Search filter
        if (searchQuery.trim()) {
            const q = searchQuery.toLowerCase();
            result = result.filter(t =>
                t.prompt.toLowerCase().includes(q) ||
                t.answer.toLowerCase().includes(q)
            );
        }

        // Quality filter
        if (filterStatus !== "all") {
            result = result.filter(t => {
                const coverage = typeof t.metrics.coverage === "number" ? t.metrics.coverage : 0;
                if (filterStatus === "high") return coverage > 0.7;
                if (filterStatus === "medium") return coverage > 0.4 && coverage <= 0.7;
                return coverage <= 0.4;
            });
        }

        return result;
    }, [traces, searchQuery, filterStatus]);

    return (
        <Card className="overflow-hidden">
            <CardHeader className="glass-panel">
                <CardTitle className="flex items-center gap-2">
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                        <ClipboardList className="h-5 w-5 text-primary" />
                    </div>
                    <span>Evaluation & Traces</span>
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
            <CardContent className="space-y-6 p-6">
                {/* Evaluation Section */}
                <div className="glass-card rounded-lg p-4">
                    <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                        <div>
                            <h4 className="font-semibold text-foreground">Quick Evaluation</h4>
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
                                <>
                                    <Sparkles className="mr-2 h-4 w-4" />
                                    Run Evaluation
                                </>
                            )}
                        </Button>
                    </div>

                    {evaluationSummary && (
                        <div className="mt-4 rounded-lg border border-border/60 bg-card/60 p-3">
                            <p className="text-sm font-medium text-foreground">{evaluationSummary}</p>
                        </div>
                    )}
                </div>

                {/* Search & Filters */}
                <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
                    <div className="relative flex-1">
                        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                        <input
                            type="text"
                            value={searchQuery}
                            onChange={(e) => setSearchQuery(e.target.value)}
                            placeholder="Search traces..."
                            className="w-full rounded-lg border border-border/60 bg-card pl-9 pr-4 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
                        />
                    </div>
                    <div className="flex items-center gap-2">
                        <Filter className="h-4 w-4 text-muted-foreground" />
                        <select
                            value={filterStatus}
                            onChange={(e) => setFilterStatus(e.target.value as typeof filterStatus)}
                            className="rounded-lg border border-border/60 bg-card px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-primary/50"
                        >
                            <option value="all">All Quality</option>
                            <option value="high">High ({'>'}70%)</option>
                            <option value="medium">Medium (40-70%)</option>
                            <option value="low">Low ({'<'}40%)</option>
                        </select>
                    </div>
                </div>

                {/* Traces Section */}
                <div className="space-y-3">
                    <div className="flex items-center justify-between">
                        <h4 className="font-semibold text-foreground">
                            Recent Traces
                            <span className="ml-2 rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">
                                {filteredTraces.length}
                            </span>
                        </h4>
                    </div>

                    {filteredTraces.length === 0 ? (
                        <div className="glass-card rounded-lg p-8 text-center fade-in">
                            <Inbox className="mx-auto h-10 w-10 text-muted-foreground" />
                            <p className="mt-3 font-medium text-foreground">
                                {traces.length === 0 ? "No traces yet" : "No matching traces"}
                            </p>
                            <p className="mt-1 text-sm text-muted-foreground">
                                {traces.length === 0
                                    ? "Run a query to capture one."
                                    : "Try adjusting your search or filters."
                                }
                            </p>
                            {traces.length === 0 && (
                                <div className="mt-4">
                                    <Button variant="secondary" size="sm" onClick={handleEvaluation}>
                                        <Sparkles className="mr-2 h-4 w-4" />
                                        Run quick evaluation
                                    </Button>
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="space-y-3 max-h-[600px] overflow-y-auto pr-2 stagger-in">
                            {filteredTraces.map(trace => (
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
