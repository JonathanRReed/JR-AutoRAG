import { useMemo } from "react";
import {
    Activity,
    AlertTriangle,
    Brain,
    Box,
    CheckCircle2,
    Database,
    FileSearch,
    GitMerge,
    HelpCircle,
    Loader2,
    Network,
    Search,
    ShieldCheck,
    Sparkles,
    Target,
    Clock,
    Zap,
} from "lucide-react";
import type { PipelineStep } from "@/types";

interface PipelineTimelineProps {
    steps: PipelineStep[];
    totalDuration: number;
    activeStage?: string | null;
    isRunning?: boolean;
}

const STEP_ICONS: Record<string, typeof Target> = {
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

const STEP_COLORS: Record<string, { bg: string; border: string; text: string }> = {
    completed: { bg: "bg-primary", border: "border-primary", text: "text-primary" },
    passed: { bg: "bg-emerald-500", border: "border-emerald-500", text: "text-emerald-600" },
    repaired: { bg: "bg-amber-500", border: "border-amber-500", text: "text-amber-600" },
    failed: { bg: "bg-red-500", border: "border-red-500", text: "text-red-600" },
    running: { bg: "bg-secondary", border: "border-secondary", text: "text-secondary-foreground" },
    skipped: { bg: "bg-muted", border: "border-muted", text: "text-muted-foreground" },
    pending: { bg: "bg-muted/50", border: "border-border", text: "text-muted-foreground" },
};

const normalizeStatus = (status: string) => {
    if (status === "passed") return "passed";
    if (status === "repaired") return "repaired";
    if (status === "failed" || status === "error") return "failed";
    return status;
};

function formatDuration(ms: number): string {
    if (ms < 1) return "<1ms";
    if (ms < 1000) return `${ms.toFixed(0)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
}

function getDurationClass(ms: number): "fast" | "medium" | "slow" {
    if (ms < 100) return "fast";
    if (ms < 500) return "medium";
    return "slow";
}

function TimelineStep({
    step,
    totalDuration,
    isLast,
    isActive,
}: {
    step: PipelineStep;
    totalDuration: number;
    isLast: boolean;
    isActive: boolean;
}) {
    const Icon = STEP_ICONS[step.name] ?? Target;
    const statusClass = isActive ? "running" : normalizeStatus(step.status);
    const statusLabel = isActive ? "running" : step.status;
    const colors = STEP_COLORS[statusClass] || STEP_COLORS.pending;
    const percentage = totalDuration > 0 ? (step.duration_ms / totalDuration) * 100 : 0;
    const durationClass = getDurationClass(step.duration_ms);
    const isSuccess = ["completed", "passed", "repaired"].includes(statusClass);

    return (
        <div className="timeline-step group">
            {/* Timeline dot */}
            <div className={`timeline-dot ${statusClass}`}>
                {statusClass === "running" ? (
                    <Loader2 className="h-3 w-3 animate-spin text-primary-foreground" />
                ) : isSuccess ? (
                    <CheckCircle2 className="h-3 w-3 text-primary-foreground" />
                ) : (
                    <Icon className="h-3 w-3 text-muted-foreground" />
                )}
            </div>

            {/* Step content */}
            <div className={`step-card ${statusClass} rounded-lg border border-border/60 bg-card p-4 ml-2`}>
                {/* Header */}
                <div className="flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2 min-w-0">
                        <span className={`flex h-7 w-7 items-center justify-center rounded-lg ${colors.bg}/10`}>
                            <Icon className={`h-4 w-4 ${colors.text}`} />
                        </span>
                        <span className="font-medium capitalize text-foreground truncate">
                            {step.name}
                        </span>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                        <span className={`text-xs font-medium ${colors.text}`}>
                            {statusLabel === "skipped" ? "skipped" : statusLabel}
                        </span>
                        {step.duration_ms > 0 && (
                            <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-xs font-medium text-foreground">
                                <Clock className="h-3 w-3" />
                                {formatDuration(step.duration_ms)}
                            </span>
                        )}
                    </div>
                </div>

                {/* Duration bar */}
                {step.duration_ms > 0 && ["completed", "passed", "repaired"].includes(statusClass) && (
                    <div className="mt-3">
                        <div className="duration-bar">
                            <div
                                className={`duration-bar-fill ${durationClass}`}
                                style={{ width: `${Math.max(percentage, 3)}%` }}
                            />
                        </div>
                        <div className="mt-1 flex justify-between text-[10px] text-muted-foreground">
                            <span>{percentage.toFixed(1)}% of total</span>
                            <span>{formatDuration(step.duration_ms)}</span>
                        </div>
                    </div>
                )}

                {/* Step details (expandable in the future) */}
                {step.details && Object.keys(step.details).length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-1.5">
                        {step.details.total_chunks !== undefined && (
                            <span className="inline-flex items-center rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">
                                {step.details.total_chunks} chunks
                            </span>
                        )}
                        {step.details.query_type && (
                            <span className="inline-flex items-center rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">
                                {step.details.query_type}
                            </span>
                        )}
                        {step.details.model && (
                            <span className="inline-flex items-center rounded-full bg-muted px-2 py-0.5 text-[10px] font-mono text-muted-foreground">
                                {step.details.model}
                            </span>
                        )}
                        {step.details.quality && (
                            <span className="inline-flex items-center rounded-full bg-primary/10 px-2 py-0.5 text-[10px] text-primary">
                                {step.details.quality}
                            </span>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}

export function PipelineTimeline({
    steps,
    totalDuration,
    activeStage,
    isRunning = false,
}: PipelineTimelineProps) {
    const enhancedSteps = useMemo(() => {
        return steps.map(step => ({
            ...step,
            isActive: isRunning && activeStage === step.name,
        }));
    }, [steps, activeStage, isRunning]);

    if (steps.length === 0) {
        return (
            <div className="glass-card rounded-lg p-6 text-center">
                <div className="flex flex-col items-center gap-3">
                    <div className="flex h-12 w-12 items-center justify-center rounded-full bg-muted">
                        <Zap className="h-6 w-6 text-muted-foreground" />
                    </div>
                    <div>
                        <p className="font-medium text-foreground">No pipeline data</p>
                        <p className="text-sm text-muted-foreground">
                            Run a query to see the pipeline timeline
                        </p>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="glass-card rounded-lg p-4">
            {/* Header */}
            <div className="flex items-center justify-between gap-3 mb-4">
                <div className="flex items-center gap-2">
                    <div className="live-indicator">
                        {isRunning && <span className="live-dot" />}
                        <span className="font-semibold text-foreground">Pipeline Timeline</span>
                    </div>
                </div>
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                    <Clock className="h-4 w-4" />
                    <span className="font-medium">{formatDuration(totalDuration)} total</span>
                </div>
            </div>

            {/* Timeline */}
            <div className="space-y-0 stagger-in">
                {enhancedSteps.map((step, idx) => (
                    <TimelineStep
                        key={`${step.name}-${idx}`}
                        step={step}
                        totalDuration={totalDuration}
                        isLast={idx === enhancedSteps.length - 1}
                        isActive={step.isActive}
                    />
                ))}
            </div>

            {/* Summary bar */}
            <div className="mt-4 pt-4 border-t border-border/60">
                <div className="flex items-center justify-between text-xs text-muted-foreground">
                    <span>
                        {steps.filter(s => ["completed", "passed", "repaired"].includes(s.status)).length} of {steps.length} steps completed
                    </span>
                    {isRunning && activeStage && (
                        <span className="inline-flex items-center gap-1.5 text-secondary-foreground">
                            <Loader2 className="h-3 w-3 animate-spin" />
                            Running: {activeStage}
                        </span>
                    )}
                </div>
            </div>
        </div>
    );
}
